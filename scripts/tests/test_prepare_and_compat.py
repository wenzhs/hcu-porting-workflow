from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import adaptation_common as common
import cmake_fastpt_convert as cmake
import cutlass_to_hytlass_prepare as hytlass
import fastpt_non_transcode_prepare as nontranscode


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prepare fixtures ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "source"
        self.repo.mkdir()
        self.out = self.base / "new-output"


class NonTranscodePrepareTests(Fixture):
    def test_static_tokens_and_versioned_paths_are_bounded(self):
        original = 'CUDA::cudart_static my_cudart_static_probe CUDA::cudart_static_extra libcudart_static.a\n'
        new, _ = nontranscode.patch_text(original)
        self.assertEqual(new, 'CUDA::cudart my_cudart_static_probe CUDA::cudart_static_extra libcudart_static.a\n')
        original = 'include_dirs = ["/usr/local/cuda-12.4/include"]\n'
        new, _ = nontranscode.patch_text(original)
        self.assertEqual(new, original)  # No unexpanded shell variable introduced.
        new, _ = nontranscode.patch_text(original, cuda_home="/opt/dtk/cuda")
        self.assertEqual(new, 'include_dirs = ["/opt/dtk/cuda/include"]\n')

    def test_root_entry_and_capability_sources_are_separate(self):
        (self.repo / "setup.py").write_text("# CUDAExtension\n")
        vendor = self.repo / "vendor"
        vendor.mkdir()
        (vendor / "CMakeLists.txt").write_text("project(dep LANGUAGES CUDA)\n")
        (self.repo / "kernel.cu").write_text("cublasLtSomething();\n")
        profile = nontranscode.detect_profile(self.repo)
        self.assertTrue(profile["has_setup_py"])
        self.assertFalse(profile["has_cmake"])
        self.assertTrue(profile["gpufusion_cuda_libs"])
        with self.assertRaises(common.AdaptationError):
            nontranscode.infer_build_commands(profile, "/fake/fastpt")

    def test_parent_named_build_does_not_hide_setup(self):
        root = self.base / "build" / "project"
        root.mkdir(parents=True)
        (root / "setup.py").write_text("# CUDAExtension\n")
        self.assertTrue(nontranscode.detect_profile(root)["has_setup_py"])

    def test_dry_run_needs_no_fastpt_and_writes_nothing(self):
        (self.repo / "setup.py").write_bytes(b'# CUDAExtension\r\ninclude_dirs=["/usr/local/cuda/include"]\r\n#\xff\n')
        before = (self.repo / "setup.py").read_bytes()
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = nontranscode.main(["--repo", str(self.repo)])
        data = json.loads(stream.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(data["status"], "analysis-only")
        self.assertEqual(data["plan"]["plan_status"], "needs-information")
        self.assertEqual((self.repo / "setup.py").read_bytes(), before)
        self.assertEqual(set(p.name for p in self.repo.iterdir()), {"setup.py"})

    def write_args(self):
        return ["--repo", str(self.repo), "--out", str(self.out), "--fastpt", "/verified/fastpt", "--gpufusion-cuda-home", "/verified/gpufusion/cuda", "--arch", "gfx936", "--build-argv", '["cmake","--build","{BUILD_DIR}"]', "--fastpt-installed", "true", "--version-compatible", "true", "--fastpt-invocable", "true", "--write"]

    def test_prepare_preserves_bytes_and_records_not_built(self):
        source = b'# CUDAExtension\r\n# user byte \xff\r\n'
        (self.repo / "setup.py").write_bytes(source)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(nontranscode.main(self.write_args()), 0)
        self.assertEqual((self.repo / "setup.py").read_bytes(), source)
        self.assertEqual((self.out / "setup.py").read_bytes(), source)
        helper = (self.out / nontranscode.BUILD_SCRIPT).read_text()
        self.assertIn('exec "${BUILD_ARGV[@]}"', helper)
        self.assertIn('set -euo pipefail', helper)
        self.assertIn('FASTPT_ACTIVATION_APPROVED', helper)
        self.assertNotIn('import torch', helper)
        result = json.loads((self.out / nontranscode.PLAN_FILE).read_text())
        self.assertFalse(result["executed"])
        self.assertEqual(result["stages"]["build"], "not-run")

    def test_mismatch_unknown_entry_or_generated_collision_fail_before_output(self):
        (self.repo / "setup.py").write_text("# CUDAExtension\n")
        cases = [self.write_args() + ["--torch-version", "2.14.0", "--required-torch-version", "2.12"], self.write_args() + ["--build-argv", "[]"]]
        for args in cases:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(nontranscode.main(args), 2)
            self.assertFalse(self.out.exists())
        (self.repo / nontranscode.BUILD_SCRIPT).write_text("user script")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(nontranscode.main(self.write_args()), 2)
        self.assertFalse(self.out.exists())

    def test_static_runtime_change_requires_opt_in(self):
        (self.repo / "setup.py").write_text("# CUDAExtension\nlibraries=['cudart_static']\n")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(nontranscode.main(self.write_args()), 2)
        self.assertFalse(self.out.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(nontranscode.main(self.write_args() + ["--allow-dynamic-runtime"]), 0)
        self.assertIn("['cudart']", (self.out / "setup.py").read_text())
        self.assertIn("['cudart_static']", (self.repo / "setup.py").read_text())


class CMakePrepareTests(Fixture):
    def basic(self):
        (self.repo / "CMakeLists.txt").write_bytes(b'cmake_minimum_required(VERSION 3.21)\r\nproject(\r\n  sample\r\n  LANGUAGES CXX CUDA\r\n)\r\nadd_library(k k.cu)\r\ntarget_link_libraries(k PRIVATE CUDA::cudart_static CUDA::cublasLt)\r\n# bytes \xff\r\n')
        (self.repo / "k.cu").write_text("// original source\n")
        (self.repo / "k.hip").write_text("// actual converter output, not compiled\n")
        return {"k.cu": "k.hip"}

    def test_multiline_project_targets_and_arch_are_preserved(self):
        mapping = self.basic()
        before = (self.repo / "CMakeLists.txt").read_bytes()
        result = cmake.prepare(self.repo, self.out, arch="gfx936", source_map=mapping, write=True)
        data = common.read_text(self.out / "CMakeLists.txt")
        self.assertLess(data.index("CMAKE_HIP_ARCHITECTURES"), data.index("project("))
        self.assertIn("LANGUAGES CXX HIP", data)
        self.assertIn("hip::host", data)
        self.assertNotIn("hip::host_static", data)
        self.assertIn("roc::hipblaslt", data)
        self.assertNotIn("roc::hipblasLt", data)
        self.assertNotIn(" FORCE", data)
        self.assertIn("k.hip", data)
        self.assertIn("LANGUAGE HIP", data)
        self.assertFalse((self.out / cmake.SCRIPT_NAME).exists())
        self.assertEqual((self.repo / "CMakeLists.txt").read_bytes(), before)
        self.assertIn(b'\xff', (self.out / "CMakeLists.txt").read_bytes())
        self.assertEqual(result["status"], "prepared-not-built")
        self.assertEqual(result["stages"]["build"], "not-run")

    def test_identity_mapping_and_explicit_helper(self):
        (self.repo / "k.cu").write_text("// source kept with CUDA spelling\n")
        (self.repo / "CMakeLists.txt").write_text("project(p LANGUAGES CXX CUDA)\nadd_library(k k.cu)\n")
        cmake.prepare(self.repo, self.out, source_map={"k.cu": "k.cu"}, arch="gfx92a", write=True, emit_build_script=True)
        self.assertIn('"k.cu" PROPERTIES LANGUAGE HIP', (self.out / "CMakeLists.txt").read_text())
        helper = (self.out / cmake.SCRIPT_NAME).read_text()
        self.assertNotIn("cuda/env.sh", helper)
        self.assertIn("gfx92a", helper)
        self.assertIn("BUILD_DIR must stay", helper)

    def test_missing_mapping_and_unsupported_flags_do_not_write(self):
        self.basic()
        with self.assertRaises(common.AdaptationError):
            cmake.prepare(self.repo, self.out, arch="gfx936", write=True)
        self.assertFalse(self.out.exists())
        (self.repo / "CMakeLists.txt").write_text('project(p LANGUAGES CUDA)\nset(CMAKE_CUDA_FLAGS "-gencode arch=compute_80,code=sm_80")\n')
        result = cmake.prepare(self.repo, self.out, arch="gfx936")
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(self.out.exists())

    def test_conditional_package_stays_in_its_scope(self):
        source = 'project(p LANGUAGES CXX HIP)\nif(OPTIONAL)\n  find_package(CUDAToolkit QUIET)\n  target_link_libraries(k PRIVATE CUDA::cublas)\nendif()\n'
        new, warnings = cmake.transform_text(source, rel_path=Path('optional/CMakeLists.txt'), arch="gfx936")
        self.assertFalse(any(w.startswith('BLOCK:') for w in warnings), warnings)
        self.assertIn('find_package(hipblas QUIET)', new)
        self.assertGreater(new.index('find_package(hipblas'), new.index('if(OPTIONAL)'))
        self.assertNotIn('hipblas REQUIRED', new)

    def test_generator_expression_and_unknown_curand_mapping_block(self):
        for source in ('add_library(k $<$<BOOL:X>:k.cu>)\n', 'target_link_libraries(k PRIVATE CUDA::curand)\n'):
            new, diagnostics = cmake.transform_text(source, rel_path=Path('CMakeLists.txt'), arch="gfx936")
            self.assertEqual(new, source)
            self.assertTrue(any(d.startswith('BLOCK:') for d in diagnostics))

    def test_numeric_arch_force_and_mapping_escape_fail(self):
        self.basic()
        for kwargs in ({"arch": "936"}, {"force": True}, {"source_map": {"../k.cu": "k.hip"}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(common.AdaptationError):
                cmake.prepare(self.repo, self.out, write=True, **kwargs)
        self.assertFalse(self.out.exists())


class HytlassPrepareTests(Fixture):
    def test_risks_keep_all_original_signals(self):
        source = 'GIT_TAG v4.0.0\nusing A=cutlass::arch::Sm100; cutlass::gemm::device::Gemm<X> g; WGMMA;\n'
        diagnostics = hytlass.scan_risks(source, Path('kernel.cu'))
        codes = {d['code'] for d in diagnostics}
        self.assertTrue({'cutlass-v4','blackwell','gemm-api','specialized-hardware'}.issubset(codes))
        self.assertTrue(all(d['original'] for d in diagnostics))
        new = hytlass.transform_source('using A = cutlass::arch::Sm90;\n', 'gfx936')
        self.assertIn('Sm90', new)
        self.assertNotIn('Gfx936', new)

    def test_lexical_include_and_namespace_only(self):
        source = '#include <cutlass/foo.h>\n// cutlass::foo\nconst char* s = "cutlass::foo";\ncutlass::Thing t; cublasLtCreate();\n'
        new = hytlass.transform_source(source, 'gfx936')
        self.assertIn('#include <hytlass/foo.h>', new)
        self.assertIn('// cutlass::foo', new)
        self.assertIn('"cutlass::foo"', new)
        self.assertIn('hytlass::Thing', new)
        self.assertIn('cublasLtCreate', new)
        self.assertEqual(hytlass.transform_dependency_text('url = https://github.com/NVIDIA/cutlass.git\n', 'other'), 'url = https://github.com/NVIDIA/cutlass.git\n')

    def make_dependency(self):
        dep = self.base / 'dependency'
        (dep / 'include' / 'hytlass').mkdir(parents=True)
        (dep / 'include' / 'hytlass' / 'foo.h').write_text('// synthetic header\n')
        (self.repo / 'kernel.cu').write_text('#include <cutlass/foo.h>\ncutlass::Thing t;\n')
        (self.repo / 'CMakeLists.txt').write_text('project(p LANGUAGES HIP)\nadd_library(k kernel.cu)\n')
        return dep

    def test_header_mode_is_explicit_and_does_not_fetch(self):
        dep = self.make_dependency()
        result = hytlass.prepare(self.repo, self.out, arch='gfx936', hytlass_root=dep, hytlass_version='0.1.0', integration_mode='headers', target='k', write=True)
        text = (self.out / 'CMakeLists.txt').read_text()
        self.assertIn('target_include_directories(k PRIVATE', text)
        self.assertNotIn('FetchContent', text)
        self.assertNotIn('add_subdirectory', text)
        self.assertNotIn('CMAKE_HIP_ARCHITECTURES "936"', text)
        self.assertTrue(result['pending_semantic_review'])
        self.assertEqual(result['status'], 'prepared-pending-semantic-review')
        self.assertEqual(result['stages']['build'], 'not-run')
        self.assertTrue((self.out / hytlass.REPORT_NAME).exists())

    def test_unpinned_or_original_risky_input_refuses_before_copy(self):
        dep = self.make_dependency()
        with self.assertRaises(common.AdaptationError):
            hytlass.prepare(self.repo, self.out, hytlass_root=dep, hytlass_version='main', integration_mode='headers', target='k', write=True)
        self.assertFalse(self.out.exists())
        (self.repo / 'kernel.cu').write_text('cutlass::arch::Sm90; WGMMA;\n')
        with self.assertRaises(common.AdaptationError):
            hytlass.prepare(self.repo, self.out, hytlass_root=dep, hytlass_version='0.1.0', integration_mode='headers', target='k', write=True)
        self.assertFalse(self.out.exists())

    def test_explicit_build_helper_uses_same_shared_interface(self):
        dep = self.make_dependency()
        result = hytlass.prepare(self.repo, self.out, arch='gfx936', hytlass_root=dep, hytlass_version='0.1.0', integration_mode='headers', target='k', write=True, emit_build_script=True, build_command=['cmake', '--build', 'build_hytlass'])
        self.assertEqual(result['stages']['build'], 'not-run')
        self.assertIn('exec cmake --build', (self.out / hytlass.BUILD_SCRIPT).read_text())


class ShellStatusTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('bash'), 'Bash unavailable')
    def test_outer_pipefail_preserves_real_process_exit(self):
        result = subprocess.run(['bash', '-c', 'set -o pipefail; (exit 7) | tee /dev/null'], capture_output=True, check=False)
        self.assertEqual(result.returncode, 7)


if __name__ == '__main__':
    unittest.main()
