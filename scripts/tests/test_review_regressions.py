"""Regression cases for the independently confirmed review findings.

Only synthetic projects, inert bytecode markers and temporary output directories
are used. No FastPT activation, project build, installation or GPU work occurs.
"""
from __future__ import annotations

import copy
import importlib.util
import importlib._bootstrap_external
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
import adaptation_common as common
import build_skill_packages as exporter
import cmake_fastpt_convert as cmake
import cutlass_to_hytlass_prepare as hytlass
import fastpt_non_transcode_prepare as nontranscode
import route_preflight
from routes import transcode


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="review regressions ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "source"
        self.repo.mkdir()
        self.out = self.base / "output"


class PackageGateRegressions(Fixture):
    def setUp(self):
        super().setUp()
        self.packages = self.base / "packages"
        exporter.export_packages(self.packages)
        (self.repo / "CMakeLists.txt").write_text("project(p LANGUAGES CUDA)\nadd_library(k k.cu)\n")
        (self.repo / "k.cu").write_text("// original\n")
        (self.repo / "k.hip").write_text("// mapped\n")
        self.mapping = self.base / "mapping.json"
        self.mapping.write_text(json.dumps({"k.cu": "k.hip"}))

    def call(self, name, script, *args, isolated=True):
        command = [sys.executable, "-B"] + (["-I"] if isolated else [])
        command += [str(self.packages / name / "scripts" / script), *args]
        return subprocess.run(command, cwd=self.repo, capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_PKG001_conflicting_valid_bytecode_never_executes(self):
        for name, module in (("hcu-native-adaptation", "adaptation_common"), ("hcu-transcode", "package_integrity")):
            folder = self.packages / name
            source = folder / "scripts" / (module + ".py")
            marker = "UNMANIFESTED_BYTECODE_EXECUTED"
            code = compile(source.read_text(encoding="utf-8") + f'\nprint("{marker}")\n', str(source), "exec")
            info = source.stat()
            data = importlib._bootstrap_external._code_to_timestamp_pyc(code, int(info.st_mtime), info.st_size)
            cache = Path(importlib.util.cache_from_source(str(source)))
            cache.parent.mkdir(exist_ok=True)
            cache.write_bytes(data)
            with self.subTest(module=module):
                result = self.call(name, "route_preflight.py", "--verify-package-only")
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertNotIn(marker, result.stdout + result.stderr)
                self.assertIn("Bytecode", result.stderr)
                with self.assertRaises(ValueError):
                    route_preflight.verify_package(folder)

    def test_PKG002_every_public_cli_checks_integrity_before_local_import(self):
        for name in exporter.NAMES:
            folder = self.packages / name
            common_file = folder / "scripts/adaptation_common.py"
            with common_file.open("a", encoding="utf-8") as stream:
                stream.write('\nprint("CORRUPTED_COMMON_LOADED")\nSCHEMA_VERSION = "999"\n')
            for relative in exporter.PUBLIC_CLIS:
                if not (folder / relative).is_file():
                    continue
                result = self.call(name, Path(relative).name, "--help")
                with self.subTest(name=name, script=relative):
                    self.assertEqual(result.returncode, 2)
                    self.assertNotIn("CORRUPTED_COMMON_LOADED", result.stdout + result.stderr)
                    self.assertIn("hash mismatch", result.stderr)

    def test_PKG002_corrupted_legacy_write_creates_no_directory(self):
        name = "hcu-transcode"
        common_file = self.packages / name / "scripts/adaptation_common.py"
        with common_file.open("a", encoding="utf-8") as stream:
            stream.write('\nSCHEMA_VERSION = "999"\n')
        result = self.call(name, "cmake_fastpt_convert.py", "--repo", str(self.repo), "--out", str(self.out), "--source-map", str(self.mapping), "--arch", "gfx936", "--write")
        self.assertEqual(result.returncode, 2)
        self.assertIn("hash mismatch", result.stderr)
        self.assertFalse(self.out.exists())

    def test_all_exported_clis_require_isolated_invocation(self):
        result = self.call("hcu-native-adaptation", "route_preflight.py", "--help", isolated=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires python -I", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_PATH001_stream_output_is_rejected_without_write(self):
        stream_name = str(self.repo) + ":preflight-review.json"
        with self.assertRaises(common.AdaptationError):
            common.validate_output_path(self.repo, stream_name)
        with self.assertRaises(common.AdaptationError):
            common.safe_write_text(self.repo, "input.txt:stream", "must not write")
        result = self.call("hcu-native-adaptation", "route_preflight.py", "--repo", str(self.repo), "--json-out", stream_name)
        self.assertEqual(result.returncode, 2)
        self.assertIn("stream", result.stderr.lower())
        self.assertFalse(Path(stream_name).exists())
        # Normal drive-qualified Windows paths remain valid (also a POSIX control).
        self.assertEqual(common.resolve_repo(self.repo), self.repo.resolve())


class SnapshotRegressions(Fixture):
    def test_PATH002_inventory_change_before_identity_is_rejected(self):
        (self.repo / "one.hip").write_text("one")
        original = common.source_identity
        def insert_file(path):
            (self.repo / "new-input.hip").write_text("added during inventory")
            return original(path)
        with patch.object(common, "source_identity", side_effect=insert_file):
            with self.assertRaises(common.AdaptationError):
                common.prepare_tree(self.repo, self.out, route="native-hip")
        self.assertFalse(self.out.exists())
        self.assertEqual((self.repo / "new-input.hip").read_text(), "added during inventory")

    def test_PATH002_added_member_during_copy_is_not_published(self):
        (self.repo / "one.hip").write_text("one")
        original = common.shutil.copy2
        def copy_and_add(src, dst, **kwargs):
            result = original(src, dst, **kwargs)
            (self.repo / "late.hip").write_text("late")
            return result
        with patch.object(common.shutil, "copy2", side_effect=copy_and_add):
            with self.assertRaises(common.AdaptationError):
                common.prepare_tree(self.repo, self.out, route="native-hip")
        self.assertFalse(self.out.exists())

    def test_R2_stale_cmake_analysis_does_not_overwrite_copied_change(self):
        config = self.repo / "CMakeLists.txt"
        config.write_text("project(p LANGUAGES CUDA)\nadd_library(k k.cu)\n")
        (self.repo / "k.cu").write_text("// original")
        (self.repo / "k.hip").write_text("// mapped")
        original = common.prepare_tree
        def changed_source(*args, **kwargs):
            with config.open("a") as stream:
                stream.write("target_compile_definitions(k PRIVATE USER_CHANGE)\n")
            return original(*args, **kwargs)
        with patch.object(common, "prepare_tree", side_effect=changed_source):
            with self.assertRaises(common.AdaptationError):
                cmake.prepare(self.repo, self.out, source_map={"k.cu": "k.hip"}, arch="gfx936", write=True)
        self.assertIn("USER_CHANGE", (self.out / "CMakeLists.txt").read_text())
        self.assertIn("LANGUAGES CUDA", (self.out / "CMakeLists.txt").read_text())
        self.assertFalse((self.out / cmake.MANIFEST_NAME).exists())

    def test_R2_new_hytlass_risks_cannot_be_erased_by_old_patch(self):
        config = self.repo / "CMakeLists.txt"
        config.write_text("project(p LANGUAGES HIP)\nadd_library(k k.cu)\n")
        kernel = self.repo / "k.cu"
        kernel.write_text("cutlass::Thing value;\n")
        dep = self.base / "dependency"
        (dep / "include/hytlass").mkdir(parents=True)
        original = common.prepare_tree
        def changed_source(*args, **kwargs):
            kernel.write_text("cutlass::Thing value; cutlass::arch::Sm90; WGMMA;\n")
            return original(*args, **kwargs)
        with patch.object(common, "prepare_tree", side_effect=changed_source):
            with self.assertRaises(common.AdaptationError):
                hytlass.prepare(self.repo, self.out, hytlass_root=dep, hytlass_version="0.1.0", integration_mode="headers", target="k", write=True)
        copied = (self.out / "k.cu").read_text()
        self.assertIn("Sm90", copied)
        self.assertIn("WGMMA", copied)
        self.assertFalse((self.out / hytlass.REPORT_NAME).exists())


class RoutingAndCMakeRegressions(Fixture):
    def request(self):
        return {"repo": str(self.repo), "arch": "gfx936", "build_command": ["cmake", "--build", "build"], "capabilities": {"hip": {"installed": True, "compiler": "hipcc"}, "converters": {"fastpt": {key: True for key in transcode.CAPABILITY_FIELDS}}}}

    def test_R1_local_and_split_provider_version_mismatches_block(self):
        (self.repo / "k.cu").write_text("// host-only CUDA source")
        for placement in ("local", "split-actual", "split-required", "conflicting-general"):
            for explicit in (True, False):
                request = self.request()
                specific = request["capabilities"]["converters"]["fastpt"]
                if placement == "local":
                    specific.update(torch_version="2.14.0", required_torch_version="2.12")
                elif placement == "split-actual":
                    specific["torch_version"] = "2.14.0"
                    request["capabilities"]["fastpt"] = {"required_torch_version": "2.12"}
                elif placement == "split-required":
                    specific["required_torch_version"] = "2.12"
                    request["capabilities"]["fastpt"] = {"torch_version": "2.14.0"}
                else:
                    specific.update(torch_version="2.12.0", required_torch_version="2.12")
                    request["capabilities"]["fastpt"] = {"torch_version": "2.14.0", "required_torch_version": "2.12"}
                if explicit:
                    request["provider"] = "fastpt"
                before = copy.deepcopy(request)
                result = transcode.plan(request)
                with self.subTest(placement=placement, explicit=explicit):
                    self.assertEqual(result["plan_status"], "blocked")
                    self.assertEqual(request, before)
                    self.assertFalse(result["executed"])

    def test_R7_multiline_and_extension_only_cuda_are_auto_candidates(self):
        for declaration in ("project(p LANGUAGES\n CXX\n CUDA\n)\n", "enable_language(\n CUDA\n)\n", ""):
            config = self.repo / "CMakeLists.txt"
            config.write_text(declaration)
            (self.repo / "host.cu").write_text("// no kernel/API markers\n")
            result = route_preflight.plan_request({"repo": str(self.repo)})
            self.assertTrue(result["scan"]["cuda_candidate"])
            self.assertEqual(result["route"], "transcode")
            self.assertEqual(result["plan_status"], "needs-information")

    def test_R4_cross_directory_source_language_uses_owning_target_scope(self):
        (self.repo / "CMakeLists.txt").write_text("project(p LANGUAGES CXX CUDA)\nadd_library(k STATIC)\nadd_subdirectory(src)\n")
        child = self.repo / "src"
        child.mkdir()
        (child / "CMakeLists.txt").write_text("target_sources(k PRIVATE k.cu)\n")
        (child / "k.cu").write_text("// original")
        for extension in ("cu", "cpp", "hip"):
            (child / ("k." + extension)).write_text("// actual mapped source")
            output = self.base / ("out-" + extension)
            cmake.prepare(self.repo, output, source_map={"src/k.cu": "src/k." + extension}, arch="gfx936", write=True)
            text = (output / "src/CMakeLists.txt").read_text()
            self.assertIn('"${CMAKE_CURRENT_SOURCE_DIR}/k.' + extension + '" TARGET_DIRECTORY k PROPERTIES LANGUAGE HIP', text)

    def test_R5_both_orders_and_split_statements_never_override_existing_hip(self):
        for declaration in (
            "set_target_properties(k PROPERTIES HIP_ARCHITECTURES gfx90a CUDA_ARCHITECTURES 80)\n",
            "set_target_properties(k PROPERTIES CUDA_ARCHITECTURES 80 HIP_ARCHITECTURES gfx90a)\n",
            "set_target_properties(k PROPERTIES HIP_ARCHITECTURES gfx90a)\nset_target_properties(k PROPERTIES CUDA_ARCHITECTURES 80)\n",
            "set_property(TARGET k PROPERTY HIP_ARCHITECTURES gfx90a)\nset_target_properties(k PROPERTIES CUDA_ARCHITECTURES 80)\n",
        ):
            text, diagnostics = cmake.transform_text(declaration, rel_path=Path("CMakeLists.txt"), arch="gfx936")
            self.assertEqual(text, declaration)
            self.assertTrue(any("BLOCK:" in d and "Existing HIP_ARCHITECTURES" in d for d in diagnostics))
        original = "set_target_properties(k PROPERTIES HIP_ARCHITECTURES gfx90a)\n"
        text, diagnostics = cmake.transform_text(original, rel_path=Path("CMakeLists.txt"), arch="gfx936")
        self.assertEqual(text, original)
        self.assertFalse(diagnostics)

    def test_R5_cross_file_arch_conflict_is_blocked(self):
        (self.repo / "CMakeLists.txt").write_text("project(p LANGUAGES HIP)\nset_target_properties(k PROPERTIES HIP_ARCHITECTURES gfx90a)\n")
        (self.repo / "other.cmake").write_text("set_target_properties(k PROPERTIES CUDA_ARCHITECTURES 80)\n")
        with self.assertRaises(common.AdaptationError):
            cmake.prepare(self.repo, self.out, arch="gfx936", write=True)
        self.assertFalse(self.out.exists())

    def test_R6_cuda_arch_options_block_but_hip_and_ordinary_options_do_not(self):
        for flag in ("-arch=sm_80", "-arch sm_80", "--gpu-architecture=compute_80", "--gpu-architecture compute_80", "--cuda-gpu-arch=sm_80", "-gencode arch=compute_80,code=sm_80", '"SHELL:-arch=sm_80"', '"SHELL:--gpu-architecture compute_80"', '"-O3;-arch=sm_80"', '"SHELL:--gpu-code=sm_80"'):
            original = f"target_compile_options(k PRIVATE {flag})\n"
            text, diagnostics = cmake.transform_text(original, rel_path=Path("CMakeLists.txt"), arch="gfx936")
            self.assertEqual(text, original)
            self.assertTrue(any(d.startswith("BLOCK:") for d in diagnostics), flag)
        for flag in ("-O3", "--offload-arch=gfx936", "-march=native", '"SHELL:-O3"', '"-O3;--offload-arch=gfx936"', '"SHELL:--offload-arch=gfx936 -O3"'):
            original = f"target_compile_options(k PRIVATE {flag})\n"
            text, diagnostics = cmake.transform_text(original, rel_path=Path("CMakeLists.txt"), arch="gfx936")
            self.assertEqual(text, original)
            self.assertFalse(diagnostics, flag)


class HelperCwdRegression(Fixture):
    @unittest.skipUnless(shutil.which("bash"), "Bash unavailable")
    def test_R3_relative_target_executable_reaches_authorization_guard(self):
        target = self.repo / "subproject"
        target.mkdir()
        (target / "build.sh").write_text("#!/usr/bin/env bash\nprintf BUILD_SHOULD_NOT_RUN\n")
        (target / "build.sh").chmod(0o755)
        launcher = self.base / "fake-fastpt"
        launcher.write_text("printf ACTIVATION_SHOULD_NOT_RUN\n")
        gpu = self.base / "gpu"
        (gpu / "include").mkdir(parents=True)
        (gpu / "lib").mkdir()
        workspace = common.prepare_tree(self.repo, self.out, route="non-transcode")
        request = {"target": "subproject", "build_command": ["./build.sh"], "gpufusion_cuda_home": gpu.as_posix(), "capabilities": {"fastpt": {"installed": True, "version_compatible": True, "invocable": True}}}
        nontranscode.write_build_script(workspace, {}, launcher.as_posix(), request=request)
        env = dict(os.environ, FASTPT_ACTIVATION_APPROVED="0", FASTPT_ISOLATED_INSTALL="0")
        result = subprocess.run(["bash", str(workspace / nontranscode.BUILD_SCRIPT)], env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Activation requires separate user authorization", result.stderr)
        self.assertNotIn("Build executable is missing", result.stderr)
        self.assertNotIn("SHOULD_NOT_RUN", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
