from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import adaptation_common as common
import adaptation_ledger as ledger
import probe_environment as probe
import route_preflight
import source_risk_scan as scanner

KERNEL = r"""
#include <cublasLt.h>
#include <cublas_v2.h>
// __shfl_sync(0xffffffff, ignored_in_comment, 0) CUDA_VERSION
/* __ballot_sync(0xffffffff, hidden) */
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
#endif
__device__ float reduce(float v, int tid) {
  int lane = tid % 32;
  v += __shfl_down_sync(0xffffffff, v, 1);
  v += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, v, 2);
  v += __shfl(v, 0);
  int w = warpSize;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
  int x = __builtin_amdgcn_ds_bpermute(0, lane);
  return v + lane + x + w;
}
__global__ void leaf(float* p) { p[0] = 1.f; }
__global__ void parent(float* p) {
  leaf<<<1, 1>>>(p);
}
void host(float* p) { leaf<<<1, 1>>>(p); }
"""

CMAKE = """\
# CUDA_HOME only in a comment
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_HIP_ARCHITECTURES gfx936)
target_link_libraries(app CUDA::cudart_static)
add_library(foo STATIC foo.cu)
add_executable(bar main.cc)
option(ENABLE_EXTRA "extra" ON)
"""


def tree_digest(root: Path) -> str:
    return hashlib.sha256(json.dumps({p.relative_to(root).as_posix(): p.read_bytes().hex() for p in sorted(root.rglob("*")) if p.is_file()}).encode()).hexdigest()


def make_project(base: Path) -> Path:
    project = base / "project"
    project.mkdir()
    (project / "k.cu").write_text(KERNEL, encoding="utf-8")
    (project / "CMakeLists.txt").write_text(CMAKE, encoding="utf-8")
    (project / "mix.cu").write_text('#include <hip/hip_runtime.h>\n#include <cuda_runtime.h>\n', encoding="utf-8")
    (project / "setup.py").write_text("ext = CUDAExtension(name='pkg._C', sources=['a.cu'])\n", encoding="utf-8")
    (project / "third_party").mkdir()
    (project / "third_party" / "dep.cu").write_text("__global__ void d() { asm volatile(\"mov.u32 %0, %%tid.x;\" : \"=r\"(x)); }\n", encoding="utf-8")
    (project / "blob.cu").write_bytes(b"\x00\x01binary")
    return project


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scan tests ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.project = make_project(self.base)
        self.before = tree_digest(self.project)
        self.result = scanner.scan_risks(self.project)

    def found(self, category, subtype=None, **where):
        return [f for f in self.result["findings"] if f["category"] == category and (subtype is None or f["subtype"] == subtype) and all(f.get(k) == v for k, v in where.items())]

    def test_read_only_and_identity_bound(self):
        self.assertEqual(tree_digest(self.project), self.before)
        self.assertFalse(self.result["rewrites"])
        self.assertEqual(self.result["source_sha256"], common.source_identity(self.project)["tree_sha256"])
        self.assertIn("No hits is not proof", self.result["coverage"])

    def test_comments_are_ignored_but_real_code_is_found(self):
        warp_lines = {f["line"] for f in self.found("warp-sync")}
        text = KERNEL.split("\n")
        self.assertTrue(all("ignored_in_comment" not in text[line - 1] and "hidden" not in text[line - 1] for line in warp_lines))
        self.assertEqual({f["subtype"] for f in self.found("warp-sync")}, {"__shfl_down_sync", "__shfl_xor_sync", "__shfl"})
        self.assertEqual(len(self.found("version-macro", "CUDA_VERSION", file="k.cu")), 1)

    def test_mask_width_is_classified_not_rewritten(self):
        masks = {f["subtype"]: f["extra"] for f in self.found("warp-sync")}
        self.assertEqual(masks["__shfl_down_sync"]["mask"], "full-32bit-literal")
        self.assertEqual(masks["__shfl_xor_sync"]["mask"], "full-64bit-literal")
        self.assertEqual(masks["__shfl"]["form"], "legacy")
        self.assertNotIn("mask", masks["__shfl"])
        self.assertIn("mask width", masks["__shfl_down_sync"]["note"])

    def test_warp_size_asm_and_builtin_candidates(self):
        self.assertTrue(self.found("warp-size", "warpSize"))
        self.assertTrue(self.found("warp-size", "numeric-near-warp"))
        self.assertEqual(len(self.found("inline-asm", "ptx")), 1)
        self.assertTrue(self.found("builtin"))

    def test_device_side_launch_not_host_launch(self):
        launches = self.found("dynamic-parallelism", "device-side-launch")
        self.assertEqual(len(launches), 1)
        self.assertEqual(KERNEL.split("\n")[launches[0]["line"] - 1].strip(), "leaf<<<1, 1>>>(p);")
        self.assertEqual(launches[0]["routes"], ["non-transcode"])
        rdc = self.found("dynamic-parallelism", "rdc-flag")
        self.assertEqual(len(rdc), 1)
        self.assertIn("separate compilation", rdc[0]["note"])

    def test_route_filter_removes_route_specific_items(self):
        # GPUFusion-specific categories removed from native route
        native = scanner.scan_risks(self.project, route="native-hip")
        categories = {f["category"] for f in native["findings"]}
        self.assertNotIn("dynamic-parallelism", categories)
        self.assertNotIn("static-runtime", categories)
        self.assertIn("warp-sync", categories)
        with self.assertRaises(common.AdaptationError):
            scanner.scan_risks(self.project, route="bogus")

    def test_dependencies_and_binary_files(self):
        self.assertFalse([f for f in self.result["findings"] if f["file"].startswith("third_party")])
        self.assertIn("blob.cu", {item["file"] for item in self.result["skipped"]})
        deep = scanner.scan_risks(self.project, include_dependencies=True)
        self.assertTrue([f for f in deep["findings"] if f["file"].startswith("third_party")])

    def test_target_and_limits_are_validated(self):
        with self.assertRaises(common.AdaptationError):
            scanner.scan_risks(self.project, target="../outside")
        capped = scanner.scan_risks(self.project, max_findings=3)
        self.assertTrue(capped["truncated"])
        self.assertEqual(len(capped["findings"]), 3)

    def test_cli_report_goes_to_a_new_file_outside_the_project(self):
        out = self.base / "report.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(scanner.main(["--repo", str(self.project), "--json-out", str(out)]), 0)
        self.assertEqual(json.loads(out.read_text())["kind"], "risk-scan")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(scanner.main(["--repo", str(self.project), "--json-out", str(out)]), 2)
            self.assertEqual(scanner.main(["--repo", str(self.project), "--json-out", str(self.project / "inside.json")]), 2)
        self.assertEqual(tree_digest(self.project), self.before)


WAVE_KERNEL = r"""
#include <cuda_runtime.h>
__global__ void bad(float* out, const float* in, int n) {
  int tid = threadIdx.x;
  int lid = tid % 32;
  float v = in[tid];
  unsigned m = __ballot_sync(0xffffffff, v > 0);
  int cnt = __popc(m);
  unsigned bit = 1u << lid;
  v += __shfl_xor_sync(0xffffffff, v, 16, 32);
  float w = __shfl_down_sync(__activemask(), v, 1);
  if (tid >= n) return;
  __syncthreads();
  out[tid] = v + w + cnt + bit;
}
__global__ void good(float* out, const float* in) {
  auto tile = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  unsigned long long all = __ballot_sync(~0ull, in[0] > 0);
  float v = __shfl_down_sync(~0ull, in[0], 1, 32);
  int c = __popcll(all);
  unsigned long long b = 1ull << threadIdx.x;
  tile.sync();
  out[0] = v + c + b;
}
"""

RAW_STRING = 'const char* s = R"x(a " b)x"; __device__ void f() { __syncwarp(); } // "\n'
PLAIN = "int pad(int x) { return x % 32; }\n"
BRACKET_CMAKE = "#[[\nset(CMAKE_CUDA_ARCHITECTURES 80)\n]]\nset(CMAKE_HIP_ARCHITECTURES gfx936)\n"


class WaveSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wave tests ")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / "project"
        self.project.mkdir()
        (self.project / "wave.cu").write_text(WAVE_KERNEL, encoding="utf-8")
        (self.project / "raw.cu").write_text(RAW_STRING, encoding="utf-8")
        (self.project / "plain.cu").write_text(PLAIN, encoding="utf-8")
        (self.project / "CMakeLists.txt").write_text(BRACKET_CMAKE, encoding="utf-8")
        self.lines = WAVE_KERNEL.split("\n")
        self.result = scanner.scan_risks(self.project, route="native-hip")

    def at(self, needle, category, subtype=None, file="wave.cu"):
        lines = {i + 1 for i, text in enumerate(self.lines) if needle in text} if file == "wave.cu" else None
        return [f for f in self.result["findings"] if f["file"] == file and f["category"] == category
                and (subtype is None or f["subtype"] == subtype) and (lines is None or f["line"] in lines)]

    def test_wave64_bug_patterns_are_found(self):
        self.assertTrue(self.at("tid % 32", "warp-size", "lane-arith"))
        self.assertTrue(self.at("unsigned m = __ballot_sync", "lane-mask", "ballot-to-32bit"))
        self.assertTrue(self.at("__popc(m)", "lane-mask", "bitcount-32bit"))
        self.assertTrue(self.at("1u << lid", "lane-mask", "lane-shift"))
        self.assertTrue(self.at("__activemask()", "warp-sync", "__activemask"))
        returns = self.at("if (tid >= n) return;", "barrier-divergence", "return-before-barrier")
        self.assertEqual(len(returns), 1)
        self.assertEqual(self.lines[returns[0]["extra"]["barrier_line"] - 1].strip(), "__syncthreads();")

    def test_shuffle_width_is_recorded(self):
        xor = self.at("__shfl_xor_sync(0xffffffff", "warp-sync", "__shfl_xor_sync")[0]
        self.assertEqual(xor["extra"]["width"], "32")
        self.assertNotIn("width_note", xor["extra"])
        down = self.at("__shfl_down_sync(__activemask()", "warp-sync", "__shfl_down_sync")[0]
        self.assertEqual(down["extra"]["width"], "default")
        self.assertIn("other half", down["extra"]["width_note"])
        self.assertEqual(down["extra"]["mask"], "expression")

    def test_legal_tile32_and_64bit_masks_are_not_flagged(self):
        self.assertFalse(self.at("unsigned long long all", "lane-mask"))
        self.assertEqual(self.at("unsigned long long all", "warp-sync")[0]["extra"]["result"], "64bit-declaration")
        self.assertFalse(self.at("__popcll", "lane-mask"))
        self.assertFalse(self.at("1ull << threadIdx.x", "lane-mask"))
        self.assertFalse(self.at("tile.sync()", "barrier-divergence"))
        self.assertNotIn("width_note", self.at("in[0], 1, 32", "warp-sync")[0]["extra"])

    def test_cards_and_native_route_without_gpufusion_flags(self):
        self.assertTrue(all(f.get("card") for f in self.result["findings"]))
        self.assertEqual(self.at("__shfl_xor_sync(0xffffffff", "warp-sync")[0]["card"], "W-05")
        self.assertFalse([f for f in self.result["findings"] if f.get("gpufusion_limit_candidate")])
        self.assertIn("W-03", self.result["summary"]["by_card"])

    def test_context_gating_raw_strings_and_cmake_brackets(self):
        self.assertFalse([f for f in self.result["findings"] if f["file"] == "plain.cu"])
        self.assertTrue([f for f in self.result["findings"] if f["file"] == "raw.cu" and f["subtype"] == "__syncwarp"])
        archs = [f for f in self.result["findings"] if f["file"] == "CMakeLists.txt"]
        self.assertEqual({f["subtype"] for f in archs}, {"hip-arch"})

    def test_summary_counts_survive_truncation(self):
        capped = scanner.scan_risks(self.project, route="native-hip", max_findings=2)
        self.assertTrue(capped["truncated"])
        self.assertEqual(len(capped["findings"]), 2)
        self.assertEqual(capped["total_findings"], self.result["total_findings"])
        self.assertEqual(capped["summary"]["by_category"], self.result["summary"]["by_category"])


class DependencyRepoTests(unittest.TestCase):
    def test_source_dependencies_are_found_with_card(self):
        with tempfile.TemporaryDirectory(prefix="dep tests ") as temp:
            project = Path(temp) / "project"
            project.mkdir()
            (project / "a.cu").write_text('#include <cuda/std/atomic>\n#include <raft/core/handle.hpp>\n#include <ck_tile/core.hpp>\n#include <vector>\n', encoding="utf-8")
            (project / "CMakeLists.txt").write_text('# libhipcxx only in a comment\nCPMAddPackage("gh:rapidsai/raft#v1")\nFetchContent_Declare(composable_kernel GIT_REPOSITORY x)\n', encoding="utf-8")
            result = scanner.scan_risks(project, route="native-hip")
        deps = [f for f in result["findings"] if f["category"] == "source-dependency"]
        self.assertEqual({(f["file"], f["subtype"]) for f in deps},
                         {("a.cu", "cuda/std/atomic"), ("a.cu", "raft/core/handle.hpp"), ("a.cu", "ck_tile/core.hpp"),
                          ("CMakeLists.txt", "raft"), ("CMakeLists.txt", "composable_kernel")})
        self.assertTrue(all(f["card"] == "S-10" for f in deps))
        self.assertEqual(result["summary"]["by_card"]["S-10"], 5)
        self.assertEqual(result["guidance"]["source-dependency"], "references/code-conversion.md#s-10")

    def test_dependency_guidance_is_packaged_and_resolves(self):
        import build_skill_packages as exporter
        reference = scanner.GUIDANCE["source-dependency"]
        document, anchor = reference.split("#")
        for name in exporter.NAMES:
            self.assertIn(document, exporter.sources_for(name))
        self.assertIn(f'id="{anchor}"', (ROOT / document).read_text(encoding="utf-8"))


ROCMINFO = """\
ROCk module is loaded
*******
Agent 1
*******
  Name:                    Intel CPU
*******
Agent 2
*******
  Name:                    gfx936
  Wavefront Size:          64(0x40)
*******
Agent 3
*******
  Name:                    gfx936
  Wavefront Size:          64(0x40)
"""
MACROS = "#define __HIP_PLATFORM_AMD__ 1\n#define __AMDGCN_WAVEFRONT_SIZE__ 64\n#define OTHER 1\n"


class FakeDistribution:
    def __init__(self, name, version, root):
        self.metadata = {"Name": name}
        self.version = version
        self._root = root

    def locate_file(self, path):
        return self._root / path


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="probe tests ")
        self.addCleanup(self.temp.cleanup)
        self.site = Path(self.temp.name) / "site"
        header = self.site / "torch/include/ATen/dtk_macros.h"
        header.parent.mkdir(parents=True)
        header.write_text("#if defined(USE_ROCM)\n#define __FPT_DTK_ARCH__ 520\n#define FPT_DTK_VERSION 11800\n#define FPT_DTKRT_VERSION 11080\n#endif\n", encoding="utf-8")
        (self.site / "torch/version.py").write_text("__version__ = '2.14.0+hcu'\nhip = '6.1.0'\ncuda = None\n", encoding="utf-8")
        hipify = self.site / "torch/utils/hipify/hipify_python.py"
        hipify.parent.mkdir(parents=True)
        hipify.write_text('HIPIFY_HEADER_FILE = "#include <ATen/dtk_macros.h>\\n"\n', encoding="utf-8")
        (hipify.parent / "version.py").write_text('__version__ = "2.0.0"\n', encoding="utf-8")
        self.calls = []

    def runner(self, argv, timeout):
        self.calls.append(argv)
        name = Path(argv[0]).name
        outputs = {"hipcc": (0, "HIP version: 6.1.0\nclang version 17\n"), "hipconfig": (0, "6.1.0\n"), "rocminfo": (0, ROCMINFO)}
        if name == "hipcc" and "-dM" in argv:
            return {"argv": argv, "exit_code": 0, "output": "", "_full": MACROS, "error": None}
        if name == "hipcc" and "-c" in argv:
            ok = "64bit" in argv[-1] or "ballot" in argv[-1]
            if ok:
                Path(argv[argv.index("-o") + 1]).write_bytes(b"synthetic object; not hardware evidence")
            return {"argv": argv, "exit_code": 0 if ok else 1, "output": "" if ok else "error: mask", "_full": "", "error": None}
        code, text = outputs.get(name, (127, ""))
        return {"argv": argv, "exit_code": code, "output": text, "_full": text, "error": None}

    def run_probe(self, *, tools=("hipcc", "hipconfig", "rocminfo"), **kwargs):
        distributions = lambda: [FakeDistribution("torch", "2.14.0", self.site), FakeDistribution("requests", "2.0", self.site)]
        return probe.probe(runner=self.runner, environ={"ROCM_PATH": "/opt/dtk", "FASTPT_MODE": "x", "SECRET_TOKEN": "hidden"}, which=lambda name: f"/opt/bin/{name}" if name in tools else None, distributions=distributions, **kwargs)

    def test_facts_are_collected_with_provenance(self):
        report = self.run_probe()
        self.assertTrue(report["read_only"])
        self.assertEqual(report["devices"]["gfx"], ["gfx936"])
        self.assertEqual(report["devices"]["wavefront_sizes"], [64])
        self.assertEqual(report["arch_suggestion"], "gfx936")
        self.assertEqual(report["hip"]["hipcc_version"], "HIP version: 6.1.0")
        self.assertEqual({p["name"] for p in report["python_packages"]}, {"torch"})
        macros = report["torch"]["dtk_macros"]
        self.assertEqual(macros["defines"], {"__FPT_DTK_ARCH__": "520", "FPT_DTK_VERSION": "11800", "FPT_DTKRT_VERSION": "11080"})
        self.assertTrue(macros["guarded_by_use_rocm"])
        self.assertIn("not the installed DTK", macros["note"])
        self.assertTrue(report["torch"]["hipify_injects_dtk_macros"])
        self.assertEqual(report["torch"]["hipify_version"], "2.0.0")
        self.assertIn("rocm_agent_enumerator", report["tools"])
        self.assertEqual(report["torch"]["version_py"]["hip"], "6.1.0")
        self.assertEqual(report["env"], {"ROCM_PATH": "/opt/dtk"})
        self.assertTrue(all(c["argv"] and "exit_code" in c for c in report["commands"]))
        self.assertNotIn("_text", json.dumps(report))

    def test_hcu_version_is_literal_evidence_not_a_support_claim(self):
        path = self.site / "torch/version.py"
        for source, expected in (("hcu: str | None = '26.10'", '26.10'),
                                 ('hcu = None', None),
                                 ('hcu = get_hcu_version()', None),
                                 ("hip = '7.2'", None)):
            with self.subTest(source=source):
                path.write_text(source + "\n", encoding="utf-8")
                report = self.run_probe(tools=())
                self.assertEqual(report["capabilities"]["hcu"]["version"], expected)
                self.assertIsNone(report["capabilities"]["hip"]["installed"])
                self.assertIsNone(report["capabilities"]["native_backend"]["state"])

    def test_probe_never_imports_or_activates_packages(self):
        report = self.run_probe()
        self.assertTrue(all(c["argv"] and "exit_code" in c for c in report["commands"]))
        self.assertNotIn("_text", json.dumps(report))

    def test_missing_tools_are_unknown_not_false(self):
        report = self.run_probe(tools=())
        self.assertIsNone(report["capabilities"]["hip"]["installed"])
        self.assertIsNone(report["capabilities"]["hip"]["compiler"])
        self.assertTrue(any("hipcc not found" in w for w in report["warnings"]))
        self.assertEqual(report["devices"]["gfx"], [])

    def test_compile_probe_reports_each_result_and_macros(self):
        report = self.run_probe(compile_probe=True)
        results = {p["name"]: p["passed"] for p in report["compile_probes"]}
        self.assertEqual(results, {"warp-sync-64bit-mask": True, "warp-sync-32bit-mask": False, "ballot-sync-64bit-mask": True})
        self.assertEqual(report["macros_defined"], {"__HIP_PLATFORM_AMD__": "1", "__AMDGCN_WAVEFRONT_SIZE__": "64"})
        self.assertTrue(all("--offload-arch=gfx936" in p["argv"] for p in report["compile_probes"]))

    def test_compile_probe_needs_one_explicit_target(self):
        report = self.run_probe(tools=("hipcc",), compile_probe=True)
        self.assertEqual(report["compile_probes"], [])
        self.assertTrue(any("--compile-probe needs" in w for w in report["warnings"]))
        with self.assertRaises(common.AdaptationError):
            self.run_probe(tools=("hipcc",), compile_probe=True, arch="sm_80")

    def test_capabilities_feed_the_route_planners(self):
        report = self.run_probe()
        project = Path(self.temp.name) / "project"
        project.mkdir()
        (project / "CMakeLists.txt").write_text("project(x LANGUAGES CXX HIP)\n", encoding="utf-8")
        request = {"repo": str(project), "arch": "gfx936", "build_command": ["cmake", "--build", "build"], "capabilities": {"native_backend": {"state": "available"}, **report["capabilities"]}}
        native = route_preflight.plan_request(dict(request), "native-hip")
        states = {c["name"]: c["status"] for c in native["checks"]}
        self.assertEqual((states["hip-installed"], states["hip-compiler"]), ("passed", "passed"))

    def test_unmodified_probe_capabilities_leave_native_unknown(self):
        report = self.run_probe()
        project = Path(self.temp.name) / "project"
        project.mkdir()
        (project / "CMakeLists.txt").write_text("project(x LANGUAGES CXX HIP)\n", encoding="utf-8")
        plan = route_preflight.plan_request({"repo": str(project), "capabilities": report["capabilities"]})
        self.assertEqual(plan["route"], "native-hip")
        self.assertEqual(plan["plan_status"], "needs-information")
        self.assertIsNone(report["capabilities"]["native_backend"]["state"])


def make_receipt(path: Path, source_sha: str, artifact: Path, covered_entries=None) -> Path:
    artifact.write_bytes(b"built artifact")
    receipt = {
        "schema_version": common.SCHEMA_VERSION, "kind": "adaptation-result", "status": "completed", "route": "native-hip", "source_sha256": source_sha,
        "covered_entries": list(covered_entries or []),
        "stages": {"build": "passed", "load": "passed", "gpu": "passed", "correctness": "passed"},
        "tests": {"total": 5, "passed": 4, "failed": 0, "skipped": 1, "gpu_passed": 4},
        "oracle": "cpu reference", "validation_contract": {"atol": 1e-5, "rtol": 1e-5, "required_stages": ["build", "gpu"]},
        "commands": [{"phase": "build", "argv": ["cmake", "--build", "b"], "exit_code": 0}, {"phase": "correctness", "argv": ["pytest"], "exit_code": 0}],
        "artifacts": [{"path": str(artifact), "sha256": common.file_sha256(artifact), "source_sha256": source_sha, "loaded": True}],
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ledger tests ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.project = make_project(self.base)
        self.path = self.base / "ledger.json"
        self.ledger = ledger.init_ledger(self.project, self.path, features=["op backward"])
        self.ids = {e["id"] for e in self.ledger["entries"]}

    def receipt(self, name="r.json"):
        return make_receipt(self.base / name, self.ledger["source_sha256"], self.base / (name + ".bin"), sorted(self.ids))

    def cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = ledger.main(list(args))
        return code, out.getvalue()

    def test_components_are_enumerated_and_all_start_owed(self):
        self.assertTrue({"library:foo", "executable:bar", "extension:pkg._C", "feature:op backward"} <= self.ids)
        self.assertTrue(all(e["status"] == "owed" for e in self.ledger["entries"]))
        self.assertFalse([e for e in self.ledger["entries"] if "third_party" in (e.get("file") or "")])

    def test_unfinished_work_cannot_be_renamed_out_of_scope(self):
        with self.assertRaises(common.AdaptationError):
            ledger.set_status(self.path, "library:foo", "scoped_out", reason="skip")
        with self.assertRaises(common.AdaptationError):
            ledger.set_status(self.path, "library:foo", "scoped_out", approval="user")
        with self.assertRaises(common.AdaptationError):
            ledger.set_status(self.path, "library:foo", "blocked")
        self.assertEqual(ledger.check_ledger(self.path)["counts"]["scoped_out"], 0)

    def test_covered_requires_a_valid_receipt_for_this_source(self):
        with self.assertRaises(common.AdaptationError):
            ledger.set_status(self.path, "library:foo", "covered")
        stale = make_receipt(self.base / "stale.json", "0" * 64, self.base / "stale.bin")
        with self.assertRaises(common.AdaptationError):
            ledger.set_status(self.path, "library:foo", "covered", receipt=stale)
        bad = json.loads(self.receipt("bad.json").read_text())
        bad["tests"]["gpu_passed"] = 0
        (self.base / "bad.json").write_text(json.dumps(bad))
        with self.assertRaises(common.AdaptationError):
            ledger.set_status(self.path, "library:foo", "covered", receipt=self.base / "bad.json")
        entry = ledger.set_status(self.path, "library:foo", "covered", route="native-hip", receipt=self.receipt())
        self.assertEqual(entry["validated_source_sha256"], self.ledger["source_sha256"])

    def test_check_exit_codes_and_completion(self):
        code, _ = self.cli("check", "--ledger", str(self.path))
        self.assertEqual(code, 3)
        receipt = self.receipt()
        for entry_id in self.ids:
            if entry_id == "feature:op backward":
                self.cli("set", "--ledger", str(self.path), "--id", entry_id, "--status", "scoped_out", "--reason", "not requested", "--approval", "user confirmed")
            else:
                self.assertEqual(self.cli("set", "--ledger", str(self.path), "--id", entry_id, "--status", "covered", "--receipt", str(receipt))[0], 0)
        code, text = self.cli("check", "--ledger", str(self.path))
        self.assertEqual(code, 0, text)
        self.assertTrue(json.loads(text)["complete"])

    def test_source_change_makes_coverage_stale(self):
        receipt = self.receipt()
        for entry_id in self.ids - {"feature:op backward"}:
            ledger.set_status(self.path, entry_id, "covered", receipt=receipt)
        ledger.set_status(self.path, "feature:op backward", "scoped_out", reason="r", approval="a")
        self.assertTrue(ledger.check_ledger(self.path)["complete"])
        with (self.project / "CMakeLists.txt").open("a", encoding="utf-8") as stream:
            stream.write("# changed\n")
        report = ledger.check_ledger(self.path)
        self.assertFalse(report["complete"])
        self.assertTrue(report["source_changed"])
        self.assertTrue(all("stale" in p["problem"] for p in report["problems"]))
        self.assertEqual(self.cli("check", "--ledger", str(self.path))[0], 2)

    def test_receipt_tampering_is_detected(self):
        receipt = self.receipt()
        for entry_id in self.ids - {"feature:op backward"}:
            ledger.set_status(self.path, entry_id, "covered", receipt=receipt)
        ledger.set_status(self.path, "feature:op backward", "scoped_out", reason="r", approval="a")
        receipt.write_text(receipt.read_text() + " ")
        report = ledger.check_ledger(self.path)
        self.assertFalse(report["complete"])
        self.assertIn("receipt", report["problems"][0]["problem"])

    def test_empty_ledger_is_never_complete(self):
        empty = self.base / "empty"
        empty.mkdir()
        (empty / "notes.txt").write_text("nothing")
        path = self.base / "empty-ledger.json"
        ledger.init_ledger(empty, path)
        report = ledger.check_ledger(path)
        self.assertFalse(report["complete"])
        self.assertTrue(report["warnings"])

    def test_ledger_must_stay_outside_the_project_and_not_be_overwritten(self):
        with self.assertRaises(common.AdaptationError):
            ledger.init_ledger(self.project, self.project / "ledger.json")
        with self.assertRaises(common.AdaptationError):
            ledger.init_ledger(self.project, self.path)
        with self.assertRaises(common.AdaptationError):
            ledger.set_status(self.path, "nope", "owed")


if __name__ == "__main__":
    unittest.main()
