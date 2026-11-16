from __future__ import annotations

import builtins
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import adaptation_common as common
import classify_fastpt_failure as classifier
import route_preflight
from routes import native_hip, transcode


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="route fixtures ")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "project"
        self.repo.mkdir()
        (self.repo / "CMakeLists.txt").write_text("project(example LANGUAGES CXX HIP)\n")
        (self.repo / "kernel.hip").write_text("__global__ void k() {}\n")

    def request(self):
        return {"repo": str(self.repo), "arch": "gfx936", "build_command": ["cmake", "--build", "build"], "capabilities": {"native_backend": {"state": "available", "reason": "fixture target"}, "hip": {"installed": True, "compiler": "hipcc"}, "fastpt": {"installed": False, "torch_version": "2.14.0", "required_torch_version": "2.12"}}}

    def cuda_only(self):
        (self.repo / "kernel.hip").unlink()
        (self.repo / "CMakeLists.txt").write_text("project(example LANGUAGES CUDA)\n")
        (self.repo / "kernel.cu").write_text("__global__ void k() {}\n")

    def test_native_ignores_fastpt_in_auto_and_explicit_modes(self):
        original_import = builtins.__import__
        def guard(name, *args, **kwargs):
            self.assertNotIn(name.split(".")[0], {"torch", "fastpt"})
            return original_import(name, *args, **kwargs)
        request = self.request()
        request["capabilities"]["fastpt"] = "irrelevant malformed provider data"
        with patch("builtins.__import__", side_effect=guard), patch.object(common.subprocess, "run", side_effect=AssertionError("no probes")):
            for selected in ("auto", "native-hip"):
                result = route_preflight.plan_request(request, selected)
                self.assertEqual(result["route"], "native-hip")
                self.assertEqual(result["selection_mode"], "auto" if selected == "auto" else "explicit")
                self.assertEqual(result["plan_status"], "planned")
                self.assertFalse(result["executed"])
                self.assertEqual(set(result["stages"].values()), {"not-run"})

    def test_native_compiler_true_without_identity_is_unknown(self):
        request = self.request()
        request["capabilities"]["hip"]["compiler"] = True
        result = native_hip.plan(request)
        self.assertEqual(result["plan_status"], "needs-information")

    def test_native_known_failure_does_not_switch(self):
        request = self.request()
        request["capabilities"]["native_backend"] = {"state": "unavailable", "reason": "required feature absent"}
        result = route_preflight.plan_request(request, "native-hip")
        self.assertEqual(result["route"], "native-hip")
        self.assertEqual(result["plan_status"], "blocked")

    def test_native_shell_activation_and_shell_c_are_rejected(self):
        for argv in (["fastpt", "-C"], ["bash", "-lc", "source fastpt -C"], ["cmake", "&&", "make"]):
            request = self.request()
            request["build_command"] = argv
            result = native_hip.plan(request)
            self.assertEqual(result["plan_status"], "blocked")

    def test_native_candidate_with_unknown_backend_stays_native(self):
        request = self.request()
        request["capabilities"]["native_backend"] = {"state": "unknown"}
        result = route_preflight.plan_request(request)
        self.assertEqual(result["route"], "native-hip")
        self.assertEqual(result["plan_status"], "needs-information")

    def test_framework_cuda_extension_is_not_forced_to_fastpt(self):
        (self.repo / "kernel.hip").unlink()
        (self.repo / "CMakeLists.txt").unlink()
        (self.repo / "setup.py").write_text("is_rocm = torch.version.hip is not None\nCUDAExtension(...)\n")
        self.assertEqual(route_preflight.plan_request(self.request())["route"], "native-hip")

    def test_cpp_hip_header_and_multiline_language_are_detected(self):
        (self.repo / "kernel.hip").unlink()
        (self.repo / "CMakeLists.txt").write_text("project(\n example\n LANGUAGES CXX\n HIP\n)\n")
        self.assertTrue(common.scan_project(self.repo)["native_candidate"])
        (self.repo / "CMakeLists.txt").unlink()
        (self.repo / "host.cpp").write_text('#include <hip/hip_runtime.h>\n')
        self.assertTrue(common.scan_project(self.repo)["native_candidate"])

    def test_auto_cuda_only_prefers_transcode_with_unknown_fastpt(self):
        self.cuda_only()
        request = self.request()
        request["capabilities"] = {}
        result = route_preflight.plan_request(request)
        self.assertEqual(result["route"], "transcode")
        self.assertEqual(result["plan_status"], "needs-information")

    def test_auto_cuda_route_does_not_depend_on_fastpt_state(self):
        self.cuda_only()
        for fastpt in (None, {}, {"installed": True}, {"installed": False}, "irrelevant"):
            request = self.request()
            request["capabilities"] = {"fastpt": fastpt}
            with self.subTest(fastpt=fastpt):
                self.assertEqual(route_preflight.plan_request(request)["route"], "transcode")

    def test_unavailable_native_requires_nonblank_reason(self):
        for reason in (None, "", " \t", False, 42):
            request = self.request()
            request["capabilities"]["native_backend"] = {"state": "unavailable", "reason": reason}
            with self.subTest(reason=reason):
                plan = route_preflight.plan_request(request)
                self.assertEqual(plan["route"], "native-hip")
                self.assertEqual(plan["plan_status"], "blocked")

    def test_unavailable_native_without_cuda_is_blocked(self):
        request = self.request()
        request["capabilities"]["native_backend"] = {"state": "unavailable", "reason": "required API missing"}
        plan = route_preflight.plan_request(request)
        self.assertEqual((plan["route"], plan["plan_status"]), ("none", "blocked"))
        (self.repo / "kernel.cu").write_text("__global__ void k() {}\n")
        self.assertEqual(route_preflight.plan_request(request)["route"], "transcode")

    def test_hcu_framework_marker_is_native_candidate_only(self):
        self.cuda_only()
        request = self.request()
        request["capabilities"].pop("native_backend")
        for expression in ("torch.version.hcu", "getattr(torch.version, 'hcu', None)",
                           'getattr( torch.version , "hip" , None )'):
            (self.repo / "setup.py").write_text(f"use_hcu = {expression} is not None\nCUDAExtension(...)\n")
            with self.subTest(expression=expression):
                plan = route_preflight.plan_request(request)
                self.assertEqual(plan["route"], "native-hip")
                self.assertEqual(plan["plan_status"], "needs-information")

    def test_auto_known_fastpt_mismatch_reaches_transcode_candidate(self):
        self.cuda_only()
        request = self.request()
        request["capabilities"].pop("native_backend")
        result = route_preflight.plan_request(request)
        self.assertEqual(result["route"], "transcode")
        self.assertIn(result["plan_status"], {"blocked", "needs-information"})
        self.assertFalse(result["executed"])

    def test_empty_cpu_scope_is_not_completed(self):
        for path in list(self.repo.iterdir()):
            path.unlink()
        (self.repo / "README.md").write_text("ordinary CPU package")
        result = route_preflight.plan_request({"repo": str(self.repo)})
        self.assertEqual(result["route"], "none")
        self.assertEqual(result["plan_status"], "needs-information")

    def test_transcode_can_use_non_fastpt_provider(self):
        request = self.request()
        request["provider"] = "hipify-clang"
        request["capabilities"]["converters"] = {"hipify-clang": {key: True for key in transcode.CAPABILITY_FIELDS}}
        before = copy.deepcopy(request)
        result = transcode.plan(request)
        self.assertEqual(request, before)
        self.assertEqual(result["provider"], "hipify-clang")
        self.assertEqual(result["plan_status"], "planned")
        self.assertFalse(result["executed"])

    def test_fixed_failing_provider_never_falls_back(self):
        request = self.request()
        request["provider"] = "fastpt"
        request["capabilities"]["converters"] = {"hipify-clang": {key: True for key in transcode.CAPABILITY_FIELDS}}
        result = transcode.plan(request)
        self.assertEqual(result["provider"], "fastpt")
        self.assertEqual(result["plan_status"], "blocked")

    def test_fastpt_series_comparison_is_not_full_string_equality(self):
        request = self.request()
        request["provider"] = "fastpt"
        request["capabilities"]["fastpt"] = {key: True for key in transcode.CAPABILITY_FIELDS}
        request["capabilities"]["fastpt"].update(torch_version="2.12.3+dtk", required_torch_version="2.12")
        result = transcode.plan(request)
        self.assertEqual(result["plan_status"], "planned")

    def test_provider_feature_unknown_and_false_are_separate(self):
        request = self.request()
        request["provider"] = "hipify-clang"
        request["required_features"] = ["special-api"]
        provider = {key: True for key in transcode.CAPABILITY_FIELDS}
        request["capabilities"]["converters"] = {"hipify-clang": provider}
        self.assertEqual(transcode.plan(request)["plan_status"], "needs-information")
        provider["features"] = {"special-api": False}
        self.assertEqual(transcode.plan(request)["plan_status"], "blocked")

    def test_legacy_hytlass_alias_and_package_route_restriction(self):
        request = self.request()
        result = route_preflight.plan_request(request, "cutlass-hytlass")
        self.assertEqual(result["route"], "transcode")
        self.assertEqual(result["specialization"], "cutlass-hytlass")
        with self.assertRaises(common.AdaptationError):
            route_preflight.plan_request(request, "transcode", available_routes=["native-hip"])
        request["specialization"] = "other"
        with self.assertRaises(common.AdaptationError):
            route_preflight.plan_request(request, "cutlass-hytlass")


class ClassifierTests(unittest.TestCase):
    def test_hip_filename_and_legal_banner_do_not_override_missing_header(self):
        text = "CUDA_HOME=/valid\nhipcc kernel.hip\nkernel.hip:42: fatal error: 'missing.h' file not found\n"
        for route in (None, "native-hip", "transcode", "non-transcode"):
            name, first, advice = classifier.classify(text, route, "build")
            self.assertEqual(name, "missing-header")
            self.assertIn("missing.h", first)
            self.assertNotIn("source fastpt", advice)

    def test_first_error_controls_later_keyword_noise(self):
        text = "  fatal error: 'a.h' file not found\n\nerror: cudart_static failed\n"
        self.assertEqual(classifier.classify(text, "non-transcode")[0], "missing-header")

    def test_static_runtime_is_route_specific(self):
        log = "ld: error: cannot find -lcudart_static"
        self.assertEqual(classifier.classify(log, "non-transcode")[0], "static-cudart")
        self.assertEqual(classifier.classify(log, "native-hip")[0], "missing-symbol")

    def test_no_error_and_mismatch(self):
        self.assertEqual(classifier.classify("hipify kernel.hip\nCUDA_HOME=/path\n")[0], "unknown")
        self.assertEqual(classifier.classify("ValueError: Error: torch version 2.14 does not match the required version 2.12")[0], "fastpt-version")


if __name__ == "__main__":
    unittest.main()
