"""Non-transcode planning tests: synthetic trees and supplied evidence only."""
import builtins
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import adaptation_common as common
from routes import non_transcode


class NonTranscodeRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="non transcode route ")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "project"
        self.repo.mkdir()
        (self.repo / "setup.py").write_text("# CUDAExtension\n", encoding="utf-8")
        (self.repo / "kernel.cu").write_bytes(b"// CUDA-facing input\n")

    def request(self, **updates):
        request = {
            "repo": str(self.repo), "selection_mode": "explicit", "arch": "gfx936",
            "build_command": ["python", "setup.py", "build_ext"],
            "capabilities": {"fastpt": {"installed": True, "version_compatible": True, "invocable": True, "feature_probed": True}},
        }
        request.update(updates)
        return request

    @staticmethod
    def checks(result):
        return {item["name"]: item["status"] for item in result["checks"]}

    def test_plan_is_read_only_and_preserves_argv(self):
        request = self.request()
        before = copy.deepcopy(request)
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            self.assertNotIn(name.split(".")[0], ("torch", "fastpt"))
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import), mock.patch.object(common.subprocess, "run", side_effect=AssertionError("no external commands")):
            result = non_transcode.plan(request)
        self.assertEqual(request, before)
        self.assertEqual(result["kind"], "adaptation-plan")
        self.assertEqual(result["plan_status"], "planned")
        self.assertFalse(result["executed"])
        self.assertTrue(all(state == "not-run" for state in result["stages"].values()))
        self.assertEqual(result["commands"][0]["argv"], request["build_command"])
        self.assertFalse(result["commands"][0]["executed"])
        self.assertFalse(result["activation"]["executed"])
        self.assertEqual(set(path.name for path in self.repo.iterdir()), {"setup.py", "kernel.cu"})
        self.assertIn("tree_sha256", result["source"])

    def test_unknown_is_not_false_or_passed(self):
        result = non_transcode.plan({"repo": str(self.repo)})
        checks = self.checks(result)
        for name in non_transcode.ACTIVATION_CHECKS:
            self.assertEqual(checks[name], "unknown")
        self.assertEqual(checks["fastpt-feature-coverage"], "unknown")
        self.assertEqual(result["plan_status"], "needs-information")
        self.assertEqual(result["blockers"], [])

    def test_known_failure_and_unknown_are_distinct(self):
        for field, name in (("installed", "fastpt-installed"), ("version_compatible", "fastpt-version-compatible"), ("invocable", "fastpt-invocable")):
            for value, status in ((None, "unknown"), (False, "failed"), (True, "passed")):
                with self.subTest(field=field, value=value):
                    request = self.request()
                    request["capabilities"]["fastpt"][field] = value
                    result = non_transcode.plan(request)
                    self.assertEqual(self.checks(result)[name], status)
                    self.assertEqual(result["plan_status"], {None: "needs-information", False: "blocked", True: "planned"}[value])

    def test_mismatch_overrides_positive_boolean(self):
        request = self.request()
        request["capabilities"]["fastpt"].update(torch_version="2.14.0+rocm7", required_torch_version="2.12.1")
        result = non_transcode.plan(request)
        self.assertEqual(result["plan_status"], "blocked")
        self.assertEqual(self.checks(result)["fastpt-version-compatible"], "failed")
        self.assertEqual(result["route"], "non-transcode")
        self.assertFalse(result["executed"])

    def test_actual_required_series_match_is_narrow_evidence(self):
        request = self.request()
        request["capabilities"]["fastpt"].update(version_compatible=None, torch_version="2.12.3+rocm", required_torch_version="2.12", invocable=None)
        result = non_transcode.plan(request)
        self.assertEqual(self.checks(result)["fastpt-version-compatible"], "unknown")
        self.assertEqual(self.checks(result)["fastpt-invocable"], "unknown")
        self.assertEqual(result["plan_status"], "needs-information")

    def test_fastpt_release_number_does_not_guess_torch_requirement(self):
        request = self.request()
        request["capabilities"]["fastpt"].update(version_compatible=None, version="2.12", torch_version="2.14")
        result = non_transcode.plan(request)
        self.assertEqual(self.checks(result)["fastpt-version-compatible"], "unknown")

    def test_hardware_validation_is_not_a_plan_prerequisite(self):
        for value in (None, False, True):
            with self.subTest(value=value):
                request = self.request()
                request["capabilities"]["fastpt"]["hardware_validated"] = value
                result = non_transcode.plan(request)
                self.assertEqual(result["plan_status"], "planned")
                self.assertEqual(result["capabilities"]["fastpt"]["hardware_validated"], value)
                self.assertEqual(result["stages"]["gpu"], "not-run")

    def test_feature_probe_is_never_inferred(self):
        request = self.request(required_features=["cublasLt"])
        request["capabilities"]["fastpt"]["feature_probed"] = None
        result = non_transcode.plan(request)
        self.assertEqual(self.checks(result)["fastpt-feature-coverage"], "unknown")
        self.assertEqual(result["plan_status"], "needs-information")

    def test_explicit_route_does_not_reroute_to_native(self):
        (self.repo / "setup.py").write_text("# torch.version.hip CUDAExtension\n", encoding="utf-8")
        request = self.request()
        request["capabilities"]["fastpt"]["invocable"] = False
        request["capabilities"]["native_backend"] = {"state": "available"}
        result = non_transcode.plan(request)
        self.assertEqual(result["route"], "non-transcode")
        self.assertEqual(result["selection_mode"], "explicit")
        self.assertEqual(result["plan_status"], "blocked")

    def test_auto_provenance_is_retained(self):
        result = non_transcode.plan(self.request(selection_mode="auto"))
        self.assertEqual(result["selection_mode"], "auto")

    def test_root_setup_is_not_displaced_by_vendor_cmake(self):
        vendor = self.repo / "vendor" / "library"
        vendor.mkdir(parents=True)
        (vendor / "CMakeLists.txt").write_text("project(vendor LANGUAGES CUDA)\n", encoding="utf-8")
        result = non_transcode.plan(self.request())
        self.assertEqual(result["scan"]["root_entries"], ["setup.py"])
        self.assertEqual(result["commands"][0]["argv"], ["python", "setup.py", "build_ext"])

    def test_shell_string_is_blocked_not_evaluated(self):
        result = non_transcode.plan(self.request(build_command="python setup.py; touch injected"))
        self.assertEqual(self.checks(result)["build-command"], "failed")
        self.assertEqual(result["commands"], [])
        self.assertEqual(result["plan_status"], "blocked")

    def test_missing_command_is_unknown(self):
        result = non_transcode.plan(self.request(build_command=None))
        self.assertEqual(self.checks(result)["build-command"], "unknown")
        self.assertEqual(result["commands"], [])

    def test_explicit_build_entry_is_checked(self):
        good = non_transcode.plan(self.request(build_entry="setup.py"))
        self.assertEqual(self.checks(good)["explicit-build-entry"], "passed")
        for value in ("missing.py", "../elsewhere.py", str(self.repo / "setup.py")):
            with self.subTest(value=value):
                result = non_transcode.plan(self.request(build_entry=value))
                self.assertEqual(self.checks(result)["explicit-build-entry"], "failed")

    def test_architecture_shorthand_is_blocked(self):
        result = non_transcode.plan(self.request(arch="936"))
        self.assertEqual(self.checks(result)["architecture"], "failed")

    def test_invalid_evidence_is_not_truthy(self):
        for field in ("installed", "version_compatible", "invocable", "feature_probed", "hardware_validated"):
            with self.subTest(field=field):
                request = self.request()
                request["capabilities"]["fastpt"][field] = "true"
                result = non_transcode.plan(request)
                self.assertEqual(result["plan_status"], "blocked")
                self.assertEqual(self.checks(result)["fastpt-evidence"], "failed")

    def test_non_utf8_source_identity_survives(self):
        (self.repo / "setup.py").write_bytes(b"# CUDAExtension \xff\r\n")
        result = non_transcode.plan(self.request())
        self.assertEqual(result["plan_status"], "planned")
        self.assertEqual(result["source"]["files"]["setup.py"], common.file_sha256(self.repo / "setup.py"))


if __name__ == "__main__":
    unittest.main()
