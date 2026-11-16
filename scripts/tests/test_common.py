from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import adaptation_common as common


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="adapt tests ")
        self.base = Path(self.temporary.name)
        self.repo = self.base / "source"
        self.repo.mkdir()
        (self.repo / "kernel.hip").write_text("__global__ void k() {}\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_protected_source_relationships(self):
        for out in (self.repo, self.base, self.repo / "out"):
            with self.subTest(out=out), self.assertRaises(common.AdaptationError):
                common.prepare_tree(self.repo, out, route="native-hip", force=True)
        self.assertTrue((self.repo / "kernel.hip").is_file())

    def test_force_never_deletes_existing_directory(self):
        out = self.base / "existing"
        out.mkdir()
        (out / "user.txt").write_text("preserve", encoding="utf-8")
        with self.assertRaises(common.AdaptationError):
            common.prepare_tree(self.repo, out, route="transcode", force=True)
        self.assertEqual((out / "user.txt").read_text(), "preserve")

    def test_copy_preserves_bytes_and_build_named_parent(self):
        (self.repo / "build").mkdir()
        (self.repo / "build" / "tracked.cu").write_bytes(b"// user input\r\n\xff\n")
        out = common.prepare_tree(self.repo, self.base / "copy", route="native-hip")
        self.assertEqual((out / "build" / "tracked.cu").read_bytes(), b"// user input\r\n\xff\n")
        self.assertEqual(common.read_text(out / "build" / "tracked.cu").encode("utf-8", "surrogateescape"), b"// user input\r\n\xff\n")
        record = json.loads((out / common.SOURCE_FILE).read_text())
        self.assertIn("build/tracked.cu", record["files"])
        self.assertEqual(json.loads((out / common.OWNERSHIP_FILE).read_text())["status"], "prepared-not-built")
        self.assertEqual(common.validate_build_dir(out, "build/objects"), out / "build" / "objects")
        with self.assertRaises(common.AdaptationError):
            common.validate_build_dir(out, "../outside")

    def test_reserved_files_fail_before_copy(self):
        (self.repo / "build.sh").write_text("user data")
        with self.assertRaises(common.AdaptationError):
            common.prepare_tree(self.repo, self.base / "out", route="non-transcode", reserved_names=("build.sh",))
        self.assertFalse((self.base / "out").exists())

    def test_safe_writer_cannot_overwrite_or_escape(self):
        common.safe_write_text(self.repo, "note.txt", "hello")
        for relative in ("note.txt", "../escaped.txt"):
            with self.subTest(relative=relative), self.assertRaises(common.AdaptationError):
                common.safe_write_text(self.repo, relative, "replacement")
        self.assertEqual((self.repo / "note.txt").read_text(), "hello")
        common.safe_write_text(self.repo, "note.txt", "allowed", overwrite=True)
        self.assertEqual((self.repo / "note.txt").read_text(), "allowed")

    def test_file_and_directory_links_are_rejected(self):
        external = self.base / "outside.txt"
        external.write_text("outside")
        link = self.repo / "linked.cu"
        try:
            link.symlink_to(external)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        with self.assertRaises(common.AdaptationError):
            common.prepare_tree(self.repo, self.base / "out", route="transcode")
        self.assertEqual(external.read_text(), "outside")
        link.unlink()
        directory = self.repo / "linked-dir"
        directory.symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(common.AdaptationError):
            list(common.source_files(self.repo))

    def test_reparse_component_is_rejected_without_windows_privilege(self):
        with patch.object(common, "_linked", side_effect=lambda path: path == self.repo):
            with self.assertRaises(common.AdaptationError):
                common.resolve_repo(self.repo)

    def test_source_changes_during_copy_abort(self):
        original = common.shutil.copy2
        def changing_copy(source, destination, **kwargs):
            result = original(source, destination, **kwargs)
            Path(destination).write_text("changed")
            return result
        with patch.object(common.shutil, "copy2", side_effect=changing_copy):
            with self.assertRaises(common.AdaptationError):
                common.prepare_tree(self.repo, self.base / "out", route="transcode")
        self.assertFalse((self.base / "out").exists())
        self.assertEqual((self.repo / "kernel.hip").read_text(), "__global__ void k() {}\n")

    def test_scan_excludes_vendor_from_target_evidence(self):
        (self.repo / "kernel.hip").unlink()
        (self.repo / "setup.py").write_text("# host only\n")
        (self.repo / "vendor").mkdir()
        (self.repo / "vendor" / "CMakeLists.txt").write_text("enable_language(HIP)\n")
        scan = common.scan_project(self.repo)
        self.assertFalse(scan["native_candidate"])
        self.assertEqual(scan["root_entries"], ["setup.py"])

    def test_pure_hip_and_framework_hipify_are_candidates(self):
        scan = common.scan_project(self.repo)
        self.assertTrue(scan["native_candidate"])
        (self.repo / "setup.py").write_text("is_rocm = torch.version.hip is not None\nCUDAExtension(...)\n")
        scan = common.scan_project(self.repo)
        self.assertTrue(scan["native_candidate"])
        self.assertTrue(scan["cuda_candidate"])

    def test_plan_does_not_claim_execution(self):
        result = common.new_plan("native-hip", {"repo": str(self.repo)})
        common.add_check(result, "compiler", "unknown", "not checked")
        common.finalize_plan(result)
        self.assertEqual(result["plan_status"], "needs-information")
        self.assertFalse(result["executed"])
        self.assertEqual(set(result["stages"].values()), {"not-run"})
        common.add_check(result, "backend", "failed", "unavailable")
        self.assertEqual(common.finalize_plan(result)["plan_status"], "blocked")

    def test_arch_and_argv_are_not_shell_or_numeric_shortcuts(self):
        self.assertEqual(common.normalize_arch("gfx92a"), "gfx92a")
        self.assertEqual(common.normalize_arch("gfx936:sramecc+:xnack-"), "gfx936:sramecc+:xnack-")
        for arch in ("936", "sm_80", "gfx936; rm -rf /", "gfx"):
            with self.subTest(arch=arch), self.assertRaises(common.AdaptationError):
                common.normalize_arch(arch)
        for command in ("cmake && make", [], ["cmake", "x\ny"]):
            with self.assertRaises(common.AdaptationError):
                common.validate_build_command(command)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "kernel.so"
        self.path.write_bytes(b"synthetic artifact, never loaded")
        self.sha = "a" * 64
        self.receipt = {"schema_version": common.SCHEMA_VERSION, "kind": "adaptation-result", "route": "native-hip", "source_sha256": self.sha, "status": "completed", "stages": {key: "passed" for key in ("build", "load", "gpu", "correctness")}, "tests": {"total": 3, "passed": 2, "failed": 0, "skipped": 1, "gpu_passed": 1}, "oracle": "independent-reference-hash", "validation_contract": {"rtol": 1e-5, "atol": 1e-6}, "commands": [{"phase": "build", "argv": ["fixture-build"], "exit_code": 0}, {"phase": "correctness", "argv": ["fixture-test"], "exit_code": 0}], "artifacts": [{"path": str(self.path), "sha256": common.file_sha256(self.path), "source_sha256": self.sha, "loaded": True}]}

    def tearDown(self):
        self.temporary.cleanup()

    def test_structurally_complete_fixture(self):
        self.assertEqual(common.validate_receipt(self.receipt, expected_source_sha256=self.sha), [])

    def test_stale_source_or_binary_is_rejected(self):
        self.assertTrue(common.validate_receipt(self.receipt, expected_source_sha256="b" * 64))
        self.path.write_bytes(b"stale")
        self.assertTrue(common.validate_receipt(self.receipt))

    def test_skips_cpu_fallback_and_only_import_cannot_complete(self):
        for mutation in ("skip", "cpu", "only-import", "build-exit"):
            record = copy.deepcopy(self.receipt)
            if mutation == "skip":
                record["tests"] = {"total": 3, "passed": 0, "failed": 0, "skipped": 3, "gpu_passed": 0}
            elif mutation == "cpu":
                record["tests"]["gpu_passed"] = 0
            elif mutation == "only-import":
                record["stages"]["correctness"] = "not-run"
            else:
                record["commands"][0]["exit_code"] = 7
            with self.subTest(mutation=mutation):
                self.assertTrue(common.validate_receipt(record))


if __name__ == "__main__":
    unittest.main()
