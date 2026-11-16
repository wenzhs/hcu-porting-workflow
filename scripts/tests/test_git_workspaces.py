from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import adaptation_common as common


@unittest.skipUnless(shutil.which("git"), "Git is required for checkout tests")
class GitWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="adapt workspace ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "source"
        self.repo.mkdir()
        (self.repo / "CMakeLists.txt").write_bytes(
            b"cmake_minimum_required(VERSION 3.21)\r\r\n"
            b"project(sample LANGUAGES CXX CUDA)\r\r\n"
            b"add_library(k k.cu)\r\r\n"
        )
        (self.repo / "k.cu").write_text("// source\r\n", encoding="utf-8")
        self.mapping = self.base / "mapping.json"
        self.mapping.write_text(json.dumps({"k.cu": "k.cu"}), encoding="utf-8")

    def git(self, *args, cwd=None):
        result = subprocess.run(
            ["git", "-c", "user.name=Workspace test", "-c",
             "user.email=workspace@example.invalid", "-c", "core.autocrlf=false",
             "-C", str(cwd or self.repo), *args],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def initialize_git(self):
        self.git("init", "-b", "main")
        self.git("add", ".")
        self.git("commit", "-m", "base")
        self.git("tag", "v1.15.1")
        self.base_sha = self.git("rev-parse", "HEAD")
        (self.repo / "later.txt").write_text("main-only\r\n")
        self.git("add", ".")
        self.git("commit", "-m", "later main")
        self.main_sha = self.git("rev-parse", "HEAD")
        self.git("switch", "-c", "hcu/v1.15.1", "v1.15.1")

    def cli(self, script, *args, repo=None):
        return subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(ROOT / "scripts" / script),
             "--repo", str(repo or self.repo), *map(str, args)],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    def test_tag_branch_cmake_cli_edits_original_checkout(self):
        self.initialize_git()
        (self.repo / "user-note.txt").write_text("preserve user content")
        result = self.cli("cmake_fastpt_convert.py", "--source-map", self.mapping, "--write")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(json.loads(result.stdout)["output"]), self.repo.resolve())
        self.assertTrue((self.repo / ".git").is_dir())
        self.assertEqual(self.git("branch", "--show-current"), "hcu/v1.15.1")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.base_sha)
        self.assertEqual(self.git("rev-parse", "main"), self.main_sha)
        self.assertEqual(self.git("rev-parse", "v1.15.1"), self.base_sha)
        self.assertIn("LANGUAGES CXX HIP", (self.repo / "CMakeLists.txt").read_text())
        self.assertIn("LANGUAGES CXX CUDA", self.git("show", "v1.15.1:CMakeLists.txt"))
        self.assertIn("CMakeLists.txt", self.git("diff", "--name-only"))
        self.assertEqual((self.repo / "user-note.txt").read_text(), "preserve user content")
        self.assertFalse((self.base / "source_fastpt_cmake_dtk").exists())
        source = json.loads((self.repo / common.SOURCE_FILE).read_text())
        owner = json.loads((self.repo / common.OWNERSHIP_FILE).read_text())
        self.assertEqual(source["git_branch"], "hcu/v1.15.1")
        self.assertEqual(source["git_head"], self.base_sha)
        self.assertEqual(source["tree_sha256"], owner["source_sha256"])

    def test_no_git_cmake_default_copies_and_preserves_input(self):
        before = common.source_identity(self.repo)["tree_sha256"]
        result = self.cli("cmake_fastpt_convert.py", "--source-map", self.mapping, "--write")
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.base / "source_fastpt_cmake_dtk"
        self.assertEqual(Path(json.loads(result.stdout)["output"]), out.resolve())
        self.assertIn("LANGUAGES CXX HIP", (out / "CMakeLists.txt").read_text())
        self.assertEqual(common.source_identity(self.repo)["tree_sha256"], before)
        self.assertFalse((self.repo / common.SOURCE_FILE).exists())

    def test_no_git_native_default_is_visible_sibling(self):
        before = common.source_identity(self.repo)["tree_sha256"]
        out = common.prepare_tree(self.repo, route="native-hip")
        self.assertEqual(out, self.base / "source_adapted")
        self.assertEqual(out.parent, self.repo.parent)
        self.assertEqual(common.source_identity(self.repo)["tree_sha256"], before)
        self.assertTrue((out / "CMakeLists.txt").is_file())

    def test_git_dry_run_does_not_write_or_switch(self):
        self.initialize_git()
        before = common.source_identity(self.repo)["tree_sha256"]
        result = self.cli("cmake_fastpt_convert.py", "--source-map", self.mapping)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(common.source_identity(self.repo)["tree_sha256"], before)
        self.assertEqual(self.git("branch", "--show-current"), "hcu/v1.15.1")
        self.assertFalse((self.repo / common.SOURCE_FILE).exists())

    def test_git_different_out_rejected_even_in_dry_run(self):
        self.initialize_git()
        for script in ("cmake_fastpt_convert.py", "cutlass_to_hytlass_prepare.py",
                       "fastpt_non_transcode_prepare.py"):
            with self.subTest(script=script):
                result = self.cli(script, "--out", self.base / "other")
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse((self.base / "other").exists())
        self.assertFalse((self.repo / common.SOURCE_FILE).exists())

    def test_detached_head_rejected_before_metadata_write(self):
        self.initialize_git()
        self.git("switch", "--detach", "v1.15.1")
        with self.assertRaisesRegex(common.AdaptationError, "detached"):
            common.prepare_tree(self.repo, route="native-hip")
        self.assertFalse((self.repo / common.SOURCE_FILE).exists())
        self.assertEqual(self.git("diff", "--name-only"), "")

    def test_existing_generated_name_is_preserved(self):
        self.initialize_git()
        (self.repo / "build-helper.sh").write_text("user helper")
        with self.assertRaises(common.AdaptationError):
            common.prepare_tree(self.repo, route="native-hip",
                                reserved_names=("build-helper.sh",))
        self.assertFalse((self.repo / common.SOURCE_FILE).exists())
        self.assertEqual((self.repo / "build-helper.sh").read_text(), "user helper")

    def test_worktree_git_file_and_project_subdirectory_stay_in_place(self):
        self.initialize_git()
        checkout = self.base / "existing-worktree"
        self.git("worktree", "add", "-b", "hcu/other", str(checkout), "v1.15.1")
        self.assertTrue((checkout / ".git").is_file())
        project = checkout / "component"
        project.mkdir()
        (project / "kernel.hip").write_text("// component\r\n")
        self.assertEqual(common.prepare_tree(project, route="native-hip"), project.resolve())
        source = json.loads((project / common.SOURCE_FILE).read_text())
        self.assertEqual(source["git_branch"], "hcu/other")
        self.assertEqual(source["git_root"], str(checkout.resolve()))
        self.assertEqual(self.git("branch", "--show-current", cwd=checkout), "hcu/other")

    def test_unavailable_or_broken_git_is_not_treated_as_plain_directory(self):
        self.initialize_git()
        project = self.repo / "component"
        project.mkdir()
        with patch.object(common.shutil, "which", return_value=None):
            with self.assertRaises(common.AdaptationError):
                common.prepare_destination(project, None)
        failed = subprocess.CompletedProcess(["git"], 128, "", "cannot read git")
        with patch.object(common.subprocess, "run", return_value=failed):
            with self.assertRaises(common.AdaptationError):
                common.prepare_destination(project, None)
        self.assertFalse((project / common.SOURCE_FILE).exists())

    def test_nontranscode_cli_defaults_in_both_modes(self):
        for use_git in (False, True):
            with self.subTest(git=use_git):
                if use_git:
                    self.initialize_git()
                (self.repo / "setup.py").write_text("# CUDAExtension\r\n")
                result = self.cli(
                    "fastpt_non_transcode_prepare.py", "--fastpt", "/verified/fastpt",
                    "--gpufusion-cuda-home", "/verified/gpufusion/cuda", "--arch", "gfx936",
                    "--build-argv", '["cmake","--build","{BUILD_DIR}"]',
                    "--fastpt-installed", "true", "--version-compatible", "true",
                    "--fastpt-invocable", "true", "--write",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                out = self.repo if use_git else self.base / "source_fastpt_non_transcode"
                self.assertEqual(Path(json.loads(result.stdout)["output"]), out.resolve())
                self.assertTrue((out / "FASTPT_NON_TRANSCODE_PLAN.json").is_file())

    def test_hytlass_cli_defaults_in_both_modes(self):
        dependency = self.base / "hytlass"
        (dependency / "include" / "hytlass").mkdir(parents=True)
        (dependency / "include" / "hytlass" / "foo.h").write_text("// fixture\r\n")
        for use_git in (False, True):
            with self.subTest(git=use_git):
                if use_git:
                    self.initialize_git()
                (self.repo / "k.cu").write_text("#include <cutlass/foo.h>\r\ncutlass::Thing t;\r\n")
                result = self.cli(
                    "cutlass_to_hytlass_prepare.py", "--hytlass-root", dependency,
                    "--hytlass-version", "0.1.0", "--integration-mode", "headers",
                    "--target", "k", "--arch", "gfx936", "--write",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                out = self.repo if use_git else self.base / "source_hytlass"
                self.assertIn("hytlass::Thing", (out / "k.cu").read_text())
                if not use_git:
                    self.assertIn("cutlass::Thing", (self.repo / "k.cu").read_text())


if __name__ == "__main__":
    unittest.main()
