from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import adaptation_common as common
import build_coverage as coverage
import inspect_build_context as context


class BuildCoverageTests(unittest.TestCase):
    def test_inline_make_recipe_is_unknown_and_never_parsed_as_headers(self):
        for body in ("a.o: a.cpp ;echo phantom.h", "a.o: a.cpp;echo phantom.h"):
            with self.subTest(body=body), self.assertRaises(common.AdaptationError):
                coverage.parse_depfile(body)

    def test_cli_reports_optional_gap_and_preserves_legacy_missing_file_error(self):
        db = self.write("cli-db.json", [{"file": "a.cpp", "directory": str(self.root),
                                        "arguments": ["c++", "-c", "a.cpp"]}])
        changes = self.write("changes.json", ["sub/k.hip"])
        cli = [sys.executable, "-B", "-I", str(ROOT / "scripts/inspect_build_context.py"),
               "--repo", str(self.root), "--compile-commands", str(db)]
        result = subprocess.run(cli + ["--changed-files", str(changes)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(result.stdout)["coverage"]["files"][0]
        self.assertEqual(row["status"], "missing_compile_unit")
        self.assertEqual(row["runtime_evidence"], "not_verified")
        legacy = subprocess.run(cli + ["--file", "sub/k.hip"], capture_output=True, text=True)
        self.assertEqual(legacy.returncode, 2)
        self.assertEqual(json.loads(legacy.stderr)["kind"], "build-context-error")

    def test_multiple_dep_rules_cannot_attach_another_objects_header(self):
        db = self.write("rules-db.json", [{"file": "a.cpp", "directory": str(self.root),
                                          "arguments": ["c++", "-c", "a.cpp"]}])
        (self.root / "multiple.d").write_text("a.o: a.cpp\nb.o: unrelated.cpp b.h\n")
        manifest = self.write("rules-deps.json", [{"file": str(self.root / "a.cpp"), "depfile": "multiple.d"}])
        row = coverage.analyze(self.root, changed_files=["b.h"], compile_commands=db,
                               dependency_manifest=manifest)["files"][0]
        self.assertEqual(row["included_by"], [])
        self.assertEqual(row["status"], "unknown_header_dependencies")

    def test_one_depfile_cannot_close_multiple_macro_configurations(self):
        db = self.write("macros-db.json", [
            {"file": "a.cpp", "directory": str(self.root), "arguments": ["c++", "-DONE", "-c", "a.cpp"]},
            {"file": "a.cpp", "directory": str(self.root), "arguments": ["c++", "-DTWO", "-c", "a.cpp"]}])
        (self.root / "one.d").write_text("one.o: a.cpp b.h\n")
        manifest = self.write("macros-deps.json", [{"file": str(self.root / "a.cpp"), "depfile": "one.d"}])
        row = coverage.analyze(self.root, changed_files=["b.h"], compile_commands=db,
                               dependency_manifest=manifest)["files"][0]
        self.assertIn("ambiguous_target_dependencies", row["gaps"])
        self.assertIn("partial_dependency_evidence", row["gaps"])
        self.assertFalse(row["compiled"])

    def test_file_api_json_reference_cannot_escape_reply_directory(self):
        index = self.write("escaped-index.json", {"objects": [
            {"kind": "codemodel", "version": {"major": 2}, "jsonFile": "../outside.json"}]})
        with self.assertRaises(common.AdaptationError):
            coverage.analyze(self.root, changed_files=["a.cpp"], cmake_index=index)
        model = {"kind": "codemodel", "version": {"major": 2},
                 "paths": {"source": str(self.root), "build": str(self.root.parent)},
                 "configurations": [{"name": "Release", "targets": [
                     {"name": "escape", "id": "escape::1", "jsonFile": "../outside.json"}]}]}
        self.write("safe-model.json", model)
        index = self.write("safe-index.json", {"objects": [
            {"kind": "codemodel", "version": {"major": 2}, "jsonFile": "safe-model.json"}]})
        with self.assertRaises(common.AdaptationError):
            coverage.analyze(self.root, changed_files=["a.cpp"], cmake_index=index)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="coverage ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "src"
        (self.root / "sub" / "generated").mkdir(parents=True)
        (self.root / "CMakeLists.txt").write_text("project(x LANGUAGES CXX HIP)\n", encoding="utf-8")
        for name in ("a.cpp", "b.h", "sub/k.hip", "sub/generated/k.cpp"):
            (self.root / name).write_text("// fixture\n", encoding="utf-8")

    def write(self, name, value):
        p = self.root.parent / name
        p.write_text(json.dumps(value), encoding="utf-8")
        return p

    def test_changed_files_and_missing_dependency_are_conservative(self):
        db = self.write("compile.json", [{"file": str(self.root / "a.cpp"), "directory": str(self.root), "arguments": ["c++", "-c", "a.cpp"]}])
        with self.assertRaises((common.AdaptationError, TypeError)):
            coverage.analyze(self.root, compile_commands=db, changed_files=json.dumps(["a.cpp", "b.h"]))
        out = coverage.analyze(self.root, compile_commands=db, changed_files=["a.cpp", "b.h"])
        by = {Path(x["file"]).name: x for x in out["files"]}
        self.assertEqual(by["a.cpp"]["status"], "configured")
        self.assertEqual(by["b.h"]["status"], "unknown_header_dependencies")
        self.assertFalse(by["b.h"]["compiled"])
        self.assertFalse(by["b.h"]["executed"])

    def test_fileapi_configuration_and_target_source_mapping(self):
        build = self.root.parent / "build"
        build.mkdir()
        index = self.write("index.json", {"objects": [{"kind": "codemodel", "version": {"major": 2, "minor": 0}, "jsonFile": "codemodel.json"}]})
        (index.parent / "target.json").write_text(json.dumps({"id": "hip-kernel::1", "name": "hip-kernel", "type": "EXECUTABLE", "paths": {"source": str(self.root), "build": str(build)}, "sources": [{"path": "sub/k.hip", "compileGroupIndex": 0}], "compileGroups": [{"language": "HIP"}]}), encoding="utf-8")
        (index.parent / "codemodel.json").write_text(json.dumps({"kind": "codemodel", "version": {"major": 2}, "paths": {"source": str(self.root), "build": str(build)}, "configurations": [{"name": "Debug", "targets": [{"id": "hip-kernel::1", "name": "hip-kernel", "jsonFile": "target.json"}]}, {"name": "Release", "targets": []}]}), encoding="utf-8")
        with self.assertRaises(common.AdaptationError):
            coverage.analyze(self.root, cmake_index=index, changed_files=["sub/k.hip"])
        out = coverage.analyze(self.root, cmake_index=index, configuration="Debug", changed_files=["sub/k.hip"])
        self.assertEqual(out["configuration"], "Debug")
        self.assertEqual(out["files"][0]["targets"], ["hip-kernel"])

    def test_transform_and_dependency_manifest_preserve_unknowns(self):
        dep = self.root / "a.d"
        dep.write_text("a.o: a.cpp b.h missing.h\\\n continued.h\n", encoding="utf-8")
        manifest = self.write("deps.json", [{"file": str(self.root / "a.cpp"), "depfile": "a.d"}])
        tm = {"a.cpp": "k.cpp"}
        db = self.write("dep-db.json", [{"file": "a.cpp", "directory": str(self.root), "arguments": ["c++", "-c", "a.cpp"]}])
        out = coverage.analyze(self.root, changed_files=["a.cpp", "b.h", "sub/generated/k.cpp"], compile_commands=db, dependency_manifest=manifest, transform_map=tm, generated_root=self.root / "sub/generated")
        by = {Path(x["file"]).name: x for x in out["files"]}
        self.assertEqual(by["k.cpp"]["status"], "unmapped_generated")
        self.assertIn(str(self.root / "a.cpp"), by["a.cpp"]["compile_units"])
        self.assertIn(str(self.root / "a.cpp"), by["b.h"]["included_by"])

    def test_paths_escape_and_legacy_file_selection_are_rejected(self):
        with self.assertRaises(common.AdaptationError):
            coverage.analyze(self.root, changed_files=["../outside.cpp"])
        db = self.write("empty.json", [])
        with self.assertRaises(common.AdaptationError):
            context.inspect_context(self.root, compile_commands=db, files=["missing.cpp"])

    def test_header_without_compile_db_is_unknown_not_pass(self):
        out = coverage.analyze(self.root, changed_files=["b.h"])
        self.assertEqual(out["files"][0]["status"], "unknown_header_dependencies")
        self.assertFalse(out["files"][0]["compiled"])

    def test_build_rule_and_optional_source_are_not_hidden(self):
        out = coverage.analyze(self.root, changed_files=["CMakeLists.txt", "sub/k.hip"])
        self.assertEqual(out["files"][0]["status"], "build_rule_requires_review")
        self.assertEqual(out["files"][1]["status"], "unknown_compile_database")

    def test_optional_existing_source_missing_from_nonempty_db_is_missing_unit(self):
        db = self.write("only-a.json", [{"file": "a.cpp", "directory": str(self.root), "arguments": ["c++", "-c", "a.cpp"]}])
        out = coverage.analyze(self.root, compile_commands=db, changed_files=["sub/k.hip"])
        self.assertEqual(out["files"][0]["status"], "missing_compile_unit")

    def test_sibling_generated_root_maps_original_to_generated_unit(self):
        gen = self.root.parent / "generated-output"
        gen.mkdir(); (gen / "k.hip").write_text("// generated\n", encoding="utf-8")
        db = self.write("generated-db.json", [{"file": str(gen / "k.hip"), "directory": str(gen), "arguments": ["hipcc", "-c", "k.hip"]}])
        out = coverage.analyze(self.root, compile_commands=db, changed_files=["a.cpp"], transform_map={"a.cpp": "k.hip"}, generated_root=gen)
        row = out["files"][0]
        self.assertEqual(row["status"], "configured")
        self.assertEqual(row["generated_file"], str(gen / "k.hip"))

    def test_two_depfiles_include_escaped_space_header(self):
        header = self.root / "b space.h"; header.write_text("// h\n", encoding="utf-8")
        (self.root / "one.d").write_text("one.o: one.cpp b\\ space.h\\\n extra.h\n", encoding="utf-8")
        (self.root / "two.d").write_text("two.o: two.cpp b\\ space.h\n", encoding="utf-8")
        db = self.write("two-db.json", [{"file": "one.cpp", "directory": str(self.root), "arguments": ["c++", "-c", "one.cpp"]}, {"file": "two.cpp", "directory": str(self.root), "arguments": ["c++", "-c", "two.cpp"]}])
        manifest = self.write("two-deps.json", [{"file": str(self.root / "one.cpp"), "depfile": "one.d"}, {"file": str(self.root / "two.cpp"), "depfile": "two.d"}])
        out = coverage.analyze(self.root, compile_commands=db, dependency_manifest=manifest, changed_files=["b space.h"])
        self.assertEqual(set(out["files"][0]["included_by"]), {str(self.root / "one.cpp"), str(self.root / "two.cpp")})

    def test_depfile_parser_handles_windows_drives_and_phony_rules(self):
        parsed = coverage.parse_depfile("C:/out.o: C:/src/a.cpp C:\\\\inc\\b\\ file.h foo\\:bar\n\nphony:\n")
        self.assertIn("C:/src/a.cpp", parsed)
        self.assertIn("foo:bar", parsed)
        self.assertNotIn("phony", parsed)

    def test_ledger_stale_identity_does_not_certify_execution(self):
        ledger = self.write("stale-ledger.json", {"kind": "scope-ledger", "repo": str(self.root), "source_sha256": "f" * 64, "entries": [{"id": "library:a", "type": "library", "name": "a", "status": "covered"}]})
        out = coverage.analyze(self.root, changed_files=["a.cpp"], scope_ledger=ledger)
        self.assertFalse(out["ledger"]["source_identity_match"])
        self.assertFalse(out["files"][0]["compiled"]); self.assertFalse(out["files"][0]["executed"])

    def test_same_source_can_belong_to_multiple_targets(self):
        build = self.root.parent / "multi-build"
        build.mkdir()
        for name in ("a.json", "b.json"):
            (build / name).write_text(json.dumps({"id": name, "name": name[0], "type": "STATIC_LIBRARY", "paths": {"source": str(self.root), "build": str(build)}, "sources": [{"path": "sub/k.hip", "compileGroupIndex": 0}], "compileGroups": [{"language": "HIP"}]}), encoding="utf-8")
        index = self.write("multi-index.json", {"objects": [{"kind": "codemodel", "version": {"major": 2}, "jsonFile": "multi-cm.json"}]})
        (index.parent / "a.json").write_text((build / "a.json").read_text(), encoding="utf-8")
        (index.parent / "b.json").write_text((build / "b.json").read_text(), encoding="utf-8")
        (index.parent / "multi-cm.json").write_text(json.dumps({"kind": "codemodel", "version": {"major": 2}, "paths": {"source": str(self.root), "build": str(build)}, "configurations": [{"name": "Release", "targets": [{"id": "a.json", "name": "a", "jsonFile": "a.json"}, {"id": "b.json", "name": "b", "jsonFile": "b.json"}]}]}), encoding="utf-8")
        row = coverage.analyze(self.root, changed_files=["sub/k.hip"], cmake_index=index)
        self.assertEqual(row["files"][0]["targets"], ["a", "b"])

    def test_unknown_configuration_and_escaped_reply_are_rejected(self):
        build = self.root.parent / "cfg-build"
        build.mkdir()
        index = self.write("cfg-index.json", {"objects": [{"kind": "codemodel", "version": {"major": 2}, "jsonFile": "cfg-cm.json"}]})
        (index.parent / "cfg-cm.json").write_text(json.dumps({"kind": "codemodel", "version": {"major": 2}, "paths": {"source": str(self.root), "build": str(build)}, "configurations": [{"name": "Release", "targets": []}]}), encoding="utf-8")
        with self.assertRaises(common.AdaptationError):
            coverage.analyze(self.root, changed_files=["a.cpp"], cmake_index=index, configuration="Debug")

    def test_scope_ledger_is_read_only_and_receipts_unvalidated(self):
        ledger = self.write("ledger.json", {"kind": "scope-ledger", "repo": str(self.root), "source_sha256": "0" * 64, "entries": [{"id": "library:a", "type": "library", "name": "a", "status": "passed"}]})
        out = coverage.analyze(self.root, changed_files=["sub/k.hip"], scope_ledger=ledger)
        self.assertFalse(out["ledger"]["receipts_validated"])

    def test_unsupported_depfile_variable_remains_unknown(self):
        db = self.write("db.json", [{"file": "a.cpp", "directory": str(self.root), "arguments": ["c++", "-c", "a.cpp"]}])
        (self.root / "bad.d").write_text("a.o: a.cpp $@\n", encoding="utf-8")
        manifest = self.write("baddeps.json", [{"file": str(self.root / "a.cpp"), "depfile": "bad.d"}])
        out = coverage.analyze(self.root, changed_files=["a.cpp"], compile_commands=db, dependency_manifest=manifest)
        self.assertTrue(out["warnings"] or out["dependency_unknown"])

    def _two_target_api(self, b_source=True):
        base = self.root.parent; build = base / "amb-build"; build.mkdir()
        for ident, name, group in (("A::1", "A", 0), ("B::1", "B", 0 if b_source else None)):
            data = {"id": ident, "name": name, "type": "STATIC_LIBRARY", "paths": {"source": str(self.root), "build": str(build)}, "sources": [{"path": "a.cpp", **({"compileGroupIndex": group} if group is not None else {})}], "compileGroups": ([{"language": "HIP"}] if group is not None else [])}
            (base / (name + "-target.json")).write_text(json.dumps(data), encoding="utf-8")
        cm = {"kind": "codemodel", "version": {"major": 2}, "paths": {"source": str(self.root), "build": str(build)}, "configurations": [{"name": "Release", "targets": [{"id": "A::1", "name": "A", "jsonFile": "A-target.json"}, {"id": "B::1", "name": "B", "jsonFile": "B-target.json"}]}]}
        (base / "amb-cm.json").write_text(json.dumps(cm), encoding="utf-8")
        return self.write("amb-index.json", {"objects": [{"kind": "codemodel", "version": {"major": 2}, "jsonFile": "amb-cm.json"}]})

    def test_ambiguous_targets_are_not_certified_by_one_compile_db_unit(self):
        index = self._two_target_api()
        (self.root / "a.d").write_text("a.o: a.cpp b.h\n", encoding="utf-8")
        db = self.write("amb-db.json", [{"file": "a.cpp", "directory": str(self.root), "arguments": ["c++", "-c", "a.cpp"]}])
        manifest = self.write("amb-deps.json", [{"file": str(self.root / "a.cpp"), "depfile": "a.d"}])
        out = coverage.analyze(self.root, changed_files=["b.h"], compile_commands=db, cmake_index=index, dependency_manifest=manifest)
        self.assertIn("ambiguous_target_dependencies", out["files"][0]["gaps"])
        self.assertIn("partial_dependency_evidence", out["files"][0]["gaps"])

    def test_target_without_compile_group_does_not_claim_header_target(self):
        index = self._two_target_api(b_source=False)
        (self.root / "a.d").write_text("a.o: a.cpp b.h\n", encoding="utf-8")
        db = self.write("one-db.json", [{"file": "a.cpp", "directory": str(self.root), "arguments": ["c++", "-c", "a.cpp"]}])
        manifest = self.write("one-deps.json", [{"file": str(self.root / "a.cpp"), "depfile": "a.d"}])
        out = coverage.analyze(self.root, changed_files=["b.h"], compile_commands=db, cmake_index=index, dependency_manifest=manifest)
        self.assertEqual(out["files"][0]["targets"], ["A"])

    def test_duplicate_ledger_names_require_mapping_review(self):
        index = self._two_target_api()
        ledger = self.write("amb-ledger.json", {"kind": "scope-ledger", "repo": str(self.root), "source_sha256": common.source_identity(self.root)["tree_sha256"], "entries": [{"id": "library:a1", "type": "library", "name": "A", "status": "owed"}, {"id": "library:a2", "type": "library", "name": "A", "status": "owed"}]})
        out = coverage.analyze(self.root, changed_files=["a.cpp"], cmake_index=index, scope_ledger=ledger)
        self.assertIn("ledger_target_mapping_requires_review", out["files"][0]["gaps"])


if __name__ == "__main__":
    unittest.main()
