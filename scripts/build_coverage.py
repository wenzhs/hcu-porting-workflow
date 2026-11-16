"""Join declared configuration/include evidence, without certifying execution."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from adaptation_common import (AdaptationError, file_sha256, literal_relative_path,
                               read_text, reject_links, resolve_repo, source_identity,
                               validate_transform_map, workspace_path)

HEADERS = {".h", ".hh", ".hpp", ".hxx", ".cuh", ".inl", ".inc"}


def _path(value, base=None):
    if not isinstance(value, (str, Path)) or not str(value):
        raise AdaptationError("Expected a nonempty path")
    p = Path(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", str(value)):
        raise AdaptationError("Foreign Windows path: inspect evidence on its native host")
    if not p.is_absolute():
        if base is None:
            raise AdaptationError(f"Expected an absolute path: {value}")
        p = base / p
    return reject_links(p).resolve()


def _load(path, expected, evidence):
    p = _path(path)
    data = json.loads(read_text(p))
    if not isinstance(data, expected):
        raise AdaptationError(f"Invalid JSON type in {p}")
    evidence.append({"path": str(p), "sha256": file_sha256(p)})
    return data


def _reply(base, name):
    name = literal_relative_path(name)
    p = _path(name, base)
    if not p.is_relative_to(base):
        raise AdaptationError("File API JSON reference escapes reply directory")
    return p


def _fileapi(index, configuration, roots, evidence):
    ip = _path(index)
    idx = _load(ip, dict, evidence)
    objects = idx.get("objects")
    if not isinstance(objects, list):
        raise AdaptationError("File API index must contain objects")
    refs = [r for r in objects if isinstance(r, dict) and r.get("kind") == "codemodel"
            and isinstance(r.get("version"), dict) and r["version"].get("major") == 2]
    if len(refs) != 1:
        raise AdaptationError("Expected one codemodel v2 object in File API index")
    model = _load(_reply(ip.parent, refs[0].get("jsonFile")), dict, evidence)
    if (model.get("kind") != "codemodel" or not isinstance(model.get("version"), dict)
            or model["version"].get("major") != 2):
        raise AdaptationError("File API object is not codemodel v2")
    paths = model.get("paths", {})
    if not isinstance(paths, dict):
        raise AdaptationError("Invalid codemodel paths")
    srcroot, buildroot = _path(paths.get("source")), _path(paths.get("build"))
    if not any(srcroot == r or srcroot.is_relative_to(r) for r in roots):
        raise AdaptationError("Codemodel source root does not belong to this source/generated tree")
    configs = model.get("configurations")
    if not isinstance(configs, list) or not configs or any(not isinstance(c, dict) for c in configs):
        raise AdaptationError("Invalid codemodel configurations")
    if configuration is None:
        if len(configs) != 1:
            raise AdaptationError("Multiple configurations require --configuration")
        cfg = configs[0]
    else:
        matches = [c for c in configs if c.get("name") == configuration]
        if len(matches) != 1:
            raise AdaptationError("Requested configuration is missing or ambiguous")
        cfg = matches[0]
    references = cfg.get("targets", [])
    if not isinstance(references, list):
        raise AdaptationError("Invalid codemodel target references")
    targets, seen = [], set()
    for ref in references:
        if not isinstance(ref, dict):
            raise AdaptationError("Invalid target reference")
        data = _load(_reply(ip.parent, ref.get("jsonFile")), dict, evidence)
        name, identity = data.get("name"), data.get("id")
        if (not isinstance(name, str) or not name or not isinstance(identity, str) or not identity
                or ref.get("id") != identity or ref.get("name") != name or identity in seen):
            raise AdaptationError("Invalid, inconsistent or duplicate target identity")
        seen.add(identity)
        target_paths = data.get("paths", {})
        if not isinstance(target_paths, dict):
            raise AdaptationError("Invalid target paths")
        target = {"id": identity, "name": name, "type": data.get("type"),
                  "source_directory": str(_path(target_paths.get("source"), srcroot)),
                  "build_directory": str(_path(target_paths.get("build"), buildroot)), "sources": []}
        groups, sources = data.get("compileGroups", []), data.get("sources", [])
        if not isinstance(groups, list) or not isinstance(sources, list):
            raise AdaptationError("Invalid target compile groups/sources")
        for source in sources:
            if not isinstance(source, dict):
                raise AdaptationError("Invalid target source")
            # Source paths are relative to the TOP source root, not target.paths.source.
            file = _path(source.get("path"), srcroot)
            gi, language = source.get("compileGroupIndex"), None
            if gi is not None:
                if type(gi) is not int or not 0 <= gi < len(groups) or not isinstance(groups[gi], dict):
                    raise AdaptationError("Invalid compileGroupIndex")
                language = groups[gi].get("language")
                if not isinstance(language, str) or not language:
                    raise AdaptationError("Missing compile-group language")
            target["sources"].append({"file": str(file), "language": language,
                                      "configured_for_compilation": gi is not None})
        targets.append(target)
    return cfg.get("name"), targets


def parse_depfile(text):
    """Read Make dependency rules; reject variables/recipes rather than evaluate.

    Handles continuations, escaped spaces/hash/colon, Windows drive colons and
    empty -MP phony rules. Returns dependency names, never output targets.
    """
    text = re.sub(r"\\\r?\n", " ", text)
    dependencies, nonempty_rules = [], 0
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        before, after, word, split, i = [], [], [], False, 0
        while i < len(line):
            char = line[i]
            if char == "\\" and i + 1 < len(line):
                nxt = line[i + 1]
                if nxt in " \t#:;|\\":
                    word.append(nxt)
                    i += 2
                    continue
                word.append(char)
            elif char == "$":
                if i + 1 < len(line) and line[i + 1] == "$":
                    word.append("$")
                    i += 2
                    continue
                raise AdaptationError("Make variable expansion is unsupported in depfiles")
            elif char == "#":
                break
            elif char in ";|":
                raise AdaptationError("Make recipes/order-only dependencies are unsupported")
            elif char == ":" and not (len(word) == 1 and word[0].isalpha()
                                        and i + 1 < len(line) and line[i + 1] in "/\\"):
                if split:
                    raise AdaptationError("Unexpected unescaped colon in dependency rule")
                if word:
                    before.append("".join(word))
                    word = []
                split = True
            elif char.isspace():
                if word:
                    (after if split else before).append("".join(word))
                    word = []
            else:
                word.append(char)
            i += 1
        if word:
            (after if split else before).append("".join(word))
        if not split or not before or any(x in {"|", ";", "="} for x in before + after):
            raise AdaptationError("Unsupported Make dependency rule")
        if after:
            nonempty_rules += 1
            if nonempty_rules > 1:
                raise AdaptationError("Multiple nonempty dependency rules require per-unit evidence")
            dependencies.extend(after)
    if not dependencies:
        raise AdaptationError("Depfile has no dependency rule")
    return list(dict.fromkeys(dependencies))


def analyze(repo, *, compile_commands=None, changed_files=None, cmake_index=None,
            configuration=None, dependency_manifest=None, transform_map=None,
            generated_root=None, scope_ledger=None):
    root, evidence, warnings = resolve_repo(repo), [], []
    if (not isinstance(changed_files, list) or not changed_files
            or any(not isinstance(f, str) for f in changed_files)):
        raise AdaptationError("changed_files must be a nonempty array of relative paths")
    selected = list(dict.fromkeys(literal_relative_path(f) for f in changed_files))
    mapping, genroot = {}, None
    if transform_map is not None:
        if generated_root is None:
            raise AdaptationError("Coverage transform map requires generated_root")
        genroot = _path(generated_root)
        mapping = validate_transform_map(root, genroot, transform_map, require_outputs=True)
    elif generated_root is not None:
        raise AdaptationError("generated_root requires a transform map")
    db = {}
    if compile_commands:
        for record in _load(compile_commands, list, evidence):
            if not isinstance(record, dict):
                raise AdaptationError("Invalid compilation entry")
            cwd = _path(record.get("directory"))
            file = _path(record.get("file"), cwd)
            db.setdefault(file, []).append({"cwd": cwd})
    cfg, targets = None, []
    if cmake_index:
        cfg, targets = _fileapi(cmake_index, configuration, [root] + ([genroot] if genroot else []), evidence)
    elif configuration is not None:
        raise AdaptationError("--configuration requires --cmake-index")
    unit_targets = {}
    for target in targets:
        for source in target["sources"]:
            if source["configured_for_compilation"]:
                unit_targets.setdefault(Path(source["file"]), set()).add(target["id"])
    dependents, dep_unknown, dep_known = {}, [], set()
    if dependency_manifest:
        for item in _load(dependency_manifest, list, evidence):
            if not isinstance(item, dict):
                raise AdaptationError("Invalid dependency manifest entry")
            unit = _path(item.get("file"))
            directories = {r["cwd"] for r in db.get(unit, [])}
            declared = item.get("directory")
            if declared is not None:
                cwd = _path(declared)
                if cwd not in directories:
                    raise AdaptationError("Dependency directory does not match compile database")
            elif len(directories) == 1:
                cwd = next(iter(directories))
            else:
                dep_unknown.append({"file": str(unit), "reason": "missing_or_ambiguous_compile_directory"})
                continue
            depfile = _path(item.get("depfile"), cwd)
            try:
                text = read_text(depfile)
                evidence.append({"path": str(depfile), "sha256": file_sha256(depfile)})
                deps = {_path(d, cwd) for d in parse_depfile(text)}
                if unit not in deps:
                    raise AdaptationError("Depfile does not name its declared compilation unit")
            except (OSError, ValueError) as error:
                dep_unknown.append({"file": str(unit), "reason": str(error)})
                continue
            if len(db.get(unit, [])) == 1:
                dep_known.add(unit)
            else:
                dep_unknown.append({"file": str(unit), "reason": "ambiguous_compile_entries_for_depfile"})
            for dep in deps:
                dependents.setdefault(dep, set()).add(unit)
    ledger_entries, ledger_state = [], {"provided": False}
    if scope_ledger:
        ledger = _load(scope_ledger, dict, evidence)
        if ledger.get("kind") != "scope-ledger" or not isinstance(ledger.get("entries"), list):
            raise AdaptationError("Invalid scope ledger")
        match = (_path(ledger.get("repo")) == root
                 and ledger.get("source_sha256") == source_identity(root)["tree_sha256"])
        ledger_state = {"provided": True, "source_identity_match": match, "receipts_validated": False}
        ledger_ids = set()
        for entry in ledger["entries"]:
            if (not isinstance(entry, dict) or not isinstance(entry.get("id"), str)
                    or not entry["id"] or entry["id"] in ledger_ids):
                raise AdaptationError("Invalid scope ledger entry")
            ledger_ids.add(entry["id"])
            ledger_entries.append(entry)
    output = []
    for relative in selected:
        original = workspace_path(root, relative)
        aliases, generated = {original}, None
        if relative in mapping:
            generated = genroot / mapping[relative]
            aliases.add(generated)
        direct = aliases & db.keys()
        included_by = set().union(*(dependents.get(p, set()) for p in aliases))
        units = direct | included_by
        associated = []
        for target in targets:
            relevant = [s for s in target["sources"] if Path(s["file"]) in aliases
                        or (Path(s["file"]) in units and s["configured_for_compilation"])]
            if relevant:
                associated.append({**{k: v for k, v in target.items() if k != "sources"},
                                   "sources": relevant, "compile_db_join": "source_path_only"})
        gaps = []
        if not original.is_file():
            status = "missing_source"
        elif units:
            status = "configured"
        elif original.suffix.lower() in HEADERS:
            status = "unknown_header_dependencies"
        elif original.name in {"CMakeLists.txt", "Makefile", "setup.py", "pyproject.toml"} or original.suffix == ".cmake":
            status = "build_rule_requires_review"
        elif genroot and relative not in mapping:
            status = "unmapped_generated"
        elif compile_commands is None:
            status = "unknown_compile_database"
        else:
            status = "missing_compile_unit"
        if status != "configured":
            gaps.append(status)
        if units and not associated:
            gaps.append("target_unknown")
        if original.suffix.lower() in HEADERS and (set(db) - dep_known or dep_unknown):
            gaps.append("partial_dependency_evidence")
        ambiguous_units = any(len(db[p]) > 1 or len(unit_targets.get(p, set())) > 1 for p in units)
        if ambiguous_units:
            gaps.append("compile_db_target_identity_unverified")
            if original.suffix.lower() in HEADERS:
                gaps.append("ambiguous_target_dependencies")
                if "partial_dependency_evidence" not in gaps:
                    gaps.append("partial_dependency_evidence")
        names = sorted({t["name"] for t in associated})
        matches = [e for e in ledger_entries if e.get("type") in {"executable", "library", "module"}
                   and e.get("name") in names]
        declared = [{"id": e["id"], "name": e.get("name"), "declared_status": e.get("status"),
                     "receipt": e.get("receipt"), "validated_source_sha256": e.get("validated_source_sha256"),
                     "association": "target_name_candidate"} for e in matches]
        if scope_ledger and any(sum(e["name"] == name for e in declared) != 1 for name in names):
            gaps.append("ledger_target_mapping_requires_review")
        output.append({"file": str(original), "relative_file": relative, "status": status,
                       "generated_file": str(generated) if generated else None,
                       "compile_units": sorted(str(p) for p in units),
                       "included_by": sorted(str(p) for p in included_by - direct),
                       "targets": names, "target_details": associated, "gaps": gaps,
                       "ledger_entries": declared, "compiled": False, "executed": False,
                       "build_evidence": "not_verified", "runtime_evidence": "not_verified"})
    if dep_unknown:
        warnings.append("Some dependency records are unknown; header coverage may be incomplete")
    return {"kind": "build-coverage", "read_only": True, "executed": False,
            "repo": str(root), "configuration": cfg, "files": output,
            "file_api_targets": len(targets), "ledger": ledger_state,
            "dependency_unknown": dep_unknown, "evidence": evidence, "warnings": warnings,
            "limits": "Declared configuration/include relationships only; no build execution, "
                      "receipt validation, template instantiation or GPU coverage is inferred."}
