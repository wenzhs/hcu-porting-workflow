#!/usr/bin/env python3
"""Scope ledger: every target/feature is covered, approved out of scope, owed or blocked.

Prevents silent scope shrinking and stale "done" claims. A component counts as
covered only with an adaptation-result receipt that validates against the
CURRENT source tree identity; any source change turns old coverage stale.
Enumeration is a static CMake/setup.py heuristic. Add features by hand with
--feature; an empty ledger checks nothing and is never reported complete.

  init   create a new ledger from the project
  set    change one entry's status (scoped_out needs reason + user approval)
  check  exit 0 complete, 3 work remains, 2 invalid/stale
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from adaptation_common import AdaptationError, SCHEMA_VERSION, STAGES, file_sha256, read_text, reject_links, resolve_repo, source_files, source_identity, validate_output_path, validate_receipt

STATUSES = ("covered", "scoped_out", "owed", "blocked")
CMAKE_COMMANDS = {
    "library": r"(?:cuda_|hip_)?add_library", "executable": r"(?:cuda_|hip_)?add_executable", "test": r"add_test",
    "module": r"pybind11_add_module|torch_python_module",
}
# CMake names that are variable/template references rather than real target names.
_CMAKE_TEMPLATE_RE = re.compile(r"\$\{|\$<")
BASE_STAGES = ["build", "load", "gpu", "correctness"]
EXTENSION = re.compile(r"\b(?:CUDAExtension|CppExtension|Extension)\s*\(\s*(?:name\s*=\s*)?[\"']([\w.]+)[\"']")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def enumerate_components(root: Path, target: str | None) -> list[dict[str, Any]]:
    base = (root / target).resolve() if target else root
    if target and (Path(target).is_absolute() or ".." in Path(target).parts or not base.is_dir()):
        raise AdaptationError("target must name a project-relative directory")
    found: list[dict[str, Any]] = []
    for path in source_files(base, include_dependencies=False):
        relative = path.relative_to(root).as_posix()
        if path.name == "CMakeLists.txt" or path.suffix == ".cmake":
            text = re.sub(r"(?m)#.*$", "", read_text(path))
            for kind, command in CMAKE_COMMANDS.items():
                for match in re.finditer(rf"(?i)\b(?:{command})\s*\(\s*(?:NAME\s+)?([^\s)]+)", text):
                    name = match.group(1).strip('\"')
                    # Skip variable/generator-expression references and common template patterns.
                    if _CMAKE_TEMPLATE_RE.search(name):
                        continue
                    found.append({"type": kind, "name": name, "file": relative, "line": text.count("\n", 0, match.start()) + 1})
        elif path.name == "setup.py":
            text = read_text(path)
            for match in EXTENSION.finditer(text):
                found.append({"type": "extension", "name": match.group(1), "file": relative, "line": text.count("\n", 0, match.start()) + 1})
    return found


def _entry(item: dict[str, Any], taken: set[str]) -> dict[str, Any]:
    base = f"{item['type']}:{item['name']}"
    entry_id = base
    if entry_id in taken:
        base += f"@{item.get('file', 'manual')}:{item.get('line', 0)}"
        entry_id = base
        ordinal = 2
        while entry_id in taken:
            entry_id = f"{base}#{ordinal}"
            ordinal += 1
    taken.add(entry_id)
    stages = BASE_STAGES + ([item["name"]] if item["type"] == "feature" and item["name"] in STAGES and item["name"] not in BASE_STAGES else [])
    return {"id": entry_id, **item, "status": "owed", "route": None, "reason": None, "approval": None,
            "required_stages": stages, "receipt": None, "validated_source_sha256": None}


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    file = reject_links(path)
    data = json.loads(read_text(file))
    if not isinstance(data, dict) or data.get("kind") != "scope-ledger" or not isinstance(data.get("entries"), list):
        raise AdaptationError("Not a scope ledger")
    ids = [e.get("id") if isinstance(e, dict) else None for e in data["entries"]]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise AdaptationError("Ledger entry IDs must be nonempty and unique; regenerate ambiguous legacy entries")
    return file, data


def _coverage_errors(record, entry, source_sha, restriction=None):
    errors = validate_receipt(record, expected_source_sha256=source_sha)
    if not isinstance(record, dict):
        return errors
    scope = record.get("covered_entries")
    if (not isinstance(scope, list) or not scope
            or any(not isinstance(i, str) or not i.strip() for i in scope)):
        errors.append("covered_entries must explicitly list full ledger IDs; target_features labels alone are not coverage")
        scope = []
    elif len(set(scope)) != len(scope):
        errors.append("covered_entries contains duplicate IDs")
    if entry["id"] not in scope:
        errors.append(f"receipt does not cover entry {entry['id']!r}")
    if restriction is not None:
        if (not isinstance(restriction, list) or not restriction
                or any(not isinstance(i, str) or not i for i in restriction)):
            errors.append("scope restriction must be a nonempty list of full entry IDs")
        elif entry["id"] not in restriction or not set(restriction).issubset(scope):
            errors.append("CLI scope must include this entry and may only narrow covered_entries")
    required = entry.get("required_stages", BASE_STAGES)
    if not isinstance(required, list) or any(s not in STAGES for s in required):
        errors.append("invalid ledger required_stages")
    else:
        required = list(dict.fromkeys(BASE_STAGES + required))
        if entry.get("type") == "feature" and entry.get("name") in STAGES:
            required = list(dict.fromkeys(required + [entry["name"]]))
        stages = record.get("stages")
        for stage in required:
            if not isinstance(stages, dict) or stages.get(stage) != "passed":
                errors.append(f"entry contract stage has not passed: {stage}")
    if entry.get("route") is not None and entry["route"] != record.get("route"):
        errors.append("entry route differs from receipt route")
    return errors


def _save(path: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = _now()
    handle, temporary = tempfile.mkstemp(prefix=".ledger-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=True, indent=2) + "\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def init_ledger(repo: str | Path, out: str | Path, *, target: str | None = None, features: list[str] | None = None,
                requirements: dict[str, list[str]] | None = None) -> dict[str, Any]:
    root, destination = validate_output_path(repo, out)
    if destination.exists():
        raise AdaptationError("Ledger already exists; choose a new path or use 'set' on it")
    taken: set[str] = set()
    items = enumerate_components(root, target) + [{"type": "feature", "name": name} for name in (features or [])]
    ledger = {
        "schema_version": SCHEMA_VERSION, "kind": "scope-ledger", "repo": str(root), "target": target,
        "source_sha256": source_identity(root)["tree_sha256"], "created_at": _now(), "history": [],
        "enumeration_coverage": "Static CMake/setup.py targets outside dependency directories; dynamically generated targets, other build systems and unlisted features must be added with --feature.",
        "entries": [_entry(item, taken) for item in items],
    }
    if requirements is not None:
        if not isinstance(requirements, dict):
            raise AdaptationError("requirements must map full entry IDs to stage lists")
        entries = {entry["id"]: entry for entry in ledger["entries"]}
        for entry_id, stages in requirements.items():
            if entry_id not in entries or not isinstance(stages, list) or any(s not in STAGES for s in stages):
                raise AdaptationError(f"Invalid entry/stage requirement: {entry_id!r}")
            entries[entry_id]["required_stages"] = list(dict.fromkeys(entries[entry_id]["required_stages"] + stages))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(ledger, ensure_ascii=True, indent=2) + "\n")
    return ledger


def set_status(ledger_path: str | Path, entry_id: str, status: str, *, route: str | None = None, reason: str | None = None, approval: str | None = None, receipt: str | Path | None = None, target_features: list[str] | None = None) -> dict[str, Any]:
    path, ledger = _load(ledger_path)
    validate_output_path(ledger["repo"], path)  # the ledger must stay outside the protected source
    entry = next((e for e in ledger["entries"] if e["id"] == entry_id), None)
    if entry is None:
        raise AdaptationError(f"No such entry: {entry_id}")
    if status not in STATUSES:
        raise AdaptationError(f"status must be one of {STATUSES}")
    if status in ("scoped_out", "blocked") and not (reason and reason.strip()):
        raise AdaptationError(f"{status} requires a reason")
    if status == "scoped_out" and not (approval and approval.strip()):
        raise AdaptationError("scoped_out requires the user's recorded approval; unfinished work is 'owed', not out of scope")
    validated = None
    if status == "covered":
        if receipt is None:
            raise AdaptationError("covered requires an adaptation-result receipt")
        receipt_path = reject_links(receipt)
        receipt_data = json.loads(read_text(receipt_path))
        current = source_identity(resolve_repo(ledger["repo"]))["tree_sha256"]
        if current != ledger["source_sha256"]:
            raise AdaptationError("Source changed since ledger init; revalidate the current source with the agreed scope")
        intended = {**entry, "route": route if route is not None else entry.get("route")}
        errors = _coverage_errors(receipt_data, intended, current, target_features)
        if errors:
            raise AdaptationError("Receipt rejected: " + "; ".join(errors))
        route = receipt_data["route"]
        validated = current
        entry["scope_restriction"] = target_features
        entry["receipt"] = {"path": str(receipt_path), "sha256": file_sha256(receipt_path)}
    else:
        entry["receipt"] = None
        entry.pop("scope_restriction", None)
    ledger["history"].append({"id": entry_id, "from": entry["status"], "to": status, "at": _now()})
    entry.update({"status": status, "route": route if route is not None else entry["route"], "reason": reason, "approval": approval if status == "scoped_out" else None, "validated_source_sha256": validated})
    _save(path, ledger)
    return entry


def check_ledger(ledger_path: str | Path, repo: str | Path | None = None) -> dict[str, Any]:
    _, ledger = _load(ledger_path)
    current = source_identity(resolve_repo(repo or ledger["repo"]))["tree_sha256"]
    source_changed = current != ledger["source_sha256"]
    problems: list[dict[str, str]] = []
    pending: list[str] = []
    for entry in ledger["entries"]:
        status = entry["status"]
        if status in ("owed", "blocked"):
            pending.append(entry["id"])
        elif status == "scoped_out":
            if not (entry.get("reason") and entry.get("approval")):
                problems.append({"id": entry["id"], "problem": "scoped_out without reason/approval"})
        elif status == "covered":
            if source_changed:
                problems.append({"id": entry["id"], "problem": "stale: source tree changed since the receipt was validated"})
                continue
            record = entry.get("receipt") or {}
            try:
                if file_sha256(record.get("path", "")) != record.get("sha256"):
                    raise AdaptationError("receipt file changed")
                errors = _coverage_errors(json.loads(read_text(record["path"])), entry, current, entry.get("scope_restriction"))
                if errors:
                    raise AdaptationError("; ".join(errors))
            except (OSError, ValueError, AdaptationError) as exc:
                problems.append({"id": entry["id"], "problem": f"receipt no longer valid: {exc}"})
        else:
            problems.append({"id": entry["id"], "problem": f"unknown status {status!r}"})
    counts = {status: sum(1 for e in ledger["entries"] if e["status"] == status) for status in STATUSES}
    warnings = [] if ledger["entries"] else ["The ledger is empty and checks nothing; add components or --feature entries before claiming completeness."]
    complete = bool(ledger["entries"]) and not pending and not problems
    return {"kind": "scope-ledger-check", "complete": complete, "source_changed": source_changed, "counts": counts, "pending": pending, "problems": problems, "warnings": warnings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--repo", required=True)
    init.add_argument("--out", required=True, help="NEW ledger path outside the project")
    init.add_argument("--target")
    init.add_argument("--feature", action="append", default=[])
    init.add_argument("--requirements", help="JSON mapping full entry IDs to additional required stages")
    update = sub.add_parser("set")
    update.add_argument("--ledger", required=True)
    update.add_argument("--id", required=True)
    update.add_argument("--status", required=True, choices=STATUSES)
    update.add_argument("--route")
    update.add_argument("--reason")
    update.add_argument("--approval", help="Where/when the user approved this scope reduction")
    update.add_argument("--receipt", help="adaptation-result JSON proving coverage")
    update.add_argument("--target-features", nargs="+", metavar="ENTRY_ID", help="Optional full entry IDs; may only narrow receipt covered_entries, never override it")
    verify = sub.add_parser("check")
    verify.add_argument("--ledger", required=True)
    verify.add_argument("--repo", help="Project to compare against (default: the ledger's repo)")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            requirements = json.loads(read_text(args.requirements)) if args.requirements is not None else None
            if args.requirements is not None and not isinstance(requirements, dict):
                raise AdaptationError("--requirements must contain a JSON object mapping entry IDs to stage lists")
            ledger = init_ledger(args.repo, args.out, target=args.target, features=args.feature, requirements=requirements)
            print(json.dumps({"kind": "scope-ledger-created", "entries": len(ledger["entries"]), "source_sha256": ledger["source_sha256"], "coverage": ledger["enumeration_coverage"]}, ensure_ascii=True, indent=2))
            return 0
        if args.command == "set":
            entry = set_status(args.ledger, args.id, args.status, route=args.route, reason=args.reason, approval=args.approval, receipt=args.receipt, target_features=getattr(args, "target_features", None))
            print(json.dumps(entry, ensure_ascii=True, indent=2))
            return 0
        report = check_ledger(args.ledger, args.repo)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0 if report["complete"] else (2 if report["problems"] else 3)
    except (OSError, ValueError, KeyError, AdaptationError) as exc:
        print(json.dumps({"kind": "scope-ledger-error", "error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
