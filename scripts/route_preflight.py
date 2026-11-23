#!/usr/bin/env python3
"""Read-only route selection and planning. No activation or builds.

Default stdout is a plan, never a success receipt. --strict returns 2 for blocked
and 3 for needs-information; without it, exit 0 only means a report was produced.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
MANIFEST_NAME = "skill-package.json"
ROUTES = ("native-hip", "non-transcode", "transcode")
MODULES = {"native-hip": "native_hip", "non-transcode": "non_transcode", "transcode": "transcode"}


# Source-tree imports are trusted development code. Exported public CLIs receive
# an inline integrity gate BEFORE this code and all package-local imports.
sys.path.insert(0, str(SCRIPT_DIR))
from package_integrity import verify_package as _verify_package


def verify_package(root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    return _verify_package(root)


def _minor(version: Any) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", str(version)) if version else None
    return (int(match[1]), int(match[2])) if match else None


def _fastpt_unavailable(capability: dict[str, Any]) -> bool:
    from adaptation_common import AdaptationError, version_series
    if not isinstance(capability, dict):
        raise AdaptationError("capabilities.fastpt must be an object when selecting a CUDA-facing route")
    if any(capability.get(key) is False for key in ("installed", "version_compatible", "invocable", "feature_probed")):
        return True
    actual, required = version_series(capability.get("torch_version")), version_series(capability.get("required_torch_version"))
    return actual is not None and required is not None and actual != required


def plan_request(request: dict[str, Any], requested_route: str = "auto", *, available_routes: tuple[str, ...] | list[str] = ROUTES) -> dict[str, Any]:
    from adaptation_common import AdaptationError, SCHEMA_VERSION, normalize_arch, resolve_repo, scan_project, source_identity, validate_build_command

    if not isinstance(request, dict) or not request.get("repo"):
        raise AdaptationError("request.repo is required")
    item = dict(request)
    repo = resolve_repo(item["repo"])
    if "arch" in item:
        item["arch"] = normalize_arch(item["arch"])
    item["build_command"] = validate_build_command(item.get("build_command"))
    capabilities = item.get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise AdaptationError("capabilities must be an object")
    # A direct route must not validate unrelated providers. Its handler owns
    # relevant capability fields; auto only inspects the fields needed to choose.
    if requested_route == "cutlass-hytlass":
        if item.get("specialization") not in (None, "cutlass-hytlass"):
            raise AdaptationError("Conflicting specialization for legacy cutlass-hytlass route")
        requested_route = "transcode"
        item["specialization"] = "cutlass-hytlass"
    if requested_route not in ("auto", *ROUTES):
        raise AdaptationError("Unknown route")
    if requested_route != "transcode" and (item.get("provider") or item.get("specialization")):
        raise AdaptationError("provider/specialization requires explicit transcode route")
    scan = scan_project(repo, item.get("target"))
    selection_mode = "auto" if requested_route == "auto" else "explicit"
    decisions = []
    if requested_route != "auto":
        selected = requested_route
        decisions.append("Explicit route is fixed; no other route will be probed or executed")
    elif scan["cuda_candidate"] and (not scan["native_candidate"] or any(
        entry["kind"] == "cuda" and not entry.get("framework_hint_only") for entry in scan["evidence"]
    )):
        selected = "transcode"
        decisions.append("CUDA target defaults to source conversion; reuse the project's HIPIFY pipeline when present")
    else:
        backend = capabilities.get("native_backend", {})
        if not isinstance(backend, dict):
            raise AdaptationError("capabilities.native_backend must be an object for automatic selection")
        state = backend.get("state")
        state = "unknown" if state is None else state
        if state not in ("available", "unavailable", "unknown"):
            raise AdaptationError("Invalid native_backend state")
        reason = backend.get("reason")
        rejected_native = state == "unavailable" and isinstance(reason, str) and bool(reason.strip())
        if not rejected_native and (scan["native_candidate"] or state in ("available", "unavailable")):
            selected = "native-hip"
            decisions.append("Native HIP candidate with no CUDA input identified in the inspected target")
        else:
            blocked = scan["native_candidate"] and rejected_native
            warning = ("Native backend unavailable: " + reason if blocked else
                       "No GPU build surface was detected; confirm target scope manually.")
            return {"schema_version": SCHEMA_VERSION, "kind": "adaptation-plan", "route": "none", "selection_mode": "auto", "scope": {"repo": str(repo), "target": item.get("target")}, "source": source_identity(repo), "scan": scan, "plan_status": "blocked" if blocked else "needs-information", "executed": False, "stages": {}, "warnings": [warning], "selection_evidence": [warning]}
    if selected not in available_routes:
        raise AdaptationError(f"Route is not present in this standalone package: {selected}")
    item["repo"] = str(repo)
    item["selection_mode"] = selection_mode
    module = importlib.import_module("routes." + MODULES[selected])
    result = module.plan(item)
    if result.get("route") != selected or result.get("selection_mode") != selection_mode or result.get("executed") is not False:
        raise AdaptationError("Route handler violated the planning contract")
    result["selection_evidence"] = decisions
    result["requested_route"] = requested_route
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="JSON request file (data only)")
    parser.add_argument("--repo")
    parser.add_argument("--route", choices=("auto", *ROUTES, "cutlass-hytlass"))
    parser.add_argument("--target")
    parser.add_argument("--arch")
    parser.add_argument("--build-argv", help="JSON argv array; never executed by preflight")
    parser.add_argument("--provider")
    parser.add_argument("--specialization", choices=("cutlass-hytlass",))
    parser.add_argument("--json-out", help="Write a NEW report outside the protected source tree")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verify-package-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        package = verify_package()
        from adaptation_common import SCHEMA_VERSION, resolve_repo, safe_write_json, validate_output_path
        if package.get("mode") != "source" and package["contract_version"] != SCHEMA_VERSION:
            raise ValueError("Loaded common contract does not match package")
        if args.verify_package_only:
            print(json.dumps({"valid": True, "mode": package.get("mode", "package"), "skill_name": package.get("skill_name"), "available_routes": package["available_routes"]}))
            return 0
        request = json.loads(Path(args.request).read_text(encoding="utf-8")) if args.request else {}
        if not isinstance(request, dict):
            raise ValueError("Request JSON must be an object")
        route_in_request = request.pop("route", None)
        if args.route and route_in_request and args.route != route_in_request:
            raise ValueError("CLI and request specify conflicting routes")
        locked = package.get("locked_route")
        selected = args.route or route_in_request or locked or "auto"
        if locked and selected != locked:
            if not (locked == "transcode" and selected == "cutlass-hytlass"):
                raise ValueError(f"This independent skill is locked to {locked}")
        for field in ("repo", "target", "arch", "provider", "specialization"):
            value = getattr(args, field)
            if value is not None:
                request[field] = value
        if args.build_argv is not None:
            request["build_command"] = json.loads(args.build_argv)
        result = plan_request(request, selected, available_routes=package["available_routes"])
        if args.json_out:
            repo = resolve_repo(request["repo"])
            _, destination = validate_output_path(repo, args.json_out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            safe_write_json(destination.parent, destination.name, result)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        if args.strict:
            return {"blocked": 2, "needs-information": 3}.get(result["plan_status"], 0)
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(json.dumps({"kind": "preflight-error", "error": str(exc), "executed": False}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
