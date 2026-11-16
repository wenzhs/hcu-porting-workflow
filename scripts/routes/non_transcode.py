"""Read-only planning for the FastPT CUDA-facing, non-transcode route.

Evidence is supplied by the caller; this module never imports Torch/FastPT,
activates a mode, probes a device, or executes a project command.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:  # Both package imports and the installed scripts/ entry point are supported.
    from .. import adaptation_common as common
except ImportError:
    import adaptation_common as common

ROUTE = "non-transcode"
ACTIVATION_CHECKS = ("fastpt-installed", "fastpt-version-compatible", "fastpt-invocable")


def _tristate(value: Any, name: str) -> str:
    if value is None:
        return "unknown"
    if type(value) is not bool:
        raise common.AdaptationError(f"{name} must be true, false, or null (unknown)")
    return "passed" if value else "failed"


def _torch_series(value: Any) -> tuple[int, int] | None:
    return common.version_series(value)


def capability_checks(request: dict[str, Any]) -> list[tuple[str, str, str, Any]]:
    """Shared evidence interpretation for the plan and legacy helper generator."""
    capabilities = request.get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise common.AdaptationError("capabilities must be an object")
    fastpt = capabilities.get("fastpt", {})
    if not isinstance(fastpt, dict):
        raise common.AdaptationError("capabilities.fastpt must be an object")
    checks = []
    for field, name in (("installed", "fastpt-installed"), ("invocable", "fastpt-invocable")):
        state = _tristate(fastpt.get(field), f"fastpt.{field}")
        checks.append((name, state, f"FastPT {field} is caller-provided evidence, not a local probe; missing evidence stays unknown.", {field: fastpt.get(field), "source": fastpt.get("source", "caller-provided")}))

    compatible = _tristate(fastpt.get("version_compatible"), "fastpt.version_compatible")
    actual = _torch_series(fastpt.get("torch_version"))
    required = _torch_series(fastpt.get("required_torch_version"))
    detail = "FastPT/Torch compatibility must be established for the installed provider; no package is imported."
    if actual is not None and required is not None:
        if actual != required:
            compatible = "failed"
            detail = "Known Torch major/minor mismatch with the provider's declared required Torch version; no executable activation helper may be written."
        elif compatible == "unknown":
            detail = "Torch major/minor requirements match, but DTK/provider compatibility is not established; version_compatible stays unknown."
    elif fastpt.get("torch_version") is not None or fastpt.get("required_torch_version") is not None:
        detail += " A complete parseable actual/required version pair was not supplied; do not infer compatibility from a FastPT release number alone."
    checks.append(("fastpt-version-compatible", compatible, detail, {key: fastpt.get(key) for key in ("version", "version_compatible", "torch_version", "required_torch_version")}))

    feature_state = _tristate(fastpt.get("feature_probed"), "fastpt.feature_probed")
    checks.append(("fastpt-feature-coverage", feature_state, "Feature coverage must refer to this target and its required features; an unknown probe is not a passed probe.", {"feature_probed": fastpt.get("feature_probed"), "required_features": request.get("required_features", [])}))
    # Hardware validation belongs to a later execution receipt, not a prerequisite
    # that would make investigation of a new target impossible.
    _tristate(fastpt.get("hardware_validated"), "fastpt.hardware_validated")
    return checks


def plan(request: dict[str, Any]) -> dict[str, Any]:
    """Return a contract plan, retaining explicit/auto provenance without rerouting."""
    if not isinstance(request, dict):
        raise common.AdaptationError("request must be an object")
    repo = common.resolve_repo(request.get("repo", ""))
    scan = common.scan_project(repo, request.get("target"))
    result = common.new_plan(ROUTE, request, scan)
    result["selection_reason"] = "Preserve CUDA-facing source and the original build; this handler never changes the requested route."
    result["warnings"].extend([
        "All capability statements are input evidence, not probes performed by this planner.",
        "FastPT -C/-E may persistently modify an installation. A new shell does not undo those changes; activation needs separate user authorization and an exclusive isolated installation.",
        "No CUDA-to-HIP rewriting, activation, build, import, GPU check, or correctness test has run.",
    ])
    cuda_sources = any(path.suffix in {".cu", ".cuh"} for path in common.source_files(Path(scan["root"]), include_dependencies=False))
    common.add_check(result, "cuda-facing-target", "passed" if scan["cuda_candidate"] or cuda_sources else "unknown", "Static CUDA-facing candidates are not proof of target membership or FastPT feature support.", {"cuda_candidate": scan["cuda_candidate"], "cuda_source_candidate": cuda_sources})
    common.add_check(result, "build-entry-discovery", "passed" if scan["root_entries"] else "not-applicable", "Only target-root build entries are candidates; vendored CMake never overrides a root setup.py. An explicit build argv is still required.", scan["root_entries"])
    try:
        command = common.validate_build_command(request.get("build_command"))
        common.add_check(result, "build-command", "passed" if command else "unknown", "Supply the project's explicit build argv; no shell string or guessed build command is executed.", command)
        if command:
            result["commands"].append({"phase": "build", "argv": command, "cwd": scan["root"], "executed": False})
    except common.AdaptationError as exc:
        common.add_check(result, "build-command", "failed", str(exc))

    build_entry = request.get("build_entry")
    if build_entry is not None:
        try:
            if not isinstance(build_entry, str) or not build_entry or Path(build_entry).is_absolute():
                raise common.AdaptationError("build_entry must name a target-relative file")
            entry = common.workspace_path(scan["root"], build_entry)
            if not entry.is_file():
                raise common.AdaptationError(f"Build entry does not exist: {build_entry}")
            result["build_entry"] = entry.relative_to(Path(scan["root"])).as_posix()
            common.add_check(result, "explicit-build-entry", "passed", "The caller selected this entry; file presence does not prove its build works.", result["build_entry"])
        except (OSError, common.AdaptationError) as exc:
            common.add_check(result, "explicit-build-entry", "failed", str(exc))
    try:
        arch = common.normalize_arch(request.get("arch"))
        result["arch"] = arch
        common.add_check(result, "architecture", "passed" if arch else "unknown", "Record the actual complete gfx target; do not map SM numbers or compatibility constants to a real SDK/GPU version.", arch)
    except common.AdaptationError as exc:
        common.add_check(result, "architecture", "failed", str(exc))
    try:
        for name, state, detail, evidence in capability_checks(request):
            common.add_check(result, name, state, detail, evidence)
        fastpt = request.get("capabilities", {}).get("fastpt", {})
        result["capabilities"] = {"fastpt": {key: fastpt.get(key) for key in ("installed", "version_compatible", "invocable", "feature_probed", "hardware_validated", "version", "torch_version", "required_torch_version", "source")}}
        if fastpt.get("hardware_validated") is not True:
            result["warnings"].append("Hardware validation is absent or not successful; this does not prevent static planning and is never reported as passed.")
    except common.AdaptationError as exc:
        common.add_check(result, "fastpt-evidence", "failed", str(exc))
    result["activation"] = {"launcher": request.get("fastpt_path"), "compile_mode": "-C", "runtime_mode": "-E", "executed": False, "requires_separate_authorization": True, "requires_isolated_installation": True}
    return common.finalize_plan(result)
