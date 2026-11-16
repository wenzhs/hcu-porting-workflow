"""Read-only planning for a project's own HIP/ROCm backend.

Capability records are supplied evidence, not probes or execution permission.
This module never imports a target framework, activates an environment, invokes
FastPT, transforms sources, or runs a compiler/build command.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import adaptation_common as common


def _object(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise common.AdaptationError(f"{name} must be an object")
    return value


def _evidence(value: Any) -> dict[str, Any]:
    return {"origin": "request", "verified_by_preflight": False, "value": deepcopy(value)}


def _check_backend(result: dict[str, Any], backend: dict[str, Any]) -> None:
    state = backend.get("state")
    state = "unknown" if state is None else state
    reason = backend.get("reason")
    if state == "available":
        status = "passed"
        detail = "The caller declares a native backend for the requested target/features; this is not a feature or GPU probe."
    elif state == "unavailable":
        status = "failed"
        if isinstance(reason, str) and reason.strip():
            detail = f"The caller declares this target's native backend unavailable: {reason}"
        else:
            detail = "native_backend.state=unavailable requires a nonempty reason; no alternate route is selected here."
    elif state == "unknown":
        status = "unknown"
        detail = "Confirm native_backend.state for this target and all required features; static markers alone do not prove availability."
    else:
        status = "failed"
        detail = "native_backend.state must be available, unavailable, or unknown."
    common.add_check(result, "native-backend", status, detail, _evidence(backend))


def _check_hip(result: dict[str, Any], hip: dict[str, Any]) -> None:
    installed = hip.get("installed")
    if installed is True:
        status, detail = "passed", "A native HIP/DTK installation is declared; its presence and runtime were not probed."
    elif installed is False:
        status, detail = "failed", "The supplied evidence declares the native HIP installation absent."
    elif installed is None:
        status, detail = "unknown", "Supply native HIP installation evidence; missing information is not proof of absence."
    else:
        status, detail = "failed", "capabilities.hip.installed must be true, false, or null."
    common.add_check(result, "hip-installed", status, detail, _evidence(installed))

    compiler = hip.get("compiler")
    if compiler is False:
        status, detail = "failed", "The supplied evidence declares the native HIP compiler unavailable."
    elif compiler is None or compiler is True or (isinstance(compiler, str) and not compiler.strip()):
        status, detail = "unknown", "Supply a native HIP compiler name/path; a boolean true without compiler identity is insufficient."
    elif isinstance(compiler, str) and not any(char in compiler for char in "\0\r\n"):
        status, detail = "passed", "A native HIP compiler name/path is declared; preflight did not locate or invoke it."
    else:
        status, detail = "failed", "capabilities.hip.compiler must be a compiler name/path, false, or null (true alone leaves identity unknown)."
    common.add_check(result, "hip-compiler", status, detail, _evidence(compiler))


def _native_build_argv(command: Any) -> list[str] | None:
    argv = common.validate_build_command(command)
    if argv is None:
        return None
    executable = argv[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if executable in {"source", ".", "fastpt", "gpufusion"} or executable.startswith("fastpt."):
        raise common.AdaptationError("Native builds must use the project's HIP build argv, not FastPT activation or conversion")
    if any(arg in {"&&", "||", ";", "|", ">", ">>", "<"} for arg in argv):
        raise common.AdaptationError("Supply build argv without shell operators; record environment setup and logging separately")
    # A script file is a legitimate original build entry; a shell -c string is
    # not an argv-form build recipe. Do not inspect or execute that script here.
    shells = {"sh", "bash", "dash", "zsh", "ksh", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}
    if executable in shells:
        for arg in argv[1:]:
            lowered = arg.lower()
            if lowered in {"-command", "-encodedcommand", "-c", "/c", "/k"} or (executable in {"sh", "bash", "dash", "zsh", "ksh"} and lowered.startswith("-") and not lowered.startswith("--") and "c" in lowered[1:]):
                raise common.AdaptationError("Supply an executable/script argv, not a shell command expression")
            if arg == "--" or not arg.startswith("-"):
                break
    return list(argv)


def plan(request: dict[str, Any]) -> dict[str, Any]:
    """Describe native prerequisites and the supplied build argv without running it.

    ``capabilities.hip.compiler`` is a declared compiler name/path, false for a
    confirmed failure, or null/missing for unknown. True without an identity is
    also unknown. FastPT metadata is deliberately not an input to native checks.
    Explicit and router-selected auto requests both stay on ``native-hip``.
    """
    result = common.new_plan("native-hip", request)
    target = request.get("target")
    if target is not None and (not isinstance(target, str) or Path(target).is_absolute()):
        raise common.AdaptationError("target must be a project-relative directory")
    scan = common.scan_project(request["repo"], target)
    result["scan"] = scan
    result["selection_reason"] = (
        "Explicit native-hip request; failure does not authorize a different route."
        if result["selection_mode"] == "explicit"
        else "Router-selected native-hip candidate; auto provenance is preserved and this handler never reroutes."
    )
    common.add_check(
        result,
        "native-candidate",
        "passed" if scan["native_candidate"] else "unknown",
        "Native markers exist in the selected project scope; actual target membership and required features still need backend evidence."
        if scan["native_candidate"]
        else "No native candidate was found in this target scope. Vendor/sibling backends are not evidence for it; confirm the original HIP build entry.",
        {"origin": "static-source-scan", "root": scan["root"], "coverage": scan["coverage"], "matches": [item for item in scan["evidence"] if item["kind"] == "native"]},
    )

    capabilities = _object(request.get("capabilities"), "capabilities")
    backend = _object(capabilities.get("native_backend"), "capabilities.native_backend")
    hip = _object(capabilities.get("hip"), "capabilities.hip")
    result["capability_evidence"] = _evidence({"native_backend": backend, "hip": hip})
    _check_backend(result, backend)
    _check_hip(result, hip)

    arch = request.get("arch")
    try:
        if arch is not None and not isinstance(arch, str):
            raise common.AdaptationError("arch must be a complete gfx target string or null")
        result["arch"] = common.normalize_arch(arch)
    except common.AdaptationError as error:
        result["arch"] = None
        common.add_check(result, "architecture", "failed", str(error), _evidence(arch))
    else:
        common.add_check(
            result,
            "architecture",
            "passed" if result["arch"] else "unknown",
            "The declared complete gfx target is syntactically valid; hardware compatibility was not tested."
            if result["arch"]
            else "Supply the target's complete gfx architecture; do not infer it from a different device or a compatibility macro.",
            _evidence(arch),
        )

    try:
        command = _native_build_argv(request.get("build_command"))
    except common.AdaptationError as error:
        common.add_check(result, "build-command", "failed", str(error), _evidence(request.get("build_command")))
    else:
        common.add_check(
            result,
            "build-command",
            "passed" if command else "unknown",
            "An argv-form original build command is supplied; it and its project-owned HIPIFY steps have not been executed."
            if command
            else "Supply the original native build argv for this target; discovered build files do not authorize a guessed command.",
            _evidence({"argv": command, "cwd": scan["root"], "root_entries": scan["root_entries"]}),
        )
        if command:
            result["commands"].append({"phase": "build", "argv": command, "cwd": scan["root"], "executed": False})

    result["warnings"].extend([
        "Capability values are caller-supplied evidence, not probes, permission to execute, or proof of GPU correctness.",
        "No FastPT metadata is required or checked. No Torch import, environment activation, source conversion, installation, cleanup, or build was performed.",
        "Plan cwd identifies the inspected source scope; an authorized later build must use a separately protected task workspace and revalidate its identities.",
    ])
    return common.finalize_plan(result)
