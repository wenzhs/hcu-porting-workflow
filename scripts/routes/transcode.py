"""Read-only transcode planning; FastPT is optional, hytlass is a specialization."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .. import adaptation_common as common
except ImportError:
    import adaptation_common as common

CAPABILITY_FIELDS = ("installed", "version_compatible", "invocable", "feature_probed")


def _state(value: Any) -> str:
    return "passed" if value is True else "failed" if value is False else "unknown"


def _provider_record(name: str, capabilities: dict[str, Any]) -> dict[str, Any]:
    converters = capabilities.get("converters", {})
    record = dict(converters.get(name, {})) if isinstance(converters, dict) and isinstance(converters.get(name), dict) else {}
    if name == "fastpt":
        specific = dict(record)
        fastpt = capabilities.get("fastpt", {})
        if not isinstance(fastpt, dict):
            raise common.AdaptationError("capabilities.fastpt must be an object")
        # Merge missing evidence, preserving negatives from either location.
        for field, value in fastpt.items():
            if value is False or field not in record or record[field] is None:
                record[field] = value
        record.pop("mismatches", None)  # Computed diagnostic, not caller-owned mutable data.
        # A complete mismatch in either original record cannot be masked, and a
        # pair split across locations must be checked in the effective record.
        for origin, evidence in (("converter", specific), ("environment", fastpt), ("effective", dict(record))):
            actual = common.version_series(evidence.get("torch_version"))
            required = common.version_series(evidence.get("required_torch_version"))
            if actual is not None and required is not None and actual != required:
                record["version_compatible"] = False
                record.setdefault("mismatches", []).append({"origin": origin, "torch_version": evidence.get("torch_version"), "required_torch_version": evidence.get("required_torch_version")})
    return record


def _feature_state(record: dict[str, Any], feature: str) -> Any:
    features = record.get("features", {})
    value = features.get(feature) if isinstance(features, dict) else None
    return value.get("supported") if isinstance(value, dict) else value


def _ready(record: dict[str, Any], features: list[str]) -> bool:
    return all(record.get(field) is True for field in CAPABILITY_FIELDS) and all(_feature_state(record, feature) is True for feature in features)


def plan(request: dict[str, Any]) -> dict[str, Any]:
    result = common.new_plan("transcode", request)
    try:
        scan = common.scan_project(request["repo"], request.get("target"))
    except (common.AdaptationError, OSError) as exc:
        common.add_check(result, "target", "failed", str(exc))
        return common.finalize_plan(result)
    result["scan"] = scan
    result["selection_reason"] = "Explicit transcode route stays fixed; no provider failure activates -T or reroutes to another adaptation route." if result["selection_mode"] == "explicit" else "Router selected transcode; provider choice is limited to supplied, complete capability evidence."
    result["warnings"].append("Capability records are caller-supplied evidence, not probes performed by this planner. All execution stages remain not-run.")
    capabilities = request.get("capabilities", {})
    if not isinstance(capabilities, dict):
        common.add_check(result, "capabilities", "failed", "capabilities must be an object; missing values remain unknown")
        capabilities = {}
    features = request.get("required_features", [])
    if not isinstance(features, list) or any(not isinstance(feature, str) or not feature for feature in features):
        common.add_check(result, "required-features", "failed", "required_features must be a list of feature names")
        features = []
    explicit = "provider" in request and request["provider"] is not None
    provider = request.get("provider")
    if explicit and (not isinstance(provider, str) or not provider.strip()):
        common.add_check(result, "provider", "failed", "provider must be a nonempty converter name")
        provider = None
    elif not explicit:
        converters = capabilities.get("converters", {})
        names = sorted(converters) if isinstance(converters, dict) else []
        if isinstance(capabilities.get("fastpt"), dict) and "fastpt" not in names:
            names.append("fastpt")
        # Deterministic, optional FastPT preference last; no tool names are guessed.
        names = sorted(names, key=lambda name: (name == "fastpt", name))
        provider = next((name for name in names if _ready(_provider_record(name, capabilities), features)), None)
        result["provider_candidates"] = [{"name": name, "capabilities": _provider_record(name, capabilities)} for name in names]
        if provider is None:
            known_failures = bool(names) and all(any(_provider_record(name, capabilities).get(field) is False for field in CAPABILITY_FIELDS) or any(_feature_state(_provider_record(name, capabilities), feature) is False for feature in features) for name in names)
            common.add_check(result, "provider-selection", "failed" if known_failures else "unknown", "No capabilities-backed converter is selectable; supply installed/version_compatible/invocable/feature_probed evidence. FastPT -T is not a fallback.")
    result["provider"] = provider
    result["provider_selection"] = "explicit" if explicit else "capability-evidence"
    if provider is not None:
        record = _provider_record(provider, capabilities)
        result["provider_capabilities"] = record
        for field in CAPABILITY_FIELDS:
            common.add_check(result, f"provider-{field}", _state(record.get(field)), f"{provider}: {field}; missing/null is unknown, not false", {"source": "request.capabilities", "value": record.get(field)})
        for feature in features:
            common.add_check(result, f"feature-{feature}", _state(_feature_state(record, feature)), f"{provider} support for required feature {feature}", {"source": "request.capabilities.converters", "feature": feature})
        if provider == "fastpt" and record.get("version_compatible") is False:
            result["warnings"].append("FastPT version/Torch mismatch blocks this provider; do not activate -T or substitute another provider without a new choice.")
        result["hardware_validated"] = record.get("hardware_validated") if type(record.get("hardware_validated")) is bool else None
    try:
        arch = common.normalize_arch(request.get("arch"))
        result["arch"] = arch
        common.add_check(result, "architecture", "passed" if arch else "unknown", "Explicit complete gfx target supplied" if arch else "Supply the actual device gfx target; no gfx936 or numeric default is assumed")
    except common.AdaptationError as exc:
        common.add_check(result, "architecture", "failed", str(exc))
    hip = capabilities.get("hip", {})
    if not isinstance(hip, dict):
        hip = {}
    common.add_check(result, "hip-installed", _state(hip.get("installed")), "HIP build toolchain availability must be recorded separately from the converter", {"source": "request.capabilities.hip", "value": hip.get("installed")})
    compiler = hip.get("compiler")
    common.add_check(result, "hip-compiler", "passed" if isinstance(compiler, str) and compiler.strip() else "failed" if compiler is False else "unknown", "Record the actual compiler path/version; no compiler or environment is activated")
    try:
        command = common.validate_build_command(request.get("build_command"))
        common.add_check(result, "build-command", "passed" if command else "unknown", "Explicit build argv retained without execution" if command else f"Original build entry needs confirmation; root entries: {', '.join(scan['root_entries']) or 'none'}")
        if command:
            result["commands"].append({"phase": "build", "argv": command, "cwd": scan["root"], "executed": False})
    except common.AdaptationError as exc:
        common.add_check(result, "build-command", "failed", str(exc))
    # Never invent converter flags from its executable name. A verified explicit
    # invocation may be recorded as optional evidence, still never executed.
    if "conversion_command" in request:
        try:
            command = common.validate_build_command(request["conversion_command"])
            if command is None:
                raise common.AdaptationError("conversion_command must be a nonempty argv array")
            result["commands"].append({"phase": "transform", "argv": command, "cwd": scan["root"], "executed": False})
            common.add_check(result, "conversion-command", "passed", "Caller-provided converter argv recorded; output mapping must be checked after conversion")
        except common.AdaptationError as exc:
            common.add_check(result, "conversion-command", "failed", str(exc))
    result["source_mapping"] = request.get("source_map", {})
    result["warnings"].append("The CMake adapter requires an explicit actual source map, including identity .cu -> .cu when compiled as HIP; planned names do not establish generated-file existence.")
    specialization = request.get("specialization")
    result["specialization"] = specialization
    if specialization is not None and specialization != "cutlass-hytlass":
        common.add_check(result, "specialization", "failed", "Only cutlass-hytlass is recognized; it is not a separate adaptation route")
    elif specialization == "cutlass-hytlass":
        try:
            try:
                from .. import cutlass_to_hytlass_prepare as hytlass
            except ImportError:
                import cutlass_to_hytlass_prepare as hytlass
            data = request.get("hytlass", {})
            if not isinstance(data, dict):
                raise common.AdaptationError("hytlass specialization configuration must be an object")
            if data.get("version") is None:
                common.add_check(result, "hytlass-version", "unknown", "A fixed dependency release or full commit hash is required")
            else:
                hytlass.validate_version(data["version"])
                common.add_check(result, "hytlass-version", "passed", "Fixed dependency identifier supplied; actual tree compatibility remains unprobed")
            mode = data.get("integration_mode")
            common.add_check(result, "hytlass-integration", "unknown" if mode is None else "passed" if mode in {"headers", "existing-target"} else "failed", "Specify actual headers or existing-target integration; no automatic root CMake/FetchContent/submodule changes")
            root = Path(scan["root"])
            diagnostics = []
            for path in hytlass.candidate_files(root):
                diagnostics.extend(hytlass.scan_risks(common.read_text(path), path.relative_to(root)))
            result["specialization_diagnostics"] = diagnostics
            hard = [d for d in diagnostics if d["severity"] == "blocked"]
            common.add_check(result, "hytlass-original-risks", "failed" if hard else "passed", "Original-text unsupported-generation signals block mechanical preparation" if hard else "No hard original-text generation signal found; this is not support validation", diagnostics)
            common.add_check(result, "hytlass-semantic-review", "unknown", "Include/namespace edits do not prove shape/mainloop/epilogue, architecture or numerical equivalence")
        except (common.AdaptationError, OSError) as exc:
            common.add_check(result, "hytlass-specialization", "failed", str(exc))
    # Existing native hytlass use alone does not set the specialization or route.
    return common.finalize_plan(result)
