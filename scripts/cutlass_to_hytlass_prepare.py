#!/usr/bin/env python3
"""Prepare a review-only CUTLASS/hytlass specialization in a new tree.

Scan ORIGINAL text before any edits. Only includes and qualified namespaces are
mechanically changed. No runtime conversion, architecture equivalence, dependency
fetch, environment activation, build or GPU validation is performed.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from . import adaptation_common as common
    from . import cmake_fastpt_convert as cmake
except ImportError:
    import adaptation_common as common
    import cmake_fastpt_convert as cmake

IGNORE_DIRS = set(common.IGNORED_DIRS)
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".hxx", ".hip"}
TEXT_SUFFIXES = SOURCE_SUFFIXES | {".cmake", ".txt", ".mk", ".sh", ".py", ".toml", ".cfg", ".md"}
TEXT_NAMES = {"CMakeLists.txt", "Makefile", "makefile", "GNUmakefile", "setup.py", "pyproject.toml", ".gitmodules"}
DEFAULT_HYTLASS_URL = "https://developer.sourcefind.cn/codes/OpenDAS/hytlass.git"
REPORT_NAME = "HYTLASS_CONVERSION_REPORT.json"
BUILD_SCRIPT = "hytlass_build.sh"
RESERVED_NAMES = (REPORT_NAME, "HYTLASS_CONVERSION_REPORT.md", BUILD_SCRIPT)
# Kept importable for legacy callers, but intentionally no longer applied.
RUNTIME_MAP: dict[str, str] = {}
CMAKE_LIB_MAP: dict[str, str] = {}
RISK_RULES = [
    ("gemm-api", re.compile(r"cutlass::gemm::device::Gemm(?:Universal)?"), "review", "Review the actual hytlass GEMM API and Arguments contract"),
    ("shape", re.compile(r"GemmShape\s*<|ShapeMMA(?:Op|Warp|ThreadBlock)|TileShape_MNK|WarpShape_MNK|InstructionShape_MNK"), "review", "Review tile/warp/instruction shapes; no NVIDIA-to-DCU equivalence is assumed"),
    ("stages", re.compile(r"NumStages\s*=|StageCount\s*<"), "review", "Review pipeline stage count and synchronization semantics"),
    ("sm-architecture", re.compile(r"\b(?:Sm|SM|sm_|compute_)\d+\b"), "review", "SM architecture tags are retained for semantic review; never mapped to Gfx by name"),
    ("cutlass-v4", re.compile(r"(?:cutlass[^\r\n]*(?:\bv?4\.|VERSION(?:_MAJOR)?\s*(?:=|\()?\s*4\b)|CUTLASS_VERSION(?:_MAJOR)?\s*(?:=|\()?\s*(?:4\b|400\b)|\bGIT_TAG\s+v?4\.)", re.I), "blocked", "CUTLASS 4.x evidence requires a version-specific support/reimplementation decision"),
    ("hopper", re.compile(r"\b(?:Sm90\w*|sm_90\w*|compute_90\w*|Hopper)\b", re.I), "blocked", "Hopper/SM90 support is unproven; mechanical preparation is blocked"),
    ("blackwell", re.compile(r"\b(?:Sm(?:100|120)\w*|sm_(?:100|120)\w*|compute_(?:100|120)\w*|Blackwell)\b", re.I), "blocked", "Blackwell/SM100/SM120 support is unproven; mechanical preparation is blocked"),
    ("specialized-hardware", re.compile(r"WGMMA|\bTMA\b|mbarrier|tcgen05|CuTeDSL|cute_dsl|nvidia-cutlass-dsl|blockscaled|MXF8|MXF6|MXF4|NVFP4", re.I), "blocked", "Hardware-specific operations need an explicit supported implementation, not token replacement"),
    ("cutlass-reference", re.compile(r"\bcutlass\b", re.I), "review", "Confirm CUTLASS target membership, fixed dependency version and semantic coverage"),
    ("cuda-api", re.compile(r"\bcuda[A-Z]\w*|\bcublas(?:Lt)?\w*"), "review", "CUDA/BLAS/Lt APIs are unchanged; use a verified source converter and library-specific review"),
    ("collective", re.compile(r"mma|Mma|Epilogue|Collective|Mainloop"), "review", "MMA/mainloop/epilogue behavior remains pending semantic review"),
    ("dependency-integration", re.compile(r"FetchContent|CPMAddPackage|add_subdirectory|github\.com[:/]NVIDIA/cutlass", re.I), "review", "Dependency URLs, submodules and root CMake execution are not automatically changed"),
]
REVIEW_PATTERNS = [(pattern, message) for _, pattern, _, message in RISK_RULES]


def ignore_tree(_: str, names: list[str]) -> set[str]:
    return set(names) & IGNORE_DIRS


def is_text_candidate(path: Path) -> bool:
    return path.is_file() and (path.name in TEXT_NAMES or path.suffix in TEXT_SUFFIXES)


def candidate_files(root: Path) -> list[Path]:
    return sorted(path for path in common.source_files(root) if is_text_candidate(path))


def normalize_arch(arch: str | None) -> tuple[str | None, str | None]:
    """Legacy tuple shape; the second value is NOT a CMAKE_HIP_ARCHITECTURES value."""
    gfx = common.normalize_arch(arch)
    return gfx, gfx[3:].split(":", 1)[0] if gfx else None


def validate_version(version: str | None) -> str:
    if not isinstance(version, str) or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64}|v?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z._-]+)?)", version):
        raise common.AdaptationError("A fixed hytlass release version or full commit hash is required; main/master/latest/HEAD/ranges are not pins")
    return version


def scan_risks(text: str, rel: Path) -> list[dict[str, Any]]:
    """Collect every matching diagnostic, on ORIGINAL text including real newlines."""
    result: list[dict[str, Any]] = []
    seen = set()
    for code, pattern, severity, message in RISK_RULES:
        for match in pattern.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            if (code, number) in seen:
                continue
            seen.add((code, number))
            start = text.rfind("\n", 0, match.start()) + 1
            end = text.find("\n", match.end())
            result.append({"code": code, "severity": severity, "file": rel.as_posix(), "line": number,
                           "text": text[start:len(text) if end < 0 else end].strip(), "detail": message, "original": True})
    return sorted(result, key=lambda item: (item["line"], item["code"]))


def word_replace(text: str, src: str, dst: str) -> str:
    return re.sub(rf"(?<![A-Za-z0-9_:.]){re.escape(src)}(?![A-Za-z0-9_:.])", dst, text)


def _namespace_code(text: str) -> str:
    # Unsupported SM tags remain wholly unchanged, rather than inventing a
    # nonexistent hytlass::arch::SmXX or claiming an equivalent Gfx architecture.
    text = re.sub(r"\bcutlass(?=\s*::)(?!\s*::\s*arch\s*::\s*Sm\d+)", "hytlass", text)
    return re.sub(r"\bcute(?=\s*::)", "hute", text)


def transform_source(text: str, gfx_arch: str | None = None) -> str:
    common.normalize_arch(gfx_arch)
    # Lexical partition keeps comments, regular/character/raw string literals
    # intact. Include directives are the sole intentional string-like edit.
    pieces: list[str] = []
    position = 0
    code_start = 0
    while position < len(text):
        include = re.match(r'#[ \t]*include[ \t]*([<"])(cutlass|cute)/([^>"\r\n]+)([>"])', text[position:]) if text[position] == "#" and not text[text.rfind("\n", 0, position) + 1:position].strip() else None
        end = None
        replacement = None
        if include:
            end = position + len(include[0])
            replacement = include[0].replace(include[2] + "/", ("hytlass" if include[2] == "cutlass" else "hute") + "/", 1)
        elif text.startswith("//", position):
            end = text.find("\n", position)
            end = len(text) if end < 0 else end
        elif text.startswith("/*", position):
            end = text.find("*/", position + 2)
            if end < 0:
                raise common.AdaptationError("Unclosed C/C++ comment needs manual review")
            end += 2
        else:
            raw = re.match(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(', text[position:])
            if raw:
                closing = ")" + raw[1] + '"'
                end = text.find(closing, position + len(raw[0]))
                if end < 0:
                    raise common.AdaptationError("Unclosed C++ raw string needs manual review")
                end += len(closing)
            elif text[position] in "\"'":
                quote = text[position]
                end = position + 1
                while end < len(text) and text[end] != quote:
                    end += 2 if text[end] == "\\" else 1
                if end >= len(text):
                    raise common.AdaptationError("Unclosed C/C++ string needs manual review")
                end += 1
        if end is None:
            position += 1
            continue
        pieces.append(_namespace_code(text[code_start:position]))
        pieces.append(text[position:end] if replacement is None else replacement)
        code_start = position = end
    pieces.append(_namespace_code(text[code_start:]))
    return "".join(pieces)


def strip_cuda_gencode(text: str, gfx_arch: str | None) -> str:
    return cmake.strip_cuda_gencode(text, gfx_arch)


def transform_dependency_text(text: str, hytlass_url: str) -> str:
    """Legacy safe no-op: URLs, pins, submodule paths and gitlinks are not equivalent."""
    return text


def transform_build_text(text: str, gfx_arch: str | None, arch_num: str | None, hytlass_root: Path | None, hytlass_url: str) -> str:
    common.normalize_arch(gfx_arch)
    # Standalone CMake adaptation is handled by the source-map-aware adapter.
    # Numeric hytlass-specific variables must never leak into standard HIP vars.
    return text


def inject_top_level_cmake(text: str, arch_num: str | None, hytlass_root: Path | None, hytlass_url: str, *, hytlass_version: str | None = None, integration_mode: str | None = None) -> str:
    common.normalize_arch(arch_num)  # Explicitly rejects the old unsafe numeric use.
    validate_version(hytlass_version)
    if integration_mode not in {"headers", "existing-target"}:
        raise common.AdaptationError("Specify actual headers or existing-target integration; automatic top-level injection is unsupported")
    return text  # In particular: no FetchContent(main), add_subdirectory or FORCE.


def transform_text(path: Path, root: Path, gfx_arch: str | None, arch_num: str | None, hytlass_root: Path | None, hytlass_url: str) -> tuple[str, list[str]]:
    old = common.read_text(path)
    diagnostics = scan_risks(old, path.relative_to(root))
    new = transform_source(old, gfx_arch) if path.suffix in SOURCE_SUFFIXES else old
    return new, [f"{d['file']}:{d['line']}: [{d['severity']}:{d['code']}] {d['detail']}: {d['text']}" for d in diagnostics]


def _dependency(root: Path | None, version: str | None, mode: str | None) -> dict[str, Any]:
    version = validate_version(version)
    if mode not in {"headers", "existing-target"}:
        raise common.AdaptationError("--integration-mode must state headers or existing-target; no implicit root CMake execution")
    if root is None:
        raise common.AdaptationError("An actual local --hytlass-root is required; preparation never downloads a dependency")
    root = common.resolve_repo(root)
    if not (root / "include" / "hytlass").is_dir():
        raise common.AdaptationError("hytlass root must contain include/hytlass for this bounded include mapping")
    identity = common.source_identity(root)
    if re.fullmatch(r"[0-9a-fA-F]{40}", version) and identity["git_head"] and identity["git_head"].lower() != version.lower():
        raise common.AdaptationError("Declared hytlass commit does not match the local dependency HEAD")
    return {"version": version, "integration_mode": mode, "root": str(root), "source": identity,
            "version_evidence": "caller-declared fixed version; tree identity recorded, release compatibility not probed"}


def integrate_headers(text: str, *, target: str | None, dependency: dict[str, Any], hytlass_target: str | None = None) -> str:
    if not isinstance(target, str) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", target):
        raise common.AdaptationError("An explicit literal --target is required for scoped hytlass integration")
    commands = cmake.parse_commands(text)
    if any(command.name in {"function", "macro", "foreach", "while"} for command in commands):
        raise common.AdaptationError("Target integration in function/macro/loop files needs manual review")
    definitions = [c for c in commands if c.name in {"add_library", "add_executable"} and c.tokens and c.tokens[0].value == target]
    if len(definitions) != 1 or any(t.value in {"ALIAS", "IMPORTED"} for t in definitions[0].tokens):
        raise common.AdaptationError("The integration file must define exactly one real, non-imported target of the supplied name")
    if dependency["integration_mode"] == "existing-target":
        if not isinstance(hytlass_target, str) or not re.fullmatch(r"[A-Za-z0-9_.:+-]+", hytlass_target):
            raise common.AdaptationError("existing-target mode requires an explicit --hytlass-target")
        if not any(c.name == "target_link_libraries" and c.tokens and c.tokens[0].value == target and any(t.value == hytlass_target for t in c.tokens[1:]) for c in commands):
            raise common.AdaptationError("existing-target integration must already link the named hytlass target; no target name is guessed")
        return text
    definition = definitions[0]
    indent = cmake._indent(text, definition)
    include = (Path(dependency["root"]) / "include").as_posix()
    # Bounded literal paths only: do not introduce CMake variable/list expansion.
    if any(character in include for character in "$;\r\n\0"):
        raise common.AdaptationError("Dependency include path needs manual CMake escaping; no automatic insertion")
    quoted = cmake._quote(include)
    newline = "\r\n" if "\r\n" in text else "\n"
    visibility = "INTERFACE" if any(t.value == "INTERFACE" for t in definition.tokens[1:]) else "PRIVATE"
    addition = f"{newline}{indent}target_include_directories({target} {visibility} {quoted})"
    return text[:definition.end] + addition + text[definition.end:]


def unified_diff(rel: Path, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=str(rel), tofile=str(rel) + ".hytlass"))


def write_build_script(root: Path, gfx_arch: str | None, arch_num: str | None, hytlass_root: Path | None, *, build_command: list[str] | None = None) -> None:
    cmake._prepared(root)
    text = cmake.render_explicit_build_script(root, build_command, "build_hytlass")
    script = common.safe_write_text(root, BUILD_SCRIPT, text)
    common.workspace_path(root, script).chmod(0o755)


def write_report(root: Path, changed: list[Path], warnings: list[str], diffs: list[str], gfx_arch: str | None, hytlass_root: Path | None, hytlass_url: str, *, diagnostics: list[dict[str, Any]] | None = None, dependency: dict[str, Any] | None = None) -> None:
    common.safe_write_json(root, REPORT_NAME, {
        "schema_version": common.SCHEMA_VERSION, "kind": "adaptation-preparation", "route": "transcode",
        "specialization": "cutlass-hytlass", "prepared": True, "executed": True,
        "execution_scope": "filesystem-preparation-only", "status": "prepared-pending-semantic-review",
        "stages": {stage: "passed" if stage in {"prepare", "transform"} else "not-run" for stage in common.STAGES}, "pending_semantic_review": True,
        "transform_scope": "bounded include/namespace edits and explicit dependency integration only",
        "arch": gfx_arch, "dependency": dependency, "repository_hint": hytlass_url,
        "changed": [p.as_posix() for p in changed], "diagnostics": diagnostics or [], "warnings": warnings,
        "limitations": ["Original risk evidence retained; namespace/include changes are not semantic validation", "No Sm-to-Gfx or BLASLt-to-BLAS equivalence", "No dependency fetched or root dependency CMake executed", "No build, GPU test or correctness validation executed"],
    })


def prepare(repo: Path, out: Path | None = None, *, arch: str | None = None, hytlass_root: Path | None = None, hytlass_url: str = DEFAULT_HYTLASS_URL, hytlass_version: str | None = None, integration_mode: str | None = None, target: str | None = None, cmake_file: str = "CMakeLists.txt", hytlass_target: str | None = None, write: bool = False, force: bool = False, emit_build_script: bool = False, build_command: list[str] | None = None) -> dict[str, Any]:
    repo = common.resolve_repo(repo)
    out, _ = common.prepare_destination(repo, out, default_suffix="_hytlass")
    if force:
        raise common.AdaptationError("--force is no longer supported; existing output is never overwritten")
    gfx_arch, arch_num = normalize_arch(arch)
    command = common.validate_build_command(build_command)
    if emit_build_script:
        if command is None:
            raise common.AdaptationError("--emit-build-script requires an explicit --build-command JSON argv array")
        cmake.render_explicit_build_script(repo, command, "build_hytlass")  # Validate before writing anything.
    analysis_identity = common.source_identity(repo)
    originals: dict[Path, str] = {}
    diagnostics: list[dict[str, Any]] = []
    # Complete original scan first, including original dependency/version text.
    for path in candidate_files(repo):
        rel = path.relative_to(repo)
        originals[rel] = common.read_text(path)
        diagnostics.extend(scan_risks(originals[rel], rel))
    blockers = [d for d in diagnostics if d["severity"] == "blocked"]
    if write and blockers:
        detail = "; ".join(f"{d['file']}:{d['line']} {d['code']}" for d in blockers)
        raise common.AdaptationError("Original-source semantic blockers; no tree written: " + detail)
    dependency = None
    if write or hytlass_version is not None or integration_mode is not None or hytlass_root is not None:
        dependency = _dependency(hytlass_root, hytlass_version, integration_mode)
    transformed = {rel: transform_source(old, gfx_arch) if rel.suffix in SOURCE_SUFFIXES else old for rel, old in originals.items()}
    if dependency is not None:
        cmake_path = common.workspace_path(repo, cmake_file)
        relative_cmake = cmake_path.relative_to(repo)
        if relative_cmake not in originals:
            raise common.AdaptationError("The explicit integration CMake file must exist in the source tree")
        transformed[relative_cmake] = integrate_headers(originals[relative_cmake], target=target, dependency=dependency, hytlass_target=hytlass_target)
        for rel, new in transformed.items():
            for include in re.findall(r'#[ \t]*include[ \t]*[<"]((?:hytlass|hute)/[^>"\r\n]+)[>"]', new):
                actual = common.workspace_path(dependency["root"], "include/" + include)
                if not actual.is_file():
                    raise common.AdaptationError(f"Mapped dependency header does not exist: {include} (referenced by {rel})")
    changed = [rel for rel in transformed if transformed[rel] != originals[rel]]
    diffs = [unified_diff(rel, originals[rel], transformed[rel]) for rel in changed]
    warnings = [f"{d['file']}:{d['line']}: [{d['severity']}:{d['code']}] {d['detail']}: {d['text']}" for d in diagnostics]
    warnings.append("Pending semantic review: fixed-version APIs, arch tags, shape/alignment, mainloop, epilogue and numerical contracts are not proven by mechanical edits")
    if write:
        workspace = common.prepare_tree(repo, out, route="transcode", force=force, reserved_names=RESERVED_NAMES)
        common.validate_prepared_inputs(workspace, analysis_identity, originals)
        for rel in changed:
            common.safe_write_text(workspace, rel, transformed[rel], overwrite=True)
        if emit_build_script:
            write_build_script(workspace, gfx_arch, arch_num, hytlass_root, build_command=command)
        write_report(workspace, changed, warnings, diffs, gfx_arch, hytlass_root, hytlass_url, diagnostics=diagnostics, dependency=dependency)
    return {"schema_version": common.SCHEMA_VERSION, "kind": "adaptation-preparation", "route": "transcode", "specialization": "cutlass-hytlass", "status": "prepared-pending-semantic-review" if write else "blocked" if blockers else "planned", "changed": [p.as_posix() for p in changed], "diffs": diffs, "diagnostics": diagnostics, "warnings": warnings, "blockers": blockers, "dependency": dependency, "pending_semantic_review": True, "executed": write, "execution_scope": "filesystem-preparation-only", "stages": {stage: "passed" if write and stage in {"prepare", "transform"} else "not-run" for stage in common.STAGES}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", help="Copy destination for non-Git input; Git stays at --repo")
    parser.add_argument("--hytlass-root")
    parser.add_argument("--hytlass-url", default=DEFAULT_HYTLASS_URL, help="Provenance hint only; never fetched or substituted")
    parser.add_argument("--hytlass-version", help="Fixed release version or full commit hash")
    parser.add_argument("--integration-mode", choices=("headers", "existing-target"))
    parser.add_argument("--target", help="Actual application CMake target, not a guessed dependency target")
    parser.add_argument("--cmake-file", default="CMakeLists.txt")
    parser.add_argument("--hytlass-target", help="Already linked dependency target for existing-target mode")
    parser.add_argument("--arch", help="Explicit complete gfx target; no numeric/default architecture")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--force", action="store_true", help="Legacy flag; unsafe overwrites are refused")
    parser.add_argument("--emit-build-script", action="store_true")
    parser.add_argument("--build-command", help="JSON argv array; only emitted on explicit request, never executed")
    args = parser.parse_args(argv)
    try:
        repo = common.resolve_repo(args.repo)
        out, _ = common.prepare_destination(repo, args.out, default_suffix="_hytlass")
        result = prepare(repo, out, arch=args.arch, hytlass_root=Path(args.hytlass_root) if args.hytlass_root else None, hytlass_url=args.hytlass_url, hytlass_version=args.hytlass_version, integration_mode=args.integration_mode, target=args.target, cmake_file=args.cmake_file, hytlass_target=args.hytlass_target, write=args.write, force=args.force, emit_build_script=args.emit_build_script, build_command=json.loads(args.build_command) if args.build_command else None)
        print("".join(result["diffs"]) or "No bounded source changes proposed.")
        for warning in result["warnings"]:
            print(warning, file=sys.stderr)
        if args.write:
            print(f"Prepared workspace: {out}; semantic review and all execution validation remain pending")
        return 2 if result["blockers"] else 0
    except (common.AdaptationError, OSError, json.JSONDecodeError) as exc:
        print(f"Blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
