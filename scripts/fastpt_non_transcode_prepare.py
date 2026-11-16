#!/usr/bin/env python3
"""Statically inspect or prepare a CUDA-facing FastPT source workspace.

Preparation is not a build: no FastPT/Torch import, activation, installation, or
project command is performed. The generated Bash helper is separately gated.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from . import adaptation_common as common
    from .routes import non_transcode
except ImportError:
    import adaptation_common as common
    from routes import non_transcode

IGNORE_DIRS = set(common.IGNORED_DIRS)
PATCH_SUFFIXES = {".py", ".txt", ".cmake", ".mk", ".sh"}
PATCH_NAMES = {"CMakeLists.txt", "Makefile", "makefile", "GNUmakefile", "setup.cfg", "pyproject.toml"}
BUILD_SCRIPT = "fastpt_non_transcode_build.sh"
REPORT_FILE = "FASTPT_NON_TRANSCODE_REPORT.md"
PLAN_FILE = "FASTPT_NON_TRANSCODE_PLAN.json"
RESERVED_NAMES = (BUILD_SCRIPT, REPORT_FILE, PLAN_FILE)
GPUFUSION_LIB_PATTERNS = re.compile(r"\b(?:thrust|cub|cublas\w*|cudnn\w*|cufft\w*|curand\w*|cusparse\w*|stdgpu)\b|cuda_runtime\.h", re.I)
# Token bounds avoid changing mycudart_static, CUDA::cudart_static_extra,
# libcudart_static.a, or a qualified target from a different namespace.
STATIC_RUNTIME = re.compile(r"(?<![\w./:+-])(?:CUDA::cudart_static|-lcudart_static|cudart_static)(?![\w.:-])")
CUDA_PATH = re.compile(r"(?<![\w./+-])/(?:usr/local|opt)/cuda(?:-\d+(?:\.\d+)*)?(?=$|[/\s\"'`;:,)\]}])")
PLACEHOLDERS = re.compile(r"\{(ROOT|BUILD_DIR|CUDA_HOME|TORCH_PREFIX|ARCH)\}")


def ignore_tree(_: str, names: list[str]) -> set[str]:
    """Compatibility helper: build/dist/.hip/egg-info inputs are never discarded."""
    return {name for name in names if name in IGNORE_DIRS}


def candidate_files(root: Path) -> list[Path]:
    """Filters are relative to root, not its ancestors (which may be named build)."""
    return sorted(path for path in common.source_files(root) if path.name in PATCH_NAMES or path.suffix in PATCH_SUFFIXES)


def _patchable_cuda_home(cuda_home: str | None) -> bool:
    # A concrete path is safe in all supported literal contexts only in this
    # bounded subset. Other paths are still safely quoted in the build helper,
    # but require project-specific source edits rather than a language parser.
    return isinstance(cuda_home, str) and cuda_home != "/" and bool(re.fullmatch(r"/[A-Za-z0-9_./+-]+", cuda_home))


def patch_text(text: str, *, cuda_home: str | None = None) -> tuple[str, list[str]]:
    """Bounded runtime fixes; optional *literal* toolkit path, never shell syntax.

    Without an explicit safe literal path, hard-coded toolkit paths are left
    intact and find_residuals reports them. No CUDA language/API is transcoded.
    """
    changes: list[str] = []
    out, count = STATIC_RUNTIME.subn(lambda match: match[0].replace("cudart_static", "cudart"), text)
    if count:
        changes.append("exact CUDA static runtime token -> dynamic runtime")
    pattern = re.compile(r'(\bset\s*\(\s*CMAKE_CUDA_RUNTIME_LIBRARY\s+)("?)Static\2(?=\s*(?:\)|CACHE\b))', re.I)
    out, count = pattern.subn(lambda match: match[1] + match[2] + "Shared" + match[2], out)
    if count:
        changes.append("CMAKE_CUDA_RUNTIME_LIBRARY Static -> Shared")
    pattern = re.compile(r'(?<![\w-])(-D)?CMAKE_CUDA_RUNTIME_LIBRARY=("?)Static\2(?![\w.-])')
    out, count = pattern.subn(lambda match: (match[1] or "") + "CMAKE_CUDA_RUNTIME_LIBRARY=" + match[2] + "Shared" + match[2], out)
    if count:
        changes.append("exact CMake runtime assignment Static -> Shared")
    if _patchable_cuda_home(cuda_home):
        out, count = CUDA_PATH.subn(lambda _: cuda_home.rstrip("/"), out)
        if count:
            changes.append("literal CUDA toolkit prefix (including numeric version) -> explicit GPUFusion path")
    return out, changes


def detect_profile(root: Path) -> dict[str, Any]:
    root = common.resolve_repo(root)
    files = list(common.source_files(root))
    entries = sorted(name for name in common.BUILD_NAMES if (root / name).is_file())
    profile: dict[str, Any] = {
        "has_setup_py": (root / "setup.py").is_file(),
        "has_cmake": (root / "CMakeLists.txt").is_file(),
        "has_pyproject": (root / "pyproject.toml").is_file(),
        "has_cuda_sources": any(path.suffix in {".cu", ".cuh"} for path in files),
        "build_entries": entries,
        "cuda_extension": False, "cmake_cuda": False, "libtorch_cuda": False,
        "cutlass": False, "hip_rocm": False, "gpufusion_cuda_libs": False,
    }
    patterns = {
        "cuda_extension": re.compile(r"\bCUDAExtension\b|torch\.utils\.cpp_extension"),
        "cmake_cuda": re.compile(r"enable_language\s*\(\s*CUDA\b|project\s*\([^)]*\bCUDA\b|\bCMAKE_CUDA\w*|CUDA::|\bCUDAToolkit\b"),
        "libtorch_cuda": re.compile(r"\b(?:lib)?(?:torch_cuda|c10_cuda)\b"),
        "cutlass": re.compile(r"\bcutlass\b", re.I),
        "hip_rocm": re.compile(r"\bUSE_ROCM\b|\bROCM\b|__HIP_PLATFORM|\bhipblas\b|\bhipcc\b|find_package\s*\(\s*hip\b", re.I),
        "gpufusion_cuda_libs": GPUFUSION_LIB_PATTERNS,
    }
    for path in files:
        if path.suffix not in common.TEXT_SUFFIXES and path.name not in common.BUILD_NAMES:
            continue
        text = common.read_text(path)
        for name, pattern in patterns.items():
            profile[name] |= bool(pattern.search(text))
    return profile


def _path_argument(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\0\r\n"):
        raise common.AdaptationError(f"{name} requires an explicit path without NUL/newlines")
    if not (value.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", value)):
        raise common.AdaptationError(f"{name} must be an explicit absolute path")
    return value


def _shell_arg(arg: str, request: dict[str, Any]) -> str:
    """Quote literal argv, expanding only documented, separately quoted tokens."""
    variables = {"ROOT": "ROOT", "BUILD_DIR": "BUILD_DIR", "CUDA_HOME": "GPUFUSION_CUDA_HOME", "TORCH_PREFIX": "TORCH_PREFIX", "ARCH": "ARCH"}
    pieces, position = [], 0
    for match in PLACEHOLDERS.finditer(arg):
        if match[1] == "TORCH_PREFIX" and not request.get("torch_prefix"):
            raise common.AdaptationError("{TORCH_PREFIX} requires --torch-prefix; no implicit Torch query is performed")
        if match[1] == "ARCH" and not request.get("arch"):
            raise common.AdaptationError("{ARCH} requires an explicit complete gfx architecture")
        if match.start() > position:
            pieces.append(shlex.quote(arg[position:match.start()]))
        pieces.append('"$' + variables[match[1]] + '"')
        position = match.end()
    if position < len(arg):
        pieces.append(shlex.quote(arg[position:]))
    return "".join(pieces) if pieces else shlex.quote(arg)


def infer_build_commands(profile: dict[str, Any], fastpt_path: str, *, request: dict[str, Any] | None = None) -> list[str]:
    """Legacy name retained; unsafe implicit build/activation defaults now fail.

    New callers supply request= with build_command argv and capability evidence.
    For programmatic compatibility those fields may instead be in profile.
    """
    request = dict(profile if request is None else request)
    command = common.validate_build_command(request.get("build_command"))
    if not command:
        raise common.AdaptationError("Unknown build command: supply explicit build_command argv / --build-argv before preparing an activation helper")
    checks = non_transcode.capability_checks(request)
    problems = [f"{name}: {state}" for name, state, _, _ in checks if (name in non_transcode.ACTIVATION_CHECKS and state != "passed") or state == "failed"]
    if problems:
        raise common.AdaptationError("Cannot prepare an executable activation helper: " + "; ".join(problems))
    fastpt_path = _path_argument(fastpt_path, "FastPT launcher")
    cuda_home = _path_argument(request.get("gpufusion_cuda_home"), "GPUFusion CUDA home")
    arch = common.normalize_arch(request.get("arch"))
    request["arch"] = arch
    torch_prefix = request.get("torch_prefix")
    if torch_prefix is not None:
        torch_prefix = _path_argument(torch_prefix, "Torch CMake prefix")
    target = request.get("target") or "."
    if not isinstance(target, str) or Path(target).is_absolute() or ".." in Path(target).parts or any(c in target for c in "\0\r\n"):
        raise common.AdaptationError("target must be a relative subdirectory without parent traversal")
    entry = request.get("build_entry")
    if entry is not None and (not isinstance(entry, str) or not entry or Path(entry).is_absolute() or ".." in Path(entry).parts or any(c in entry for c in "\0\r\n")):
        raise common.AdaptationError("build_entry must name a target-relative file without parent traversal")
    quoted_command = " ".join(_shell_arg(arg, request) for arg in command)
    lines = [
        '# Generation is preparation only. Do not run until separately authorized.',
        'if (( $# )); then printf "%s\\n" "Pass build arguments when preparing, not as unrecorded helper arguments" >&2; exit 2; fi',
        "FASTPT_ENTRY=" + shlex.quote(fastpt_path),
        "GPUFUSION_CUDA_HOME=" + shlex.quote(cuda_home),
        "TORCH_PREFIX=" + shlex.quote(torch_prefix or ""),
        "ARCH=" + shlex.quote(arch or ""),
        'BUILD_CWD="$(realpath -e -- "$ROOT"/' + shlex.quote(target) + ')"',
        'case "$BUILD_CWD" in "$ROOT"|"$ROOT"/*) ;; *) printf "%s\\n" "Build target escapes output" >&2; exit 2 ;; esac',
        'cd -- "$BUILD_CWD"',
        "BUILD_ARGV=(" + quoted_command + ")",
        '[[ -f "$FASTPT_ENTRY" && -r "$FASTPT_ENTRY" ]] || { printf "%s\\n" "FastPT launcher is not a readable file" >&2; exit 2; }',
        '[[ -d "$GPUFUSION_CUDA_HOME/include" ]] || { printf "%s\\n" "GPUFusion include directory is missing" >&2; exit 2; }',
        '[[ -d "$GPUFUSION_CUDA_HOME/lib64" || -d "$GPUFUSION_CUDA_HOME/lib" ]] || { printf "%s\\n" "GPUFusion library directory is missing" >&2; exit 2; }',
        'command -v -- "${BUILD_ARGV[0]}" >/dev/null || { printf "%s\\n" "Build executable is missing; activation was not attempted" >&2; exit 2; }',
    ]
    if entry:
        lines.append('[[ -f "$BUILD_CWD"/' + shlex.quote(entry) + ' ]] || { printf "%s\\n" "Selected build entry is missing" >&2; exit 2; }')
    if torch_prefix:
        lines.append('[[ -d "$TORCH_PREFIX" ]] || { printf "%s\\n" "Explicit Torch prefix is missing" >&2; exit 2; }')
    lines.extend([
        '[[ "${FASTPT_ACTIVATION_APPROVED:-}" == 1 && "${FASTPT_ISOLATED_INSTALL:-}" == 1 ]] || { printf "%s\\n" "Activation requires separate user authorization and an exclusive isolated installation; capability JSON is not authorization" >&2; exit 2; }',
        'cd -- "$BUILD_CWD"',
        'source "$FASTPT_ENTRY" -C',
        'set -euo pipefail',
        'export GPUFUSION_CUDA_HOME',
        'export CUDA_HOME="$GPUFUSION_CUDA_HOME" CUDA_PATH="$GPUFUSION_CUDA_HOME"',
        'export CUDAToolkit_ROOT="$GPUFUSION_CUDA_HOME" CUDA_TOOLKIT_ROOT_DIR="$GPUFUSION_CUDA_HOME"',
        'export CPATH="$GPUFUSION_CUDA_HOME/include${CPATH:+:$CPATH}"',
        'export LIBRARY_PATH="$GPUFUSION_CUDA_HOME/lib64:$GPUFUSION_CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"',
        'export LD_LIBRARY_PATH="$GPUFUSION_CUDA_HOME/lib64:$GPUFUSION_CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
    ])
    if torch_prefix:
        lines.append('export CMAKE_PREFIX_PATH="$TORCH_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"')
    lines.append('exec "${BUILD_ARGV[@]}"')
    return lines


def _owned_workspace(root: Path) -> Path:
    root = common.resolve_repo(root)
    marker = common.workspace_path(root, common.OWNERSHIP_FILE)
    if not marker.is_file():
        raise common.AdaptationError("Write helpers only into an output created by common.prepare_tree")
    metadata = json.loads(common.read_text(marker))
    if metadata.get("route") != non_transcode.ROUTE or metadata.get("status") != "prepared-not-built":
        raise common.AdaptationError("Output ownership/route/status does not match non-transcode preparation")
    return root


def write_build_script(root: Path, profile: dict[str, Any], fastpt_path: str, *, request: dict[str, Any] | None = None) -> None:
    lines = common.bash_build_prelude("build_fastpt_non_transcode")
    lines.extend(infer_build_commands(profile, fastpt_path, request=request))
    root = _owned_workspace(root)
    script = common.safe_write_text(root, BUILD_SCRIPT, "\n".join(lines) + "\n")
    common.workspace_path(root, script).chmod(0o755)


def write_report(root: Path, profile: dict[str, Any], patches: list[str], residuals: list[str], fastpt_path: str, *, request: dict[str, Any] | None = None, plan_record: dict[str, Any] | None = None) -> None:
    root = _owned_workspace(root)
    provenance = json.loads(common.read_text(common.workspace_path(root, common.SOURCE_FILE)))
    lines = [
        "# FastPT non-transcode preparation\n\n",
        "Status: **prepared-not-built**. No activation, build, load, GPU, numerical, integration, performance, or original-platform validation has run.\n\n",
        "## Source provenance\n",
        f"- Source: `{provenance['path']}`\n- Tree SHA-256: `{provenance['tree_sha256']}`\n",
        f"- Git HEAD: `{provenance.get('git_head')}`; describe: `{provenance.get('git_describe')}`\n",
        "- Git projects retain their checkout and history in place; non-Git projects use a separate source copy. Existing artifacts are inputs, not validated build outputs.\n\n",
        "## Static profile\n",
    ]
    lines.extend(f"- {key}: `{value}`\n" for key, value in profile.items())
    lines.append("\n## Bounded changes\n")
    lines.extend(f"- {item}\n" for item in patches or ["No bounded runtime/path changes found."])
    lines.append("\n## Residuals / unsupported patterns\n")
    lines.extend(f"- {item}\n" for item in residuals or ["No listed residual found; this is not a completeness or compatibility claim."])
    lines.extend([
        "\n## Deferred build\n",
        f"- Launcher: `{fastpt_path}`; helper: `{BUILD_SCRIPT}`; static plan: `{PLAN_FILE}`.\n",
        f"- Plan status: `{plan_record.get('plan_status') if plan_record else 'not-recorded'}`.\n",
        "- Every plan execution stage remains `not-run`. Caller capability fields are not activation authorization.\n",
        "- Check the target's required features and actual provider/SDK versions before execution. Use an exclusive isolated FastPT installation; a new shell does not undo persistent provider modifications.\n",
        "- The helper requires separate approval plus isolation acknowledgments (`FASTPT_ACTIVATION_APPROVED=1` and `FASTPT_ISOLATED_INSTALL=1`); those variables do not grant permission by themselves.\n",
        "- Preserve build status with an outer pipeline: `set -o pipefail; bash ./fastpt_non_transcode_build.sh 2>&1 | tee build.log`. Record the real exit status immediately.\n",
        "- Runtime mode `-E` is a separate, authorized step in the matching isolated environment; import alone, CPU fallback, stale artifacts, and all-skipped GPU tests are not acceptance.\n",
        "- No HIP/source cleanup, converter, Torch-prefix query, or automatic fallback route is performed.\n",
    ])
    if request:
        lines.append("- Recorded argv: `" + json.dumps(request.get("build_command"), ensure_ascii=True) + "`\n")
    common.safe_write_text(root, REPORT_FILE, "".join(lines))


def find_residuals(root: Path) -> list[str]:
    residuals: list[str] = []
    patterns = [
        (re.compile(r"cudart_static"), "static runtime or unsupported similar identifier; review exact target/link semantics"),
        (re.compile(r"CMAKE_CUDA_RUNTIME_LIBRARY.*Static"), "unhandled runtime expression; no general CMake parser is used"),
        (re.compile(r"\bcutlass\b", re.I), "CUTLASS requires target-specific coverage evidence; do not silently transcode or disable it"),
        (re.compile(r"/(?:usr/local|opt)/cuda"), "hard-coded/unsupported toolkit path remains; use a reviewed concrete GPUFusion path"),
        (re.compile(r"\$\{GPUFUSION_CUDA_HOME\}"), "shell-style path expression requires language-aware review (not expanded in Python strings)"),
        (GPUFUSION_LIB_PATTERNS, "confirm CUDA-facing headers and dynamic libraries belong to the intended GPUFusion installation"),
    ]
    for path in candidate_files(root):
        text = common.read_text(path)
        for number, line in enumerate(text.splitlines(), 1):
            for pattern, message in patterns:
                if pattern.search(line):
                    residuals.append(f"{path.relative_to(root).as_posix()}:{number}: {message}: {line.strip()}")
    return residuals


def _known(value: str) -> bool | None:
    return {"true": True, "false": False, "unknown": None}[value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Application source directory; read only unless --write")
    parser.add_argument("--out", help="Copy destination for non-Git input; Git stays at --repo")
    parser.add_argument("--fastpt", help="Explicit FastPT launcher path; not sourced by this tool")
    parser.add_argument("--write", action="store_true", help="Write a prepared-not-built tree, never build it")
    parser.add_argument("--force", action="store_true", help="Compatibility option; never deletes or overwrites existing output")
    parser.add_argument("--target", help="Project-relative target directory")
    parser.add_argument("--build-entry", help="Optional target-relative build file (not a guessed command)")
    parser.add_argument("--build-argv", "--build-command", dest="build_argv", help="Explicit build argv as a JSON array, not shell text")
    parser.add_argument("--gpufusion-cuda-home", help="Explicit CUDA-facing GPUFusion toolkit directory")
    parser.add_argument("--torch-prefix", help="Optional known Torch CMake prefix; never queried implicitly")
    parser.add_argument("--arch", help="Actual complete gfx target, not SM/numeric shorthand")
    parser.add_argument("--allow-dynamic-runtime", action="store_true", help="Explicitly accept replacing static CUDA runtime tokens with dynamic linkage; not activation authorization")
    parser.add_argument("--capabilities-json", help="Contract capabilities JSON object; evidence is not permission")
    for flag in ("fastpt-installed", "version-compatible", "fastpt-invocable", "feature-probed", "hardware-validated"):
        parser.add_argument("--" + flag, choices=("true", "false", "unknown"), default=None)
    parser.add_argument("--fastpt-version")
    parser.add_argument("--torch-version")
    parser.add_argument("--required-torch-version")
    args = parser.parse_args(argv)
    try:
        repo = common.resolve_repo(args.repo)
        out, _ = common.prepare_destination(repo, args.out, default_suffix="_fastpt_non_transcode")
        capabilities = json.loads(args.capabilities_json) if args.capabilities_json else {}
        if not isinstance(capabilities, dict) or not isinstance(capabilities.get("fastpt", {}), dict):
            raise common.AdaptationError("capabilities and capabilities.fastpt must be JSON objects")
        fastpt = capabilities.setdefault("fastpt", {})
        for flag, field in (("fastpt_installed", "installed"), ("version_compatible", "version_compatible"), ("fastpt_invocable", "invocable"), ("feature_probed", "feature_probed"), ("hardware_validated", "hardware_validated")):
            if getattr(args, flag) is not None:
                fastpt[field] = _known(getattr(args, flag))
        for flag, field in (("fastpt_version", "version"), ("torch_version", "torch_version"), ("required_torch_version", "required_torch_version")):
            if getattr(args, flag) is not None:
                fastpt[field] = getattr(args, flag)
        request = {
            "repo": str(repo), "selection_mode": "explicit", "target": args.target,
            "build_entry": args.build_entry, "build_command": json.loads(args.build_argv) if args.build_argv else None,
            "arch": args.arch, "fastpt_path": args.fastpt, "gpufusion_cuda_home": args.gpufusion_cuda_home,
            "torch_prefix": args.torch_prefix, "capabilities": capabilities,
        }
        plan_record = non_transcode.plan(request)
        scan_root = Path(plan_record["scan"]["root"])
        profile = detect_profile(scan_root)
        patches: list[str] = []
        replacements = []
        for path in candidate_files(scan_root):
            old = common.read_text(path)
            new, changes = patch_text(old, cuda_home=args.gpufusion_cuda_home)
            if new != old:
                relative = path.relative_to(repo)
                replacements.append((relative, old, new))
                patches.extend(f"{relative.as_posix()}: {change}" for change in changes)
        residuals = find_residuals(scan_root)
        if args.gpufusion_cuda_home and not _patchable_cuda_home(args.gpufusion_cuda_home):
            residuals.append("Configured CUDA home is not in the safe literal patch subset; paths with quoting/spacing require language-aware manual edits. The Bash helper still quotes this path safely.")
        if not args.write:
            print(json.dumps({"status": "analysis-only", "profile": profile, "suggested_patches": patches, "input_residuals": residuals, "plan": plan_record}, ensure_ascii=True, indent=2))
            return 0
        if plan_record["plan_status"] == "blocked":
            raise common.AdaptationError("Preparation blocked: " + "; ".join(item["detail"] for item in plan_record["blockers"]))
        if not args.allow_dynamic_runtime and any("runtime" in patch.lower() for patch in patches):
            raise common.AdaptationError("Static-to-dynamic runtime changes require explicit --allow-dynamic-runtime after reviewing the delivery contract")
        # Validate argv, critical evidence and all helper configuration BEFORE
        # writing, and certainly before any possible future activation.
        infer_build_commands(profile, args.fastpt, request=request)
        work_root = common.prepare_tree(repo, out, route=non_transcode.ROUTE, force=args.force, reserved_names=RESERVED_NAMES)
        prepared_identity = json.loads(common.read_text(common.workspace_path(work_root, common.SOURCE_FILE)))
        if prepared_identity["tree_sha256"] != plan_record["source"]["tree_sha256"]:
            raise common.AdaptationError("Source changed since static planning; workspace retained for inspection, no helper generated")
        for relative, old, new in replacements:
            if common.read_text(common.workspace_path(work_root, relative)) != old:
                raise common.AdaptationError(f"Input changed while preparing {relative}; no stale patch applied")
            common.safe_write_text(work_root, relative, new, overwrite=True)
        output_scan_root = common.workspace_path(work_root, args.target, allow_root=True) if args.target else work_root
        residuals = find_residuals(output_scan_root)
        write_build_script(work_root, profile, args.fastpt, request=request)
        common.safe_write_json(work_root, PLAN_FILE, plan_record)
        write_report(work_root, profile, patches, residuals, args.fastpt, request=request, plan_record=plan_record)
        print(json.dumps({"status": "prepared-not-built", "output": str(work_root), "automatic_fixes": len(patches), "residuals": residuals, "plan_status": plan_record["plan_status"], "executed": False}, ensure_ascii=True, indent=2))
        return 0
    except (common.AdaptationError, OSError, ValueError) as exc:
        print("non-transcode preparation error: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
