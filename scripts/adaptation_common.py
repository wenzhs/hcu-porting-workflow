"""Small, standard-library-only contracts and filesystem safeguards.

No imports of Torch/FastPT, environment activation, project imports or builds.
Plans describe evidence and commands; they never claim those commands ran.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

from package_integrity import CONTRACT_VERSION as SCHEMA_VERSION
ROUTES = ("native-hip", "non-transcode", "transcode")
STAGES = ("prepare", "transform", "build", "load", "gpu", "correctness", "integration", "performance", "original-platform")
OWNERSHIP_FILE = ".adaptation-owned.json"
SOURCE_FILE = ".adaptation-source.json"
# Build/dist are deliberately NOT ignored: they can contain tracked inputs.
IGNORED_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__", ".pytest_cache"})
TEXT_SUFFIXES = frozenset({".py", ".toml", ".cfg", ".cmake", ".txt", ".sh", ".mk", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".hip", ".h", ".hpp", ".rst", ".md"})
BUILD_NAMES = frozenset({"CMakeLists.txt", "Makefile", "makefile", "GNUmakefile", "meson.build", "setup.py", "pyproject.toml"})
DEPENDENCY_DIRS = frozenset({"vendor", "third_party", "thirdparty", "external", "extern", "_deps", "node_modules"})


class AdaptationError(ValueError):
    """A request cannot safely be handled automatically."""


def _linked(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def reject_links(path: str | Path) -> Path:
    """Reject stream syntax and existing symlink/reparse components."""
    raw = os.fspath(path)
    drive, remainder = os.path.splitdrive(raw)
    if ":" in remainder:
        raise AdaptationError("Alternate data stream/colon path syntax is not allowed")
    if drive and not os.path.isabs(raw):
        raise AdaptationError("Drive-relative paths are ambiguous; use an absolute path")
    value = Path(os.path.abspath(raw))
    for component in reversed((value, *value.parents)):
        if _linked(component):
            raise AdaptationError(f"Linked/reparse path is not allowed: {component}")
    return value


def resolve_repo(repo: str | Path) -> Path:
    if not isinstance(repo, (str, Path)) or not str(repo).strip():
        raise AdaptationError("An explicit repository path is required")
    path = reject_links(repo).resolve()
    if not path.is_dir():
        raise AdaptationError(f"Repository directory does not exist: {path}")
    return path


def _within(path: Path, root: Path) -> bool:
    # normcase is needed on Windows, where spelling changes must not evade checks.
    child, parent = os.path.normcase(str(path)), os.path.normcase(str(root))
    try:
        return os.path.commonpath((child, parent)) == parent
    except ValueError:
        return False


def validate_output_path(repo: str | Path, out: str | Path) -> tuple[Path, Path]:
    source = resolve_repo(repo)
    target = reject_links(out).resolve()
    if _within(target, source) or _within(source, target):
        raise AdaptationError("Copy output must not equal, contain, or be inside the protected source directory")
    return source, target


def git_checkout_root(repo: str | Path) -> Path | None:
    """Return the Git checkout root, or None for an ordinary source tree."""
    source = resolve_repo(repo)
    marker = next((parent / ".git" for parent in (source, *source.parents) if (parent / ".git").exists()), None)
    if marker is None:
        return None
    if not shutil.which("git"):
        raise AdaptationError("Git metadata exists but git is unavailable; refusing to copy the checkout")
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=15, check=False)
    if result.returncode != 0:
        if marker is not None:
            raise AdaptationError("Git metadata exists but the checkout cannot be inspected")
        return None
    return Path(result.stdout.strip()).resolve()


def prepare_destination(repo: str | Path, out: str | Path | None, *, default_suffix: str = "_adapted") -> tuple[Path, bool]:
    """Select the in-place Git checkout or a fresh copy destination."""
    source = resolve_repo(repo)
    git_root = git_checkout_root(source)
    if git_root is not None:
        if out is not None and reject_links(out).resolve() != source:
            raise AdaptationError("Git source is edited in place; --out must equal --repo")
        return source, True
    target = Path(out) if out is not None else source.parent / (source.name + default_suffix)
    validate_output_path(source, target)
    return target.resolve(), False


def workspace_path(root: str | Path, relative: str | Path, *, allow_root: bool = False) -> Path:
    base = resolve_repo(root)
    value = Path(relative)
    candidate = reject_links(value if value.is_absolute() else base / value).resolve()
    if not _within(candidate, base) or (candidate == base and not allow_root):
        raise AdaptationError(f"Path escapes its workspace or names its root: {candidate}")
    return candidate


def validate_build_dir(workspace: str | Path, directory: str | Path) -> Path:
    """A build subdirectory inside the selected source workspace is legitimate."""
    return workspace_path(workspace, directory)


def source_files(root: str | Path, *, include_dependencies: bool = True) -> Iterator[Path]:
    base = resolve_repo(root)
    for current, directories, names in os.walk(base, followlinks=False):
        current_path = Path(current)
        keep = []
        for name in sorted(directories):
            if name in IGNORED_DIRS:
                continue
            path = current_path / name
            if _linked(path):
                raise AdaptationError(f"Source contains a linked directory: {path}")
            if include_dependencies or name not in DEPENDENCY_DIRS:
                keep.append(name)
        directories[:] = keep
        for name in sorted(names):
            if name in {".git", ".hg", ".svn"}:
                continue
            path = current_path / name
            if _linked(path):
                raise AdaptationError(f"Source contains a linked file: {path}")
            if not path.is_file():
                raise AdaptationError(f"Source contains a non-regular file: {path}")
            yield path


def literal_relative_path(value: str) -> str:
    """Validate a portable literal file path before resolving any components."""
    if (not isinstance(value, str) or not value or value.startswith("/")
            or re.search(r'[:$;\r\n\x00"\\]', value)
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise AdaptationError(f"Expected a literal normalized project-relative path: {value!r}")
    return value


def parse_predefined_macros(text: str) -> dict[str, str]:
    """Object-like defines only; empty macros never consume the next line.

    Call separately for host/device evidence. A merged compiler dump cannot
    establish which compilation pass supplied a definition.
    """
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r"[ \t]*#[ \t]*define[ \t]+(\w+)(?:[ \t]+(.*))?", line)
        if match:
            result[match.group(1)] = (match.group(2) or "").strip()
    return result


def validate_transform_map(repo: str | Path, generated_root: str | Path,
                           mapping: object, *, require_outputs: bool = False) -> dict[str, str]:
    """Check the declared transform read/write set, not converter side effects.

    Conversion may consume only owned project files. External compiler include
    paths must not be promoted to transformation inputs. Outputs are fresh files
    below a separate generated root; this is not an OS write sandbox.
    """
    source = resolve_repo(repo)
    output = reject_links(generated_root).resolve()
    if output == source or _within(source, output):
        raise AdaptationError("Generated root must not equal or contain the source root")
    if not isinstance(mapping, dict) or not mapping:
        raise AdaptationError("Transform map must be a nonempty source-to-output object")
    if any(parent.exists() and not parent.is_dir() for parent in (output, *output.parents)):
        raise AdaptationError("Generated root and each existing ancestor must be a directory")
    seen: set[Path] = set()
    clean = {}
    for original, generated in mapping.items():
        original, generated = literal_relative_path(original), literal_relative_path(generated)
        src = workspace_path(source, original)
        dst = reject_links(output / generated).resolve()
        if not src.is_file() or not _within(dst, output):
            raise AdaptationError("Transform input missing or output escapes generated root")
        if _within(src, output):
            raise AdaptationError("A generated file cannot also be an original transform input")
        key = Path(os.path.normcase(str(dst)))
        if any(key.is_relative_to(other) or other.is_relative_to(key) for other in seen):
            raise AdaptationError(f"Transform output collision: {generated}")
        for parent in dst.parents:
            if parent.exists() and not parent.is_dir():
                raise AdaptationError(f"Transform output parent is not a directory: {generated}")
            if parent == output:
                break
        seen.add(key)
        if require_outputs and not dst.is_file():
            raise AdaptationError(f"Generated output missing: {generated}")
        if not require_outputs and dst.exists():
            raise AdaptationError(f"Refusing to plan overwrite of an existing output: {generated}")
        clean[original] = generated
    return clean


def read_text(path: str | Path) -> str:
    """Preserve undecodable bytes and original line endings for bounded patches."""
    file = reject_links(path)
    data = file.read_bytes()
    if b"\0" in data:
        raise AdaptationError(f"Binary input cannot be text-patched: {file}")
    return data.decode("utf-8", errors="surrogateescape")


def safe_write_text(root: str | Path, relative: str | Path, text: str, *, overwrite: bool = False) -> Path:
    target = workspace_path(root, relative)
    if target.exists() and (not overwrite or not target.is_file()):
        raise AdaptationError(f"Refusing to overwrite an existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    workspace_path(root, target)  # Recheck after parent creation.
    mode = "wb" if overwrite else "xb"
    with target.open(mode) as stream:
        stream.write(text.encode("utf-8", errors="surrogateescape"))
    return target


def safe_write_json(root: str | Path, relative: str | Path, data: Any, *, overwrite: bool = False) -> Path:
    return safe_write_text(root, relative, json.dumps(data, ensure_ascii=True, indent=2) + "\n", overwrite=overwrite)


def file_sha256(path: str | Path) -> str:
    file = reject_links(path)
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(repo: str | Path) -> dict[str, Any]:
    root = resolve_repo(repo)
    hashes = {p.relative_to(root).as_posix(): file_sha256(p) for p in source_files(root)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    record: dict[str, Any] = {"path": str(root), "tree_sha256": digest, "files": hashes, "git_head": None, "git_describe": None, "git_branch": None}
    # Git reference metadata is optional. Avoid --dirty/index refresh, which can
    # consult worktree filters/fsmonitor; the file-tree digest carries dirty data.
    record["source_state"] = "working-tree-content"
    git_root = git_checkout_root(root)
    record["git_root"] = str(git_root) if git_root else None
    if git_root is not None:
        for field, args in (("git_head", ["rev-parse", "HEAD"]), ("git_describe", ["describe", "--always", "--tags"])):
            result = subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", str(root), *args], capture_output=True, text=True, timeout=15, check=False)
            if result.returncode == 0:
                record[field] = result.stdout.strip()
        branch = subprocess.run(["git", "-C", str(root), "symbolic-ref", "--quiet", "--short", "HEAD"], capture_output=True, text=True, timeout=15, check=False)
        record["git_branch"] = branch.stdout.strip() if branch.returncode == 0 else None
    record["git_metadata_copied"] = False
    record["source_state"] = "git-working-tree" if git_root is not None else "working-tree-content"
    record["version_note"] = "Git checkouts are edited in place on the selected adaptation branch; non-Git sources use a separate copy."
    return record


def prepare_tree(repo: str | Path, out: str | Path | None = None, *, route: str, force: bool = False, reserved_names: tuple[str, ...] = ()) -> Path:
    """Prepare the selected Git branch in place; copy only non-Git input.

    The caller selects the adaptation branch before analysis. Existing outputs
    and user files are never removed by --force.
    """
    if route not in ROUTES:
        raise AdaptationError(f"Unknown route: {route}")
    source = resolve_repo(repo)
    target, in_place = prepare_destination(source, out)
    if in_place:
        if force:
            raise AdaptationError("--force is not used for in-place Git adaptation")
        result = subprocess.run(["git", "-C", str(source), "symbolic-ref", "--quiet", "--short", "HEAD"], capture_output=True, text=True, timeout=15, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise AdaptationError("Git checkout is detached; create an adaptation branch before writing")
        for name in (OWNERSHIP_FILE, SOURCE_FILE, *reserved_names):
            if (source / name).exists() or _linked(source / name):
                raise AdaptationError(f"Source collides with a reserved generated file: {name}")
        identity = source_identity(source)
        safe_write_json(source, SOURCE_FILE, identity)
        safe_write_json(source, OWNERSHIP_FILE, {"schema_version": SCHEMA_VERSION, "route": route, "status": "prepared-not-built", "source_sha256": identity["tree_sha256"]})
        return source
    if target.exists():
        raise AdaptationError("Output already exists; choose a fresh path. --force does not authorize deletion.")
    files = list(source_files(source))
    for name in (OWNERSHIP_FILE, SOURCE_FILE, *reserved_names):
        if (source / name).exists() or _linked(source / name):
            raise AdaptationError(f"Source collides with a reserved generated file: {name}")
    identity = source_identity(source)
    if {path.relative_to(source).as_posix() for path in files} != set(identity["files"]):
        raise AdaptationError("Source inventory changed before copying; no output created")
    target.parent.mkdir(parents=True, exist_ok=True)
    reject_links(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        for path in files:
            reject_links(path)
            rel = path.relative_to(source)
            destination = workspace_path(staging, rel)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination, follow_symlinks=False)
            if file_sha256(destination) != identity["files"][rel.as_posix()]:
                raise AdaptationError(f"Source changed during copy: {rel}")
        copied = {path.relative_to(staging).as_posix() for path in source_files(staging)}
        if copied != set(identity["files"]) or source_identity(source)["tree_sha256"] != identity["tree_sha256"]:
            raise AdaptationError("Source inventory/content changed during preparation; output not published")
        safe_write_json(staging, SOURCE_FILE, identity)
        safe_write_json(staging, OWNERSHIP_FILE, {"schema_version": SCHEMA_VERSION, "route": route, "status": "prepared-not-built", "source_sha256": identity["tree_sha256"]})
        if target.exists():
            raise AdaptationError("Output appeared during preparation; refusing to replace it")
        staging.rename(target)
    except BaseException:
        # This is the freshly allocated directory owned by this invocation only.
        if staging.is_dir() and not _linked(staging):
            shutil.rmtree(staging)
        raise
    return target


def validate_prepared_inputs(workspace: str | Path, analysis_identity: dict[str, Any], originals: dict[Path, str]) -> None:
    """Reject stale analysis before applying ANY of its stored patches."""
    root = resolve_repo(workspace)
    copied = json.loads(read_text(workspace_path(root, SOURCE_FILE)))
    if copied.get("tree_sha256") != analysis_identity.get("tree_sha256"):
        raise AdaptationError("Source changed after analysis; workspace retained without applying patches")
    for relative, expected in originals.items():
        if read_text(workspace_path(root, relative)) != expected:
            raise AdaptationError(f"Workspace input changed since analysis: {relative}; no stale patches applied")


def normalize_arch(arch: str | None) -> str | None:
    if not arch:
        return None
    if not isinstance(arch, str) or not re.fullmatch(r"gfx[0-9a-f]+(?::(?:sramecc|xnack)[+-])*", arch.lower()):
        raise AdaptationError("Use an explicit complete gfx target (for example gfx936 or gfx92a), not an SM or numeric shorthand")
    return arch.lower()


def scan_project(repo: str | Path, target: str | None = None) -> dict[str, Any]:
    """Static candidate evidence, NOT a statement of backend availability."""
    root = resolve_repo(repo)
    if target is not None and (not isinstance(target, str) or not target or Path(target).is_absolute() or Path(target).drive or ".." in Path(target).parts):
        raise AdaptationError("target must be a project-relative directory without parent traversal")
    scan_root = workspace_path(root, target, allow_root=True) if target else root
    if not scan_root.is_dir():
        raise AdaptationError("target must name a project-relative directory")
    entries = [name for name in BUILD_NAMES if (scan_root / name).is_file()]
    evidence = []
    native = re.compile(r"enable_language\s*\(\s*HIP\b|LANGUAGES[^)\n]*\bHIP\b|find_package\s*\(\s*hip\b|\bCMAKE_HIP\w*|[<\"]hip/hip_runtime|\bUSE_ROCM\b|torch\.version\.(?:hip|hcu)|getattr\s*\(\s*torch\.version\s*,\s*['\"](?:hip|hcu)['\"]|IS_HIP_EXTENSION|\w+_WITH_HIP|BUILD_VARIANT[^\n]*rocm|COMPUTE_BACKEND[^\n]*hip|BUILD_HIP|\bhipcc\b", re.I)
    # __global__/__device__ are shared with HIP and cannot justify a CUDA
    # conversion fallback when the actual native backend is unavailable.
    cuda = re.compile(r"CUDAExtension|CUDA::|CMAKE_CUDA|enable_language\s*\(\s*CUDA|LANGUAGES[^)\n]*\bCUDA\b|\bnvcc\b|\bcuda[A-Z]\w*|[<\"]cuda(?:_runtime(?:_api)?)?\.h[>\"]", re.I)
    for path in source_files(scan_root, include_dependencies=False):
        if path.suffix not in TEXT_SUFFIXES and path.name not in BUILD_NAMES:
            continue
        text = read_text(path)
        if path.name == "CMakeLists.txt" or path.suffix == ".cmake":
            for kind, language in (("native", "HIP"), ("cuda", "CUDA")):
                for match in re.finditer(r"(?:project|enable_language)\s*\([^)]*\b" + language + r"\b[^)]*\)", text, re.I):
                    evidence.append({"kind": kind, "file": path.relative_to(root).as_posix(), "line": text.count("\n", 0, match.start()) + 1, "text": match.group(0), "note": "multiline language candidate; confirm actual target/configuration"})
        for number, line in enumerate(text.splitlines(), 1):
            for kind, pattern in (("native", native), ("cuda", cuda)):
                if pattern.search(line):
                    entry = {"kind": kind, "file": path.relative_to(root).as_posix(), "line": number, "text": line.strip()}
                    # PyTorch uses the same extension class for HIP and CUDA inputs.
                    if kind == "cuda" and "CUDAExtension" in line and not cuda.search(line.replace("CUDAExtension", "")):
                        entry["framework_hint_only"] = True
                    evidence.append(entry)
        if path.suffix == ".hip":
            evidence.append({"kind": "native", "file": path.relative_to(root).as_posix(), "line": None, "text": "HIP source file; confirm target membership"})
        elif path.suffix in {".cu", ".cuh"}:
            evidence.append({"kind": "cuda", "file": path.relative_to(root).as_posix(), "line": None, "text": "CUDA source/header candidate; confirm target membership and backend"})
    kinds = {entry["kind"] for entry in evidence}
    native_candidate = "native" in kinds
    cuda_candidate = "cuda" in kinds
    return {"root": str(scan_root), "root_entries": sorted(entries), "native_candidate": native_candidate, "cuda_candidate": cuda_candidate, "evidence": evidence, "coverage": "static target-directory candidates; conditionals, dependency backends and actual membership require confirmation"}


def new_plan(route: str, request: dict[str, Any], scan: dict[str, Any] | None = None) -> dict[str, Any]:
    if route not in ROUTES:
        raise AdaptationError(f"Unknown route: {route}")
    if not isinstance(request, dict) or not isinstance(request.get("repo"), (str, Path)):
        raise AdaptationError("request.repo is required")
    features = request.get("required_features", [])
    if not isinstance(features, list) or any(not isinstance(feature, str) or not feature.strip() for feature in features):
        raise AdaptationError("required_features must be a list of nonempty feature names")
    selection = request.get("selection_mode", "explicit")
    if selection not in ("explicit", "auto"):
        raise AdaptationError("selection_mode must be explicit or auto")
    repo = resolve_repo(request["repo"])
    return {"schema_version": SCHEMA_VERSION, "kind": "adaptation-plan", "route": route, "selection_mode": selection, "scope": {"repo": str(repo), "target": request.get("target"), "required_features": request.get("required_features", [])}, "source": source_identity(repo), "scan": scan or {}, "checks": [], "blockers": [], "warnings": [], "commands": [], "stages": {stage: "not-run" for stage in STAGES}, "plan_status": "needs-information", "executed": False}


def add_check(plan: dict[str, Any], name: str, status: str, detail: str, evidence: Any = None) -> None:
    if status not in {"passed", "failed", "unknown", "not-applicable"}:
        raise AdaptationError(f"Invalid check status: {status}")
    check = {"name": name, "status": status, "detail": detail}
    if evidence is not None:
        check["evidence"] = evidence
    plan["checks"].append(check)
    if status == "failed":
        plan["blockers"].append({"name": name, "detail": detail})


def finalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    plan["plan_status"] = "blocked" if plan["blockers"] else ("needs-information" if not plan["checks"] or any(c["status"] == "unknown" for c in plan["checks"]) else "planned")
    return plan


def validate_build_command(command: Any) -> list[str] | None:
    if command is None:
        return None
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg or "\0" in arg or "\n" in arg or "\r" in arg for arg in command):
        raise AdaptationError("build_command must be a non-empty argv array, not a shell expression")
    if any(arg in {"&&", "||", ";", "|", ">", ">>", "<"} for arg in command):
        raise AdaptationError("Shell operators are not build argv; record logging/setup separately")
    executable = command[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if executable in {"bash", "sh", "dash", "zsh", "ksh", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        for argument in command[1:]:
            flag = argument.lower()
            if flag in {"-c", "/c", "/k", "-command", "-encodedcommand"} or (executable in {"bash", "sh", "dash", "zsh", "ksh"} and flag.startswith("-") and not flag.startswith("--") and "c" in flag[1:]):
                raise AdaptationError("Use an original script/executable argv, not a shell command expression")
            if argument == "--" or not argument.startswith("-"):
                break
    return list(command)


def version_series(value: Any) -> tuple[int, int] | None:
    """Parse a declared major/minor series; equality is not full compatibility."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise AdaptationError("Version values must be strings or null")
    match = re.fullmatch(r"v?(\d+)\.(\d+)(?:\.\d+)?(?:[a-zA-Z+._-][a-zA-Z0-9+._-]*)?", value.strip())
    if not match:
        raise AdaptationError(f"Unrecognized declared version: {value!r}")
    return int(match[1]), int(match[2])


def bash_build_prelude(default_dir: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", default_dir):
        raise AdaptationError("Invalid default build directory name")
    return ["#!/usr/bin/env bash", "set -euo pipefail", 'ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"', 'cd "$ROOT"', f'BUILD_DIR="${{BUILD_DIR:-$ROOT/{default_dir}}}"', 'BUILD_DIR="$(realpath -m -- "$BUILD_DIR")"', 'case "$BUILD_DIR" in "$ROOT"/*) ;; *) printf "%s\\n" "BUILD_DIR must stay inside this task workspace" >&2; exit 2 ;; esac']


def validate_receipt(record: dict[str, Any], *, expected_source_sha256: str | None = None) -> list[str]:
    """Validate completeness/freshness, not authenticity of user-supplied evidence."""
    if not isinstance(record, dict):
        return ["receipt must be an object"]
    errors = []
    if record.get("status") not in ("completed", "failed", "blocked", "partial", "not-run"):
        errors.append("invalid result status")
    if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != "adaptation-result":
        errors.append("receipt schema/kind mismatch")
    if record.get("route") not in ROUTES:
        errors.append("invalid route")
    source_sha = record.get("source_sha256")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        errors.append("source identity missing")
    if expected_source_sha256 is not None and source_sha != expected_source_sha256:
        errors.append("source identity is stale")
    if record.get("status") != "completed":
        errors.append(f"receipt is not completed (status={record.get('status')!r}); only completed receipts may mark an entry covered")
        return errors
    stages = record.get("stages", {})
    if not isinstance(stages, dict):
        errors.append("stages must be an object")
        stages = {}
    for stage in ("build", "load", "gpu", "correctness"):
        if stages.get(stage) != "passed":
            errors.append(f"required stage has not passed: {stage}")
    tests = record.get("tests", {})
    if not isinstance(tests, dict):
        tests = {}
    counts = [tests.get(key) for key in ("total", "passed", "failed", "skipped", "gpu_passed")]
    if any(type(n) is not int or n < 0 for n in counts):
        errors.append("test counts missing/invalid")
    elif counts[0] != sum(counts[1:4]) or counts[1] == 0 or counts[2] != 0 or not 0 < counts[4] <= counts[1]:
        errors.append("no successful GPU test coverage or inconsistent test counts")
    oracle, contract = record.get("oracle"), record.get("validation_contract")
    if not isinstance(oracle, (str, dict)) or not oracle or not isinstance(contract, dict) or not contract:
        errors.append("oracle/validation contract missing")
    if isinstance(contract, dict):
        for name in ("atol", "rtol"):
            if name in contract and (type(contract[name]) not in (int, float) or not math.isfinite(contract[name]) or contract[name] < 0):
                errors.append(f"invalid tolerance: {name}")
        required = contract.get("required_stages", [])
        if not isinstance(required, list) or any(stage not in STAGES for stage in required):
            errors.append("invalid required stages")
        else:
            for stage in required:
                if stages.get(stage) != "passed":
                    errors.append(f"contract stage has not passed: {stage}")
    commands = record.get("commands", [])
    if not isinstance(commands, list) or not commands:
        errors.append("command receipts missing")
    else:
        for stage in ("build", "correctness"):
            if not any(isinstance(c, dict) and c.get("phase") == stage and c.get("argv") and type(c.get("exit_code")) is int and c["exit_code"] == 0 for c in commands):
                errors.append(f"successful command receipt missing: {stage}")
    artifacts = record.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifact identity missing")
    else:
        loaded = False
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                errors.append("invalid artifact record")
                continue
            try:
                current = file_sha256(artifact.get("path", ""))
                if current != artifact.get("sha256") or artifact.get("source_sha256") != source_sha:
                    errors.append("artifact is stale or mismatched")
                elif artifact.get("loaded") is True:
                    loaded = True
            except (OSError, ValueError):
                errors.append("artifact cannot be verified")
        if not loaded:
            errors.append("no verified loaded artifact")
    return errors
