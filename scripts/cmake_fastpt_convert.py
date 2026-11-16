#!/usr/bin/env python3
"""Bounded CMake adaptation using an explicit, existing source mapping.

This is not a CMake interpreter or a source converter. Unsupported constructs
block ALL writes, rather than leaving a partly converted project. No builds run.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shlex
import sys
from pathlib import Path, PurePosixPath
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptation_common as common

LIB_MAP = {
    "CUDA::cudart_static": "hip::host", "CUDA::cudart": "hip::host",
    "CUDA::cublasLt": "roc::hipblaslt", "CUDA::cublas": "roc::hipblas",
    "CUDA::cufft": "hip::hipfft",
    "CUDA::cusparse": "roc::hipsparse",
    "cudart_static": "hip::host", "cudart": "hip::host",
    "cublasLt": "roc::hipblaslt", "cublas": "roc::hipblas",
    "cufft": "hip::hipfft", "cusparse": "roc::hipsparse",
}
# cuRAND APIs normally need the HIP wrapper, not a guessed rocRAND target.
# Unverified wrapper/export names are deliberately refused for manual review.
PACKAGE_MAP = {
    "hip::host": "hip", "roc::hipblas": "hipblas", "roc::hipblaslt": "hipblaslt",
    "hip::hipfft": "hipfft", "roc::hipsparse": "hipsparse",
}
VARIABLE_MAP = {"CMAKE_CUDA_STANDARD": "CMAKE_HIP_STANDARD",
                "CMAKE_CUDA_STANDARD_REQUIRED": "CMAKE_HIP_STANDARD_REQUIRED",
                "CMAKE_CUDA_EXTENSIONS": "CMAKE_HIP_EXTENSIONS"}
REPORT_NAME = "FASTPT_CMAKE_CONVERSION_REPORT.md"
MANIFEST_NAME = "fastpt_cmake_conversion.json"
SCRIPT_NAME = "fastpt_dtk_cmake_build.sh"
RESERVED_NAMES = (REPORT_NAME, MANIFEST_NAME, SCRIPT_NAME)
IGNORE_DIRS = common.IGNORED_DIRS
REVIEW_PATTERNS = [(re.compile(r"cutlass", re.I), "CUTLASS requires a separate, version-bound semantic review")]


class Token(NamedTuple):
    value: str
    start: int
    end: int
    quoted: bool = False
    opaque: bool = False


class Command(NamedTuple):
    name: str
    start: int
    end: int
    tokens: list[Token]
    line: int


def ignore_tree(_: str, names: list[str]) -> set[str]:
    """Legacy helper; build/dist/.hip are deliberately not disposable."""
    return set(names).intersection(IGNORE_DIRS)


def cmake_files(root: Path) -> list[Path]:
    return sorted(p for p in common.source_files(root) if p.name == "CMakeLists.txt" or p.suffix == ".cmake")


def _bracket_end(text: str, pos: int) -> int | None:
    opening = re.match(r"\[(=*)\[", text[pos:])
    if not opening:
        return None
    close = "]" + opening.group(1) + "]"
    end = text.find(close, pos + len(opening.group()))
    if end < 0:
        raise common.AdaptationError("Unterminated CMake bracket argument/comment")
    return end + len(close)


def _skip(text: str, pos: int) -> int:
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
        elif text[pos] == "#":
            bracket = _bracket_end(text, pos + 1)
            end = text.find("\n", pos)
            pos = bracket if bracket is not None else (len(text) if end < 0 else end + 1)
        else:
            break
    return pos


def parse_commands(text: str) -> list[Command]:
    """Lex balanced commands, retaining exact spans, comments and line endings.

    Nested expression tokens are opaque: the transformer refuses to interpret
    them when a rewrite would require it. No CMake evaluation is performed.
    """
    commands = []
    pos = 1 if text.startswith("﻿") else 0
    while (pos := _skip(text, pos)) < len(text):
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text[pos:])
        if not match:
            raise common.AdaptationError(f"Unsupported CMake syntax at line {text.count(chr(10), 0, pos) + 1}")
        start = pos
        name = match.group(1).lower()
        pos += len(match.group())
        tokens: list[Token] = []
        while True:
            pos = _skip(text, pos)
            if pos >= len(text):
                raise common.AdaptationError(f"Unterminated {name} command")
            if text[pos] == ")":
                pos += 1
                break
            begin = pos
            bracket = _bracket_end(text, pos)
            if bracket is not None:
                pos = bracket
                tokens.append(Token(text[begin:pos], begin, pos, opaque=True))
            elif text[pos] == '"':
                pos += 1
                while pos < len(text) and text[pos] != '"':
                    pos += 2 if text[pos] == "\\" else 1
                if pos >= len(text):
                    raise common.AdaptationError("Unterminated CMake quoted argument")
                pos += 1
                value = text[begin + 1:pos - 1]
                tokens.append(Token(value, begin, pos, True, "\\" in value))
            else:
                depth = 0
                escaped = False
                while pos < len(text):
                    char = text[pos]
                    if char == "\\":
                        escaped = True
                        pos += 2
                        continue
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        if depth == 0:
                            break
                        depth -= 1
                    if char.isspace() and depth == 0:
                        break
                    pos += 1
                value = text[begin:pos]
                if not value:
                    raise common.AdaptationError("Unsupported empty CMake token")
                tokens.append(Token(value, begin, pos, opaque=escaped or "(" in value or ")" in value))
        commands.append(Command(name, start, pos, tokens, text.count("\n", 0, start) + 1))
    return commands


def replace_word(text: str, src: str, dst: str) -> str:
    # Qualified imported targets also need BOTH boundaries; substring replacement
    # corrupts cudart_static/cublasLt and similarly named user targets.
    return re.sub(rf"(?<![A-Za-z0-9_:+.-]){re.escape(src)}(?![A-Za-z0-9_:+.-])", lambda _: dst, text)


def strip_cuda_gencode(line: str, arch: str | None) -> str:
    """Diagnose known NVCC options, including literal CMake option groups.

    This never evaluates a CMake expression or removes an option. Unknown dynamic
    SHELL groups and ambiguous quoted/list forms are refused for manual review.
    """
    pattern = re.compile(r"(?:^|[\s:])(?:-gencode|--generate-code|-arch|--gpu-architecture|--cuda-gpu-arch|-code|--gpu-code)(?==|\s|$)")
    for element in line.split(";"):
        element = element.strip()
        if element.startswith("SHELL:"):
            body = element[len("SHELL:"):]
            if "$" in body or "\\;" in body:
                raise common.AdaptationError("Dynamic/escaped CMake SHELL options require manual review")
            try:
                values = shlex.split(body, posix=True)
            except ValueError as exc:
                raise common.AdaptationError("Ambiguous CMake SHELL option group requires manual review") from exc
        else:
            values = [element]
        if any(pattern.search(value) for value in values):
            raise common.AdaptationError("CUDA code-generation flags need an explicit, reviewed HIP replacement; they are not removed heuristically")
    return line


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _token_text(token: Token, value: str) -> str:
    return _quote(value) if token.quoted or re.search(r"\s", value) else value


def _patch(text: str, edits: list[tuple[int, int, str]]) -> str:
    for start, end, value in sorted(edits, reverse=True):
        text = text[:start] + value + text[end:]
    return text


def _newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _indent(text: str, command: Command) -> str:
    prefix = text[text.rfind("\n", 0, command.start) + 1:command.start]
    if prefix.strip():
        raise common.AdaptationError("Commands sharing one physical line cannot receive scoped insertions")
    return prefix


def _map_reference(value: str, rel_path: Path, mapping: dict[str, str]) -> str | None:
    prefix = ""
    path = value
    base = PurePosixPath(rel_path.as_posix()).parent
    for marker, origin in (("${CMAKE_CURRENT_SOURCE_DIR}/", base), ("${CMAKE_SOURCE_DIR}/", PurePosixPath("."))):
        if value.startswith(marker):
            prefix, path, base = marker, value[len(marker):], origin
            break
    if "$" in path or ";" in path or "\\" in path or Path(path).is_absolute():
        raise common.AdaptationError(f"Source reference is not a literal project-relative path: {value}")
    key = os.path.normpath(str(base / path)).replace("\\", "/")
    destination = mapping.get(key)
    if destination is None:
        return None
    relative = os.path.relpath(destination, str(base)).replace("\\", "/")
    return prefix + relative


def validate_source_map(repo: Path, mapping: object) -> dict[str, str]:
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
        raise common.AdaptationError("source_map must map original project-relative paths to actual project-relative files")
    clean = {}
    for source, destination in mapping.items():
        for path in (source, destination):
            common.literal_relative_path(path)
        actual = common.workspace_path(repo, destination)
        if not actual.is_file():
            raise common.AdaptationError(f"Mapped source does not exist: {destination}; this adapter does not create HIP sources")
        if Path(source).suffix not in {".cu", ".cuh"} or actual.suffix not in {".cu", ".hip", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cuh"}:
            raise common.AdaptationError(f"Unsupported source mapping: {source} -> {destination}")
        if Path(source).suffix == ".cu" and actual.suffix not in {".cu", ".hip", ".cpp", ".cc", ".cxx"}:
            raise common.AdaptationError("A CUDA translation unit must map to a compilable source, not a header")
        clean[source] = destination
    return clean


def detect_packages(text: str) -> set[str]:
    """Legacy inspection only. Never aggregate this across subdirectories."""
    values = {token.value for command in parse_commands(text) for token in command.tokens}
    return {f"find_package({package} REQUIRED)" for target, package in PACKAGE_MAP.items() if target in values}


def insert_top_level_block(text: str, arch: str | None, packages: set[str]) -> str:
    """Insert an optional architecture *default* before HIP initialization.

    The compatibility package argument permits only hip at this local entry;
    dependency packages must be discovered in their original target scope.
    """
    arch = common.normalize_arch(arch)
    if any(not re.fullmatch(r"find_package\(hip(?: CONFIG)?(?: REQUIRED| QUIET)?\)", p) for p in packages):
        raise common.AdaptationError("Do not hoist dependency packages to the root; retain their target/conditional scope")
    commands = parse_commands(text)
    enable = next((c for c in commands if (c.name == "project" and any(t.value == "HIP" for t in c.tokens[1:])) or (c.name == "enable_language" and any(t.value == "HIP" for t in c.tokens))), None)
    newline = _newline(text)
    edits = []
    if arch and enable and "FASTPT_DTK_CMAKE_BEGIN" not in text:
        indent = _indent(text, enable)
        block = newline.join(["# FASTPT_DTK_CMAKE_BEGIN", "if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)", f'  set(CMAKE_HIP_ARCHITECTURES "{arch}" CACHE STRING "HIP GPU architectures")', "endif()", "# FASTPT_DTK_CMAKE_END", ""])
        block = block.replace(newline, newline + indent)
        edits.append((enable.start, enable.start, block))
    if packages:
        project = next((c for c in commands if c.name == "project"), None)
        if not project:
            raise common.AdaptationError("Cannot insert local hip discovery without a project command")
        if not any(c.name == "find_package" and c.tokens and c.tokens[0].value == "hip" for c in commands):
            edits.append((project.end, project.end, newline + next(iter(sorted(packages))) + newline))
    return _patch(text, edits)


def hip_arch_targets(commands: list[Command]) -> set[str]:
    """Conservative inventory: do not overwrite an existing HIP target setting."""
    found: set[str] = set()
    for command in commands:
        values = [token.value for token in command.tokens]
        targets: list[str] = []
        if command.name == "set_target_properties" and "PROPERTIES" in values:
            marker = values.index("PROPERTIES")
            if "HIP_ARCHITECTURES" in values[marker + 1::2]:
                targets = values[:marker]
        elif command.name == "set_property" and values and values[0] == "TARGET" and "PROPERTY" in values:
            marker = values.index("PROPERTY")
            if marker + 1 < len(values) and values[marker + 1] == "HIP_ARCHITECTURES":
                targets = [value for value in values[1:marker] if value not in {"APPEND", "APPEND_STRING"}]
        for target in targets:
            found.add(target if re.fullmatch(r"[A-Za-z0-9_.:+-]+", target) else "*")
    return found


def transform_text(text: str, *, rel_path: Path, arch: str | None = None,
                   cuda_root: str | None = None, dtk_root: str | None = None,
                   source_map: dict[str, str] | None = None,
                   established_hip_targets: set[str] | None = None) -> tuple[str, list[str]]:
    """Pure bounded rewrite; callers must verify source_map with the filesystem.

    Returns unchanged text on any BLOCK diagnostic. Metadata/conditions are never
    evaluated, package discovery is local, and generated source files are never
    guessed from an extension. Identity .cu mappings get explicit LANGUAGE HIP.
    """
    diagnostics: list[str] = []
    mapping = source_map or {}
    try:
        arch = common.normalize_arch(arch)
        if cuda_root or dtk_root:
            raise common.AdaptationError("--cuda-root/--dtk-root implicit directory replacement is unsafe; supply an actual --source-map")
        commands = parse_commands(text)
    except common.AdaptationError as exc:
        return text, [f"BLOCK: {rel_path}: {exc}"]
    existing_hip = hip_arch_targets(commands) | set(established_hip_targets or ())
    edits: list[tuple[int, int, str]] = []
    stack: list[str] = []
    # Each conditional branch has its own discovery scope. An optional package
    # query stays optional, and a disabled vendor never becomes root REQUIRED.
    discovery: list[dict[str, str]] = [{}]
    newline = _newline(text)
    active = any(re.search(r"CUDA|CUDAToolkit|\.cu(?:h)?(?:$|[;>])", t.value) for c in commands for t in c.tokens)
    for command in commands:
        name, tokens = command.name, command.tokens
        values = [t.value for t in tokens]
        before: list[str] = []
        after: list[str] = []
        local: list[tuple[int, int, str]] = []
        replace_command: str | None = None
        scope = {key: value for layer in discovery for key, value in layer.items()}
        try:
            if name in {"target_compile_options", "add_compile_options", "set", "set_property", "set_target_properties", "add_custom_command", "execute_process"} or name.startswith("cuda_"):
                for token in tokens:
                    strip_cuda_gencode(token.value, arch)
            if name in {"else", "elseif"}:
                if not stack or stack[-1] != "if":
                    raise common.AdaptationError("Unbalanced conditional")
                discovery[-1] = {}
                scope = {key: value for layer in discovery for key, value in layer.items()}
            relevant = any(re.search(r"CUDA|CUDAToolkit|\.cu(?:h)?(?:$|[;>])|-gencode|--generate-code|\bnvcc\b", v) for v in values) or (name == "target_link_libraries" and any(v in LIB_MAP for v in values))
            if relevant and any(t.opaque or "$<" in t.value or ";" in t.value for t in tokens):
                # A literal semicolon architecture list is explicitly handled below.
                arch_set = name == "set" and values and values[0] == "CMAKE_CUDA_ARCHITECTURES" and not any(t.opaque or "$" in t.value for t in tokens)
                if not arch_set:
                    raise common.AdaptationError("Generator expressions, bracket/escaped arguments or list-valued CUDA expressions require project-specific review")
            if relevant and any(frame in {"function", "macro", "foreach", "while", "block"} for frame in stack):
                raise common.AdaptationError("CUDA edits inside functions/macros/loops/blocks are outside the bounded adapter")
            if name == "project":
                language_start = values.index("LANGUAGES") + 1 if "LANGUAGES" in values else 1
                for index in range(language_start, len(tokens)):
                    if values[index] == "CUDA":
                        local.append((tokens[index].start, tokens[index].end, _token_text(tokens[index], "HIP")))
            elif name == "enable_language":
                for token in tokens:
                    if token.value == "CUDA":
                        local.append((token.start, token.end, _token_text(token, "HIP")))
            elif name == "find_package" and values:
                package = values[0]
                if package in {"CUDA", "CUDAToolkit"}:
                    if any(v not in {"REQUIRED", "QUIET", "CONFIG", "NO_MODULE"} for v in values[1:]):
                        raise common.AdaptationError("CUDA package versions/components/search paths cannot be transferred to HIP discovery")
                    local.append((tokens[0].start, tokens[0].end, "hip"))
                    package = "hip"
                discovery[-1][package] = "QUIET" if "QUIET" in values and "REQUIRED" not in values else "REQUIRED"
            elif name == "target_link_libraries" and values:
                for token in tokens[1:]:
                    if token.value in LIB_MAP:
                        target = LIB_MAP[token.value]
                        local.append((token.start, token.end, _token_text(token, target)))
                        package = PACKAGE_MAP[target]
                        if package not in scope:
                            policy = scope.get("hip", "REQUIRED")
                            before.append(f"find_package({package} {policy})")
                            scope[package] = policy
                            discovery[-1][package] = policy
                    elif re.search(r"CUDA::|(?:^|-l)(?:cudart|cublas|cufft|curand|cusparse|cudnn|nccl|nvrtc)", token.value):
                        raise common.AdaptationError(f"No exact, verified library target mapping for {token.value}")
            elif name == "set" and values and values[0] in {"CMAKE_CUDA_ARCHITECTURES", "CMAKE_HIP_ARCHITECTURES"}:
                if values[0] == "CMAKE_CUDA_ARCHITECTURES":
                    if not arch:
                        raise common.AdaptationError("CUDA architectures cannot determine a HIP architecture; supply --arch gfx...")
                    if len(values) < 2 or any("$" in v for v in values[1:]) or "PARENT_SCOPE" in values:
                        raise common.AdaptationError("Dynamic/inherited CUDA architectures need explicit project-specific configuration")
                    replace_command = newline.join(["if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)", f'  set(CMAKE_HIP_ARCHITECTURES "{arch}" CACHE STRING "HIP GPU architectures")', "endif()"])
                else:
                    if len(values) < 2:
                        raise common.AdaptationError("Empty HIP architecture configuration")
                    end = values.index("CACHE") if "CACHE" in values else len(values)
                    for value in values[1:end]:
                        for item in value.split(";"):
                            common.normalize_arch(item)
            elif name == "set_target_properties" and "CUDA_ARCHITECTURES" in values:
                if not arch or "PROPERTIES" not in values:
                    raise common.AdaptationError("Target CUDA architectures require an explicit gfx target")
                property_start = values.index("PROPERTIES") + 1
                target_names = values[:property_start - 1]
                if "*" in existing_hip or any(target in existing_hip for target in target_names):
                    raise common.AdaptationError("Existing HIP_ARCHITECTURES for this target must be preserved; resolve the CUDA/HIP property combination explicitly")
                if not target_names or any(not re.fullmatch(r"[A-Za-z0-9_.:+-]+", target) for target in target_names):
                    raise common.AdaptationError("Target architecture edits require literal target names")
                if (len(values) - property_start) % 2:
                    raise common.AdaptationError("Malformed target properties")
                for index in range(property_start, len(values), 2):
                    if values[index] == "CUDA_ARCHITECTURES":
                        local.extend([(tokens[index].start, tokens[index].end, "HIP_ARCHITECTURES"), (tokens[index + 1].start, tokens[index + 1].end, _quote(arch))])
                    elif "CUDA" in values[index]:
                        raise common.AdaptationError(f"Unsupported target property {values[index]}")
            elif name in {"add_library", "add_executable", "target_sources", "set"}:
                if name in {"add_library", "add_executable", "target_sources"} and active:
                    for token in tokens[1:]:
                        if "$" in token.value and not token.value.startswith(("${CMAKE_CURRENT_SOURCE_DIR}/", "${CMAKE_SOURCE_DIR}/")):
                            raise common.AdaptationError("Indirect/generated target source lists require explicit project-specific conversion")
                    if name == "target_sources" and any(v in {"FILE_SET", "BASE_DIRS"} for v in values):
                        raise common.AdaptationError("FILE_SET source semantics are outside the bounded adapter")
                for token in tokens[1:]:
                    if re.search(r"\.cu(?:h)?$", token.value):
                        mapped = _map_reference(token.value, rel_path, mapping)
                        if mapped is None:
                            raise common.AdaptationError(f"Missing actual source mapping for {token.value}; no .hip file is assumed")
                        local.append((token.start, token.end, _token_text(token, mapped)))
                        if token.value.endswith(".cu"):
                            if name == "target_sources":
                                if not values or not re.fullmatch(r"[A-Za-z0-9_.+-]+", values[0]):
                                    raise common.AdaptationError("target_sources needs a literal owning target for HIP language scope")
                                if any(value in {"PUBLIC", "INTERFACE"} for value in values[1:]) and not mapped.endswith(".hip"):
                                    raise common.AdaptationError("Public/interface identity or C++ source mappings need consumer-directory language review")
                                absolute = mapped if mapped.startswith(("${CMAKE_CURRENT_SOURCE_DIR}/", "${CMAKE_SOURCE_DIR}/")) else "${CMAKE_CURRENT_SOURCE_DIR}/" + mapped
                                after.append(f"set_source_files_properties({_quote(absolute)} TARGET_DIRECTORY {values[0]} PROPERTIES LANGUAGE HIP)")
                            else:
                                after.append(f"set_source_files_properties({_quote(mapped)} PROPERTIES LANGUAGE HIP)")
            elif relevant:
                # Conditions involving CUDA_FOUND, custom FindCUDA macros, source
                # GLOBs, custom commands and compiler flags cannot be ported blind.
                raise common.AdaptationError(f"Unsupported CUDA-dependent {name} command")

            # Only the language-standard family has equivalent bounded semantics.
            for token in tokens:
                replacement = token.value
                for old, new in VARIABLE_MAP.items():
                    replacement = replace_word(replacement, old, new)
                if replacement != token.value:
                    if name != "set":
                        raise common.AdaptationError("CUDA standard references outside a literal set need review")
                    local.append((token.start, token.end, _token_text(token, replacement)))
            if relevant and not local and replace_command is None and name not in {"project", "enable_language", "find_package"}:
                raise common.AdaptationError(f"Unresolved CUDA inputs in {name}; no partial target rewrite is allowed")
            if name == "set" and values and "CUDA" in values[0] and values[0] not in {"CMAKE_CUDA_ARCHITECTURES", *VARIABLE_MAP}:
                raise common.AdaptationError(f"Unsupported CUDA build variable {values[0]}")
            if name == "set_target_properties" and "HIP_ARCHITECTURES" in values:
                index = values.index("HIP_ARCHITECTURES") + 1
                if index >= len(values):
                    raise common.AdaptationError("Missing HIP_ARCHITECTURES value")
                for value in values[index].split(";"):
                    common.normalize_arch(value)
            if before or after or replace_command:
                indent = _indent(text, command)
                if replace_command:
                    local = [(command.start, command.end, replace_command.replace(newline, newline + indent))]
                if before:
                    local.append((command.start, command.start, (newline + indent).join(dict.fromkeys(before)) + newline + indent))
                if after:
                    local.append((command.end, command.end, newline + indent + (newline + indent).join(dict.fromkeys(after))))
            edits.extend(local)
        except common.AdaptationError as exc:
            diagnostics.append(f"BLOCK: {rel_path}:{command.line}: {exc}")
        if name in {"if", "function", "macro", "foreach", "while", "block"}:
            stack.append(name)
            discovery.append({})
        elif name in {"endif", "endfunction", "endmacro", "endforeach", "endwhile", "endblock"}:
            expected = name[3:]
            if not stack or stack[-1] != expected:
                diagnostics.append(f"BLOCK: {rel_path}:{command.line}: Unbalanced {name}")
            else:
                stack.pop()
                discovery.pop()
    if stack:
        diagnostics.append(f"BLOCK: {rel_path}: Unclosed CMake scope")
    for pattern, detail in REVIEW_PATTERNS:
        if pattern.search(text):
            diagnostics.append(f"REVIEW: {rel_path}: {detail}")
    if any(item.startswith("BLOCK:") for item in diagnostics):
        return text, diagnostics
    transformed = _patch(text, edits)
    try:
        transformed = insert_top_level_block(transformed, arch, set())
    except common.AdaptationError as exc:
        return text, diagnostics + [f"BLOCK: {rel_path}: {exc}"]
    return transformed, diagnostics


def transform_line(line: str, arch: str | None, cuda_root: str | None, dtk_root: str | None) -> str:
    """Legacy small-string entry; unsupported cases now refuse clearly."""
    transformed, diagnostics = transform_text(line, rel_path=Path("CMakeLists.txt"), arch=arch, cuda_root=cuda_root, dtk_root=dtk_root)
    if any(item.startswith("BLOCK:") for item in diagnostics):
        raise common.AdaptationError("\n".join(diagnostics))
    return transformed


def unified_diff(path: Path, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=str(path), tofile=str(path) + ".fastpt-dtk"))


def _prepared(root: Path) -> None:
    marker = common.workspace_path(root, common.OWNERSHIP_FILE)
    if not marker.is_file() or json.loads(common.read_text(marker)).get("route") != "transcode":
        raise common.AdaptationError("Generated outputs require a common.prepare_tree transcode workspace")


def write_build_script(out_dir: Path, arch: str | None) -> None:
    _prepared(out_dir)
    arch = common.normalize_arch(arch)
    if not arch:
        raise common.AdaptationError("An explicitly requested CMake build helper requires --arch; no architecture default exists")
    if not (out_dir / "CMakeLists.txt").is_file():
        raise common.AdaptationError("A CMake build helper requires a root CMakeLists.txt")
    lines = common.bash_build_prelude("build_fastpt_dtk")
    lines += ['cmake -S "$ROOT" -B "$BUILD_DIR" ' + shlex.quote("-DCMAKE_HIP_ARCHITECTURES=" + arch), 'cmake --build "$BUILD_DIR"', ""]
    script = common.safe_write_text(out_dir, SCRIPT_NAME, "\n".join(lines))
    common.workspace_path(out_dir, script).chmod(0o755)


def render_explicit_build_script(root: Path, command: list[str], default_dir: str) -> str:
    """An explicit build-only CMake argv; no shell expressions/configure guesses.

    hytlass uses this only after its integration has been configured separately.
    Restriction to cmake --build avoids pretending to contain arbitrary scripts'
    output paths. The actual build remains NOT RUN.
    """
    command = common.validate_build_command(command)
    if not command or len(command) < 3 or command[:2] != ["cmake", "--build"]:
        raise common.AdaptationError("Build helper supports explicit ['cmake', '--build', '<relative-build-dir>', ...] only; use the original build manually for other entries")
    directory = command[2]
    if Path(directory).is_absolute() or re.match(r"[A-Za-z]:", directory):
        raise common.AdaptationError("Build helper directory must be relative to the selected workspace")
    common.validate_build_dir(root, directory)
    # Do not let options overwrite --build or run install/native shell actions.
    for argument in command[3:]:
        if not re.fullmatch(r"[A-Za-z0-9_.:+-]+", argument) or argument in {"--", "--build", "--install", "-E", "-P"}:
            raise common.AdaptationError(f"Unsupported build-helper argument: {argument}")
    lines = common.bash_build_prelude(default_dir)
    # The declared directory, not an ambient BUILD_DIR, is used by this command.
    lines += ["BUILD_DIR=\"$ROOT\"/" + shlex.quote(directory), 'BUILD_DIR="$(realpath -m -- "$BUILD_DIR")"', 'case "$BUILD_DIR" in "$ROOT"/*) ;; *) printf "%s\\n" "Build directory escapes the workspace" >&2; exit 2 ;; esac', 'exec cmake --build "$BUILD_DIR"' + (" " + shlex.join(command[3:]) if command[3:] else ""), ""]
    return "\n".join(lines)


def write_report(report_path: Path, changed: list[Path], warnings: list[str], sources: list[str]) -> None:
    _prepared(report_path.parent)
    lines = ["# Bounded CMake preparation\n\n", "Preparation only; no build, load, GPU or correctness validation ran.\n\n", "## Changed files\n"]
    lines.extend(f"- `{path}`\n" for path in changed)
    lines.extend(["\n## Pending review\n", "- Verify installed HIP package targets, original numerical contract, actual converter outputs and target coverage.\n"])
    lines.extend(f"- {item}\n" for item in warnings)
    lines.extend(["\n## Mapping and scope\n", "- Source references consume an explicit verified mapping; .cu identity mappings are compiled as HIP.\n", "- Dependency discovery stays in its original file/conditional; build/dist inputs and undecodable bytes are retained.\n", "- No FastPT mode activation, environment sourcing or architecture guessing.\n"])
    lines.extend(f"- {item}\n" for item in sources)
    common.safe_write_text(report_path.parent, report_path.name, "".join(lines))


def prepare(repo: Path, out: Path | None = None, *, arch: str | None = None, source_map: dict[str, str] | None = None,
            write: bool = False, force: bool = False, emit_build_script: bool = False,
            cuda_root: str | None = None, dtk_root: str | None = None) -> dict:
    repo = common.resolve_repo(repo)
    if force:
        raise common.AdaptationError("--force is no longer supported; existing output is never overwritten")
    out, _ = common.prepare_destination(repo, out, default_suffix="_fastpt_cmake_dtk")
    mapping = validate_source_map(repo, source_map or {})
    arch = common.normalize_arch(arch)
    if emit_build_script and (not arch or not (repo / "CMakeLists.txt").is_file()):
        raise common.AdaptationError("--emit-build-script requires an explicit arch and a root CMakeLists.txt")
    analysis_identity = common.source_identity(repo)
    originals = {path.relative_to(repo): common.read_text(path) for path in cmake_files(repo)}
    transformed, diagnostics = {}, []
    existing_hip: set[str] = set()
    for rel, old in originals.items():
        try:
            existing_hip.update(hip_arch_targets(parse_commands(old)))
        except common.AdaptationError as exc:
            diagnostics.append(f"BLOCK: {rel}: {exc}")
    for rel, old in originals.items():
        new, warnings = transform_text(old, rel_path=rel, arch=arch, cuda_root=cuda_root, dtk_root=dtk_root, source_map=mapping, established_hip_targets=existing_hip)
        transformed[rel] = new
        diagnostics.extend(warnings)
    if not originals:
        diagnostics.append("BLOCK: No CMake files were found; this adapter cannot infer a build entry")
    blocked = any(item.startswith("BLOCK:") for item in diagnostics)
    changed = [rel for rel in transformed if originals[rel] != transformed[rel]]
    report = {"kind": "adaptation-preparation", "schema_version": common.SCHEMA_VERSION, "route": "transcode", "output": str(out), "status": "blocked" if blocked else "planned", "executed": False, "stages": {stage: "not-run" for stage in common.STAGES}, "source_map": mapping, "diagnostics": diagnostics, "changed": [p.as_posix() for p in changed], "diffs": [unified_diff(p, originals[p], transformed[p]) for p in changed]}
    if write and blocked:
        raise common.AdaptationError("CMake conversion blocked before creating output:\n" + "\n".join(diagnostics))
    if write:
        root = common.prepare_tree(repo, out, route="transcode", force=force, reserved_names=RESERVED_NAMES)
        common.validate_prepared_inputs(root, analysis_identity, originals)
        for rel in changed:
            common.safe_write_text(root, rel, transformed[rel], overwrite=True)
        if emit_build_script:
            write_build_script(root, arch)
        report["status"] = "prepared-not-built"
        report["executed"] = True
        report["stages"]["prepare"] = "passed"
        report["stages"]["transform"] = "passed"
        report["transform_scope"] = "bounded CMake edits only; source conversion and semantic equivalence are not validated"
        common.safe_write_json(root, MANIFEST_NAME, report)
        write_report(root / REPORT_NAME, changed, diagnostics, [])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", help="Copy destination for non-Git input; Git stays at --repo")
    parser.add_argument("--arch", help="Actual complete gfx target; no environment/default architecture is inferred")
    parser.add_argument("--source-map", help="JSON object mapping original relative paths to actual existing relative source files")
    parser.add_argument("--cuda-root", help="Legacy option; unsafe implicit mapping is refused")
    parser.add_argument("--dtk-root", help="Legacy option; unsafe implicit mapping is refused")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--force", action="store_true", help="Legacy option; refused, never deletes output")
    parser.add_argument("--emit-build-script", action="store_true", help="Explicitly emit, but never run, a bounded CMake helper")
    args = parser.parse_args()
    try:
        repo = common.resolve_repo(args.repo)
        mapping = json.loads(common.read_text(args.source_map)) if args.source_map else {}
        report = prepare(repo, Path(args.out) if args.out else None, arch=args.arch, source_map=mapping, write=args.write, force=args.force, emit_build_script=args.emit_build_script, cuda_root=args.cuda_root, dtk_root=args.dtk_root)
        # JSON escaping also preserves undecodable source bytes in dry-run diffs.
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 2 if report["status"] == "blocked" else 0
    except (common.AdaptationError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
