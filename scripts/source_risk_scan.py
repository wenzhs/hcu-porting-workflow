#!/usr/bin/env python3
"""Read-only inventory of CUDA/HIP porting risks in a project. It never rewrites code.

Every hit is a candidate location, not a verdict: version macros, warp/wave
intrinsics, inline asm, builtins, library names, device-side launches and the
GPUFusion (non-transcode) limits each need a target-specific check recorded in
the scope ledger. Heuristic text matching; macro expansion, generated code and
dependency directories (unless requested) are not covered.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from adaptation_common import AdaptationError, BUILD_NAMES, SCHEMA_VERSION, literal_relative_path, read_text, resolve_repo, safe_write_json, source_files, source_identity, validate_output_path, workspace_path

ALL = ("native-hip", "non-transcode", "transcode")
NT = ("non-transcode",)
CODE, BUILD, PYTHON = "code", "build", "python"
CODE_SUFFIXES = frozenset({".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx", ".hip", ".inl", ".cl"})
BUILD_SUFFIXES = frozenset({".cmake", ".mk", ".sh", ".toml", ".cfg", ".ini", ".yml", ".yaml"})
MAX_FILE_BYTES = 4 * 1024 * 1024

# Guidance uses stable explicit card anchors in the consolidated references.

LIB_NOTES = {
}

RULE_NOTES = {
    ("dynamic-parallelism", "rdc-flag"): "Relocatable device code is also used for plain separate compilation; it is only a Dynamic Parallel hint. Look for device-side launches.",
    ("arch-flag", "hip-arch"): "HYTLASS_HIPCC_ARCHS uses its own numeric form (for example 936); keep it apart from the standard complete gfx target.",
}


# Handling card per category in references/instruction-porting.md and references/code-conversion.md.
CATEGORY_CARDS = {
    "warp-size": "W-01", "lane-mask": "W-03", "barrier-divergence": "W-09", "cooperative-groups": "W-13", "memory-semantics": "W-08",
    "version-macro": "S-01", "platform-macro": "S-01", "inline-asm": "S-02", "builtin": "S-03", "cuda-library": "S-04",
    "runtime-feature": "S-05", "driver-api": "S-05", "texture": "S-05", "matrix-core": "S-06", "cutlass": "S-06", "low-precision-type": "S-06",
    "arch-flag": "S-07", "toolkit-path": "S-08", "hip-path-marker": "S-08", "source-dependency": "S-10",
    "dynamic-parallelism": "S-09", "static-runtime": "S-09", "mixed-hip-cuda": "S-09", "backend-dependency": "S-11",
}

def card_reference(card: str) -> str:
    document = "instruction-porting" if card.startswith("W-") else "code-conversion"
    return f"references/{document}.md#{card.lower()}"


GUIDANCE = {category: card_reference(card) for category, card in CATEGORY_CARDS.items()}
GUIDANCE["warp-sync"] = card_reference("W-04")

SUBTYPE_CARDS = {("warp-size", "numeric-near-warp"): "W-02", ("warp-size", "thread-index-arith"): "W-02", ("warp-size", "lane-arith"): "W-02"}


def card_for(category: str, subtype: str) -> str | None:
    if category == "warp-sync":
        if subtype.startswith("__shfl"):
            return "W-05"
        if subtype == "__activemask":
            return "W-07"
        if subtype == "__syncwarp":
            return "W-08"
        return "W-06"
    return SUBTYPE_CARDS.get((category, subtype), CATEGORY_CARDS.get(category))


def _rule(category: str, subtype: str | None, pattern: str, kinds: tuple[str, ...], routes: tuple[str, ...] = ALL, *, limit: bool = False, flags: int = 0) -> tuple:
    return (category, subtype, re.compile(pattern, flags), frozenset(kinds), routes, limit)


LINE_RULES = (
    _rule("version-macro", None, r"\b(CUDA_VERSION|CUDART_VERSION|__CUDACC_VER_MAJOR__|__CUDACC_VER_MINOR__|__CUDA_ARCH__|__CUDA_ARCH_LIST__)\b", (CODE,)),
    _rule("version-macro", None, r"\b(ROCM_VERSION|ROCM_VERSION_MAJOR|HIP_VERSION|HIP_VERSION_MAJOR|HIP_VERSION_MINOR|HIP_VERSION_PATCH|TORCH_HIP_VERSION)\b", (CODE,)),
    _rule("version-macro", None, r"\b(FPT_DTK_VERSION|FPT_DTKRT_VERSION|__FPT_DTK_ARCH__)\b", (CODE,)),
    _rule("platform-macro", None, r"\b(USE_ROCM|__HIP_PLATFORM_AMD__|__HIP_PLATFORM_HCC__|__HIP_PLATFORM_NVIDIA__|__HIPCC__|__HIP_DEVICE_COMPILE__|__NVCC__|__CUDACC__)\b", (CODE,)),
    _rule("warp-size", None, r"\b(warpSize|WARP_SIZE|kWarpSize|WARPSIZE|WARP_SZ|kWarpSz)\b", (CODE,)),
    _rule("warp-size", "numeric-near-warp", r"(?i:warp|lane)\w*[^;{}\n]*\b(?:32|31|0x1f)\b|\b(?:32|31)\b[^;{}\n]*(?i:warp|lane)", (CODE,)),
    _rule("warp-size", "thread-index-arith", r"\bthreadIdx\.x\s*(?:%|&|/|>>)\s*(?:32|31|5|0x1f)\b", (CODE,)),
    _rule("cooperative-groups", "tiled_partition", r"\btiled_partition\s*<\s*(\d+)\s*>", (CODE,)),
    _rule("cooperative-groups", "namespace", r"\bcooperative_groups\b", (CODE,)),
    _rule("builtin", None, r"\b(__builtin_(?:amdgcn|hcu|mmac|lingshu)\w*)", (CODE,)),
    _rule("builtin", "nvidia", r"\b(__nvvm_\w+)", (CODE,)),
    _rule("cuda-library", None, r"#\s*include\s*[<\"]((?:cublasLt|cublas\w*|cufftw|cufft\w*|cusparseLt|cusparse\w*|curand\w*|cusolver\w*|cudnn\w*|cutensor\w*|nccl\w*|nvtx\w*|nvToolsExt|cupti\w*|cub/\w+|thrust/\w+|nvrtc|nvjpeg|cudss)[^>\"]*)[>\"]", (CODE,)),
    _rule("driver-api", None, r"\b(cu(?:Init|ModuleLoad\w*|ModuleGetFunction|LaunchKernel|Ctx\w+|MemAlloc\w*|MemcpyHtoD\w*|MemcpyDtoH\w*|StreamCreate|DeviceGet\w*|LinkAddData))\s*\(", (CODE,)),
    _rule("texture", None, r"\b(texture\s*<|tex[123]D\w*\s*\(|cudaTextureObject_t|cudaCreateTextureObject|cudaBindTexture\w*|cudaResourceDesc)", (CODE,)),
    _rule("dynamic-parallelism", "device-launch-api", r"\b(cudaLaunchDevice|cudaStreamTailLaunch|cudaStreamFireAndForget)\b", (CODE,), NT),
    _rule("matrix-core", None, r"(nvcuda::wmma|\bwmma::|\bmma_sync\b|\bldmatrix\b|\bmma\.sync\b|\bcp\.async\b)", (CODE,)),
    _rule("cutlass", None, r"((?:cutlass|cute)::|#\s*include\s*[<\"](?:cutlass|cute)/)", (CODE,)),
    _rule("low-precision-type", None, r"\b(__nv_fp8\w*|__nv_fp4\w*|float8_e[45]m[23](?:fnuz|fn)?|float8e[45](?:b8|nv|b15)?|float4_e2m1fn_x2)\b", (CODE, PYTHON)),
    _rule("backend-dependency", None, r"\b(?:from|import)\s+(triton|flash_attn(?:_2_cuda)?|flydsl)\b", (PYTHON,)),
    _rule("runtime-feature", None, r"\b(cudaGraph\w*|cudaMallocAsync|cudaFreeAsync|cudaMemPool\w+|cudaStreamBeginCapture|cudaLaunchCooperativeKernel|cudaLaunchKernelEx|cudaFuncSetAttribute|cudaDeviceSetLimit|cudaMemAdvise|cudaMallocManaged|cudaIpc\w+)\b", (CODE,)),
    _rule("memory-semantics", None, r"\b(__threadfence_system|__nanosleep|__ldg|__ldcg|__ldcs|__stcg|__stcs|__stwt|cuda::atomic\w*|cuda::barrier|cuda::pipeline|cuda::std)\b", (CODE,)),
    # Build-system and Python build scripts.
    _rule("source-dependency", None, r"#\s*include\s*[<\"]((?:cuda/std/|cuda/(?:atomic|barrier|semaphore|latch|pipeline)\b|libcudacxx/|libhipcxx/|hip/std/|raft/|cuvs/|hipvs/|ck/|ck_tile/|rmm/)[^>\"]*)[>\"]", (CODE,)),
    _rule("source-dependency", None, r"\b(libcudacxx|libhipcxx|composable_kernel|ck_tile|hipvs|cuvs|raft|rmm|rapids[-_]cmake|rapids_cpm_\w+)\b", (BUILD, PYTHON), flags=re.I),
    _rule("arch-flag", "nvidia-arch", r"\b(CMAKE_CUDA_ARCHITECTURES|CUDA_ARCHITECTURES|TORCH_CUDA_ARCH_LIST|CUDA_NVCC_FLAGS|CMAKE_CUDA_FLAGS)\b|-arch[= ]\s*sm_\w+|-gencode\b|--generate-code\b|\b(?:compute|sm)_\d{2,3}[a-z]?\b", (BUILD, PYTHON)),
    _rule("arch-flag", "hip-arch", r"\b(CMAKE_HIP_ARCHITECTURES|AMDGPU_TARGETS|GPU_TARGETS|PYTORCH_ROCM_ARCH|HYTLASS_HIPCC_ARCHS|HIP_ARCHITECTURES)\b|--offload-arch\b", (BUILD, PYTHON)),
    _rule("static-runtime", None, r"\b(cudart_static|CUDA_USE_STATIC_CUDA_RUNTIME|culibos|cudadevrt)\b|CMAKE_CUDA_RUNTIME_LIBRARY[^\n]*Static|-cudart[= ]\s*static", (BUILD, PYTHON), NT),
    _rule("dynamic-parallelism", "rdc-flag", r"(-rdc\s*=\s*true|--relocatable-device-code|\bCUDA_SEPARABLE_COMPILATION\b)", (BUILD, PYTHON), NT),
    _rule("cuda-library", "link", r"\bCUDA::(\w+)|(?<![\w-])-l(cublasLt|cublas|cufftw|cufft|cusparse|curand|cusolver|cudnn|nccl|nvrtc|cupti|nvToolsExt)\b", (BUILD, PYTHON)),
    _rule("toolkit-path", None, r"\b(CUDA_HOME|CUDA_PATH|CUDA_TOOLKIT_ROOT_DIR|CUDAToolkit_ROOT)\b|(/usr/local/cuda)", (BUILD, PYTHON)),
    _rule("hip-path-marker", None, r"\b(hipify\w*|IS_HIP_EXTENSION|ROCM_HOME|HIP_HOME|\w+_WITH_HIP|USE_ROCM)\b", (BUILD, PYTHON)),
)

WARP_CALL = re.compile(r"\b(__(?:shfl(?:_up|_down|_xor)?|ballot|any|all|match_any|match_all)(?:_sync)?|__reduce_(?:add|min|max|and|or|xor)_sync|__activemask|__syncwarp)\b")
ASM_CALL = re.compile(r"\b(?:__asm__|__asm|asm)\s*(?:__volatile__|volatile)?\s*\(")
PTX_TEXT = re.compile(r"\b(?:mov|ld|st|cvt|mma|bar|red|atom|shfl|vote|prmt|lop3|cp\.async|fma|add|mul)\.[a-z0-9.]+|%laneid|%tid|%smid|\.reg\b|\.pred\b")
# Raw strings first so their bodies are kept verbatim and never parsed as comments.
C_TOKENS = re.compile(r"\bR\"([^()\\\s]{0,16})\(.*?\)\1\"|//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S)
PY_TOKENS = re.compile(r"\"\"\".*?\"\"\"|'''.*?'''|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|#[^\n]*", re.S)
# CMake bracket comments: #[[ ... ]] and #[==[ ... ]==].
BUILD_TOKENS = re.compile(r"#\[(=*)\[.*?\]\1\]|\"(?:\\.|[^\"\\\n])*\"|'[^'\n]*'|#[^\n]*", re.S)
# Context-gated rules: only applied in files that already use warp-level operations, to keep noise down.
BITCOUNT_32 = re.compile(r"\b(__popc|__ffs|__clz|__brev|__fns)\s*\(")
LANE_SHIFT = re.compile(r"(?<![\w.])1[uU]?\s*<<\s*\(?\s*([A-Za-z_][\w.]*)")
LANE_ARITH = re.compile(r"\b[A-Za-z_][\w.]*\s*(?:%\s*32|&\s*(?:31|0x1[fF]))\b")
WARP_CONTEXT = re.compile(r"\b(warpSize|WARP_SIZE|kWarpSize|tiled_partition|__lane_id|laneid)\b")
TYPE64 = re.compile(r"(?:\bunsigned\s+long(?:\s+long)?(?:\s+int)?|\buint64_t|\bint64_t|\blong\s+long|\blane_mask_t|\bsize_t)\s+\w+\s*=\s*$")
TYPE32 = re.compile(r"(?:\bunsigned(?:\s+int)?|\buint32_t|\bint32_t|\bint|\buint)\s+\w+\s*=\s*$")
AUTO_DECL = re.compile(r"\bauto\s+\w+\s*=\s*$")
CAST32 = re.compile(r"(?:\(\s*(?:unsigned(?:\s+int)?|uint32_t|int|uint)\s*\)|static_cast\s*<\s*(?:unsigned(?:\s+int)?|uint32_t|int|uint)\s*>\s*\()\s*$")
BARRIER = re.compile(r"\b(?:__syncthreads(?:_count|_and|_or)?|__syncwarp|__barrier_sync)\s*\(|\.sync\s*\(\s*\)|\bsync\s*\(\s*\w+\s*\)")
RETURN = re.compile(r"\breturn\b")
HIP_INCLUDE = re.compile(r"#\s*include\s*[<\"]hip/")
CUDA_INCLUDE = re.compile(r"#\s*include\s*[<\"](?:cuda_runtime\w*|cuda\.h|cuda_fp16\.h|cuda_bf16\.h|cuda/)")


def _blank(match: re.Match[str]) -> str:
    # Keep newlines so line numbers still refer to the original file.
    return re.sub(r"[^\n]", " ", match.group(0))


def strip_comments(text: str, kind: str) -> str:
    if kind == CODE:
        return C_TOKENS.sub(lambda m: _blank(m) if m.group(0).startswith(("//", "/*")) else m.group(0), text)
    tokens = PY_TOKENS if kind == PYTHON else BUILD_TOKENS
    return tokens.sub(lambda m: _blank(m) if m.group(0).startswith("#") else m.group(0), text)


def call_args(text: str, open_paren: int, limit: int = 4000) -> list[str] | None:
    """Split the top-level arguments of a call whose '(' is at open_paren; None if unbalanced."""
    depth, start, args = 0, open_paren + 1, []
    for index in range(open_paren, min(len(text), open_paren + limit)):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                args.append(text[start:index].strip())
                return [] if args == [""] else args
        elif char == "," and depth == 1:
            args.append(text[start:index].strip())
            start = index + 1
    return None


def result_width(text: str, call_start: int) -> str:
    """Classify the declared type that receives a call result in the same statement."""
    begin = max(text.rfind(c, 0, call_start) for c in ";{}")
    prefix = text[begin + 1: call_start]
    if CAST32.search(prefix):
        return "cast-32bit"
    if TYPE64.search(prefix):
        return "64bit-declaration"
    if TYPE32.search(prefix):
        return "32bit-declaration"
    if AUTO_DECL.search(prefix):
        return "auto"
    return "other"


def file_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in CODE_SUFFIXES:
        return CODE
    if suffix == ".py":
        return PYTHON
    if path.name in BUILD_NAMES or suffix in BUILD_SUFFIXES or path.name == "Dockerfile":
        return BUILD
    return None


def classify_mask(argument: str) -> str:
    value = re.sub(r"\s+", "", argument).lower()
    if re.fullmatch(r"0xf{16}(?:ull|ul|llu|lu|ll|l|u)?", value) or value in {"~0ull", "~0ul", "-1ll", "-1ull"}:
        return "full-64bit-literal"
    if re.fullmatch(r"0xf{8}u?", value) or value in {"~0u", "~0"}:
        return "full-32bit-literal"
    if value in {"-1", "(unsigned)-1"}:
        return "all-ones-int"
    return "expression" if value else "missing"


def _brace_end(text: str, start: int) -> int:
    depth = 0
    for index in range(start, min(len(text), start + 400_000)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _scan_file(relative: str, kind: str, original: str, route: str | None) -> list[dict[str, Any]]:
    text = strip_comments(original, kind)
    original_lines = original.split("\n")
    found: dict[tuple, dict[str, Any]] = {}

    def add(category: str, subtype: str, line: int, routes: tuple[str, ...], *, note: str | None = None, extra: dict[str, Any] | None = None) -> None:
        if route and route not in routes:
            return
        key = (category, subtype, line)
        if key in found:
            return
        entry: dict[str, Any] = {"category": category, "subtype": subtype, "file": relative, "line": line, "text": original_lines[line - 1].strip()[:200], "routes": list(routes)}
        card = card_for(category, subtype)
        if card:
            entry["card"] = card
        if note:
            entry["note"] = note
        if extra:
            entry["extra"] = extra
        found[key] = entry

    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        for category, subtype, pattern, kinds, routes, limit in LINE_RULES:
            if kind not in kinds:
                continue
            match = pattern.search(line)
            if not match:
                continue
            label = subtype or next((g for g in match.groups() if g), match.group(0))
            note = next((text_ for key, text_ in LIB_NOTES.items() if key in str(label).lower()), None) if category == "cuda-library" else None
            note = note or RULE_NOTES.get((category, str(label)))
            add(category, str(label), number, routes, note=note)
    if kind != CODE:
        return list(found.values())

    warp_calls = list(WARP_CALL.finditer(text))
    for match in warp_calls:
        name = match.group(1)
        form = "mask-query" if name in {"__activemask", "__syncwarp"} else ("sync" if name.endswith("_sync") else "legacy")
        extra: dict[str, Any] = {"form": form}
        paren = re.match(r"\s*\(", text[match.end():])
        args = call_args(text, match.end() + paren.end() - 1) if paren else None
        if form == "sync":
            extra["mask"] = classify_mask(args[0] if args else "")
            if extra["mask"] == "full-32bit-literal":
                extra["note"] = "32-bit full mask: confirm mask width and participating lanes for the target wave size; do not rewrite mechanically."
        if name.startswith("__shfl") and args is not None:
            width_index = 3 if form == "sync" else 2
            width = args[width_index] if len(args) > width_index else None
            extra["width"] = re.sub(r"\s+", "", width) if width else "default"
            if width is None and ("_up" in name or "_down" in name):
                extra["width_note"] = "Default width is the physical wave (W); an up/down shift that assumed 32-lane groups crosses into the other half on wave64."
        if name.startswith(("__ballot", "__activemask")):
            extra["result"] = result_width(text, match.start())
            if extra["result"] in {"32bit-declaration", "cast-32bit"}:
                add("lane-mask", "ballot-to-32bit", _line_of(text, match.start()), ALL,
                    note="Lane mask result stored in 32 bits: on wave64 the upper lanes are lost unless only a 32-lane group's bit range is intended.")
        add("warp-sync", name, _line_of(text, match.start()), ALL, extra=extra)

    # Rules that are only meaningful next to warp-level code.
    if warp_calls or WARP_CONTEXT.search(text):
        for match in BITCOUNT_32.finditer(text):
            add("lane-mask", "bitcount-32bit", _line_of(text, match.start()), ALL,
                note=f"{match.group(1)} takes 32 bits; a 64-lane mask needs the ll variant or an explicit group bit range.")
        for match in LANE_SHIFT.finditer(text):
            add("lane-mask", "lane-shift", _line_of(text, match.start()), ALL,
                note="32-bit shift by a variable next to warp code: if it is a lane index, lanes 32..63 overflow on wave64.",
                extra={"shift_by": match.group(1)})
        warp_size_lines = {key[2] for key in found if key[0] == "warp-size"}
        for match in LANE_ARITH.finditer(text):
            line = _line_of(text, match.start())
            if line not in warp_size_lines:
                add("warp-size", "lane-arith", line, ALL,
                    note="%32 / &31 next to warp code: preserve format/API width; for HCU scheduling prefer an equivalent whole-wave mapping, otherwise phase the contract-bound logical groups.")
    for match in ASM_CALL.finditer(text):
        end = text.find(";", match.end(), match.end() + 4000)
        snippet = text[match.start(): end if end != -1 else match.end() + 400]
        is_ptx = bool(PTX_TEXT.search(snippet))
        add("inline-asm", "ptx" if is_ptx else "asm", _line_of(text, match.start()), ALL,
            note="PTX cannot be translated by hipify; non-transcode needs the explicit PTX switch and a feature check (DTK26.10 manual)." if is_ptx else None)
    for match in re.finditer(r"\b__global__\b", text):
        cursor, depth_paren = match.end(), 0
        while cursor < len(text) and not (depth_paren == 0 and text[cursor] in "{;"):
            depth_paren += (text[cursor] == "(") - (text[cursor] == ")")
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            continue
        end = _brace_end(text, cursor)
        body = text[cursor: end if end != -1 else len(text)]
        for launch in re.finditer(r"<<<", body):
            add("dynamic-parallelism", "device-side-launch", _line_of(text, cursor + launch.start()), NT,
                note="Kernel launch inside a __global__ body (heuristic): GPUFusion does not support Dynamic Parallel.")
        barriers = [b.start() for b in BARRIER.finditer(body)]
        for ret in RETURN.finditer(body):
            later = [b for b in barriers if b > ret.start()]
            if later:
                add("barrier-divergence", "return-before-barrier", _line_of(text, cursor + ret.start()), ALL,
                    note="Early return before a barrier in the same kernel (heuristic, ignores nested lambdas and called functions).",
                    extra={"barrier_line": _line_of(text, cursor + later[0])})
    if HIP_INCLUDE.search(text) and (cuda_include := CUDA_INCLUDE.search(text)):
        add("mixed-hip-cuda", "same-file-includes", _line_of(text, cuda_include.start()), NT,
            note="Both HIP and CUDA headers are included; GPUFusion does not support mixing. Confirm the branches are mutually exclusive.")
    return list(found.values())


def scan_risks(repo: str | Path, target: str | None = None, *, route: str | None = None, include_dependencies: bool = False, exclude_dirs: list[str] | None = None, max_findings: int = 5000, files_from: list[str] | None = None) -> dict[str, Any]:
    if route is not None and route not in ALL:
        raise AdaptationError(f"Unknown route: {route}")
    root = resolve_repo(repo)
    base = root
    if target:
        if Path(target).is_absolute() or ".." in Path(target).parts:
            raise AdaptationError("target must be a project-relative directory")
        base = (root / target).resolve()
        if not base.is_dir() or root not in (base, *base.parents):
            raise AdaptationError("target must name a project-relative directory")
    # Resolve excluded directories to absolute paths inside the root.
    excluded: set[Path] = set()
    for raw in (exclude_dirs or []):
        if Path(raw).is_absolute() or ".." in Path(raw).parts:
            raise AdaptationError(f"exclude_dir must be a project-relative path: {raw!r}")
        candidate = (root / raw).resolve()
        if root not in candidate.parents and candidate != root:
            raise AdaptationError(f"exclude_dir escapes the project root: {raw!r}")
        excluded.add(candidate)
    findings: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    scanned = 0
    paths = source_files(base, include_dependencies=include_dependencies)
    if files_from is not None:
        if not isinstance(files_from, list) or not files_from:
            raise AdaptationError("files_from must be a nonempty JSON list of project-relative files")
        chosen = []
        for item in files_from:
            path = workspace_path(root, literal_relative_path(item))
            if not path.is_file() or not path.is_relative_to(base):
                raise AdaptationError(f"Selected file is missing or outside target: {item}")
            chosen.append(path)
        paths = sorted(set(chosen))
    for path in paths:
        # Skip files inside explicitly excluded directories (e.g. build/, hipify output).
        if excluded and any(path.resolve() == ex or ex in path.resolve().parents for ex in excluded):
            skipped.append({"file": path.relative_to(root).as_posix(), "reason": "excluded directory"})
            continue
        kind = file_kind(path)
        if kind is None:
            continue
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_FILE_BYTES:
            skipped.append({"file": relative, "reason": "larger than 4 MiB"})
            continue
        try:
            text = read_text(path)
        except AdaptationError as error:
            skipped.append({"file": relative, "reason": str(error)})
            continue
        scanned += 1
        findings.extend(_scan_file(relative, kind, text, route))
    findings.sort(key=lambda f: (f["file"], f["line"] or 0, f["category"]))
    # Summaries cover every finding; only the detailed list is capped.
    by_category: dict[str, int] = {}
    by_card: dict[str, int] = {}
    for item in findings:
        by_category[item["category"]] = by_category.get(item["category"], 0) + 1
        if item.get("card"):
            by_card[item["card"]] = by_card.get(item["card"], 0) + 1
    groups: dict[tuple, dict] = {}
    for item in findings:
        key = (item["category"], item["subtype"], item.get("card"))
        group = groups.setdefault(key, {"category": key[0], "subtype": key[1], "card": key[2],
                                        "count": 0, "files": set(), "examples": [], "status": "unreviewed"})
        group["count"] += 1
        group["files"].add(item["file"])
        if len(group["examples"]) < 3:
            group["examples"].append({"file": item["file"], "line": item["line"]})
    grouped = [{**g, "files": sorted(g["files"])} for g in groups.values()]
    total_findings = len(findings)
    files_with_findings = len({f["file"] for f in findings})
    truncated = total_findings > max_findings
    findings = findings[:max_findings]
    return {
        "schema_version": SCHEMA_VERSION, "kind": "risk-scan", "analysis": "static-heuristic", "rewrites": False,
        "repo": str(root), "target": target, "route_filter": route, "source_sha256": source_identity(root)["tree_sha256"],
        "files_scanned": scanned, "files_with_findings": files_with_findings,
        "scope": {"files_from": files_from, "exclude_dirs": exclude_dirs or [], "identity": "whole-repo-tree; selection does not narrow identity"},
        "groups": grouped, "skipped": skipped, "truncated": truncated, "total_findings": total_findings,
        "coverage": "Line/function text heuristics over source and build files; no macro expansion, no generated or dependency code unless requested. No hits is not proof of absence.",
        "cards": "Each finding.card points to a handling card in references/instruction-porting.md or references/code-conversion.md; guidance provides the category link.",
        "summary": {"by_category": dict(sorted(by_category.items())), "by_card": dict(sorted(by_card.items()))},
        "guidance": {category: GUIDANCE[category] for category in sorted(by_category)},
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--target", help="Project-relative directory to scan")
    parser.add_argument("--route", choices=ALL, help="Only report hits relevant to this route")
    parser.add_argument("--include-dependencies", action="store_true", help="Also scan vendor/third_party style directories")
    parser.add_argument("--exclude-dir", action="append", default=[], metavar="REL_PATH", dest="exclude_dirs", help="Project-relative directory to exclude from scanning (e.g. build, hipify_output); may be repeated")
    parser.add_argument("--max-findings", type=int, default=5000)
    parser.add_argument("--json-out", help="Write a NEW report outside the protected source tree")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--files-from", help="JSON array of exact project-relative files")
    selection.add_argument("--transform-map", help="JSON source-to-generated map; scope is relative to --repo")
    parser.add_argument("--map-side", choices=("source", "generated"), default="source")
    parser.add_argument("--summary", action="store_true", help="Print grouped summary; --json-out saves details subject to --max-findings / truncated")
    args = parser.parse_args(argv)
    try:
        selected = json.loads(read_text(args.files_from)) if args.files_from else None
        if args.files_from and not isinstance(selected, list):
            raise AdaptationError("--files-from must contain a nonempty JSON array")
        if args.map_side != "source" and not args.transform_map:
            raise AdaptationError("--map-side requires --transform-map")
        if args.max_findings < 1:
            raise AdaptationError("--max-findings must be positive")
        if args.transform_map:
            mapping = json.loads(read_text(args.transform_map))
            if not isinstance(mapping, dict) or not mapping:
                raise AdaptationError("Transform map must be a nonempty object")
            for original, generated in mapping.items():
                literal_relative_path(original)
                literal_relative_path(generated)
            selected = list(mapping if args.map_side == "source" else mapping.values())
        result = scan_risks(args.repo, args.target, route=args.route, include_dependencies=args.include_dependencies,
                            exclude_dirs=args.exclude_dirs, max_findings=max(1, args.max_findings), files_from=selected)
        if args.json_out:
            _, destination = validate_output_path(args.repo, args.json_out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            safe_write_json(destination.parent, destination.name, result)
        if args.summary:
            skipped_reasons: dict[str, int] = {}
            for item in result["skipped"]:
                reason = item["reason"]
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            result = {key: value for key, value in result.items() if key not in {"findings", "skipped"}}
            result["groups"] = [{**{k: v for k, v in group.items() if k != "files"},
                                 "file_count": len(group["files"])} for group in result["groups"]]
            if result["scope"]["files_from"] is not None:
                result["scope"]["selected_file_count"] = len(set(result["scope"].pop("files_from")))
                result["scope"]["selection_file"] = args.files_from or args.transform_map
            result["skipped_by_reason"] = skipped_reasons
            result["detail_report"] = args.json_out
            result["detail_note"] = "Groups remain unreviewed. Detailed findings may be capped: check truncated, then raise --max-findings or narrow scope."
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"kind": "risk-scan-error", "error": str(exc), "rewrites": False}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
