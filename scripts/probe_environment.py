#!/usr/bin/env python3
"""Inspect the CURRENT shell/container for HCU/DTK adaptation evidence.

Read-only by default; explicit compile probes/evidence output create local files.

Run it inside the target environment (after the project's normal DTK activation,
with the Python that owns Torch). It records versions, paths, devices and macro
facts with the command and exit code behind each, and emits a ``capabilities``
fragment for route_preflight. It never imports Torch, installs anything or
touches the project. Values the probe cannot establish stay null (unknown)
rather than false.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from adaptation_common import AdaptationError, SCHEMA_VERSION, file_sha256, normalize_arch, parse_predefined_macros, reject_links

SOURCE = "probe_environment.py"
TOOLS = ("hipcc", "hipconfig", "rocminfo", "rocm_agent_enumerator", "hy-smi", "cmake", "ninja", "make", "nvcc")
ENV_NAMES = ("ROCM_PATH", "HIP_PATH", "DTK_ROOT", "CUDA_HOME", "CUDA_PATH", "CMAKE_PREFIX_PATH", "LD_LIBRARY_PATH", "HSA_TOOLS_LIB", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
ENV_PREFIXES = ("HIP_DISABLE", "HIP_ENABLE", "DTK")
PACKAGE_PATTERN = re.compile(r"^(torch|torchvision|torchaudio|triton|numpy|ninja|cmake|scikit-build\w*|setuptools|hip\w*|dtk\w*)$", re.I)
MACROS = ("USE_ROCM", "FPT_DTK_VERSION", "FPT_DTKRT_VERSION", "__FPT_DTK_ARCH__", "__HIP_DEVICE_COMPILE__", "ROCM_VERSION", "HIP_VERSION", "__HIP_PLATFORM_AMD__", "HIP_DISABLE_WARP_SYNC_BUILTINS", "HIP_ENABLE_WARP_SYNC_BUILTINS", "__AMDGCN_WAVEFRONT_SIZE__")
WARP_PROBES = (
    ("warp-sync-64bit-mask", "__shfl_sync(0xffffffffffffffffULL, v, 0)"),
    ("warp-sync-32bit-mask", "__shfl_sync(0xffffffffu, v, 0)"),
    ("ballot-sync-64bit-mask", "(float)__ballot_sync(0xffffffffffffffffULL, v > 0.f)"),
)
KERNEL = "#include <hip/hip_runtime.h>\n__global__ void k(float* o) {{ float v = o[threadIdx.x]; v = {expr}; o[threadIdx.x] = v; }}\n"
MAX_EXCERPT = 1500

Runner = Callable[[list[str], float], dict[str, Any]]


def run_command(argv: list[str], timeout: float = 20) -> dict[str, Any]:
    """Run one argv without a shell. Failures are data, not exceptions."""
    record: dict[str, Any] = {"argv": argv, "exit_code": None, "output": "", "error": None,
                              "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              "_full": "", "_stderr": "", "timed_out": False}
    try:
        done = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout, check=False)
        record.update(exit_code=done.returncode, _full=done.stdout, _stderr=done.stderr)
    except FileNotFoundError:
        record["error"] = "not-found"
    except subprocess.TimeoutExpired as exc:
        record.update(error=f"timeout after {timeout}s", timed_out=True)
        for key, data in (("_full", exc.stdout), ("_stderr", exc.stderr)):
            record[key] = data.decode(errors="replace") if isinstance(data, bytes) else (data or "")
    except OSError as exc:
        record["error"] = str(exc)
    record["output"] = (record["_full"] + record["_stderr"])[:MAX_EXCERPT]
    record["ended"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return record


def parse_rocminfo(text: str) -> list[dict[str, Any]]:
    agents = []
    for block in re.split(r"(?m)^[ \t]*Agent[ \t]+\d+[ \t]*$", text)[1:]:
        name = re.search(r"(?m)^\s*Name:\s+(gfx[0-9a-f]+)", block)
        if name:
            wave = re.search(r"(?m)^\s*Wavefront Size:\s+(\d+)", block)
            agents.append({"gfx": name.group(1), "wavefront_size": int(wave.group(1)) if wave else None})
    return agents


def parse_launch_limits(text: str) -> dict[str, Any]:
    """Extract per-agent workgroup/thread launch limits from rocminfo output."""
    limits: dict[str, Any] = {}
    for block in re.split(r"(?m)^[ \t]*Agent[ \t]+\d+[ \t]*$", text)[1:]:
        name = re.search(r"(?m)^\s*Name:\s+(gfx[0-9a-f]+)", block)
        if not name:
            continue
        gfx = name.group(1)
        max_wg = re.search(r"(?m)^[ \t]*(?:Workgroup Max Size|Max Workgroup Size):[ \t]+(\d+)", block)
        max_cu = re.search(r"(?m)^[ \t]*(?:Compute Unit|Max Compute Units):[ \t]+(\d+)", block)
        max_waves = re.search(r"(?m)^[ \t]*Max Waves Per CU:[ \t]+(\d+)", block)
        if max_wg or max_cu or max_waves:
            limits[gfx] = {
                "max_workgroup_size": int(max_wg.group(1)) if max_wg else None,
                "max_compute_units": int(max_cu.group(1)) if max_cu else None,
                "max_waves_per_cu": int(max_waves.group(1)) if max_waves else None,
            }
    return limits


def inspect_torch_files(root: Path) -> dict[str, Any]:
    """Parse files shipped with the installed Torch; the package is not imported."""
    result: dict[str, Any] = {"dtk_macros": None, "version_py": None, "hipify_injects_dtk_macros": None, "hipify_version": None, "hcu_version": None}
    header = root / "torch/include/ATen/dtk_macros.h"
    if header.is_file():
        data = header.read_bytes()
        text = data.decode("utf-8", errors="replace")
        result["dtk_macros"] = {
            "path": str(header), "sha256": hashlib.sha256(data).hexdigest(),
            "defines": dict(re.findall(r"#\s*define\s+(FPT_DTK_VERSION|FPT_DTKRT_VERSION|__FPT_DTK_ARCH__)\s+(\d+)", text)),
            "guarded_by_use_rocm": bool(re.search(r"#\s*if\s+defined\s*\(\s*USE_ROCM\s*\)|#\s*ifdef\s+USE_ROCM", text)),
            "note": "Compatibility constants, not the installed DTK/HIP/gfx version.",
        }
    version_py = root / "torch/version.py"
    if version_py.is_file():
        text = version_py.read_text(encoding="utf-8", errors="replace")
        values: dict[str, Any] = {}
        try:
            statements = ast.parse(text).body
        except SyntaxError:
            statements = []
        for statement in statements:
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {"__version__", "hip", "hcu", "cuda"}:
                    try:
                        value = ast.literal_eval(statement.value)
                    except (ValueError, TypeError, SyntaxError):
                        continue
                    if value is None or isinstance(value, str):
                        values[target.id] = value
        result["hcu_version"] = values.get("hcu")
        result["version_py"] = {"path": str(version_py), **values}
    hipify = root / "torch/utils/hipify/hipify_python.py"
    if hipify.is_file():
        result["hipify_injects_dtk_macros"] = "dtk_macros.h" in hipify.read_text(encoding="utf-8", errors="replace")
    hipify_version = root / "torch/utils/hipify/version.py"
    if hipify_version.is_file():
        found = re.search(r"(?m)^__version__\s*=\s*['\"]([^'\"]+)['\"]", hipify_version.read_text(encoding="utf-8", errors="replace"))
        result["hipify_version"] = found.group(1) if found else None
    return result


def probe(*, arch: str | None = None, compile_probe: bool = False, timeout: float = 20,
          runner: Runner = run_command, environ: dict[str, str] | None = None, which: Callable[[str], str | None] = shutil.which,
          distributions: Callable[[], Any] = importlib.metadata.distributions,
          evidence_dir: str | Path | None = None) -> dict[str, Any]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise AdaptationError("timeout must be finite and positive")
    env = dict(os.environ if environ is None else environ)
    commands: list[dict[str, Any]] = []
    warnings: list[str] = []
    evidence = reject_links(evidence_dir) if evidence_dir else None
    if evidence:
        if evidence.exists():
            raise AdaptationError("Evidence directory already exists; choose a new one")
        evidence.mkdir(parents=True)

    def run(argv: list[str]) -> dict[str, Any]:
        record = dict(runner(argv, timeout))
        complete = "_full" in record and "_stderr" in record
        full, stderr = record.pop("_full", ""), record.pop("_stderr", "")
        if evidence:
            logs = []
            for label, text in (("stdout", full), ("stderr", stderr)):
                path = evidence / f"{len(commands) + 1:04d}.{label}.log"
                path.write_text(text, encoding="utf-8")
                logs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
            record.update(logs=logs, complete_streams=complete)
        commands.append(record)
        record["_text"] = full
        return record

    tools = {name: which(name) for name in TOOLS}
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "kind": "environment-probe",
        "read_only": not (compile_probe or evidence), "source": SOURCE,
        "platform": {"system": platform.system(), "machine": platform.machine(), "python": sys.version.split()[0], "python_executable": sys.executable},
        "env": {name: value for name, value in sorted(env.items()) if name in ENV_NAMES or name.startswith(ENV_PREFIXES)},
        "tools": tools,
    }
    if tools["nvcc"]:
        warnings.append("nvcc is on PATH: record whether it is a real CUDA toolkit or a compatibility layer before judging the environment native.")

    hip: dict[str, Any] = {"hipcc_version": None, "hipconfig_version": None}
    if tools["hipcc"]:
        record = run([tools["hipcc"], "--version"])
        hip["hipcc_version"] = next((line.strip() for line in record["_text"].splitlines() if line.strip()), None) if record["exit_code"] == 0 else None
    if tools["hipconfig"]:
        record = run([tools["hipconfig"], "--version"])
        hip["hipconfig_version"] = record["_text"].strip() or None if record["exit_code"] == 0 else None
    result["hip"] = hip
    if not tools["hipcc"]:
        warnings.append("hipcc not found on PATH: activate the DTK environment the project normally uses (for example source /opt/dtk/env.sh) and rerun; absence here is unknown, not proof of absence.")

    agents: list[dict[str, Any]] = []
    launch_limits: dict[str, Any] = {}
    if tools["rocminfo"]:
        record = run([tools["rocminfo"]])
        if record["exit_code"] == 0:
            agents = parse_rocminfo(record["_text"])
            launch_limits = parse_launch_limits(record["_text"])
    gfx = sorted({a["gfx"] for a in agents})
    result["devices"] = {"agents": agents, "gfx": gfx, "wavefront_sizes": sorted({a["wavefront_size"] for a in agents if a["wavefront_size"]}), "launch_limits": launch_limits, "source": "rocminfo" if agents else None}

    packages = []
    torch_root: Path | None = None
    for dist in distributions():
        name = (dist.metadata["Name"] or "").strip()
        if not PACKAGE_PATTERN.match(name):
            continue
        location = Path(str(dist.locate_file("")))
        packages.append({"name": name, "version": dist.version, "location": str(location)})
        if name.lower() == "torch":
            torch_root = location
    result["python_packages"] = sorted(packages, key=lambda p: p["name"].lower())
    torch_files = inspect_torch_files(torch_root) if torch_root else {"dtk_macros": None, "version_py": None, "hipify_injects_dtk_macros": None, "hipify_version": None, "hcu_version": None}
    torch_version = next((p["version"] for p in packages if p["name"].lower() == "torch"), None)
    result["torch"] = {"version": torch_version, **torch_files}

    if compile_probe:
        probes, macros = [], {}
        selected = normalize_arch(arch) if arch else (gfx[0] if len(gfx) == 1 else None)
        if not tools["hipcc"] or not selected:
            warnings.append("--compile-probe needs hipcc and a single complete gfx target (use --arch); skipped.")
        else:
            scratch_context = nullcontext(str(evidence)) if evidence else tempfile.TemporaryDirectory(prefix="hcu-probe-")
            with scratch_context as scratch:
                for name, expr in WARP_PROBES:
                    source = Path(scratch) / f"{name}.cpp"
                    source.write_text(KERNEL.format(expr=expr), encoding="utf-8")
                    artifact = Path(scratch) / f"{name}.o"
                    record = run([tools["hipcc"], "-c", f"--offload-arch={selected}", "-o", str(artifact), str(source)])
                    produced = artifact.is_file() and artifact.stat().st_size > 0
                    probes.append({"name": name, "arch": selected, "argv": record["argv"], "exit_code": record["exit_code"],
                                   "passed": record["exit_code"] == 0 and produced and not record.get("error") and not record.get("timed_out"),
                                   "output": record["output"],
                                   "source_sha256": file_sha256(source), "source_path": str(source) if evidence else None,
                                   "artifact_sha256": file_sha256(artifact) if produced else None,
                                   "artifact_path": str(artifact) if evidence and produced else None,
                                   "artifact_nonempty": produced, "evidence_persisted": bool(evidence), "gpu_validated": False})
                source = Path(scratch) / "macros.cpp"
                source.write_text(KERNEL.format(expr="v"), encoding="utf-8")
                record = run([tools["hipcc"], "-dM", "-E", f"--offload-arch={selected}", "-x", "hip", str(source)])
                if record["exit_code"] == 0:
                    defined = parse_predefined_macros(record["_text"])
                    result["macro_context"] = "compiler-default-dump; may combine passes, not a host/device comparison"
                    macros = {name: defined.get(name) for name in MACROS if name in defined}
                    result["macros_checked"] = list(MACROS)
        result["compile_probes"] = probes
        result["macros_defined"] = macros

    result["capabilities"] = {
        "native_backend": {"state": None, "reason": None, "source": SOURCE},
        "hip": {"installed": True if tools["hipcc"] else None, "compiler": tools["hipcc"], "version": hip["hipcc_version"], "source": SOURCE},
        "hcu": {"version": torch_files.get("hcu_version"), "source": torch_files.get("version_py", {}).get("path") if isinstance(torch_files.get("version_py"), dict) else None},
    }
    result["devices"]["kernel_launch_limits"] = None
    result["devices"]["launch_limits_note"] = "Hardware limits only; inspect the actual kernel metadata and launch argv separately."
    result["evidence_dir"] = str(evidence) if evidence else None
    result["arch_suggestion"] = gfx[0] if len(gfx) == 1 else None
    if len(gfx) > 1:
        warnings.append("Several gfx targets are visible; choose the target explicitly.")
    result["commands"] = [{key: value for key, value in c.items() if key != "_text"} for c in commands]
    result["warnings"] = warnings
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", help="Complete gfx target for --compile-probe (default: the single visible device)")
    parser.add_argument("--compile-probe", action="store_true", help="Compile tiny kernels in a temp directory and list predefined macros")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--out", help="Write the report to a NEW file")
    parser.add_argument("--evidence-dir", help="NEW directory preserving complete command logs and compile inputs/outputs")
    args = parser.parse_args(argv)
    try:
        result = probe(arch=args.arch, compile_probe=args.compile_probe, timeout=args.timeout, evidence_dir=args.evidence_dir)
        text = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
        if args.out:
            destination = reject_links(args.out)
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(text)
        sys.stdout.write(text)
        return 0
    except (OSError, ValueError, AdaptationError) as exc:
        print(json.dumps({"kind": "environment-probe-error", "error": str(exc),
                          "read_only": not (args.compile_probe or args.evidence_dir)}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
