#!/usr/bin/env python3
"""Record one explicitly authorized argv in the current environment.

No SSH, shell activation, installation, retries or inferred build commands.
The caller supplies an executable argv; this recorder grants no authorization.
It records stdout/stderr and the actual child exit status for foreground argv.
If cleanup cannot establish quiescence, evidence is explicitly unfinalized with
no stable log hashes or artifact verdict. Detached work requires caller checks.
A record is execution evidence, not proof of a correct backend or GPU numerics.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptation_common import AdaptationError, file_sha256, reject_links, resolve_repo

ENV_KEYS = ("PATH", "PYTHONPATH", "ROCM_PATH", "HIP_PATH", "DTK_ROOT", "LD_LIBRARY_PATH",
            "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
            "HSA_DISABLE_CACHE", "CMAKE_PREFIX_PATH")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def record_command(argv: list[str], *, cwd: str | Path, out_dir: str | Path,
                   timeout: float = 120, expected_artifacts: list[str] | None = None) -> dict:
    if not argv or any(not isinstance(a, str) or not a or "\0" in a for a in argv):
        raise AdaptationError("An explicit nonempty executable argv is required")
    if not math.isfinite(timeout) or timeout <= 0:
        raise AdaptationError("timeout must be finite and positive")
    work = resolve_repo(cwd)
    output = reject_links(out_dir)
    if output.exists():
        raise AdaptationError("Evidence directory already exists; choose a new directory")
    artifacts = [reject_links(work / p) for p in (expected_artifacts or [])]
    # Recorder-owned files cannot prove that the child produced a build artifact.
    if any(p.is_relative_to(output) or output.is_relative_to(p) for p in artifacts):
        raise AdaptationError("Expected artifacts must not overlap the recorder evidence directory")
    if len(set(artifacts)) != len(artifacts):
        raise AdaptationError("Expected artifact paths must be unique")
    # A pre-existing artifact must not turn rc0-without-output into success.
    if any(p.exists() for p in artifacts):
        raise AdaptationError("Expected artifacts must be new; an existing artifact is not fresh output")
    output.mkdir(parents=True)
    env = dict(os.environ)
    result = {"kind": "command-record", "argv": list(argv), "cwd": str(work),
              "env": {k: env.get(k) for k in ENV_KEYS}, "started": now(),
              "exit_code": None, "timed_out": False, "interrupted": False, "error": None,
              "executed": False,
              "stdout": str(output / "stdout.log"), "stderr": str(output / "stderr.log"),
              "gpu_validated": False}
    start = time.monotonic()
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        try:
            child = subprocess.Popen(argv, cwd=work, env=env, stdout=stdout, stderr=stderr,
                                     start_new_session=(os.name == "posix"))
            result.update(pid=child.pid, executed=True)
            try:
                result["exit_code"] = child.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                result["timed_out"] = isinstance(exc, subprocess.TimeoutExpired)
                result["interrupted"] = isinstance(exc, KeyboardInterrupt)
                if os.name == "posix":
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    child.kill()
                result["exit_code"] = child.wait()
                cleanup = "owned-process-group" if os.name == "posix" else "direct-child-only"
                result["timeout_cleanup" if result["timed_out"] else "interrupt_cleanup"] = cleanup
        except OSError as exc:
            result["error"] = str(exc)
    result.update(ended=now(), wall_seconds=time.monotonic() - start)
    aborted = result["timed_out"] or result["interrupted"]
    cleanup = result.get("timeout_cleanup") or result.get("interrupt_cleanup")
    finalized = not (result["executed"] and (result["error"] is not None
                                            or (aborted and cleanup != "owned-process-group")))
    result["evidence_finalized"] = finalized
    result["evidence_note"] = (
        "Snapshot after foreground command/owned-group cleanup; detached workers require separate verification."
        if finalized else
        "Unfinalized: descendants may still write logs/artifacts. No stable hashes or artifact verdict; verify owned processes before reuse."
    )
    result["artifacts"] = []
    for p in artifacts:
        entry = {"path": str(p), "nonempty": False if finalized else None, "finalized": finalized}
        if finalized:
            try:
                p = reject_links(p)
                if p.is_file():
                    entry.update(size=p.stat().st_size, nonempty=p.stat().st_size > 0, sha256=file_sha256(p))
            except (OSError, ValueError) as exc:
                entry["error"] = str(exc)
        result["artifacts"].append(entry)
    result["command_succeeded"] = (result["exit_code"] == 0 and not result["timed_out"]
                                   and not result["interrupted"] and result["error"] is None)
    result["artifact_check_requested"] = bool(artifacts)
    result["expected_artifacts_present"] = all(a["nonempty"] for a in result["artifacts"]) if artifacts and finalized else None
    result["logs"] = [{"path": str(output / name), "finalized": finalized,
                       "bytes": (output / name).stat().st_size if finalized else None,
                       "sha256": file_sha256(output / name) if finalized else None}
                      for name in ("stdout.log", "stderr.log")]
    (output / "command.json").write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--out-dir", required=True, help="NEW evidence directory")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--expected-artifact", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- executable arg ...")
    args = parser.parse_args(argv)
    try:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        report = record_command(command, cwd=args.cwd, out_dir=args.out_dir,
                                timeout=args.timeout, expected_artifacts=args.expected_artifact)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        if report["interrupted"]:
            return 130
        if report["timed_out"]:
            return 124
        if report["error"]:
            return 127
        if report["exit_code"]:
            return report["exit_code"] if report["exit_code"] > 0 else 128 - report["exit_code"]
        return 2 if report["expected_artifacts_present"] is False else 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"kind": "command-record-error", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
