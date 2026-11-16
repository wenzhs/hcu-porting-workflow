#!/usr/bin/env python3
"""Inspect declared compilation evidence and planned conversion paths, without execution.

Reads compile_commands.json, optional framework header and separate macro dumps.
No project imports, shell evaluation, compiler invocation or automatic patches.
Unknown command wrappers/response files require examination of the real argv.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptation_common import (AdaptationError, file_sha256, literal_relative_path,
                               parse_predefined_macros, read_text, reject_links,
                               resolve_repo, validate_transform_map, workspace_path)
from build_coverage import analyze as analyze_coverage

CONTEXT_MACROS = ("__CUDA_ARCH__", "__FPT_DTK_ARCH__", "__HIP_DEVICE_COMPILE__",
                  "__CUDACC__", "__HIPCC__", "USE_ROCM", "USE_HCU")
STANDARD_YEAR = {"98": 1998, "03": 2003, "11": 2011, "0x": 2011, "14": 2014,
                 "1y": 2014, "17": 2017, "1z": 2017, "20": 2020, "2a": 2020,
                 "23": 2023, "2b": 2023, "26": 2026, "2c": 2026}


def evidence_file(path):
    p = reject_links(path)
    return {"path": str(p), "sha256": file_sha256(p)}


def inspect_context(repo, *, compile_commands=None, files=None, requirement_header=None,
                    host_macros=None, device_macros=None, transform_map=None, generated_root=None,
                    outputs_exist=False, changed_files=None, cmake_index=None,
                    configuration=None, dependency_manifest=None, scope_ledger=None):
    root = resolve_repo(repo)
    if transform_map is None and (generated_root is not None or outputs_exist):
        raise AdaptationError("Generated root / output audit requires an explicit transform map")
    selected = {workspace_path(root, literal_relative_path(f)) for f in (files or [])}
    report = {"kind": "build-context", "read_only": True, "executed": False,
              "repo": str(root), "units": [], "warnings": [], "evidence": [],
              "requirement": None, "macro_contexts": {}, "transform": None}
    if requirement_header:
        text = read_text(requirement_header)
        requirements = []
        for number, line in enumerate(text.splitlines(), 1):
            if re.match(r"\s*#\s*error\b", line):
                for match in re.finditer(r"C\+\+(\d{2})(?:\s+or\s+(?:later|higher|newer))?", line):
                    year = STANDARD_YEAR.get(match.group(1))
                    if year:
                        requirements.append({"year": year, "line": number, "text": line.strip()})
        report["requirement"] = {"declared_errors": requirements,
                                 "minimum_candidate": max((r['year'] for r in requirements), default=None),
                                 "note": "Header diagnostic evidence only; surrounding feature conditions require review."}
        report["evidence"].append(evidence_file(requirement_header))
    for context, path in (("host", host_macros), ("device", device_macros)):
        if path:
            values = parse_predefined_macros(read_text(path))
            report["macro_contexts"][context] = {k: values.get(k) for k in CONTEXT_MACROS}
            report["evidence"].append({"context": context, **evidence_file(path)})
    if host_macros and device_macros:
        host, device = report["macro_contexts"]["host"], report["macro_contexts"]["device"]
        if host.get("__FPT_DTK_ARCH__") is not None:
            report["warnings"].append("Architecture compatibility macro exists in host context; it is not a device-pass predicate.")
        report["device_context_comparison"] = {"host": host.get("__HIP_DEVICE_COMPILE__"),
                                               "device": device.get("__HIP_DEVICE_COMPILE__"),
                                               "verified_execution": False}
    if compile_commands:
        records = json.loads(read_text(compile_commands))
        if not isinstance(records, list) or not records:
            raise AdaptationError("compile_commands must contain a nonempty array")
        report["evidence"].append(evidence_file(compile_commands))
        for record in records:
            unit = inspect_unit(root, record, report["requirement"])
            if not selected or Path(unit["file"]) in selected:
                report["units"].append(unit)
        missing = selected - {Path(u["file"]) for u in report["units"]}
        if missing:
            raise AdaptationError("Selected files absent from compile_commands: " + ", ".join(str(p) for p in sorted(missing)))
    elif selected:
        raise AdaptationError("--file requires --compile-commands")
    if transform_map is not None:
        if generated_root is None:
            raise AdaptationError("Transform map requires an explicit generated root")
        report["transform"] = {"mapping": validate_transform_map(root, generated_root, transform_map,
                                                                   require_outputs=outputs_exist),
                               "generated_root": str(reject_links(generated_root)),
                               "outputs_checked": outputs_exist, "write_sandbox": False}
    if changed_files is not None:
        if transform_map is not None and not outputs_exist:
            raise AdaptationError("Coverage requires existing generated outputs: use --outputs-exist")
        report["coverage"] = analyze_coverage(root, compile_commands=compile_commands,
            changed_files=changed_files, cmake_index=cmake_index, configuration=configuration,
            dependency_manifest=dependency_manifest, transform_map=transform_map,
            generated_root=generated_root, scope_ledger=scope_ledger)
    elif any((cmake_index, configuration is not None, dependency_manifest, scope_ledger)):
        raise AdaptationError("Coverage evidence requires --changed-files")
    if not any((compile_commands, requirement_header, host_macros, device_macros,
                transform_map is not None, changed_files is not None)):
        raise AdaptationError("Provide compilation, macro, header or conversion evidence")
    return report


def inspect_unit(root, record, requirement):
    if not isinstance(record, dict) or not isinstance(record.get("file"), str):
        raise AdaptationError("Each compilation entry needs a file")
    directory = record.get("directory")
    if not isinstance(directory, str) or not Path(directory).is_absolute():
        raise AdaptationError("Compilation directory must be absolute")
    directory = reject_links(directory)
    source = reject_links(directory / record["file"])
    argv = record.get("arguments")
    origin = "arguments"
    if argv is None:
        command = record.get("command")
        if not isinstance(command, str) or not command.strip():
            raise AdaptationError("Compilation entry needs arguments or command")
        argv, origin = shlex.split(command, posix=True), "command-posix-tokenized-not-executed"
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a or "\0" in a for a in argv):
        raise AdaptationError("Invalid compilation argv")
    standards, defines, includes, architectures = [], {}, [], []
    warnings = []
    consumed = set()
    for i, arg in enumerate(argv):
        if i in consumed:
            continue
        value, standard_flag = None, None
        if arg in {"-std", "--std"}:
            standard_flag = arg
            if i + 1 < len(argv) and not argv[i + 1].startswith(("-", "/")):
                value = argv[i + 1]
                standard_flag += " " + value
                consumed.add(i + 1)
        elif arg.startswith(("-std=", "--std=", "/std:")):
            standard_flag = arg
            value = arg.split(":" if arg.startswith("/") else "=", 1)[1]
        if standard_flag is not None:
            match = re.fullmatch(r"(?:c\+\+|gnu\+\+)(\w+)", value or "")
            year = STANDARD_YEAR.get(match.group(1)) if match else None
            standards.append({"flag": standard_flag, "year": year})
            if year is None:
                warnings.append("Unknown or missing language-standard value: inspect the effective compiler argv")
        if arg.split("=", 1)[0] in {"-Xcompiler", "--compiler-options", "-Xclang", "-Xpreprocessor"}:
            warnings.append("Forwarded compiler options: inspect the expanded host/device argv before judging compatibility")
        for flag, values in (("-I", includes), ("-isystem", includes), ("--offload-arch=", architectures),
                             ("--cuda-gpu-arch=", architectures)):
            if arg == flag and i + 1 < len(argv):
                values.append(argv[i + 1])
            elif arg.startswith(flag) and arg != flag:
                values.append(arg[len(flag):])
        if arg.startswith("-D"):
            value = argv[i + 1] if arg == "-D" and i + 1 < len(argv) else arg[2:]
            name, _, val = value.partition("=")
            defines[name] = val if "=" in value else "1"
        if arg.startswith("@") or arg in {"&&", ";", "|"} or "$" in arg:
            warnings.append("Response file or shell syntax: inspect expanded real argv before deciding compatibility")
    if origin != "arguments":
        warnings.append("Only POSIX tokenization was applied; wrappers and shell expansion were not evaluated")
    effective = standards[-1]["year"] if standards else None
    minimum = (requirement or {}).get("minimum_candidate")
    state = "unknown"
    if minimum and effective and not warnings:
        state = "possible-standard-mismatch" if effective < minimum else "explicit-standard-meets-header-candidate"
    return {"file": str(source), "owned_source": source.is_relative_to(root), "cwd": str(directory),
            "compiler_entry": argv[0], "argv": argv, "argv_source": origin,
            "standards": standards, "explicit_standard_year": effective, "header_comparison": state,
            "defines": defines, "includes": includes, "architectures": architectures,
            "warnings": warnings, "compiled": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--compile-commands")
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--requirement-header", help="Actual framework header containing a standard diagnostic")
    parser.add_argument("--host-macros", help="Separate dump from the actual host compile context")
    parser.add_argument("--device-macros", help="Separate dump from the actual device compile context")
    parser.add_argument("--transform-map", help="JSON object mapping original to generated relative files")
    parser.add_argument("--generated-root")
    parser.add_argument("--outputs-exist", action="store_true", help="Audit existing outputs instead of a fresh planned write set")
    parser.add_argument("--out", help="NEW output JSON; stdout remains the report")
    parser.add_argument("--changed-files", help="JSON array of project-relative changed files")
    parser.add_argument("--cmake-index", help="Existing CMake File API index JSON")
    parser.add_argument("--configuration", help="Exact File API configuration; required for multi-config")
    parser.add_argument("--dependency-manifest", help="JSON array of file/depfile records; see docs/tools.md")
    parser.add_argument("--scope-ledger", help="Read declared target status only; does not validate receipts")
    args = parser.parse_args(argv)
    try:
        mapping = json.loads(read_text(args.transform_map)) if args.transform_map else None
        if args.transform_map and not isinstance(mapping, dict):
            raise AdaptationError("Transform map must contain a JSON object")
        changed = json.loads(read_text(args.changed_files)) if args.changed_files else None
        result = inspect_context(args.repo, compile_commands=args.compile_commands, files=args.file,
                                 requirement_header=args.requirement_header, host_macros=args.host_macros,
                                 device_macros=args.device_macros, transform_map=mapping,
                                 generated_root=args.generated_root, outputs_exist=args.outputs_exist,
                                 changed_files=changed, cmake_index=args.cmake_index,
                                 configuration=args.configuration, dependency_manifest=args.dependency_manifest,
                                 scope_ledger=args.scope_ledger)
        if args.changed_files:
            if changed is None:
                raise AdaptationError("--changed-files must contain a nonempty array, not null")
            result["evidence"].append(evidence_file(args.changed_files))
        text = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
        if args.out:
            with reject_links(args.out).open("x", encoding="utf-8") as stream:
                stream.write(text)
        print(text, end="")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"kind": "build-context-error", "error": str(exc), "executed": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
