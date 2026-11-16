#!/usr/bin/env python3
"""Export four self-contained skill directories from one maintained source tree.

This is an exporter, not an installer. It never writes user-global skills folders
by default and never replaces an existing destination. No exported code is run.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import textwrap

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
from adaptation_common import AdaptationError, SCHEMA_VERSION, reject_links, source_identity, validate_output_path

NAMES = ("hcu-porting-workflow", "hcu-native-adaptation", "fastpt-non-transcode", "hcu-transcode")
NAME_ROUTE = {"hcu-porting-workflow": None, "hcu-native-adaptation": "native-hip", "fastpt-non-transcode": "non-transcode", "hcu-transcode": "transcode"}
COMMON = (
    "docs/contract.md", "docs/tools.md", "docs/pre-build-analysis.md", "docs/adaptation-summary.md",
    "references/resources.md", "references/build-adaptation.md", "references/code-conversion.md", "references/instruction-porting.md", "references/common-errors-and-solutions.md",
    "references/pytorch-hipify.md",
    "scripts/adaptation_common.py", "scripts/package_integrity.py", "scripts/route_preflight.py", "scripts/routes/__init__.py",
    "scripts/probe_environment.py", "scripts/source_risk_scan.py", "scripts/adaptation_ledger.py",
    "scripts/inspect_build_context.py", "scripts/record_command.py",
    "scripts/build_coverage.py",
)
ROUTE_FILES = {
    "native-hip": ("scripts/routes/native_hip.py",),
    "non-transcode": ("scripts/routes/non_transcode.py", "scripts/fastpt_non_transcode_prepare.py", "scripts/classify_fastpt_failure.py", "references/fastpt-non-transcode.md"),
    "transcode": ("scripts/routes/transcode.py", "scripts/cmake_fastpt_convert.py", "scripts/cutlass_to_hytlass_prepare.py", "scripts/classify_fastpt_failure.py"),
}
ROUTER_REFERENCES = ()
PUBLIC_CLIS = frozenset({
    "scripts/route_preflight.py", "scripts/classify_fastpt_failure.py",
    "scripts/fastpt_non_transcode_prepare.py", "scripts/cmake_fastpt_convert.py",
    "scripts/cutlass_to_hytlass_prepare.py", "scripts/probe_environment.py",
    "scripts/source_risk_scan.py", "scripts/adaptation_ledger.py",
    "scripts/inspect_build_context.py", "scripts/record_command.py",
})


def guarded_cli(payload: bytes, integrity_source: str) -> bytes:
    """Embed the single maintained gate before any package-local CLI imports.

    Script entrypoints execute from source, not import caches. Inlining this gate
    avoids importing an unverified bootstrap module (or its .pyc) to verify itself.
    Generated copies are never maintained separately from package_integrity.py.
    """
    text = payload.decode("utf-8")
    tree = ast.parse(text)
    boundary = 0
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            boundary = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            boundary = node.end_lineno
        else:
            break
    prefix = (
        "\n# Generated from the common package integrity source; do not edit this copy.\n"
        "if __name__ == '__main__':\n"
        "    import sys as _guard_sys\n"
        "    if not _guard_sys.flags.isolated:\n"
        "        print('Exported CLI requires python -I (recommended: -B -I); no package modules were loaded', file=_guard_sys.stderr)\n"
        "        raise SystemExit(2)\n"
        "    _guard_sys.dont_write_bytecode = True\n"
        + textwrap.indent(integrity_source.rstrip() + "\n", "    ")
        + "    try:\n"
        "        exported_cli_bootstrap(__file__)\n"
        "    except (OSError, ValueError, TypeError, KeyError) as _guard_error:\n"
        "        print(_integrity_json.dumps({'kind': 'package-integrity-error', 'error': str(_guard_error), 'executed': False}), file=_guard_sys.stderr)\n"
        "        raise SystemExit(2)\n\n"
    )
    lines = text.splitlines(keepends=True)
    result = "".join(lines[:boundary]) + prefix + "".join(lines[boundary:])
    ast.parse(result)
    return result.encode("utf-8")


def sources_for(name: str) -> dict[str, str]:
    if name not in NAME_ROUTE:
        raise AdaptationError(f"Unknown skill name: {name}")
    locked = NAME_ROUTE[name]
    files = set(COMMON)
    for route in ([locked] if locked else ROUTE_FILES):
        files.update(ROUTE_FILES[route])
    if locked is None:
        files.update(ROUTER_REFERENCES)
    result = {relative: relative for relative in sorted(files)}
    result["SKILL.md"] = "SKILL.md" if locked is None else f"docs/entrypoints/{name}/SKILL.md"
    return result


def check_links(folder: Path) -> None:
    for document in folder.rglob("*.md"):
        text = document.read_text(encoding="utf-8")
        if text.count("```") % 2:
            raise AdaptationError(f"Unbalanced code fence: {document.relative_to(folder)}")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            relative = re.sub(r":\d+(?:[-–]\d+)?$", "", target.split("#", 1)[0])
            if not relative:
                continue
            candidate = (document.parent / relative).resolve()
            if not candidate.is_relative_to(folder.resolve()) or not candidate.is_file():
                raise AdaptationError(f"Unbundled/escaped reference in {document.relative_to(folder)}: {target}")
        for old_path in ("/mnt/d/AG/", "/home/wenzs/", "D:\\AG\\skills\\"):
            if old_path in text:
                raise AdaptationError(f"Developer-machine path remains in {document.relative_to(folder)}")


def export_packages(out: str | Path, names: list[str] | tuple[str, ...] = NAMES, *, source_root: Path = SOURCE_ROOT) -> dict:
    source, destination = validate_output_path(source_root, out)
    if destination.exists():
        raise AdaptationError("Export destination exists; choose a new directory. Nothing will be overwritten.")
    if not names or len(set(names)) != len(names):
        raise AdaptationError("Select a nonempty set of unique skill names")
    maps = {name: sources_for(name) for name in names}
    for name, mapping in maps.items():
        for relative in mapping.values():
            path = reject_links(source / relative)
            if not path.resolve().is_relative_to(source) or not path.is_file():
                raise AdaptationError(f"Missing source for {name}: {relative}")
        entry = (source / mapping["SKILL.md"]).read_text(encoding="utf-8")
        if not re.search(r"^name:\s*" + re.escape(name) + r"\s*$", entry, re.M):
            raise AdaptationError(f"Skill frontmatter name mismatch: {name}")
    identity = source_identity(source)
    integrity_bytes = (source / "scripts/package_integrity.py").read_bytes()
    if hashlib.sha256(integrity_bytes).hexdigest() != identity["files"]["scripts/package_integrity.py"]:
        raise AdaptationError("Integrity source changed during export")
    integrity_source = integrity_bytes.decode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reject_links(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.export-", dir=destination.parent))
    summary = {"package_format": 1, "contract_version": SCHEMA_VERSION, "source_tree_sha256": identity["tree_sha256"], "source_git_head": identity["git_head"], "installed": False, "packages": {}}
    try:
        for name, mapping in maps.items():
            folder = staging / name
            folder.mkdir()
            for target, relative in mapping.items():
                source_file = reject_links(source / relative)
                payload = source_file.read_bytes()
                if hashlib.sha256(payload).hexdigest() != identity["files"][relative]:
                    raise AdaptationError(f"Source changed during export: {relative}")
                if target == "SKILL.md" and relative != "SKILL.md":
                    # Child source entrypoints live three directories below the
                    # maintained root; their package entrypoint lives at root.
                    payload = re.sub(r"(\[[^\]]+\]\()\.\./\.\./\.\./", r"\1", payload.decode("utf-8")).encode("utf-8")
                if target in PUBLIC_CLIS:
                    payload = guarded_cli(payload, integrity_source)
                target_file = folder / target
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with target_file.open("xb") as stream:
                    stream.write(payload)
            locked = NAME_ROUTE[name]
            readme = (
                f"# {name}\n\n"
                "这是同源生成的自包含安装目录，不要单独手工改写其中公共组件。\n\n"
                "将整个目录放入所用宿主支持的 skills 目录；本导出器没有执行安装。"
                "安装后按宿主要求刷新/重开会话，确认入口被发现。\n\n"
                f"固定路线：`{locked or 'auto / explicit'}`。\n\n"
                "入口与执行边界见 [SKILL.md](SKILL.md)，公共规则见 [共同契约](docs/contract.md)。\n\n"
                "计划检查不执行构建、环境激活或GPU测试；planned不是适配通过。"
                "所有导出CLI以python -I运行（推荐-B -I），在包内模块导入前校验资源并拒绝字节码缓存。"
                "不要用删除manifest或修改哈希的方式绕过资源一致性检查。\n"
            )
            (folder / "README.md").write_text(readme, encoding="utf-8")
            check_links(folder)
            hashes = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}
            manifest = {"package_format": 1, "contract_version": SCHEMA_VERSION, "skill_name": name, "locked_route": locked, "available_routes": [locked] if locked else list(ROUTE_FILES), "source_tree_sha256": identity["tree_sha256"], "source_git_head": identity["git_head"], "files": hashes}
            (folder / "skill-package.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summary["packages"][name] = {"files": len(hashes), "locked_route": locked, "manifest_sha256": hashlib.sha256((folder / "skill-package.json").read_bytes()).hexdigest()}
        (staging / "bundle-manifest.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if destination.exists():
            raise AdaptationError("Destination appeared during export; refusing to overwrite")
        staging.rename(destination)
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return {**summary, "output": str(destination)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="NEW directory outside source tree")
    parser.add_argument("--skills", nargs="+", choices=NAMES, default=list(NAMES))
    args = parser.parse_args(argv)
    try:
        result = export_packages(args.out, args.skills)
    except (OSError, ValueError) as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
