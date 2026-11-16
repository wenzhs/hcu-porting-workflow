"""Standard-library package integrity gate, also embedded in exported CLIs.

The exporter inserts this single maintained source after future imports and
before any package-local imports in every public CLI. It is an integrity check,
not a signature/authenticity mechanism or a sandbox for project build commands.
"""
import hashlib as _integrity_hashlib
import json as _integrity_json
import os as _integrity_os
from pathlib import Path as _IntegrityPath, PurePosixPath as _IntegrityPurePath
import re as _integrity_re
import sys as _integrity_sys

_INTEGRITY_MANIFEST = "skill-package.json"
CONTRACT_VERSION = "1.0"
_INTEGRITY_ROUTES = {"native-hip": "native_hip", "non-transcode": "non_transcode", "transcode": "transcode"}


def _integrity_linked(path):
    try:
        return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)
    except FileNotFoundError:
        return False


def verify_package(root, *, allow_source=True):
    root = _IntegrityPath(root).resolve()
    manifest_path = root / _INTEGRITY_MANIFEST
    if _integrity_linked(manifest_path):
        raise ValueError("Package manifest must not be a linked/reparse file")
    if not manifest_path.is_file():
        definitions = ("hcu-native-adaptation", "fastpt-non-transcode", "hcu-transcode")
        if allow_source and (root / "scripts/build_skill_packages.py").is_file() and (root / "docs/contract.md").is_file() and (root / "docs/tools.md").is_file() and all((root / "docs/entrypoints" / name / "SKILL.md").is_file() for name in definitions):
            return {"mode": "source", "locked_route": None, "available_routes": list(_INTEGRITY_ROUTES)}
        raise ValueError("Package manifest missing; install a complete exported package")
    data = _integrity_json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Package manifest must be an object")
    if type(data.get("package_format")) is not int or data["package_format"] != 1 or data.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Unsupported package or common contract version")
    routes, locked = data.get("available_routes"), data.get("locked_route")
    if not isinstance(routes, list) or not routes or any(not isinstance(route, str) or route not in _INTEGRITY_ROUTES for route in routes) or len(routes) != len(set(routes)):
        raise ValueError("Invalid package route manifest")
    if locked is not None and (not isinstance(locked, str) or locked not in _INTEGRITY_ROUTES or routes != [locked]):
        raise ValueError("Standalone package has an inconsistent route lock")
    files = data.get("files")
    required = {"SKILL.md", "docs/contract.md", "docs/tools.md", "scripts/adaptation_common.py", "scripts/package_integrity.py", "scripts/route_preflight.py", "scripts/routes/__init__.py"}
    required.update(f"scripts/routes/{_INTEGRITY_ROUTES[route]}.py" for route in routes)
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("Package resource manifest is incomplete")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or not _integrity_re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid resource digest entry")
        parts = _IntegrityPurePath(relative)
        if not relative or parts.is_absolute() or ".." in parts.parts or ":" in relative or "\\" in relative or parts.suffix.lower() in {".pyc", ".pyo"} or "__pycache__" in parts.parts:
            raise ValueError("Invalid package resource or forbidden bytecode cache")
        path = root / relative
        if not path.resolve().is_relative_to(root) or _integrity_linked(path) or not path.is_file():
            raise ValueError(f"Missing/escaped package resource: {relative}")
        if _integrity_hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Package resource hash mismatch: {relative}")
    actual = set()
    for current, directories, names in _integrity_os.walk(root, followlinks=False):
        for name in directories:
            candidate = _IntegrityPath(current) / name
            if name == "__pycache__":
                raise ValueError("Bytecode cache is forbidden in an exported package; re-export a clean package")
            if _integrity_linked(candidate) or not candidate.resolve().is_relative_to(root):
                raise ValueError(f"Linked package directory: {candidate}")
        for name in names:
            path = _IntegrityPath(current) / name
            if path.suffix.lower() in {".pyc", ".pyo"}:
                raise ValueError("Unverified bytecode cache is forbidden in an exported package")
            if _integrity_linked(path):
                raise ValueError(f"Linked package resource: {path}")
            relative = path.relative_to(root).as_posix()
            if relative != _INTEGRITY_MANIFEST:
                actual.add(relative)
    if actual != set(files):
        raise ValueError("Package contains missing or unmanifested resources; re-export it")
    return data


def exported_cli_bootstrap(entry_file):
    """Called from an uncached inline prefix, before adaptation module imports."""
    _integrity_sys.dont_write_bytecode = True
    return verify_package(_IntegrityPath(entry_file).resolve().parent.parent, allow_source=False)
