#!/usr/bin/env python3
"""audit_public.py - safety gate. Stdlib only.

Scans text files for secrets (ERROR) and machine-local absolute paths (WARN).
Exposes audit_findings(root) -> list[(level, path, msg)] and a CLI (--json).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ERROR, WARN = "error", "warn"

SECRET_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"gh[oprsu]_[A-Za-z0-9]{20,}"), "GitHub token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)(password|passwd|secret|api[_-]?key)\s*[:=]\s*['\"][^'\"]{6,}['\"]"),
     "inline credential"),
]
PATH_PATTERNS = [
    (re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+"), "Windows user path"),
    (re.compile(r"/Users/[^/\s]+"), "macOS user path"),
    (re.compile(r"/home/[^/\s]+"), "Linux user path"),
]

# files allowed to contain secret-like patterns (docs about the patterns themselves)
SECRET_ALLOW = {
    "scripts/audit_public.py",
    "Schema/lint-checklist.md",
    "Schema/frontmatter-contract.md",
}
# path checks are skipped here: docs are illustrative; the registry's job is to hold paths.
PATH_SKIP_PREFIXES = ("docs/", "Schema/", "templates/")
PATH_SKIP_FILES = {"Schema/sources.jsonl", ".wiki-local.json", "scripts/audit_public.py"}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".npy", ".docx",
                 ".pptx", ".xlsx", ".ico", ".woff", ".woff2"}


def _iter_text_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        # don't scan staged binary drops
        if rel.startswith("inbox/pending/Files/") or rel.startswith("inbox_personal/pending/Files/"):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        yield rel, text


def audit_findings(root: Path):
    findings = []
    for rel, text in _iter_text_files(root):
        if rel not in SECRET_ALLOW:
            for rx, label in SECRET_PATTERNS:
                if rx.search(text):
                    findings.append((ERROR, rel, f"possible secret: {label}"))
        skip_path = rel in PATH_SKIP_FILES or rel.startswith(PATH_SKIP_PREFIXES)
        if not skip_path:
            for rx, label in PATH_PATTERNS:
                if rx.search(text):
                    findings.append((WARN, rel, f"machine-local path: {label}"))
                    break
    return findings


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="audit_public")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    findings = audit_findings(Path(args.root))
    if args.json:
        print(json.dumps([{"level": l, "path": p, "msg": m} for l, p, m in findings]))
    else:
        for level, path, msg in findings:
            print(f"[{level}] {path}: {msg}")
        if not findings:
            print("clean")
    return 1 if any(f[0] == ERROR for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
