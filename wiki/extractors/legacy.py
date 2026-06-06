"""Optional legacy-format bridge: convert binary .doc/.ppt/.xls to modern OOXML via headless
LibreOffice, then let the matching modern extractor handle the result. No Office COM automation.

This is an opt-in power-up that needs a LibreOffice/soffice *binary* on the machine (not a Python
package). If none is found, convert() raises ExtractorUnavailable with guidance and the caller falls
back to the agent's own file skill - exactly as before this bridge existed.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from base import ExtractorUnavailable

# legacy binary -> modern OOXML target extension
LEGACY_TARGET = {".doc": "docx", ".ppt": "pptx", ".xls": "xlsx"}

# soffice is frequently not on PATH (especially on Windows); probe the usual install locations too.
_WINDOWS_CANDIDATES = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
)


def find_soffice():
    """Locate a LibreOffice/soffice binary, or None if not installed."""
    for name in ("soffice", "libreoffice"):
        exe = shutil.which(name)
        if exe:
            return exe
    for cand in _WINDOWS_CANDIDATES:
        if Path(cand).exists():
            return cand
    return None


def soffice_available() -> bool:
    return find_soffice() is not None


def convert(path, out_dir=None, soffice=None) -> Path:
    """Convert a legacy .doc/.ppt/.xls to its modern equivalent; return the converted file path.

    Raises ExtractorUnavailable if the suffix isn't a known legacy one, or no converter is installed.
    Raises RuntimeError if the converter runs but produces no output.
    """
    src = Path(path)
    target = LEGACY_TARGET.get(src.suffix.lower())
    if target is None:
        raise ExtractorUnavailable(f"not a supported legacy format: {src.suffix}")
    exe = soffice or find_soffice()
    if not exe:
        raise ExtractorUnavailable(
            f"legacy {src.suffix} needs LibreOffice to convert to .{target}. Install LibreOffice "
            "(an optional power-up; provides `soffice`), convert the file manually, or use the "
            "agent's own file skill.")
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="llmwiki-legacy-"))
    out.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "--headless", "--convert-to", target, "--outdir", str(out), str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "unknown error").strip()
        raise RuntimeError(f"LibreOffice conversion failed for {src.name}: {details}")
    converted = out / f"{src.stem}.{target}"
    if not converted.exists():
        # soffice usually names output by the source stem; if it differs, take the newest match.
        matches = sorted(out.glob(f"*.{target}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            raise RuntimeError(f"LibreOffice reported success but produced no .{target} for {src.name}")
        converted = matches[0]
    return converted
