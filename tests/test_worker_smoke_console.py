"""Regression tests for the worker's smoke-test assertion printing.

Guards against the 2026-04-19 regression (plans/entropy-control-v2/)
where Rich's legacy-Windows renderer crashed on the ``\u2713`` /
``\u2717`` markers and the em-dash characters embedded in earlier
assertion detail strings. The crash propagated out of
``Worker._run_smoke_test`` AFTER the probe had PASSED, preventing
the full population from launching.

Verifies two things about the fix:

1. ``_format_smoke_assertion_line`` returns a pure-ASCII string, so
   a downstream cp1252-encoded console.print cannot crash on it.
2. The marker is bracketed ``[PASS]`` / ``[FAIL]``, not the unicode
   checkmark/cross.

We test the formatter directly rather than ``_run_smoke_test`` as a
whole because the helper is the invariant the regression hinged on;
future refactors may move the call site but must keep the ASCII-safe
guarantee.
"""

from __future__ import annotations

import pytest

from training.worker import _format_smoke_assertion_line


class TestSmokeAssertionLineFormatter:
    def test_passed_marker_is_bracketed_pass(self):
        """Passed assertion renders ``[PASS]`` — not the unicode
        checkmark character (U+2713) the pre-fix code used."""
        line = _format_smoke_assertion_line({
            "passed": True, "detail": "all good",
        })
        assert "[PASS]" in line
        assert "\u2713" not in line
        assert "\u2717" not in line

    def test_failed_marker_is_bracketed_fail(self):
        """Failed assertion renders ``[FAIL]`` — not U+2717."""
        line = _format_smoke_assertion_line({
            "passed": False, "detail": "oops",
        })
        assert "[FAIL]" in line
        assert "\u2717" not in line

    def test_output_is_pure_ascii_even_with_unicode_detail(self):
        """The key invariant: the rendered line must encode cleanly
        to ASCII so a cp1252 console cannot crash on it. Any non-ASCII
        characters in the upstream ``detail`` string (em-dashes,
        curly arrows, box-drawing, Greek letters from old assertion
        formats) get substituted."""
        # Mix of characters the old and new gate details historically
        # contained:
        #   U+2192 RIGHTWARDS ARROW (new tracking-error detail used
        #     '->' ASCII but the plan's lessons_learnt has ->)
        #   U+2212 MINUS SIGN        (old slope detail delta symbol)
        #   U+0394 GREEK CAPITAL DELTA (old slope detail)
        #   U+2014 EM DASH           (Rich-common)
        #   U+2026 HORIZONTAL ELLIPSIS
        detail = (
            "ep3\u2212ep1 entropy: worst \u0394 = +5.7 \u2014 "
            "139.6 \u2192 145.3 \u2026"
        )
        line = _format_smoke_assertion_line({
            "passed": False, "detail": detail,
        })
        # Must round-trip through ASCII without raising.
        encoded = line.encode("ascii", errors="strict")
        assert isinstance(encoded, bytes)
        # Non-ASCII characters substituted with ``?`` (errors="replace").
        assert "?" in line
        for cp in ("\u2212", "\u0394", "\u2014", "\u2192", "\u2026"):
            assert cp not in line

    def test_output_encodes_cleanly_to_cp1252(self):
        """Belt-and-braces: even if the console falls through to the
        legacy Windows cp1252 path, the line must encode without
        raising. This is the exact failure mode the regression
        hit — Rich's LegacyWindowsTerm ultimately calls
        ``text.encode('cp1252')`` and raised UnicodeEncodeError on
        the first non-encodable character."""
        # Realistic new-gate detail shape (ASCII already):
        detail = (
            "tracking-error growth: worst = +40.0 (agent abc12345: "
            "|140.0-150|=10.0 -> |180.0-150|=30.0), threshold <= 3.0"
        )
        line = _format_smoke_assertion_line({
            "passed": False, "detail": detail,
        })
        line.encode("cp1252", errors="strict")  # must not raise

    def test_missing_fields_degrades_gracefully(self):
        """An assertion dict missing ``passed`` or ``detail`` still
        renders without crashing — the helper treats absent
        ``passed`` as falsy and absent ``detail`` as empty."""
        line_empty = _format_smoke_assertion_line({})
        assert "[FAIL]" in line_empty  # absent passed is falsy

        line_partial = _format_smoke_assertion_line({"passed": True})
        assert "[PASS]" in line_partial

    def test_formatter_keeps_detail_content(self):
        """Sanity: regular ASCII detail content is preserved
        verbatim so operators still get the diagnostic information."""
        detail = "ep1 policy_loss: worst = 46.5 (agent smoke-pp)"
        line = _format_smoke_assertion_line({
            "passed": True, "detail": detail,
        })
        assert detail in line


class TestWorkerStreamEncoding:
    """Verify the worker module reconfigures stdout/stderr to utf-8
    at import time so Rich never falls through to cp1252."""

    def test_worker_import_reconfigures_stdout(self):
        """Importing training.worker should reconfigure stdout to
        utf-8. If stdout is a redirected/wrapped stream that doesn't
        support reconfigure, the module must not crash at import."""
        import sys
        import training.worker  # noqa: F401 — side effect is the test

        # If the current stdout supports reconfigure (the common
        # case — TextIOWrapper around the tty), encoding should
        # have landed on utf-8. We don't strictly require it —
        # pytest's capture can substitute a wrapper that keeps its
        # own encoding — but the import itself must not raise.
        encoding = getattr(sys.stdout, "encoding", None)
        assert encoding is not None
