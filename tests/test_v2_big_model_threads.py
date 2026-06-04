"""Size-aware multiprocess threading (--big-model-threads, 2026-06-04).

Big LSTMs (d_model >= 512) get N BLAS/OMP threads so their large matmuls can
use the cores that small agents free as they finish — shrinking the lone
h1024 straggler that otherwise gates a whole generation. Everything else
stays single-threaded. Default N=1 == single-thread everywhere == bit-identical
to the golden multiprocess path.

Three load-bearing pieces, one guard each:
1. the size->threads policy (`_threads_for_hidden`),
2. the worker actually applying it AND not leaking `_num_threads` into
   `train_one_agent` (would TypeError every agent),
3. the import-time BLAS/OMP cap following SES_MP_MAX_THREADS (without it,
   torch.set_num_threads can't grow past an OMP cap of 1, so the feature
   silently no-ops).
"""

from __future__ import annotations

import subprocess
import sys

from training_v2.cohort.runner import _threads_for_hidden


class TestThreadsForHidden:
    def test_default_one_is_single_thread_every_size(self):
        # big_model_threads=1 -> always 1 -> byte-identical golden path.
        for h in (64, 128, 256, 512, 1024):
            assert _threads_for_hidden(h, 1) == 1

    def test_big_get_n_small_stay_one(self):
        assert _threads_for_hidden(64, 4) == 1
        assert _threads_for_hidden(256, 4) == 1   # below the 512 threshold
        assert _threads_for_hidden(512, 4) == 4   # at threshold
        assert _threads_for_hidden(1024, 4) == 4

    def test_threshold_boundary_is_512(self):
        assert _threads_for_hidden(256, 6) == 1
        assert _threads_for_hidden(512, 6) == 6


class TestWorkerConsumesNumThreads:
    def _fake_result(self):
        class _T:
            def __init__(self):
                self.wall_time_sec = 0.0

        class _R:
            def __init__(self):
                self.train = _T()

        return _R()

    def test_sets_threads_and_does_not_leak_to_train(self, monkeypatch):
        import torch

        import training_v2.cohort.worker as worker_mod
        from training_v2.cohort import multiproc_worker as mw

        calls = []
        monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
        captured: dict = {}
        res = self._fake_result()

        def fake(**kw):
            captured.update(kw)
            return res

        monkeypatch.setattr(worker_mod, "train_one_agent", fake)

        mw._train_agent_worker({"_num_threads": 6, "agent_id": "a", "seed": 1})
        assert calls == [6]                     # threads set from the spec
        assert "_num_threads" not in captured    # consumed, NOT passed onward
        assert captured["agent_id"] == "a"       # other kwargs flow through

    def test_defaults_to_one_when_absent(self, monkeypatch):
        import torch

        import training_v2.cohort.worker as worker_mod
        from training_v2.cohort import multiproc_worker as mw

        calls = []
        monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
        res = self._fake_result()
        monkeypatch.setattr(worker_mod, "train_one_agent", lambda **kw: res)

        mw._train_agent_worker({"agent_id": "a"})
        assert calls == [1]   # absent -> single-thread (bit-identical default)


class TestEnvCapPropagation:
    """Import-time BLAS/OMP cap must follow SES_MP_MAX_THREADS, else
    torch.set_num_threads(N>1) can't take effect and the feature no-ops."""

    def _omp_after_import(self, ses_value):
        setenv = (
            f"os.environ['SES_MP_MAX_THREADS']={ses_value!r};"
            if ses_value is not None else ""
        )
        code = (
            "import os;" + setenv
            + "import training_v2.cohort.multiproc_worker;"
            + "print(os.environ['OMP_NUM_THREADS'])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=180,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        return out.stdout.strip().splitlines()[-1]

    def test_default_caps_at_one(self):
        assert self._omp_after_import(None) == "1"

    def test_raised_cap_follows_env(self):
        assert self._omp_after_import("4") == "4"
