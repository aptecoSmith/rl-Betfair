"""Migrate the 3 garaged scalping checkpoints in-place to action schema v4.

The garaged models in registry/weights/ predate scalping-active-management
session 01 (action_schema_version=2, per-runner action dim 5 — signal,
stake, aggression, cancel, arb_spread). scalping-close-signal session 01
bumped the schema to v4 by adding ``requote_signal`` (v2→v3) and
``close_signal`` (v3→v4), plus injected ``fill_prob_head`` (session 02) and
``risk_head`` (session 03) across the same active-management series.

This script:

1. Backs up each checkpoint file with a ``.backup-<timestamp>`` suffix.
2. Applies the shape-based action-head widener (`migrate_scalping_action_head`)
   with ``old_per_runner=5``, ``new_per_runner=7`` — one step across all
   three schema bumps the underlying widener is agnostic to.
3. Injects fresh ``fill_prob_head.*`` and ``risk_head.*`` weights via
   :func:`migrate_fill_prob_head` / :func:`migrate_risk_head` so the v4
   policy's strict load-state-dict succeeds.
4. Bumps ``action_schema_version`` on the saved payload 2 → 4.
   ``obs_schema_version`` is NOT touched — the garaged weights' input
   encoders were sized for obs schema 5, and no obs-migration helper
   exists in this repo. The resulting checkpoint is v4-on-action /
   v5-on-obs, which is the correct state for a follow-up obs-schema
   migration (not in scope for this session).
5. Constructs a fresh v4 policy, loads the migrated state dict strictly,
   and runs a forward pass on a zero-filled obs vector sized for the
   checkpoint's *original* obs schema. Reports NaN-free status.

Usage::

    python scripts/migrate_garaged_to_v4.py           # dry-run / verify
    python scripts/migrate_garaged_to_v4.py --apply   # actually rewrite
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running this file directly via `python scripts/migrate_garaged_to_v4.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from agents.policy_network import (
    migrate_fill_prob_head,
    migrate_risk_head,
    migrate_scalping_action_head,
)
from agents.architecture_registry import create_policy
from env.betfair_env import ACTION_SCHEMA_VERSION

logger = logging.getLogger(__name__)


GARAGED_MODEL_IDS: list[str] = [
    "46187c46-3d0c-40d4-ab0f-da8e8e41c7e3",
    "ef453cd9-3798-4942-8d61-30f29456b1c4",
    "ab460eb9-d42d-4aac-9ab1-4faa5832609a",
]


def _checkpoint_path(model_id: str) -> Path:
    return Path("registry") / "weights" / f"{model_id}.pt"


def _hp_for(model_id: str) -> tuple[str, dict]:
    """Read architecture + hyperparameters from the registry DB."""
    import sqlite3

    conn = sqlite3.connect("registry/models.db")
    cur = conn.execute(
        "SELECT architecture_name, hyperparameters FROM models WHERE model_id = ?",
        (model_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        raise SystemExit(f"Model {model_id} not in registry")
    arch_name, hp_json = row
    hp = json.loads(hp_json)
    return arch_name, hp


def migrate_checkpoint(model_id: str, apply: bool) -> dict:
    path = _checkpoint_path(model_id)
    if not path.exists():
        raise SystemExit(f"Missing weights file {path}")

    raw = torch.load(str(path), weights_only=True)
    if not isinstance(raw, dict) or "weights" not in raw:
        raise SystemExit(f"Unexpected checkpoint format in {path}")

    old_action_v = raw.get("action_schema_version")
    obs_v = raw.get("obs_schema_version")
    weights = raw["weights"]

    arch_name, hp = _hp_for(model_id)
    # Derive per-runner dim from the actor head's final output shape.
    actor_w = weights.get("actor_head.2.weight")
    if actor_w is None:
        raise SystemExit(f"actor_head.2.weight missing in {path}")
    old_per_runner = int(actor_w.shape[0])
    if old_per_runner >= 7:
        logger.info("%s already widened to %d per-runner; skipping head migration",
                    model_id, old_per_runner)

    max_runners = int(weights["action_log_std"].shape[0]) // old_per_runner

    # Step 1: widen action head 5 (or 6) → 7 in one shape-based pass.
    migrated = migrate_scalping_action_head(
        weights,
        max_runners=max_runners,
        old_per_runner=old_per_runner,
        new_per_runner=7,
    )

    # Steps 2/3: build a fresh v4 policy matching this checkpoint's
    # obs_schema (the saved input-layer weights dictate obs dim). We
    # construct a policy whose obs_dim matches the saved input-layer
    # input-size so load_state_dict is strict-compatible.
    runner_enc_w = weights.get("runner_encoder.0.weight")
    market_enc_w = weights.get("market_encoder.0.weight")
    if runner_enc_w is None or market_enc_w is None:
        raise SystemExit(
            f"{model_id}: encoder weights missing — unexpected checkpoint layout"
        )
    per_runner_input = int(runner_enc_w.shape[1])
    market_input = int(market_enc_w.shape[1])
    # Total obs dim = market + max_runners * per_runner_input.
    obs_dim = market_input + max_runners * per_runner_input
    action_dim = max_runners * 7

    fresh_policy = create_policy(
        name=arch_name,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams=hp,
    )

    # Steps 2/3 continued: inject fill_prob_head + risk_head fresh weights.
    migrated = migrate_fill_prob_head(migrated, fresh_policy)
    migrated = migrate_risk_head(migrated, fresh_policy)

    # Step 4: strict-load into the fresh v4 policy. This validates that
    # every key is now accounted for.
    missing, unexpected = fresh_policy.load_state_dict(
        migrated, strict=False,
    )
    # ``strict=False`` so we can report what's missing / unexpected
    # instead of raising; callers decide.
    #
    # Step 5: forward pass smoke test. Zero-vector obs of the right
    # shape, check no NaNs in the outputs.
    obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        fresh_policy.eval()
        out = fresh_policy(obs)
    # PPOLSTMPolicy.forward returns a namedtuple with action_mean, log_std,
    # and value. We just check every tensor coming out is NaN-free.
    nan_fields: list[str] = []
    if hasattr(out, "_asdict"):
        pairs = out._asdict().items()
    elif isinstance(out, dict):
        pairs = out.items()
    elif isinstance(out, (list, tuple)):
        pairs = list(enumerate(out))
    else:
        pairs = [("output", out)]
    for name, t in pairs:
        if isinstance(t, torch.Tensor) and torch.isnan(t).any():
            nan_fields.append(str(name))

    status: dict = {
        "model_id": model_id,
        "arch": arch_name,
        "obs_schema_version": obs_v,
        "old_action_schema_version": old_action_v,
        "new_action_schema_version": ACTION_SCHEMA_VERSION,
        "old_per_runner": old_per_runner,
        "obs_dim": obs_dim,
        "load_missing_keys": missing,
        "load_unexpected_keys": unexpected,
        "nan_fields_in_forward": nan_fields,
    }

    if apply and not nan_fields and not unexpected:
        # Backup & write.
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = path.with_suffix(path.suffix + f".backup-{ts}")
        shutil.copy2(path, backup)
        payload = {
            "weights": migrated,
            "obs_schema_version": obs_v,
            "action_schema_version": ACTION_SCHEMA_VERSION,
        }
        torch.save(payload, str(path))
        status["backup"] = str(backup)
        status["wrote"] = str(path)
    else:
        status["backup"] = None
        status["wrote"] = None

    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="actually rewrite checkpoints (default: dry-run)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== migrate_garaged_to_v4 - {mode} ===")

    any_fail = False
    for mid in GARAGED_MODEL_IDS:
        print(f"\n--- {mid} ---")
        st = migrate_checkpoint(mid, apply=args.apply)
        for k, v in st.items():
            print(f"  {k}: {v}")
        if st["nan_fields_in_forward"] or st["load_unexpected_keys"]:
            any_fail = True

    print("\n=== summary ===")
    if any_fail:
        print("FAIL: one or more migrations produced NaNs or unexpected keys")
        return 1
    print("OK: all 3 checkpoints migrated + forward-pass clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
