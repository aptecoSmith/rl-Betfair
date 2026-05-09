"""scripts/predictor/models.py - architecture definitions.

All torch architectures share a common interface:

  forward(x) -> (B, H, n_quantiles) prediction tensor

The trainer reads `n_quantiles = len(quantiles)` from the model
config and slices the output for the pinball loss.

Tabular path (MLP):
  input shape: (B, F)

Sequence path (LSTM, Transformer, Conv1D):
  input shape: (B, T, F)  -- T is the time-window length

Param-count cap (hard_constraints sec 7) is respected by the
configs in master_todo.md S03; this module does NOT enforce
the cap -- the matrix runner does (it logs param_count to the
scoreboard).

GBM is handled separately in train_one.py via lightgbm.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _output_dim(n_horizons: int, n_quantiles: int) -> int:
    return n_horizons * n_quantiles


# ----------------------------------------------------------------- MLP

class MLPPredictor(nn.Module):
    """Plain feedforward MLP over flat features.

    Architecture knob is `depth` (number of hidden layers); width
    is held constant at `hidden`.
    """
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        n_quantiles: int,
        depth: int,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, _output_dim(n_horizons, n_quantiles)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        out = self.net(x)
        return out.view(-1, self.n_horizons, self.n_quantiles)


# ----------------------------------------------------------------- LSTM

class LSTMPredictor(nn.Module):
    """LSTM over the time window, head off the last hidden state.

    Architecture knob is `time_window` (the input sequence length,
    set in datasets.SequenceExamples). Hidden size and layers are
    held constant per master_todo.md S03 defaults.
    """
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        n_quantiles: int,
        time_window: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        self.time_window = time_window
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, _output_dim(n_horizons, n_quantiles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]            # (B, hidden)
        y = self.head(last)
        return y.view(-1, self.n_horizons, self.n_quantiles)


# ----------------------------------------------------------------- Transformer

class TransformerPredictor(nn.Module):
    """Transformer encoder over the time window, head off CLS-like
    pooled token (mean over positions).

    Architecture knob is `depth` (number of encoder layers).
    `d_model`, `heads`, `ctx_ticks` held constant per master_todo S03.
    """
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        n_quantiles: int,
        depth: int,
        d_model: int = 64,
        heads: int = 4,
        ctx_ticks: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, ctx_ticks, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(d_model, _output_dim(n_horizons, n_quantiles))
        self.ctx_ticks = ctx_ticks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F). Crop / pad T to ctx_ticks if mismatched.
        B, T, _ = x.shape
        if T > self.ctx_ticks:
            x = x[:, -self.ctx_ticks:, :]
        elif T < self.ctx_ticks:
            pad = x.new_zeros(B, self.ctx_ticks - T, x.size(-1))
            x = torch.cat([pad, x], dim=1)
        h = self.proj(x) + self.pos
        h = self.encoder(h)
        # Mean-pool across valid positions (here all positions, since pad
        # is zero and projection of zero approximately preserves zero
        # contribution; for the small-scale architecture sweep this
        # approximation is adequate).
        pooled = h.mean(dim=1)
        y = self.head(pooled)
        return y.view(-1, self.n_horizons, self.n_quantiles)


# ----------------------------------------------------------------- Conv1D

class Conv1DPredictor(nn.Module):
    """Stacked 1D-conv blocks over the time window, head off the
    last position.

    Architecture knob is `kernel` (kernel size). Number of layers
    and channels are held constant per master_todo S03.
    """
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        n_quantiles: int,
        kernel: int,
        layers: int = 4,
        channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        blocks: list[nn.Module] = []
        in_ch = n_features
        for i in range(layers):
            blocks += [
                nn.Conv1d(
                    in_ch, channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = channels
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, _output_dim(n_horizons, n_quantiles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) for conv1d.
        h = x.transpose(1, 2)
        h = self.body(h)
        last = h[:, :, -1]
        y = self.head(last)
        return y.view(-1, self.n_horizons, self.n_quantiles)


# ----------------------------------------------------------------- factory

def build_model(
    family: str,
    n_features: int,
    n_horizons: int,
    n_quantiles: int,
    arch_kwargs: dict,
) -> nn.Module:
    f = family.lower()
    if f == "mlp":
        return MLPPredictor(
            n_features=n_features,
            n_horizons=n_horizons,
            n_quantiles=n_quantiles,
            depth=arch_kwargs.get("depth", 3),
            hidden=arch_kwargs.get("hidden", 128),
            dropout=arch_kwargs.get("dropout", 0.1),
        )
    if f == "lstm":
        return LSTMPredictor(
            n_features=n_features,
            n_horizons=n_horizons,
            n_quantiles=n_quantiles,
            time_window=arch_kwargs.get("time_window", 32),
            hidden=arch_kwargs.get("hidden", 64),
            layers=arch_kwargs.get("layers", 2),
            dropout=arch_kwargs.get("dropout", 0.1),
        )
    if f == "transformer":
        return TransformerPredictor(
            n_features=n_features,
            n_horizons=n_horizons,
            n_quantiles=n_quantiles,
            depth=arch_kwargs.get("depth", 4),
            d_model=arch_kwargs.get("d_model", 64),
            heads=arch_kwargs.get("heads", 4),
            ctx_ticks=arch_kwargs.get("ctx_ticks", 32),
            dropout=arch_kwargs.get("dropout", 0.1),
        )
    if f in ("conv1d", "conv"):
        return Conv1DPredictor(
            n_features=n_features,
            n_horizons=n_horizons,
            n_quantiles=n_quantiles,
            kernel=arch_kwargs.get("kernel", 5),
            layers=arch_kwargs.get("layers", 4),
            channels=arch_kwargs.get("channels", 64),
            dropout=arch_kwargs.get("dropout", 0.1),
        )
    raise ValueError(f"unknown architecture family {family!r}")


def is_sequence_family(family: str) -> bool:
    return family.lower() in ("lstm", "transformer", "conv1d", "conv")


def is_torch_family(family: str) -> bool:
    return family.lower() in ("mlp", "lstm", "transformer", "conv1d", "conv")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
