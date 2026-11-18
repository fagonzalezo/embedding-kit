"""EmbeddingRefiner: a PCA-initialised linear map (default), a residual adapter, or an MLP."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

ARCHITECTURES = ("mlp", "residual", "linear")


class EmbeddingRefiner(nn.Module):
    """Maps ``input_dim`` features to ``target_dim``, L2-normalised by default.

    Parameters
    ----------
    architecture:
        - ``"linear"`` (default): ``z = W x̃``. ``x̃`` is the input standardised by fixed
          (non-trainable) statistics and ``W`` a linear map (with bias).
        - ``"residual"``: ``z = W x̃ + g(x̃)`` (review 4.5, WP 5.1), with ``g`` the MLP below and
          its output layer initialised to zero.
        - ``"mlp"``: ``z = g(x)``, an MLP with ``n_layers`` hidden layers of ``hidden_dim`` units
          (Linear, BatchNorm, ReLU, Dropout) and a linear output layer. The default before E6.

        ``hidden_dim``, ``n_layers`` and ``dropout`` configure ``g``; ``"linear"`` ignores them.

        For ``"linear"`` and ``"residual"``, :meth:`initialize_from_data` (called by
        ``Trainer.fit``) sets the standardisation statistics and initialises ``W`` from the top
        ``target_dim`` principal directions of the training data, so that before any training
        step the output equals ``PCAReduce(target_dim)`` followed by L2 normalisation. Training
        starts from PCA. A custom training loop must call it first; training without it (a
        forward pass in training mode) warns once, because ``W`` is then a random map.

        E6 (``docs/BENCHMARK_E1_E2.md``): ``"linear"`` and ``"residual"`` beat ``"mlp"`` on 5 of 6
        datasets and lost on none; ``"linear"`` was at least as good as ``"residual"`` on 5 of 6.
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        normalize_output: bool = True,
        architecture: Literal["mlp", "residual", "linear"] = "linear",
    ):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.architecture = architecture
        self.normalize_output = normalize_output

        self.net: nn.Sequential | None = None
        if architecture in ("mlp", "residual"):
            layers: list[nn.Module] = []
            in_d = input_dim
            for _ in range(n_layers):
                layers += [
                    nn.Linear(in_d, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
                in_d = hidden_dim
            layers.append(nn.Linear(in_d, target_dim))
            self.net = nn.Sequential(*layers)

        self._data_initialized = False
        self._warned_uninitialized = False
        self.linear: nn.Linear | None = None
        if architecture in ("residual", "linear"):
            self.linear = nn.Linear(input_dim, target_dim)
            self.register_buffer("input_mean", torch.zeros(input_dim))
            self.register_buffer("input_scale", torch.ones(input_dim))
            if self.net is not None:
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def initialize_from_data(self, X: np.ndarray) -> None:
        """Set the standardisation statistics and initialise ``W`` at PCA (residual / linear).

        ``W = C diag(s)``, with ``C`` the top ``target_dim`` principal directions of the centred
        data and ``s`` the per-feature standard deviation, so ``W x̃ = C (x - mean)``: the PCA
        projection, whatever the standardisation. Directions beyond the rank of the data have
        zero rows. The output layer of ``g`` is reset to zero. No-op for ``"mlp"``.
        """
        if self.linear is None:
            return
        from embedkit.baselines import _pca_fit  # avoid an import cycle at module load

        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        scale = np.where(std > 1e-8 * max(std.max(), 1e-300), std, 1.0)
        comps, _, rank = _pca_fit(X - mean)
        W = np.zeros((self.target_dim, self.input_dim))
        k = min(self.target_dim, rank)
        W[:k] = comps[:k] * scale

        dev = self.linear.weight.device
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32, device=dev))
        self.input_scale.copy_(torch.as_tensor(scale, dtype=torch.float32, device=dev))
        self.linear.weight.copy_(torch.as_tensor(W, dtype=torch.float32, device=dev))
        self.linear.bias.zero_()
        if self.net is not None:
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
        self._data_initialized = True

    def load_state_dict(self, state_dict, *args, **kwargs):
        out = super().load_state_dict(state_dict, *args, **kwargs)
        if "input_mean" in state_dict:  # a saved, initialised model
            self._data_initialized = True
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            z = self.net(x)
        else:
            if self.training and not self._data_initialized and not self._warned_uninitialized:
                self._warned_uninitialized = True
                warnings.warn(
                    f"EmbeddingRefiner(architecture={self.architecture!r}) is being trained "
                    "without initialize_from_data(X_train), so it starts from a random linear map "
                    "instead of PCA. Trainer.fit calls it; call it yourself in a custom loop.",
                    UserWarning, stacklevel=2,
                )
            x = (x - self.input_mean) / self.input_scale
            z = self.linear(x)
            if self.net is not None:
                z = z + self.net(x)
        if self.normalize_output:
            z = nn.functional.normalize(z, dim=1)
        return z
