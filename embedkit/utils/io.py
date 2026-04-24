"""Save and load embeddings, models, and analysis bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_embeddings(path: str | Path, X: np.ndarray) -> None:
    np.save(str(path), X)


def load_embeddings(path: str | Path) -> np.ndarray:
    return np.load(str(path))


def save_bundle(
    directory: str | Path,
    model,
    config: dict,
    report=None,
) -> None:
    import torch
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    if report is not None:
        with open(d / "report.json", "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=_json_default)


def load_bundle(directory: str | Path) -> dict:
    import torch
    d = Path(directory)
    state_dict = torch.load(d / "model.pt", map_location="cpu")
    with open(d / "config.json") as f:
        config = json.load(f)
    report_dict = None
    if (d / "report.json").exists():
        with open(d / "report.json") as f:
            report_dict = json.load(f)
    return {"state_dict": state_dict, "config": config, "report_dict": report_dict}


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Not serializable: {type(obj)}")
