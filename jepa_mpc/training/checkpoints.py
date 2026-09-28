from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml

from jepa_mpc.data.latent_cache import ActionNormalizer
from jepa_mpc.models.two_clock import TwoClockWorldModel


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML config with ``${ENV_VAR}`` expansion in string values."""

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            expanded = os.path.expandvars(value)
            if "${" in expanded:
                raise KeyError(f"unset environment variable in config value {value!r}")
            return expanded
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand(yaml.safe_load(Path(path).read_text()))


def build_two_clock(model_config: dict[str, Any], repr_dim: int, action_dim: int) -> TwoClockWorldModel:
    return TwoClockWorldModel(repr_dim=repr_dim, action_dim=action_dim, **model_config)


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)  # atomic: a crash never leaves a truncated checkpoint


def load_two_clock_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[TwoClockWorldModel, ActionNormalizer, dict[str, Any]]:
    """Returns (model in eval mode, action normalizer, full checkpoint payload)."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = build_two_clock(payload["config"]["model"], payload["repr_dim"], payload["action_dim"])
    model.load_state_dict(payload["model"])
    model.eval()
    return model, ActionNormalizer.from_state_dict(payload["normalizer"]), payload
