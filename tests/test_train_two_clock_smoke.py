import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from jepa_mpc.training.checkpoints import load_two_clock_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def write_synthetic_cache(directory: Path, episodes: int = 12, length: int = 60, dim: int = 8) -> None:
    rng = np.random.default_rng(0)
    directory.mkdir(parents=True)
    actions = rng.normal(size=(episodes * length, 2)).astype(np.float32)
    projection = rng.normal(size=(2, dim)).astype(np.float32)
    emb = np.zeros((episodes * length, dim), dtype=np.float32)
    for episode in range(episodes):
        offset = episode * length
        emb[offset] = rng.normal(size=dim)
        for t in range(1, length):
            emb[offset + t] = 0.9 * emb[offset + t - 1] + 0.1 * actions[offset + t - 1] @ projection
    np.save(directory / "emb.npy", emb.astype(np.float16))
    np.save(directory / "action.npy", actions)
    np.save(directory / "state.npy", rng.normal(size=(episodes * length, 5)).astype(np.float32))
    np.save(directory / "ep_len.npy", np.full(episodes, length))
    np.save(directory / "ep_offset.npy", np.arange(episodes) * length)


def write_config(path: Path, cache_dir: Path) -> None:
    stage = dict(epochs=2, steps_per_epoch=5, batch_size=8, lr=1e-3, weight_decay=0.0)
    config = {
        "experiment": {"seed": 0},
        "data": dict(cache_dir=str(cache_dir), frameskip=2, window=8, val_fraction=0.2,
                     split_seed=0, start_stride=1, num_workers=0),
        "model": dict(hidden_dim=32, macro_dim=4, max_segment=4, gate_width=32,
                      slow_width=32, init_boundary_rate=0.25),
        "fast": dict(stage, teacher_weight=1.0, open_loop_weight=1.0),
        "slow": dict(stage, segment_mode="learned", fixed_length=2, boundary_cost=0.5,
                     uniform_weight=0.1, macro_kl_weight=1e-3, duration_weight=0.1),
    }
    path.write_text(yaml.safe_dump(config))


def run(*args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/train_two_clock.py"), *args],
                   check=True, cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("mode", ["learned", "fixed"])
def test_fast_then_slow_training_and_resume(tmp_path, mode):
    cache = tmp_path / "cache"
    write_synthetic_cache(cache)
    config = tmp_path / "config.yaml"
    write_config(config, cache)

    run("--config", str(config), "--stage", "fast", "--out", str(tmp_path / "fast"))
    fast_ckpt = tmp_path / "fast" / "best.pt"
    assert fast_ckpt.exists()

    slow_out = tmp_path / "slow"
    run("--config", str(config), "--stage", "slow", "--fast-checkpoint", str(fast_ckpt),
        "--segment-mode", mode, "--out", str(slow_out))
    lines = (slow_out / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert "val/slow/error" in json.loads(lines[-1])

    # The frozen fast clock is carried over unchanged.
    fast_model, _, _ = load_two_clock_checkpoint(fast_ckpt)
    slow_model, normalizer, payload = load_two_clock_checkpoint(slow_out / "last.pt")
    for a, b in zip(fast_model.fast.parameters(), slow_model.fast.parameters()):
        assert np.allclose(a.detach().numpy(), b.detach().numpy())
    assert payload["config"]["slow"]["segment_mode"] == mode
    assert set(payload["train_episodes"]).isdisjoint(payload["val_episodes"])

    # Resuming with more epochs continues from the saved epoch.
    run("--config", str(config), "--stage", "slow", "--fast-checkpoint", str(fast_ckpt),
        "--segment-mode", mode, "--out", str(slow_out), "--epochs", "3")
    lines = (slow_out / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [0, 1, 2]

    # Offline C1/C2 analysis runs on the trained checkpoint.
    analysis = tmp_path / "analysis"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_offline.py"), "--checkpoint",
         f"{mode}={slow_out / 'last.pt'}", "--out", str(analysis), "--horizons", "3", "6"],
        check=True, cwd=ROOT, capture_output=True, text=True,
    )
    summary = json.loads((analysis / "summary.json").read_text())[mode]
    assert np.isfinite(summary["h6/slow_mse"]) and np.isfinite(summary["h6/flat_mse"])
    assert "precision" in summary and "h3/slow_block_pos_err" in summary
