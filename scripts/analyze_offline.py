"""Offline evaluation for C1 and C2 on validation episodes.

    python scripts/analyze_offline.py \
        --checkpoint learned=runs/slow_learned_b0.5/best.pt \
        --checkpoint fixed5=runs/slow_fixed_k5/best.pt \
        --checkpoint random=runs/slow_random/best.pt \
        --out runs/analysis

C1: for every start block and horizon H (blocks), predict z[s+H] by chaining
    slow jumps along each method's own segmentation, versus the flat fast-clock
    rollout; report latent MSE and ridge-probe block-pose error.
C2: precision/recall of boundaries against block-motion onset/release events
    within ±tolerance blocks, versus a rate-matched random baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_mpc.data.latent_cache import LatentCache  # noqa: E402
from jepa_mpc.evaluation.boundaries import (  # noqa: E402
    RidgeProbe,
    alignment_scores,
    block_motion,
    block_pose_errors,
    block_pose_targets,
    chained_slow_prediction,
    contact_during_blocks,
    episode_blocks,
    motion_events,
    random_alignment,
    segment_episode,
)
from jepa_mpc.training.checkpoints import load_two_clock_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C1/C2 offline analysis")
    parser.add_argument("--checkpoint", action="append", required=True, help="label=path")
    parser.add_argument("--cache", type=Path, help="defaults to the first checkpoint's cache_dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 15], help="in blocks")
    parser.add_argument("--start-stride", type=int, default=5, help="blocks between prediction starts")
    parser.add_argument("--tolerance", type=int, default=1, help="blocks")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--position-threshold", type=float, default=1.0)
    parser.add_argument("--angle-threshold", type=float, default=0.02)
    parser.add_argument("--probe-frames", type=int, default=100_000)
    parser.add_argument("--timelines", type=int, default=6, help="episodes to plot")
    return parser.parse_args()


def fit_probe(cache: LatentCache, episodes: list[int], max_frames: int) -> RidgeProbe | None:
    if cache.state is None:
        return None
    frames = cache.episode_frames(np.asarray(episodes))
    rng = np.random.default_rng(0)
    if len(frames) > max_frames:
        frames = np.sort(rng.choice(frames, max_frames, replace=False))
    x = np.asarray(cache.emb[frames], dtype=np.float32)
    return RidgeProbe.fit(x, block_pose_targets(np.asarray(cache.state[frames])))


def plot_timelines(path: Path, timelines: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(timelines), 1, figsize=(10, 1.8 * len(timelines)), squeeze=False)
    for axis, timeline in zip(axes[:, 0], timelines):
        steps = np.arange(len(timeline["speed"]))
        axis.fill_between(steps, 0, timeline["moving"], step="post", color="0.85", label="contact / block moving")
        axis.plot(steps, timeline["speed"] / max(timeline["speed"].max(), 1e-6), color="0.4", lw=1, label="block speed (norm.)")
        hazard = timeline["hazard"]
        axis.plot(np.arange(len(hazard)), hazard, color="tab:blue", lw=1.2, label="gate hazard")
        for boundary in timeline["boundaries"]:
            axis.axvline(boundary, color="tab:red", lw=1.2)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel(f"ep {timeline['episode']}", fontsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=7, ncol=3)
    axes[-1, 0].set_xlabel("block (5 env steps)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for item in args.checkpoint:
        label, path = item.split("=", 1)
        model, normalizer, payload = load_two_clock_checkpoint(path, map_location=device)
        runs.append((label, model.to(device), normalizer, payload))

    first = runs[0][3]
    cache = LatentCache.load(args.cache or first["config"]["data"]["cache_dir"])
    frameskip = first["config"]["data"]["frameskip"]
    val_episodes = first["val_episodes"]
    for _, _, _, payload in runs[1:]:
        if payload["val_episodes"] != val_episodes:
            raise SystemExit("checkpoints use different validation splits")
    if args.max_episodes:
        val_episodes = val_episodes[: args.max_episodes]
    probe = fit_probe(cache, first["train_episodes"], args.probe_frames)

    summary: dict[str, dict] = {}
    for label, model, normalizer, payload in runs:
        slow_config = payload["config"]["slow"]
        mode, fixed_length = slow_config["segment_mode"], slow_config["fixed_length"]
        rng = np.random.default_rng(0)
        stats: dict[str, list] = defaultdict(list)
        timelines = []
        for episode in val_episodes:
            blocks = episode_blocks(cache, episode, frameskip, normalizer)
            if blocks is None:
                continue
            num_blocks = len(blocks.actions)
            latents = torch.from_numpy(blocks.latents).to(device)
            actions = torch.from_numpy(blocks.actions).to(device)
            with torch.no_grad():
                hiddens = model.fast_filter(latents[None], actions[None]).hiddens[0]
            boundaries, hazard = segment_episode(
                model, blocks, mode, fixed_length, rng=rng, device=device, hiddens=hiddens
            )
            stats["num_blocks"].append(num_blocks)
            stats["num_boundaries"].append(len(boundaries))

            # ---------------- C2: alignment with block-motion events
            if blocks.contacts is not None or blocks.state is not None:
                # Prefer simulator contact counts; fall back to block motion.
                if blocks.contacts is not None:
                    moving = contact_during_blocks(blocks.contacts).astype(np.float32)
                else:
                    moving = block_motion(
                        blocks.state,
                        position_threshold=args.position_threshold,
                        angle_threshold=args.angle_threshold,
                    ).astype(np.float32)
                events = motion_events(moving > 0)
                stats["num_events"].append(len(events))
                precision, recall = alignment_scores(np.asarray(boundaries), events, args.tolerance)
                chance = random_alignment(len(boundaries), num_blocks, events, args.tolerance, rng)
                stats["precision"].append(precision)
                stats["recall"].append(recall)
                stats["chance_precision"].append(chance[0])
                stats["chance_recall"].append(chance[1])
                stats["fraction_moving"].append(float(moving.mean()))
                if len(timelines) < args.timelines and blocks.state is not None:
                    speed = np.linalg.norm(np.diff(blocks.state[:, 2:4], axis=0), axis=1)
                    timelines.append(
                        dict(episode=episode, speed=speed, moving=moving, hazard=hazard,
                             boundaries=boundaries, events=events)
                    )

            # ---------------- C1: chained slow jumps vs flat rollout
            for horizon in args.horizons:
                for start in range(0, num_blocks - horizon + 1, args.start_stride):
                    local, _ = segment_episode(
                        model, blocks, mode, fixed_length, rng=rng, device=device,
                        start=start, hiddens=hiddens,
                    )
                    result = chained_slow_prediction(model, blocks, local, horizon, start, device)
                    true_end = result["true_end"]
                    stats[f"h{horizon}/slow_mse"].append(float(np.mean((result["slow_end"] - true_end) ** 2)))
                    stats[f"h{horizon}/flat_mse"].append(float(np.mean((result["flat_end"] - true_end) ** 2)))
                    stats[f"h{horizon}/num_jumps"].append(result["num_jumps"])
                    if probe is not None and blocks.state is not None:
                        target = block_pose_targets(blocks.state[start + horizon][None])
                        for name in ("slow", "flat"):
                            position, angle = block_pose_errors(probe(result[f"{name}_end"][None]), target)
                            stats[f"h{horizon}/{name}_block_pos_err"].append(float(position[0]))
                            stats[f"h{horizon}/{name}_block_angle_err"].append(float(angle[0]))

        blocks_total = float(np.sum(stats["num_blocks"]))
        result = {
            "event_source": "n_contacts" if cache.contacts is not None else "block_motion",
            "mode": mode,
            "fixed_length": fixed_length if mode == "fixed" else None,
            "boundary_cost": slow_config["boundary_cost"] if mode == "learned" else None,
            "episodes": len(stats["num_blocks"]),
            "mean_segment_length_blocks": blocks_total / max(float(np.sum(stats["num_boundaries"])) + len(stats["num_blocks"]), 1),
        }
        for key, values in stats.items():
            if key in ("num_blocks", "num_boundaries"):
                continue
            result[key] = float(np.nanmean(values)) if len(values) else float("nan")
        if "precision" in result:
            p, r = result["precision"], result["recall"]
            cp, cr = result["chance_precision"], result["chance_recall"]
            result["f1"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
            result["chance_f1"] = 2 * cp * cr / (cp + cr) if cp + cr > 0 else 0.0
        summary[label] = result
        if timelines and mode == "learned":
            plot_timelines(args.out / f"timelines_{label}.png", timelines)
            np.savez(args.out / f"timelines_{label}.npz", timelines=np.asarray(timelines, dtype=object))
        print(f"== {label}\n{json.dumps(result, indent=2)}", flush=True)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
