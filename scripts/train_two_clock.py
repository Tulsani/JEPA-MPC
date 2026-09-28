"""Train the two-clock world model on cached latents.

Stage 1 (fast clock only):
    python scripts/train_two_clock.py --stage fast --out runs/fast

Stage 2 (gate + slow clock, fast clock frozen from stage 1):
    python scripts/train_two_clock.py --stage slow --fast-checkpoint runs/fast/best.pt \
        --segment-mode learned --boundary-cost 0.5 --out runs/slow_learned_b0.5

Every run writes ``metrics.jsonl``, ``last.pt`` (resumable) and ``best.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_mpc.data.latent_cache import (  # noqa: E402
    ActionNormalizer,
    LatentBlockDataset,
    LatentCache,
    split_episodes,
)
from jepa_mpc.training.checkpoints import (  # noqa: E402
    build_two_clock,
    load_config,
    load_two_clock_checkpoint,
    save_checkpoint,
)
from jepa_mpc.training.two_clock_losses import SEGMENT_MODES, fast_losses, slow_losses  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two-clock world model")
    parser.add_argument("--config", type=Path, default=Path("configs/pusht_two_clock.yaml"))
    parser.add_argument("--stage", choices=("fast", "slow"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fast-checkpoint", type=Path, help="required for --stage slow")
    parser.add_argument("--segment-mode", choices=SEGMENT_MODES)
    parser.add_argument("--fixed-length", type=int)
    parser.add_argument("--boundary-cost", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    stage = config[args.stage]
    for name in ("epochs", "steps_per_epoch", "batch_size"):
        value = getattr(args, name)
        if value is not None:
            stage[name] = value
    if args.stage == "slow":
        for name in ("segment_mode", "fixed_length", "boundary_cost"):
            value = getattr(args, name)
            if value is not None:
                stage[name] = value
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_data(config: dict, seed: int):
    data = config["data"]
    cache = LatentCache.load(data["cache_dir"])
    # The latent cache is small; load it into RAM so workers avoid mmap page faults.
    cache.emb = np.asarray(cache.emb)
    cache.action = np.asarray(cache.action)
    train_episodes, val_episodes = split_episodes(cache.num_episodes, data["val_fraction"], data["split_seed"])
    normalizer = ActionNormalizer.fit(np.asarray(cache.action)[cache.episode_frames(train_episodes)])
    common = dict(window=data["window"], frameskip=data["frameskip"], normalizer=normalizer)
    train = LatentBlockDataset(cache, train_episodes, start_stride=data["start_stride"], **common)
    val = LatentBlockDataset(cache, val_episodes, start_stride=data["frameskip"], **common)
    print(
        f"[data] {cache.num_episodes} episodes ({len(train_episodes)} train / {len(val_episodes)} val), "
        f"{len(train)} train windows, {len(val)} val windows, embed_dim={cache.embed_dim}",
        flush=True,
    )
    return cache, train, val, normalizer, train_episodes, val_episodes


def compute_losses(model, batch, args, stage_config):
    latents, actions = batch["latents"], batch["actions"]
    if args.stage == "fast":
        return fast_losses(
            model,
            latents,
            actions,
            teacher_weight=stage_config["teacher_weight"],
            open_loop_weight=stage_config["open_loop_weight"],
        )
    return slow_losses(
        model,
        latents,
        actions,
        mode=stage_config["segment_mode"],
        fixed_length=stage_config["fixed_length"],
        boundary_cost=stage_config["boundary_cost"],
        uniform_weight=stage_config["uniform_weight"],
        macro_kl_weight=stage_config["macro_kl_weight"],
        duration_weight=stage_config["duration_weight"],
    )


@torch.no_grad()
def validate(model, loader, args, stage_config, device, max_batches=100) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}
        loss, metrics = compute_losses(model, batch, args, stage_config)
        totals["loss"] += loss.item()
        for key, value in metrics.items():
            totals[key] += float(value)
        count += 1
    model.train()
    return {f"val/{key}": value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    stage_config = config[args.stage]
    seed = config["experiment"]["seed"]
    seed_everything(seed)
    device = select_device()
    args.out.mkdir(parents=True, exist_ok=True)

    cache, train_set, val_set, normalizer, train_episodes, val_episodes = build_data(config, seed)
    repr_dim = cache.embed_dim
    action_dim = cache.action_dim * config["data"]["frameskip"]
    model = build_two_clock(config["model"], repr_dim, action_dim).to(device)

    if args.stage == "slow":
        if args.fast_checkpoint is None:
            raise SystemExit("--stage slow requires --fast-checkpoint")
        fast_model, fast_normalizer, fast_payload = load_two_clock_checkpoint(args.fast_checkpoint)
        if fast_payload["config"]["model"] != config["model"]:
            raise SystemExit("model config differs from the fast checkpoint's config")
        if not np.allclose(fast_normalizer.mean, normalizer.mean):
            raise SystemExit("action normalization differs from the fast checkpoint (different split?)")
        model.fast.load_state_dict(fast_model.fast.state_dict())
        model.fast.requires_grad_(False)
        trainable = [p for name, p in model.named_parameters() if not name.startswith("fast.")]
    else:
        trainable = list(model.fast.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=stage_config["lr"], weight_decay=stage_config["weight_decay"])
    total_steps = stage_config["epochs"] * stage_config["steps_per_epoch"]
    warmup = min(1000, total_steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(warmup, 1))
        * 0.5
        * (1 + math.cos(math.pi * min(step, total_steps) / max(total_steps, 1))),
    )

    start_epoch, global_step, best = 0, 0, float("inf")
    last_path, best_path = args.out / "last.pt", args.out / "best.pt"
    if last_path.exists() and not args.no_resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, global_step, best = state["epoch"] + 1, state["global_step"], state["best"]
        print(f"[resume] from epoch {start_epoch}, step {global_step}", flush=True)

    sampler = RandomSampler(
        train_set,
        replacement=True,
        num_samples=stage_config["steps_per_epoch"] * stage_config["batch_size"],
        generator=torch.Generator().manual_seed(seed),
    )
    loader_kwargs = dict(
        batch_size=stage_config["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["data"]["num_workers"] > 0,
        drop_last=True,
    )
    train_loader = DataLoader(train_set, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(
        val_set, shuffle=True, generator=torch.Generator().manual_seed(seed + 1), **loader_kwargs
    )

    def payload(epoch: int, metrics: dict) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best": best,
            "config": config,
            "stage": args.stage,
            "repr_dim": repr_dim,
            "action_dim": action_dim,
            "normalizer": normalizer.state_dict(),
            "train_episodes": train_episodes.tolist(),
            "val_episodes": val_episodes.tolist(),
            "fast_checkpoint": str(args.fast_checkpoint) if args.fast_checkpoint else None,
            "metrics": metrics,
        }

    model.train()
    log_path = args.out / "metrics.jsonl"
    for epoch in range(start_epoch, stage_config["epochs"]):
        started = time.time()
        running: dict[str, float] = defaultdict(float)
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}
            loss, metrics = compute_losses(model, batch, args, stage_config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running["loss"] += loss.item()
            for key, value in metrics.items():
                running[key] += float(value)

        steps = stage_config["steps_per_epoch"]
        record = {f"train/{key}": value / steps for key, value in running.items()}
        record.update(validate(model, val_loader, args, stage_config, device))
        record.update(epoch=epoch, step=global_step, lr=scheduler.get_last_lr()[0], seconds=time.time() - started)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in record.items()}), flush=True)

        # Selection metric: the stage's prediction error, not the gate objective
        # (which is negative by design and not comparable across boundary costs).
        selection = record["val/fast/open_loop"] if args.stage == "fast" else record["val/slow/error"]
        if selection < best:
            best = selection
            save_checkpoint(best_path, **payload(epoch, record))
        save_checkpoint(last_path, **payload(epoch, record))

    print(f"[done] best selection metric {best:.5f}; checkpoints in {args.out}")


if __name__ == "__main__":
    main()
