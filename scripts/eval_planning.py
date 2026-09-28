"""Closed-loop Push-T evaluation through stable-worldmodel (claims C3/C4).

Protocol (LeWM / Hi-LeWM / DINO-WM): start from a dataset state, goal = the
dataset observation ``goal_offset`` env steps later, succeed within
``eval_budget`` env steps.

    # sanity: reproduce flat LeWM (published ~94% at d=25)
    python scripts/eval_planning.py --planner lewm --goal-offset 25 --eval-budget 50

    # ours / baselines on one shared low level
    python scripts/eval_planning.py --planner two_clock --checkpoint runs/slow_learned/best.pt \
        --mode learned --goal-offset 75 --eval-budget 150
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_mpc.envs.lewm import encode_images, load_lewm  # noqa: E402
from jepa_mpc.planning.hierarchical import PLANNER_MODES, PlannerConfig, TwoClockPlanner  # noqa: E402
from jepa_mpc.training.checkpoints import load_two_clock_checkpoint  # noqa: E402

CHECKPOINT_MODE = {"learned": "learned", "fixed": "fixed", "random": "duration"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop Push-T evaluation")
    parser.add_argument("--planner", choices=("lewm", "two_clock"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=PLANNER_MODES, help="default: from the checkpoint")
    parser.add_argument("--fixed-length", type=int, help="default: from the checkpoint")
    parser.add_argument("--cost-mode", choices=("terminal", "prefix_min"), default="prefix_min")
    parser.add_argument("--low-horizon", type=int, default=5)
    parser.add_argument("--high-jumps", type=int, default=3)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--elites", type=int, default=30)
    parser.add_argument("--goal-offset", type=int, default=25, help="env steps")
    parser.add_argument("--eval-budget", type=int, default=50, help="env steps")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-only", action="store_true", help="start states from our val episodes only")
    parser.add_argument("--env-name", default="swm/PushT-v1")
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--dataset-dir", type=Path, default=Path(os.environ.get("DATA_ROOT", ".")) / "raw")
    parser.add_argument("--encoder-repo", default="quentinll/lewm-pusht")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--out", type=Path, required=True, help="results directory")
    parser.add_argument("--video", action="store_true")
    return parser.parse_args()


def last_frame(value: np.ndarray) -> np.ndarray:
    """``[E, T, H, W, C]`` history → ``[E, H, W, C]`` current frame."""
    value = np.asarray(value)
    return value[:, -1] if value.ndim == 5 else value


def per_env(value, count: int) -> np.ndarray:
    return np.asarray(value).reshape(count, -1)


class TwoClockPolicy:
    """stable-worldmodel policy adapter: images → LeWM latents → planner → env actions."""

    def __init__(self, planner_config, model, normalizer, encoder, args, device):
        self.type = "two_clock"
        self.planner_config = planner_config
        self.model = model
        self.normalizer = normalizer
        self.encoder = encoder
        self.args = args
        self.device = device
        self.records: list[dict] = []
        self.plan_seconds: list[float] = []
        self.env = None

    def set_env(self, env) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.planner = TwoClockPlanner(self.model, self.planner_config, self.num_envs, self.device, self.args.seed)
        self.queues = [deque() for _ in range(self.num_envs)]
        self.low = np.asarray(env.single_action_space.low, dtype=np.float32)
        self.high = np.asarray(env.single_action_space.high, dtype=np.float32)
        self.block_count = np.zeros(self.num_envs, dtype=np.int64)

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        n = self.num_envs
        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            reset_ids = [i for i in range(n) if bool(np.asarray(flush[i]).any())]
            for i in reset_ids:
                self.queues[i].clear()
                self.block_count[i] = 0
            if reset_ids:
                self.planner.reset(reset_ids)
        dead = per_env(info_dict.get("terminated", np.zeros(n, dtype=bool)), n)[:, -1].astype(bool)
        replan = [i for i in range(n) if not self.queues[i] and not dead[i]]
        if replan:
            started = time.time()
            pixels = last_frame(info_dict["pixels"])[replan]
            goals = last_frame(info_dict["goal"])[replan]
            latent = encode_images(self.encoder, pixels, self.args.img_size)
            goal = encode_images(self.encoder, goals, self.args.img_size)
            action, info = self.planner.act(latent, goal, replan)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.plan_seconds.append(time.time() - started)
            blocks = action.cpu().numpy().reshape(len(replan), self.args.frameskip, -1)
            raw = np.clip(self.normalizer.denormalize(blocks), self.low, self.high)
            contacts = per_env(info_dict["n_contacts"], n)[:, -1] if "n_contacts" in info_dict else None
            for row, env_id in enumerate(replan):
                self.queues[env_id].extend(raw[row])
                record = {"env": env_id, "block": int(self.block_count[env_id])}
                for key, value in info.items():
                    record[key] = float(value[row])
                if contacts is not None:
                    record["n_contacts"] = float(contacts[env_id])
                self.records.append(record)
                self.block_count[env_id] += 1
        actions = np.full((n, len(self.low)), np.nan, dtype=np.float32)
        for i in range(n):
            if not dead[i] and self.queues[i]:
                actions[i] = self.queues[i].popleft()
        return actions.reshape(*self.env.action_space.shape)


def build_lewm_policy(args, dataset, device):
    """Flat LeWM + CEM exactly as in LeWM's eval.py (for reproduction)."""
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=args.img_size),
    ])
    process = {}
    for column in ("action", "proprio", "state"):
        data = dataset.get_col_data(column)
        data = data[~np.isnan(data).any(axis=1)]
        process[column] = preprocessing.StandardScaler().fit(data)
        if column != "action":
            process[f"goal_{column}"] = process[column]
    model = swm.wm.utils.load_pretrained(args.encoder_repo).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    solver_kwargs = dict(batch_size=1, num_samples=args.samples, var_scale=1.0, n_steps=args.iterations,
                         topk=args.elites, device=str(device), seed=args.seed)
    key = "model" if "model" in inspect.signature(swm.solver.CEMSolver.__init__).parameters else "cost"
    solver = swm.solver.CEMSolver(**{key: model}, **solver_kwargs)
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=args.frameskip)
    return swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform={"pixels": transform, "goal": transform}
    )


def sample_starts(dataset, args, allowed_episodes=None):
    """Same sampling as LeWM eval.py, optionally restricted to given episodes."""
    column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = dataset.get_col_data(column)
    step_ids = dataset.get_col_data("step_idx")
    unique = np.unique(episode_ids)
    lengths = {ep: int(step_ids[episode_ids == ep].max()) + 1 for ep in unique}
    max_start = np.array([lengths[ep] - args.goal_offset - 1 for ep in episode_ids])
    valid = step_ids <= max_start
    if allowed_episodes is not None:
        valid &= np.isin(episode_ids, np.asarray(allowed_episodes))
    valid_rows = np.flatnonzero(valid)
    rng = np.random.default_rng(args.seed)
    rows = np.sort(valid_rows[rng.choice(len(valid_rows) - 1, size=args.num_eval, replace=False)])
    data = dataset.get_row_data(rows)
    return data[column].tolist(), data["step_idx"].tolist()


def summarize_records(records: list[dict]) -> dict:
    if not records:
        return {}
    chosen = np.array([r["chosen_horizon"] for r in records])
    cap = np.array([r["horizon_cap"] for r in records])
    switched = np.array([r["switched"] for r in records])
    summary = {
        "replans": len(records),
        "mean_chosen_horizon": float(chosen.mean()),
        "mean_horizon_cap": float(cap.mean()),
        "subgoal_switch_rate": float(switched.mean()),
    }
    if "n_contacts" in records[0]:
        contact = np.array([r["n_contacts"] > 0 for r in records])
        for name, mask in (("contact", contact), ("free", ~contact)):
            if mask.any():
                summary[f"chosen_horizon_{name}"] = float(chosen[mask].mean())
                summary[f"horizon_cap_{name}"] = float(cap[mask].mean())
                summary[f"switch_rate_{name}"] = float(switched[mask].mean())
        summary["fraction_contact"] = float(contact.mean())
    return summary


def main() -> None:
    args = parse_args()
    import stable_worldmodel as swm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    # Pass the file path directly: with cache_dir, swm looks under <cache_dir>/datasets/.
    dataset_path = args.dataset_dir / f"{args.dataset_name}.h5"
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found: {dataset_path}")
    dataset = swm.data.HDF5Dataset(keys_to_cache=["action", "proprio", "state"], path=dataset_path)
    world = swm.World(env_name=args.env_name, num_envs=args.num_eval, image_shape=(args.img_size, args.img_size),
                      max_episode_steps=2 * args.eval_budget)

    run_config = vars(args).copy()
    allowed = None
    if args.planner == "lewm":
        policy = build_lewm_policy(args, dataset, device)
        label = "lewm_flat"
    else:
        if args.checkpoint is None:
            raise SystemExit("--planner two_clock requires --checkpoint")
        model, normalizer, payload = load_two_clock_checkpoint(args.checkpoint, map_location=device)
        slow = payload["config"]["slow"]
        mode = args.mode or CHECKPOINT_MODE[slow["segment_mode"]]
        fixed_length = args.fixed_length or slow["fixed_length"]
        planner_config = PlannerConfig(
            mode=mode, fixed_length=fixed_length, low_horizon=args.low_horizon, high_jumps=args.high_jumps,
            cost_mode=args.cost_mode, low_samples=args.samples, low_iterations=args.iterations,
            low_elites=args.elites, high_samples=args.samples, high_iterations=args.iterations,
            high_elites=args.elites,
        )
        encoder = load_lewm(args.encoder_repo, device)
        policy = TwoClockPolicy(planner_config, model.to(device), normalizer, encoder, args, device)
        if args.val_only:
            allowed = payload["val_episodes"]
        label = f"two_clock_{mode}"
        run_config["planner_config"] = vars(planner_config)

    episodes, starts = sample_starts(dataset, args, allowed)
    world.set_policy(policy)
    started = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=starts,
        goal_offset=args.goal_offset,
        eval_budget=args.eval_budget,
        episodes_idx=episodes,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
        video=args.out / "videos" if args.video else None,
    )
    elapsed = time.time() - started

    result = {
        "label": label,
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
        "wall_seconds": elapsed,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in run_config.items()},
    }
    if isinstance(policy, TwoClockPolicy):
        result["mean_plan_seconds"] = float(np.mean(policy.plan_seconds)) if policy.plan_seconds else None
        result["horizon_stats"] = summarize_records(policy.records)
        with (args.out / "replans.jsonl").open("w") as handle:
            for record in policy.records:
                handle.write(json.dumps(record) + "\n")
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "episode_successes"}, indent=2))


if __name__ == "__main__":
    main()
