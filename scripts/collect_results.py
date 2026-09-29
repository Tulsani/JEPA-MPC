"""Print a compact report of every run under RUNS_ROOT (paste-able for review).

    python scripts/collect_results.py                 # uses $RUNS_ROOT
    python scripts/collect_results.py --runs-root /path/to/runs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def training_runs(root: Path) -> None:
    for run in sorted(root.glob("pusht_*")):
        log = run / "metrics.jsonl"
        if not log.exists():
            print(f"\n## {run.name}: no metrics.jsonl (not started or failed)")
            continue
        records = read_jsonl(log)
        stage = "fast" if "_fast_" in run.name else "slow"
        key = "val/fast/open_loop" if stage == "fast" else "val/slow/error"
        best = min(records, key=lambda r: r.get(key, float("inf")))
        first, last = records[0], records[-1]
        print(f"\n## {run.name}  ({len(records)} epochs, {fmt(sum(r['seconds'] for r in records) / 60, 1)} min)")
        print(f"   {key}: first={fmt(first.get(key))} best={fmt(best.get(key))} (epoch {best['epoch']}) last={fmt(last.get(key))}")
        print(f"   train loss: first={fmt(first.get('train/loss'))} last={fmt(last.get('train/loss'))}")
        if stage == "fast":
            horizons = [k for k in best if k.startswith("val/fast/open_loop_h")]
            print("   per-horizon (best):", ", ".join(f"{k.split('_')[-1]}={fmt(best[k])}" for k in sorted(horizons, key=lambda k: int(k.split('h')[-1]))))
            print(f"   teacher-forced (best): {fmt(best.get('val/fast/teacher'))}")
        else:
            for name in ("segments/mean_length", "segments/hazard", "slow/duration_ce", "slow/macro_kl", "gate/objective"):
                if f"val/{name}" in last:
                    print(f"   {name}: first={fmt(first.get(f'val/{name}'))} last={fmt(last.get(f'val/{name}'))}")
            per_length = sorted(k for k in last if k.startswith("val/slow/error_j"))
            print("   slow error by jump length (last):", ", ".join(f"{k.split('_')[-1]}={fmt(last[k])}" for k in per_length))


def analysis(root: Path) -> None:
    for summary_path in sorted(root.glob("analysis*/summary.json")):
        summary = json.loads(summary_path.read_text())
        print(f"\n## {summary_path.parent.name}")
        columns = ["mean_segment_length_blocks", "precision", "chance_precision", "recall",
                   "chance_recall", "f1", "chance_f1"]
        horizon_keys = sorted({k.split("/")[0] for s in summary.values() for k in s if k.startswith("h") and "/" in k},
                              key=lambda h: int(h[1:]))
        print("   label | " + " | ".join(columns))
        for label, result in summary.items():
            print(f"   {label} | " + " | ".join(fmt(result.get(c), 3) for c in columns))
        print("   event source:", next(iter(summary.values())).get("event_source"))
        for horizon in horizon_keys:
            print(f"   {horizon} (blocks): label | slow_mse | flat_mse | jumps | slow_pos_err | flat_pos_err | slow_ang_err | flat_ang_err")
            for label, result in summary.items():
                keys = ["slow_mse", "flat_mse", "num_jumps", "slow_block_pos_err", "flat_block_pos_err",
                        "slow_block_angle_err", "flat_block_angle_err"]
                print(f"      {label} | " + " | ".join(fmt(result.get(f"{horizon}/{k}"), 3) for k in keys))


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * (p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5 / denominator
    return 100 * (center - half), 100 * (center + half)


def pooled_evaluations(root: Path) -> None:
    """Pool evaluation seeds per method and offset; report Wilson 95% intervals."""
    import re
    from collections import defaultdict

    groups: dict[str, list[int]] = defaultdict(list)
    for path in sorted((root / "eval").glob("*/result.json")):
        name = re.sub(r"_seed\d+$", "", path.parent.name)
        groups[name].extend(json.loads(path.read_text())["episode_successes"])
    if not groups:
        return
    print("\n## pooled over evaluation seeds (success %, 95% Wilson interval)")
    for name, outcomes in sorted(groups.items()):
        low, high = wilson(sum(outcomes), len(outcomes))
        print(f"   {name}: {100 * sum(outcomes) / len(outcomes):.1f}% [{low:.1f}, {high:.1f}] (n={len(outcomes)})")


def evaluations(root: Path) -> None:
    results = sorted((root / "eval").glob("*/result.json"))
    if not results:
        return
    print("\n## closed-loop evaluations")
    for path in results:
        result = json.loads(path.read_text())
        stats = result.get("horizon_stats", {})
        extra = ""
        if stats:
            extra = (f" chosenH={fmt(stats.get('mean_chosen_horizon'), 2)}"
                     f" (contact={fmt(stats.get('chosen_horizon_contact'), 2)}, free={fmt(stats.get('chosen_horizon_free'), 2)})"
                     f" switch={fmt(stats.get('subgoal_switch_rate'), 2)}"
                     f" plan_s={fmt(result.get('mean_plan_seconds'), 3)}")
        print(f"   {path.parent.name}: success={fmt(result['success_rate'], 1)}%{extra}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path(os.environ.get("RUNS_ROOT", "runs")))
    args = parser.parse_args()
    print(f"# Results under {args.runs_root}")
    training_runs(args.runs_root)
    analysis(args.runs_root)
    evaluations(args.runs_root)
    pooled_evaluations(args.runs_root)


if __name__ == "__main__":
    main()
