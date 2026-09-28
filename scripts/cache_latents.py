"""Download a LeWM dataset + checkpoint and cache frozen-encoder embeddings.

Example (Push-T):

    python scripts/cache_latents.py \
        --dataset-repo quentinll/lewm-pusht --dataset-file pusht_expert_train.h5.zst \
        --model-repo quentinll/lewm-pusht --data-root $DATA_ROOT \
        --out $DATA_ROOT/latents/pusht_lewm

Writes the files documented in ``jepa_mpc/data/latent_cache.py`` and runs two
sanity checks whose output belongs in the job log:

1. LeWM's own predictor error on the cached embeddings (checks that image
   preprocessing matches what LeWM was trained with);
2. a ridge probe from embedding to simulator state on held-out episodes
   (checks that block pose is linearly decodable from the frozen latent).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa_mpc.envs.lewm import load_lewm, preprocess  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-repo", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset-file", default="pusht_expert_train.h5.zst")
    parser.add_argument("--model-repo", default="quentinll/lewm-pusht")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pixel-key", default="pixels")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------- data
def fetch_dataset(repo: str, filename: str, data_root: Path, skip: bool) -> Path:
    """Download + decompress; returns the .h5 path (archives are extracted)."""
    data_root.mkdir(parents=True, exist_ok=True)
    is_tar = filename.endswith(".tar.zst")
    target = data_root / (filename.removesuffix(".tar.zst") + ".h5" if is_tar else filename.removesuffix(".zst"))
    if target.exists():
        print(f"[data] using existing {target}")
        return target
    if is_tar:
        found = sorted(data_root.rglob("*.h5"))
        if found:
            print(f"[data] using existing extracted {found[0]}")
            return found[0]
    if skip:
        raise FileNotFoundError(f"{target} not found and --skip-download was given")

    from huggingface_hub import hf_hub_download

    print(f"[data] downloading {repo}/{filename} ...", flush=True)
    archive = Path(
        hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=data_root)
    )
    if is_tar:
        import tarfile

        import zstandard

        print(f"[data] extracting {archive} into {data_root}", flush=True)
        with archive.open("rb") as source:
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(source)
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(data_root)
        found = sorted(data_root.rglob("*.h5"))
        if not found:
            raise FileNotFoundError(f"no .h5 file inside {archive}")
        print(f"[data] extracted {[str(path) for path in found]}; using {found[0]}")
        return found[0]
    print(f"[data] decompressing {archive} -> {target}", flush=True)
    if shutil.which("zstd"):
        subprocess.run(["zstd", "-d", "--long=31", "-f", str(archive), "-o", str(target)], check=True)
    else:
        import zstandard

        with archive.open("rb") as source, target.open("wb") as sink:
            zstandard.ZstdDecompressor(max_window_size=2**31).copy_stream(source, sink)
    return target


def describe_h5(handle) -> None:
    print("[data] HDF5 columns:")
    for key in handle.keys():
        dataset = handle[key]
        print(f"    {key:24s} shape={dataset.shape} dtype={dataset.dtype}")


# -------------------------------------------------------------------- model
@torch.no_grad()
def encode_frames(model, pixels, count: int, args, device, out_path: Path) -> np.ndarray:
    first = model.encode({"pixels": preprocess(pixels[0:1], args.img_size, device)[:, None]})
    dim = first["emb"].shape[-1]
    emb = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(count, dim))
    start_time = time.time()
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        images = preprocess(pixels[start:stop], args.img_size, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model.encode({"pixels": images[:, None]})
        emb[start:stop] = output["emb"][:, 0].float().cpu().numpy().astype(np.float16)
        if (start // args.batch_size) % 50 == 0:
            rate = stop / max(time.time() - start_time, 1e-6)
            print(f"[encode] {stop}/{count} frames ({rate:.0f} frames/s)", flush=True)
    emb.flush()
    return np.load(out_path, mmap_mode="r")


# ------------------------------------------------------------ sanity checks
@torch.no_grad()
def lewm_predictor_check(model, emb, action, ep_len, ep_offset, frameskip, device, samples=2048):
    """LeWM one-step prediction MSE on cached embeddings, relative to variance."""
    rng = np.random.default_rng(0)
    history = 3
    span = history * frameskip  # need history+1 latents
    finite = action[np.isfinite(action).all(axis=1)]
    mean, std = finite.mean(0), finite.std(0) + 1e-6
    valid = np.flatnonzero(ep_len > span)
    if not len(valid):
        print("[check] episodes too short for predictor check")
        return
    latents, actions = [], []
    for _ in range(samples):
        episode = rng.choice(valid)
        start = ep_offset[episode] + rng.integers(0, ep_len[episode] - span)
        frames = start + frameskip * np.arange(history + 1)
        latents.append(np.asarray(emb[frames], dtype=np.float32))
        block = np.nan_to_num((action[start : start + span] - mean) / std)
        actions.append(block.reshape(history, -1))
    latents = torch.from_numpy(np.stack(latents)).to(device)
    actions = torch.from_numpy(np.stack(actions)).float().to(device)
    act_emb = model.action_encoder(actions)
    predicted = model.predict(latents[:, :history], act_emb)
    mse = (predicted - latents[:, 1:]).square().mean().item()
    copy_mse = (latents[:, :-1] - latents[:, 1:]).square().mean().item()
    variance = latents.var(dim=(0, 1)).mean().item()
    print(
        f"[check] LeWM predictor MSE={mse:.4f}  copy-last MSE={copy_mse:.4f}  "
        f"latent var={variance:.4f}  (predictor should beat copy-last clearly)"
    )
    return {"lewm_pred_mse": mse, "copy_mse": copy_mse, "latent_var": variance}


def ridge_probe(emb, state, ep_len, ep_offset, max_frames=200_000, alpha=1.0):
    """Held-out R^2 of a ridge probe from embedding to each state dimension."""
    rng = np.random.default_rng(0)
    episodes = rng.permutation(len(ep_len))
    num_val = max(1, len(episodes) // 10)

    def frames(episode_ids):
        index = np.concatenate([np.arange(ep_offset[e], ep_offset[e] + ep_len[e]) for e in episode_ids])
        if len(index) > max_frames:
            index = np.sort(rng.choice(index, max_frames, replace=False))
        return index

    train_index, val_index = frames(episodes[num_val:]), frames(episodes[:num_val])
    x_train = np.asarray(emb[train_index], dtype=np.float64)
    x_val = np.asarray(emb[val_index], dtype=np.float64)
    y_train, y_val = state[train_index].astype(np.float64), state[val_index].astype(np.float64)
    x_mean, y_mean = x_train.mean(0), y_train.mean(0)
    xc = x_train - x_mean
    weights = np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ (y_train - y_mean))
    prediction = (x_val - x_mean) @ weights + y_mean
    residual = ((prediction - y_val) ** 2).sum(0)
    total = ((y_val - y_val.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - residual / total
    print("[check] ridge probe R^2 per state dim (held-out episodes):", np.round(r2, 3).tolist())
    return r2.tolist()


def main() -> None:
    args = parse_args()
    import h5py

    try:
        import hdf5plugin  # noqa: F401  (registers compression filters)
    except ImportError:
        print("[warn] hdf5plugin not installed; compressed columns may fail to read")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_path = fetch_dataset(args.dataset_repo, args.dataset_file, args.data_root, args.skip_download)
    args.out.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as handle:
        describe_h5(handle)
        ep_len = handle["ep_len"][:].astype(np.int64)
        ep_offset = handle["ep_offset"][:].astype(np.int64)
        if args.limit_episodes:
            # Keep the first episodes by storage order so frames stay a prefix.
            order = np.argsort(ep_offset)[: args.limit_episodes]
            ep_len, ep_offset = ep_len[order], ep_offset[order]
        count = int((ep_offset + ep_len).max())

        model = load_lewm(args.model_repo, device)
        emb = encode_frames(model, handle[args.pixel_key], count, args, device, args.out / "emb.npy")

        # Keep every small per-frame numeric column (action, state, proprio,
        # n_contacts, ...) for diagnostics; skip images and episode metadata.
        saved = {}
        for column in handle.keys():
            dataset = handle[column]
            if column in (args.pixel_key, "ep_len", "ep_offset") or dataset.ndim > 2:
                continue
            if dataset.shape[0] < count or dataset.dtype.kind not in "fiub":
                continue
            if dataset.ndim == 2 and dataset.shape[1] > 64:
                continue
            values = dataset[:count].astype(np.float32)
            np.save(args.out / f"{column}.npy", values)
            saved[column] = list(values.shape)
        if "action" not in saved:
            raise KeyError("dataset has no 'action' column")
        np.save(args.out / "ep_len.npy", ep_len)
        np.save(args.out / "ep_offset.npy", ep_offset)

    action = np.load(args.out / "action.npy")
    checks = {"predictor": lewm_predictor_check(model, emb, action, ep_len, ep_offset, args.frameskip, device)}
    if "state" in saved:
        checks["state_probe_r2"] = ridge_probe(emb, np.load(args.out / "state.npy"), ep_len, ep_offset)

    meta = {
        "source_dataset": f"{args.dataset_repo}/{args.dataset_file}",
        "encoder": args.model_repo,
        "embed_dim": int(emb.shape[1]),
        "num_frames": int(count),
        "num_episodes": int(len(ep_len)),
        "img_size": args.img_size,
        "frameskip_hint": args.frameskip,
        "columns": saved,
        "checks": checks,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
