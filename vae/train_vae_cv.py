"""Train the spectral VAE (loss = WMSE + beta * KL) with K-fold cross-fitting.

Model k trains on every fold except k and later scores fold k, so every spectrum in
the catalogue gets a score from a model that never trained on it.

    python train_vae.py --fold 0             # one fold, with the settings below
    sbatch --array=0 train_vae.slurm         # the same as a batch job
    sbatch --array=1-4 train_vae.slurm       # the other folds, once fold 0 looks right

Rerunning the same command resumes from the run folder's latest.pt.
Exit code 99 means the run was stopped by SIGUSR1/SIGTERM (e.g. Slurm's time-limit
warning) after saving a checkpoint; train_vae.slurm requeues the job when it sees it.
The notebook imports build_run(), fold_runs(), train() and run_status() from here.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

import vae_model_cv as vm

# Data: files, keys and cuts are copied from this checkpoint. The split is set below.
# Set to None to use DATA_CONFIG instead.
BASE_CHECKPOINT = "/pscratch/sd/z/zixinq/vae_v2_run1/best.pt"
DATA_CONFIG = dict(
    path="/global/cfs/cdirs/desi/users/zixinq/data/dr2_catalog/loa_spectra_preprocessed",
    flux_key="X", weight_key="W", targetid_key="TARGETID", spectype_key="SPECTYPE",
    keep_spectypes=None, val_fraction=0.10, seed=42, nfeat=1000,
    batch_size=256, block_rows=4096, min_valid_pixels=50, recursive=False,
    n_folds=None, fold=None, exclude_targetids=None,
    mask_key=None, mask_nonzero_valid=True, mix_batches=1,
)

# Cross-fitting split. Identical for every fold; only `fold` changes between runs.
N_FOLDS = 5          # each model trains on (1 - 1/N_FOLDS) * (1 - VAL_FRACTION) of TARGETIDs
VAL_FRACTION = 0.02  # early-stopping set; these TARGETIDs are never trained on by any model

# Repacked shards (repack_folds.py). None: read the original files as before.
# Shards change the data source, so use a new OUT_ROOT when you switch.
SHARD_MANIFEST ="/global/cfs/cdirs/desi/users/zixinq/data/dr2_catalog/loa_spectra_preprocessed/5_fold/manifest.json"

# Known transients: their TARGETIDs are left out of training and validation, and still get
# holdout scores (as in the stars test). None: nothing excluded.
TRANSIENT_FILE = None   # HDF5 file with a targetid / TARGETID dataset

OUT_ROOT = "/pscratch/sd/z/zixinq/vae_cv_run2"
SEED = 0           # weight init, dropout and sampling noise (the split uses the data seed)
NUM_WORKERS = 4    # parallel HDF5 readers; must stay the same when resuming

# nfeat is taken from the data config. mlp_hidden=0: linear bottleneck (no MLP).
MODEL_CONFIG = dict(ncode=64, ch1=32, ch2=64, ch3=128, ch4=256, dropout=0.02, use_bn=True,
                    mlp_hidden=0)
# Learning rate: linear warm-up over lr_warmup_steps optimizer steps, then cosine decay to
# lr_final_frac * learning_rate at the end of epoch `epochs`. The schedule spans exactly
# `epochs`, so epochs is locked once a fold starts: choose it before. patience = epochs
# keeps early stopping from cutting the decay short; best.pt is still the best val epoch.
# lr_schedule="wsd" (constant, then linear decay over the last lr_decay_frac of steps) or
# "plateau" (the old ReduceLROnPlateau) are the alternatives.
TRAIN_CONFIG = dict(epochs=30, learning_rate=3e-4, beta_max=1e-2, warmup_epochs=5,
                    patience=10, max_grad_norm=3e7, log_every=2000, checkpoint_minutes=20,
                    lr_schedule="cosine", lr_warmup_steps=2000, lr_final_frac=0.02,
                    lr_decay_frac=0.2)


def run_prefix(model_config, seed):
    mlp = f"_mlp{model_config['mlp_hidden']}" if model_config.get("mlp_hidden") else ""
    return f"ncode{model_config['ncode']}{mlp}_seed{seed}"


def build_run(fold, model_config=None, train_config=None, seed=SEED, n_folds=N_FOLDS,
              val_fraction=VAL_FRACTION, out_root=OUT_ROOT, base_checkpoint=BASE_CHECKPOINT,
              shard_manifest=SHARD_MANIFEST, transient_file=TRANSIENT_FILE):
    """Full settings of one fold's run (nothing is read from the HDF5 files yet).

    fold=None with n_folds=None: a single model on a plain train/val split (no scores
    for the whole catalogue).
    """
    if (fold is None) != (n_folds is None) or (fold is not None and not 0 <= fold < n_folds):
        raise ValueError(f"fold must be 0..{n_folds - 1}" if n_folds else "pass fold=None with n_folds=None")
    base = vm.load_checkpoint(base_checkpoint)["data_config"] if base_checkpoint else DATA_CONFIG
    data_config = {**base, "n_folds": n_folds, "fold": fold, "val_fraction": val_fraction}
    if transient_file:
        data_config["exclude_targetids"] = transient_targetids(transient_file)
    if shard_manifest:
        if n_folds is None:
            raise ValueError("Shards hold a cross-fitting split; use n_folds, or no manifest.")
        manifest_path = Path(shard_manifest).expanduser().resolve()
        made = json.loads(manifest_path.read_text())
        # Shards store X/W with invalid pixels zeroed and rows already cut, under fixed keys.
        data_config.update(made["keys"], path=str(manifest_path.parent),
                           shard_manifest=str(manifest_path), mask_key=None,
                           valid_rows_key=None, seed=made["config"]["seed"])
    model_config = {**MODEL_CONFIG, **(model_config or {}), "nfeat": data_config["nfeat"]}
    train_config = {**TRAIN_CONFIG, **(train_config or {})}
    vm.VAE(**model_config)  # fail early on a bad model setting
    name = run_prefix(model_config, seed) + (f"_fold{fold}of{n_folds}" if n_folds else "")
    return dict(data_config=data_config, model_config=model_config, train_config=train_config,
                seed=seed, out_dir=Path(out_root).expanduser() / name,
                base_checkpoint=base_checkpoint)


def transient_targetids(path):
    """Sorted unique TARGETIDs of the known transients (a plain list, so it fits in JSON)."""
    import h5py
    import numpy as np
    with h5py.File(Path(path).expanduser(), "r") as h:
        key = vm._first_key(h, "targetid", "TARGETID")
        if key is None:
            raise KeyError(f"{path}: no targetid / TARGETID dataset")
        ids = np.unique(np.asarray(vm.h5_field(h, key)[:]).reshape(-1))
    return [int(i) for i in ids]


def fold_runs(**kwargs):
    """build_run for every fold, with the same settings."""
    n_folds = kwargs.get("n_folds", N_FOLDS)
    return [build_run(k, **kwargs) for k in range(n_folds)]


def _run_info(run):
    return {k: v for k, v in run.items() if k != "out_dir"}


def _comparable(info):
    """The settings that must match across folds: everything except the fold index and
    settings that only affect logging or how long training may run."""
    data = {k: v for k, v in info["data_config"].items() if k != "fold"}
    train = vm.locked_train_settings(info["train_config"])
    return json.dumps(dict(data=data, model=info["model_config"], train=train),
                      sort_keys=True, default=str)


def check_fold_consistency(run):
    """Refuse to start a fold whose settings differ from any fold already started in the
    same OUT_ROOT. One OUT_ROOT holds one cross-fitted set of models."""
    if not run["data_config"]["n_folds"]:
        return
    mine = _comparable(_run_info(run))
    for info_file in sorted(run["out_dir"].parent.glob("*/run_info.json")):
        if info_file.parent == run["out_dir"]:
            continue
        info = json.loads(info_file.read_text())
        if info["data_config"].get("n_folds") and _comparable(info) != mine:
            raise ValueError(
                f"Settings differ from {info_file.parent.name} (see its run_info.json). "
                "All folds need the same model, data and locked training settings, or "
                "their scores cannot be combined. Use a new OUT_ROOT to change them.")


def describe_data(data_config, train_loader):
    ds = train_loader.dataset
    print(f"data: {ds.n_files} HDF5 file(s), {ds.n_rows:,} rows in total")
    notes = []
    if data_config.get("n_folds"):
        k, n = data_config["fold"], data_config["n_folds"]
        frac = (1 - 1 / n) * (1 - data_config["val_fraction"])
        notes.append(f"fold {k} of {n} held out for scoring; trains on ~{frac:.0%} of TARGETIDs")
    if data_config.get("keep_spectypes"):
        notes.append(f"only SPECTYPE in {data_config['keep_spectypes']}")
    if data_config.get("exclude_targetids") is not None:
        notes.append(f"{len(data_config['exclude_targetids']):,} TARGETIDs excluded from training")
    print("      " + ("; ".join(notes) if notes else "single train/val split, no cuts"))


def hours_left_in_job():
    """Hours until the current Slurm job (batch job or Jupyter session) ends, or None."""
    import os
    import subprocess
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        left = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not left or left in ("UNLIMITED", "NOT_SET", "INVALID"):
        return None
    days, _, hms = left.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days or 0) * 24) + parts[0] + parts[1] / 60 + parts[2] / 3600


def train(run, device=None, num_workers=NUM_WORKERS, deadline="auto"):
    """Train (or resume) one run. Returns (model, history, val_loader).

    deadline: a time.time() value after which no new epoch is started if it could not
    finish in time; "auto" uses the end of the current Slurm job or Jupyter session
    (minus 10 minutes), None disables it. Stopping at an epoch boundary means the next
    resume does not re-read the batches already trained in that epoch.
    """
    if deadline == "auto":
        hours = hours_left_in_job()
        deadline = None if hours is None else time.time() + hours * 3600 - 600
    if deadline is not None:
        print(f"will stop at an epoch boundary before "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(deadline))}", flush=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = run["out_dir"]
    if not (out_dir / "latest.pt").exists():
        check_fold_consistency(run)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_info.json").write_text(json.dumps(_run_info(run), indent=2, default=str))

    train_loader, val_loader = vm.make_loaders(run["data_config"], device, num_workers)
    describe_data(run["data_config"], train_loader)
    vm.seed_everything(run["seed"])
    model = vm.VAE(**run["model_config"]).to(device)
    print(f"model: {vm.n_params(model) / 1e6:.2f}M parameters | {run['model_config']}")
    print(f"training: {run['train_config']} | device {device} | num_workers {num_workers}",
          flush=True)
    history = vm.fit_vae(model, train_loader, val_loader, device, out_dir, run["data_config"],
                         **run["train_config"], deadline=deadline)
    return model, history, val_loader


def run_status(run):
    """Progress of one run, read from history.json (no checkpoint is loaded)."""
    out_dir, cfg = run["out_dir"], run["train_config"]
    row = dict(run=out_dir.name, status="not started", epochs_done=0)
    if not (out_dir / "latest.pt").exists():
        return row
    age = (time.time() - (out_dir / "latest.pt").stat().st_mtime) / 3600
    row.update(status="started", last_saved_h_ago=round(age, 1))
    if not (out_dir / "history.json").exists():
        return row
    history = json.loads((out_dir / "history.json").read_text())
    best_val, bad = float("inf"), 0
    for r in history:                       # same rule as fit_vae
        if r["val_loss"] < best_val:
            best_val, bad = r["val_loss"], 0
        elif r["epoch"] > cfg["warmup_epochs"]:
            bad += 1
    best = min(history, key=lambda r: r["val_loss"])
    last = history[-1]
    done = bad >= cfg["patience"] or last["epoch"] >= cfg["epochs"]
    row.update(status="early stopped" if bad >= cfg["patience"] else
               ("finished" if done else "training"),
               epochs_done=last["epoch"], best_epoch=best["epoch"],
               val_rchi2=best["val_rchi2"], train_rchi2=best["train_rchi2"],
               val_kl=best["val_kl"], lr=last["lr"], clipped=last["train_clip_frac"],
               still_improving=best["epoch"] == last["epoch"])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fold", type=int, help=f"0..{N_FOLDS - 1}")
    p.add_argument("--no-folds", action="store_true", help="single train/val split instead")
    p.add_argument("--ncode", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--base-checkpoint", default=BASE_CHECKPOINT,
                   help="'none' to use DATA_CONFIG from this file")
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--device")
    p.add_argument("--hours", type=float,
                   help="stop at an epoch boundary before this many hours from now "
                        "(default: the end of the Slurm job, if any)")
    args = p.parse_args()
    if args.no_folds == (args.fold is not None):
        p.error("pass --fold K, or --no-folds")

    base = None if str(args.base_checkpoint).lower() == "none" else args.base_checkpoint
    run = build_run(
        None if args.no_folds else args.fold,
        model_config={} if args.ncode is None else dict(ncode=args.ncode),
        train_config={} if args.epochs is None else dict(epochs=args.epochs),
        seed=args.seed, n_folds=None if args.no_folds else N_FOLDS,
        out_root=args.out_root, base_checkpoint=base)
    deadline = "auto" if args.hours is None else time.time() + args.hours * 3600
    try:
        train(run, args.device, args.num_workers, deadline)
    except vm.TrainingStopped as stop:
        print(stop, flush=True)
        # 99: time ran out, requeue/rerun. 0: someone asked it to stop. 130: Ctrl-C.
        sys.exit({"SIGUSR1": 99, "SIGTERM": 99, "deadline": 99, "stop_file": 0}
                 .get(stop.signal_name, 130))


if __name__ == "__main__":
    main()