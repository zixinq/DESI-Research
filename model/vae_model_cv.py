"""Spectral VAE, v4: model, WMSE + KL loss, HDF5 streaming, training, scoring.

Changes from v3
* Loss is the weighted MSE plus beta * KL only. The contrastive loss, projection head
  and error floor are gone, and so is weight_power (it had to be 1.0 anyway).
* The loss is unchanged: LOSS_CONFIG and the split/batching code are identical to v3,
  so runs are comparable with earlier ones and checkpoints load in either version.
* Checkpoints are saved every `checkpoint_minutes` instead of every 200 batches.
* Ctrl-C, SIGTERM and SIGUSR1 (Slurm's pre-timeout warning) save a checkpoint at the
  end of the current batch and stop cleanly; rerunning resumes from it.
* Training resumes automatically from out_dir/latest.pt. Settings that change the
  result must match; epochs, patience, logging and checkpoint interval may change.
* DataLoader workers are allowed (the batch order stays deterministic as long as
  num_workers is the same when resuming).
* Metrics are accumulated on the GPU, so each batch syncs once instead of ~7 times.
"""
from pathlib import Path
from os import PathLike
import hashlib
import inspect
import json
import math
import os
import random
import signal
import threading
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


# Identical to v3, so v3 checkpoints are recognized as the same loss.
LOSS_CONFIG = {
    "name": "direct_ivar_batch_sum_v1",
    "reconstruction": "0.5 * sum_batch_pixels(ivar * residual**2)",
    "kl": "sum_batch_latent",
    "weight_power": 1.0,
}

# Keys of older model configs that v4 ignores (contrastive projection head).
LEGACY_MODEL_KEYS = ("proj_dim", "proj_hidden")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path):
    return torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)


# Model

def kaiming_init(module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.kaiming_uniform_(module.weight, a=0.2, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VAE(nn.Module):
    """1D convolutional VAE for spectra on a fixed grid of nfeat pixels.

    Encoder: four conv layers (stride 1, 2, 2, 2) to (ch4, ceil(nfeat / 8)).
    mlp_hidden=0: linear bottleneck, z <-> flattened conv features (original design).
    mlp_hidden>0: nonlinear bottleneck. A 1x1 conv reduces the features to
        mlp_channels, a two-layer MLP of width mlp_hidden gives mu and logvar; the
        decoder maps z through a two-layer MLP plus a linear skip back to an
        (mlp_channels, L) map and expands it to ch4 with a 1x1 conv.
    Decoder: three transposed convs mirroring the encoder, then a 1-channel conv.
    """

    def __init__(self, nfeat=1000, ncode=16, ch1=32, ch2=64, ch3=128, ch4=256,
                 dropout=0.02, use_bn=True, mlp_hidden=0, mlp_channels=32):
        super().__init__()
        self.nfeat, self.ncode = int(nfeat), int(ncode)
        self.ch4 = ch4
        self.mlp_hidden, self.mlp_channels = int(mlp_hidden), int(mlp_channels)
        self.config = dict(
            nfeat=self.nfeat, ncode=self.ncode, ch1=ch1, ch2=ch2, ch3=ch3, ch4=ch4,
            dropout=dropout, use_bn=use_bn,
            mlp_hidden=self.mlp_hidden, mlp_channels=self.mlp_channels,
        )
        norm = nn.BatchNorm1d if use_bn else (lambda _: nn.Identity())
        act = lambda: nn.LeakyReLU(0.2)

        self.enc = nn.Sequential(
            nn.Conv1d(1, ch1, 9, stride=1, padding=4), norm(ch1), act(),
            nn.Conv1d(ch1, ch2, 7, stride=2, padding=3), norm(ch2), act(),
            nn.Conv1d(ch2, ch3, 5, stride=2, padding=2), norm(ch3), act(),
            nn.Conv1d(ch3, ch4, 3, stride=2, padding=1), norm(ch4), act(),
        )
        self.L_enc = (self.nfeat + 7) // 8
        self.enc_ln = nn.LayerNorm([ch4, self.L_enc])
        self.flat = nn.Flatten()
        self.enc_dropout = nn.Dropout(dropout)

        if self.mlp_hidden:
            H, C = self.mlp_hidden, self.mlp_channels
            map_dim = C * self.L_enc
            self.enc_reduce = nn.Conv1d(ch4, C, 1)
            self.enc_mlp = nn.Sequential(
                nn.Linear(map_dim, H), act(), nn.Dropout(dropout),
                nn.Linear(H, H), act(),
            )
            self.mu = nn.Linear(H, self.ncode)
            self.lv = nn.Linear(H, self.ncode)
            self.fc = nn.Sequential(
                nn.Linear(self.ncode, H), act(),
                nn.Linear(H, H), act(),
                nn.Linear(H, map_dim),
            )
            self.fc_skip = nn.Linear(self.ncode, map_dim, bias=False)
            self.dec_expand = nn.Conv1d(C, ch4, 1)
        else:
            enc_dim = ch4 * self.L_enc
            self.mu = nn.Linear(enc_dim, self.ncode)
            self.lv = nn.Linear(enc_dim, self.ncode)
            self.fc = nn.Linear(self.ncode, enc_dim)

        self.dec_preln = nn.LayerNorm([ch4, self.L_enc])
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(ch4, ch3, 3, stride=2, padding=1, output_padding=1), act(),
            nn.ConvTranspose1d(ch3, ch2, 5, stride=2, padding=2, output_padding=1), act(),
            nn.ConvTranspose1d(ch2, ch1, 7, stride=2, padding=3, output_padding=1), act(),
            nn.Conv1d(ch1, 1, 9, padding=4),
        )

        self.apply(kaiming_init)
        nn.init.zeros_(self.lv.weight)
        nn.init.zeros_(self.lv.bias)

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        """Build a model from a checkpoint dict or path, in eval mode.

        Also loads v3 checkpoints trained with a projection head: the head only fed the
        contrastive loss, so dropping it leaves encode/decode unchanged.
        """
        if not isinstance(checkpoint, dict):
            checkpoint = load_checkpoint(checkpoint)
        config = {k: v for k, v in checkpoint["model_config"].items()
                  if k not in LEGACY_MODEL_KEYS}
        model = cls(**config)
        state = {k: v for k, v in checkpoint["model_state_dict"].items()
                 if not k.startswith("proj.")}
        model.load_state_dict(state)
        return model.to(device).eval()

    def encode(self, x):
        h = self.enc_ln(self.enc(x.unsqueeze(1) if x.ndim == 2 else x))
        if self.mlp_hidden:
            h = self.enc_mlp(self.enc_dropout(self.flat(self.enc_reduce(h))))
        else:
            h = self.enc_dropout(self.flat(h))
        return self.mu(h), self.lv(h).clamp(-8.0, 8.0)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        if self.mlp_hidden:
            h = (self.fc(z) + self.fc_skip(z)).reshape(len(z), self.mlp_channels, self.L_enc)
            h = self.dec_expand(h)
        else:
            h = self.fc(z).reshape(len(z), self.ch4, self.L_enc)
        return self.dec(self.dec_preln(h))[..., :self.nfeat]

    def forward(self, x, sample=None):
        """Returns (recon (B, 1, nfeat), mu, logvar, z). sample=None: sample when training."""
        mu, logvar = self.encode(x)
        sample = self.training if sample is None else sample
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar, z


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# Loss

def _tensor_spectra_2d(x):
    return x[:, 0].float() if x.ndim == 3 else x.float()


def masked_inputs(x, w=None):
    """Zero flux and weight wherever either is non-finite or the weight is <= 0."""
    x = _tensor_spectra_2d(x)
    w = torch.ones_like(x) if w is None else _tensor_spectra_2d(w)
    valid = torch.isfinite(x) & torch.isfinite(w) & (w > 0)
    return torch.where(valid, x, 0.0), torch.where(valid, w, 0.0)


def vae_loss(x, recon, mu, logvar, w=None, beta=1e-3):
    """Weighted MSE + beta * KL, both summed over the batch. Returns (total, wmse, kl).

    wmse = 0.5 * sum(w * (recon - x)**2) over valid pixels. With w = inverse variance
           this is chi^2 / 2, i.e. the Gaussian negative log-likelihood up to a constant.
    kl   = sum over spectra and latents of KL(N(mu, exp(logvar)) || N(0, 1)).
    """
    x, w = masked_inputs(x, w)
    residual = torch.where(w > 0, _tensor_spectra_2d(recon), 0.0) - x
    mu, logvar = mu.float(), logvar.float()
    wmse = 0.5 * (w * residual.square()).sum()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum()
    return wmse + beta * kl, wmse, kl


def reconstruction_scores(x, recon, w=None):
    """Per-spectrum WMSE with weights normalized to sum to 1 (for comparing spectra)."""
    x, w = masked_inputs(x, w)
    residual = torch.where(w > 0, _tensor_spectra_2d(recon), 0.0) - x
    weights = w / w.sum(dim=1, keepdim=True).clamp_min(1e-30)
    return (weights * residual.square()).sum(dim=1)


# HDF5 data (unchanged from v3: same files, keys, validity cuts, split and batch order)

def resolve_hdf5_files(source, recursive=False):
    sources = [source] if isinstance(source, (str, PathLike)) else source
    files = set()

    for item in sources:
        path = Path(item).expanduser()
        candidates = path.rglob("*") if recursive else path.glob("*")
        if path.is_dir():
            files.update(
                str(p.resolve())
                for p in candidates
                if p.is_file() and p.suffix.lower() in {".h5", ".hdf5", ".hdf"}
            )
        else:
            files.add(str(path.resolve()))

    if not files:
        raise FileNotFoundError("No HDF5 files found.")
    return tuple(sorted(files))


def h5_field(handle, key):
    # Supports dataset paths and compound fields such as META::TARGETID.
    dataset, separator, field = key.partition("::")
    return handle[dataset].fields(field) if separator else handle[dataset]


def _first_key(handle, *names):
    """First key whose dataset exists in the file (supports 'a/b' and 'DS::FIELD')."""
    for name in names:
        if name and name.partition("::")[0] in handle:
            return name
    return None


def load_spectra_h5(
    source, flux_key="X", weight_key="W", targetid_key="TARGETID",
    spectype_key="SPECTYPE", mask_key=None, mask_nonzero_valid=True,
    min_valid_pixels=50, nfeat=None, verbose=True, **_ignored,
):
    """Load every spectrum of one HDF5 file into memory (for small sets such as known
    transients), with the same keys and validity rules as H5SpectralBatches.

    A training data_config can be passed as keyword arguments; entries that do not
    apply here (path, split settings, batch sizes, ...) are ignored. If a configured
    key is missing, common alternatives are tried and reported.

    Returns a dict with X and W (invalid pixels set to 0), TARGETID, SPECTYPE,
    n_valid, wave (or None), and all_targetids (every TARGETID in the file, including
    rows dropped by the valid-pixel cut, for excluding them from training).
    """
    with h5py.File(source, "r") as handle:
        keys = {
            "flux": _first_key(handle, flux_key, "X", "flux", "FLUX"),
            "weight": _first_key(handle, weight_key, "W", "ivar", "IVAR"),
            "targetid": _first_key(handle, targetid_key, "TARGETID", "targetid"),
            "spectype": _first_key(handle, spectype_key, "SPECTYPE", "spectype",
                                   "redshift/spectype"),
            "mask": _first_key(handle, mask_key) if mask_key else None,
        }
        for role in ("flux", "weight", "targetid"):
            if keys[role] is None:
                raise KeyError(f"No {role} array found in {source}; pass {role}_key=...")
        x = _spectra_2d(np.asarray(h5_field(handle, keys["flux"])[:]), "flux").astype(np.float32)
        w = _spectra_2d(np.asarray(h5_field(handle, keys["weight"])[:]), "weight").astype(np.float32)
        ids = np.asarray(h5_field(handle, keys["targetid"])[:]).reshape(-1)
        labels = (
            np.full(len(ids), "UNKNOWN", dtype="U7") if keys["spectype"] is None
            else np.char.upper(np.char.strip(
                np.asarray(h5_field(handle, keys["spectype"])[:]).reshape(-1).astype(str)))
        )
        mask = (
            None if keys["mask"] is None
            else _spectra_2d(np.asarray(h5_field(handle, keys["mask"])[:]), "mask") != 0
        )
        wave = np.asarray(handle["wave_grid"][:]) if "wave_grid" in handle else None

    if x.shape != w.shape or len(ids) != len(x):
        raise ValueError(f"Shape mismatch: flux {x.shape}, weight {w.shape}, ids {ids.shape}.")
    if nfeat is not None and x.shape[1] != nfeat:
        raise ValueError(f"Spectra have {x.shape[1]} pixels; the model expects {nfeat}.")
    if ids.dtype.kind == "f":
        raise TypeError("TARGETID must be integer or text; floats lose precision.")

    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask is not None:
        valid &= mask if mask_nonzero_valid else ~mask
    n_valid = valid.sum(axis=1)
    keep = n_valid >= min_valid_pixels
    out = dict(
        X=np.where(valid, x, 0.0).astype(np.float32)[keep],
        W=np.where(valid, w, 0.0).astype(np.float32)[keep],
        TARGETID=ids[keep], SPECTYPE=labels[keep], n_valid=n_valid[keep],
        wave=wave, all_targetids=np.unique(ids), keys=keys,
    )
    if verbose:
        print(f"{Path(source).name}: {len(ids):,} spectra, {keep.sum():,} with >= "
              f"{min_valid_pixels} valid pixels, {len(np.unique(ids[keep])):,} unique TARGETIDs")
        print("  keys used:", {k: v for k, v in keys.items() if v})
        names, counts = np.unique(labels[keep], return_counts=True)
        print("  SPECTYPE counts:", {str(n): int(c) for n, c in zip(names, counts)})
    return out


def compare_to_training(transients, training_file, flux_key="X", weight_key="W",
                        n_rows=5000, **_ignored):
    """Print scale checks between a loaded set (load_spectra_h5) and one training file.

    The preprocessing must match: same wavelength grid, and flux and inverse variance
    on the same scale. Ratios far from 1 mean the transients were normalized
    differently, and their scores cannot be compared with the training spectra.
    """
    with h5py.File(training_file, "r") as handle:
        x = _spectra_2d(np.asarray(h5_field(handle, flux_key)[:n_rows]), "flux")
        w = _spectra_2d(np.asarray(h5_field(handle, weight_key)[:n_rows]), "weight")
        wave = np.asarray(handle["wave_grid"][:]) if "wave_grid" in handle else None
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    tw = transients["W"] > 0
    if wave is not None and transients["wave"] is not None:
        same = wave.shape == transients["wave"].shape and np.allclose(wave, transients["wave"])
        print(f"  wavelength grid identical to training: {same}")
        if not same:
            print("  WARNING: different wavelength grids; the model cannot score these spectra.")
    else:
        print("  wavelength grid: not stored in one of the files, cannot compare")
    checks = {
        "median |X|": (np.median(np.abs(transients["X"][tw])), np.median(np.abs(x[ok]))),
        "median W": (np.median(transients["W"][tw]), np.median(w[ok])),
        "median valid pixels": (np.median(tw.sum(1)), np.median(ok.sum(1))),
    }
    for name, (a, b) in checks.items():
        ratio = a / b if b else float("nan")
        flag = "" if 1 / 3 < ratio < 3 else "   <-- check preprocessing"
        print(f"  {name}: transients {a:.4g} | training {b:.4g} | ratio {ratio:.2f}{flag}")


def inspect_hdf5(path):
    with h5py.File(path, "r") as handle:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(name, obj.shape, obj.dtype)

        handle.visititems(show)


def _spectra_2d(array, name="x"):
    array = np.asarray(array, dtype=np.float32)
    return array[:, 0] if array.ndim == 3 else array


def stable_id_hash(ids, seed=42):
    # Keeps repeated TARGETIDs in the same split across files.
    ids = np.asarray(ids).reshape(-1)

    if ids.dtype.kind in "iu":
        values = ids.astype(np.uint64, copy=True)
        with np.errstate(over="ignore"):
            values += np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
            values = (
                values ^ (values >> np.uint64(30))
            ) * np.uint64(0xBF58476D1CE4E5B9)
            values = (
                values ^ (values >> np.uint64(27))
            ) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))

    if ids.dtype.kind == "f":
        raise TypeError("TARGETID must be integer or text; floats lose precision.")
    if ids.dtype.kind not in "SUO":
        raise TypeError(f"Unsupported TARGETID dtype: {ids.dtype}.")

    result = np.empty(len(ids), dtype=np.uint64)
    for i, value in enumerate(ids):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, (str, int, np.integer)) or not str(value).strip():
            raise ValueError("TARGETID contains a missing or unsupported value.")
        payload = f"{seed}:{value}".encode("utf-8")
        result[i] = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "little"
        )
    return result


SPLIT_CODES = {"train": 0, "val": 1, "holdout": 2}


def split_codes(ids, seed=42, val_fraction=0.10, n_folds=None, fold=None):
    """Assign each TARGETID to a split. 0 = train, 1 = val, 2 = holdout.

    n_folds=None: the original train/val split (hash < val_fraction -> val).
    n_folds=K, fold=k: TARGETIDs with hash % K == k are the holdout fold, which this
    model never sees and later scores. The validation set (for early stopping) is drawn
    from the remaining folds with an independent hash. Repeated observations of one
    TARGETID always land in the same split.
    """
    h = stable_id_hash(ids, seed)
    cutoff = np.uint64(int(val_fraction * (1 << 64)))
    if n_folds is None:
        return np.where(h < cutoff, 1, 0).astype(np.int8)
    holdout = (h % np.uint64(n_folds)) == np.uint64(fold)
    is_val = stable_id_hash(ids, seed + 1) < cutoff
    return np.where(holdout, 2, np.where(is_val, 1, 0)).astype(np.int8)


class H5SpectralBatches(IterableDataset):
    def __init__(
        self, path, flux_key="X", weight_key="W", targetid_key="TARGETID",
        spectype_key="SPECTYPE", valid_rows_key=None, keep_spectypes=None,
        split="train", val_fraction=0.10, seed=42, nfeat=1000,
        batch_size=256, block_rows=4096, min_valid_pixels=50,
        shuffle=False, recursive=False,
        n_folds=None, fold=None, exclude_targetids=None,
        mask_key=None, mask_nonzero_valid=True, mix_batches=1,
    ):
        super().__init__()
        # mix_batches > 1 (shuffled loaders only): pool this many consecutive batches,
        # which span several blocks/files, shuffle the rows and re-batch. Needed when a
        # single block holds mostly one SPECTYPE, e.g. for the contrastive loss.
        self.mix_batches = max(1, int(mix_batches))
        if split not in ("train", "val", "holdout", "all"):
            raise ValueError("split must be train, val, holdout or all.")
        if n_folds is not None and not (n_folds >= 2 and 0 <= int(fold) < n_folds):
            raise ValueError("Cross-fitting needs n_folds >= 2 and 0 <= fold < n_folds.")
        if split == "holdout" and n_folds is None:
            raise ValueError("split='holdout' needs n_folds and fold.")
        self.n_folds, self.fold = n_folds, None if fold is None else int(fold)
        # Known anomalies (e.g. transients) are removed from train/val only; they are
        # still yielded for split="holdout"/"all" so they get out-of-sample scores.
        self.exclude_targetids = (
            None if exclude_targetids is None else np.unique(np.asarray(exclude_targetids))
        )
        # Optional pixel mask. mask_nonzero_valid=True: M != 0 marks valid pixels.
        self.mask_key, self.mask_nonzero_valid = mask_key, mask_nonzero_valid
        self.paths = resolve_hdf5_files(path, recursive)
        self.path = self.paths[0] if len(self.paths) == 1 else self.paths
        self.n_files = len(self.paths)

        self.flux_key, self.weight_key = flux_key, weight_key
        self.targetid_key, self.spectype_key = targetid_key, spectype_key
        self.valid_rows_key = valid_rows_key
        self.keep_spectypes = (
            None if keep_spectypes is None
            else tuple(str(s).strip().upper() for s in keep_spectypes)
        )

        self.split, self.val_fraction, self.seed = split, val_fraction, seed
        self.nfeat, self.batch_size = nfeat, batch_size
        self.block_rows, self.min_valid_pixels = block_rows, min_valid_pixels
        self.shuffle, self.epoch = shuffle, 0

        self.n_rows_by_file = []
        for path in self.paths:
            with h5py.File(path, "r") as handle:
                self.n_rows_by_file.append(h5_field(handle, flux_key).shape[0])

        self.n_rows = sum(self.n_rows_by_file)
        self.file_manifest = [
            dict(path=p, rows=n)
            for p, n in zip(self.paths, self.n_rows_by_file)
        ]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        worker = get_worker_info()
        worker_id, n_workers = (
            (0, 1) if worker is None else (worker.id, worker.num_workers)
        )

        order_rng = np.random.default_rng(self.seed + self.epoch)
        row_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, worker_id])
        )

        file_order = np.arange(self.n_files)
        if self.shuffle:
            order_rng.shuffle(file_order)

        mix_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, worker_id, 1])
        )
        batches = self._iter_batches(file_order, worker_id, n_workers, order_rng, row_rng)
        if self.shuffle and self.mix_batches > 1:
            batches = self._mix(batches, mix_rng)
        yield from batches

    def _mix(self, batches, rng):
        pool = []
        for batch in batches:
            pool.append(batch)
            if len(pool) >= self.mix_batches:
                pool = yield from self._reshuffle(pool, rng, keep_remainder=True)
        if pool:
            yield from self._reshuffle(pool, rng, keep_remainder=False)

    def _reshuffle(self, pool, rng, keep_remainder):
        """Shuffle the pooled rows and yield full batches. The leftover rows are returned
        as the start of the next pool, so partial batches only occur at the very end."""
        x = torch.cat([b[0] for b in pool])
        w = torch.cat([b[1] for b in pool])
        labels = np.concatenate([b[2] for b in pool])
        ids = np.concatenate([b[3] for b in pool])
        perm = rng.permutation(len(ids))
        n_full = len(perm) - len(perm) % self.batch_size if keep_remainder else len(perm)
        for offset in range(0, n_full, self.batch_size):
            rows = perm[offset:offset + self.batch_size]
            index = torch.from_numpy(rows)
            yield x[index], w[index], labels[rows], ids[rows]
        rest = perm[n_full:]
        if not len(rest):
            return []
        index = torch.from_numpy(rest)
        return [(x[index], w[index], labels[rest], ids[rest])]

    def _iter_batches(self, file_order, worker_id, n_workers, order_rng, row_rng):
        block_offset = 0

        for file_index in file_order:
            n_rows = self.n_rows_by_file[file_index]
            starts = np.arange(0, n_rows, self.block_rows, dtype=np.int64)
            if self.shuffle:
                order_rng.shuffle(starts)

            first_block = (worker_id - block_offset) % n_workers
            block_offset += len(starts)
            starts = starts[first_block::n_workers]

            if len(starts):
                yield from self._iter_file_batches(
                    self.paths[file_index], n_rows, starts, row_rng
                )

    def _iter_file_batches(self, filename, n_rows, starts, row_rng):
        with h5py.File(filename, "r") as handle:
            flux_ds = h5_field(handle, self.flux_key)
            weights_ds = (
                None if self.weight_key is None
                else h5_field(handle, self.weight_key)
            )
            ids_ds = h5_field(handle, self.targetid_key)
            labels_ds = (
                None if self.spectype_key is None
                else h5_field(handle, self.spectype_key)
            )
            valid_rows_ds = (
                None if self.valid_rows_key is None
                else h5_field(handle, self.valid_rows_key)
            )
            mask_ds = None if self.mask_key is None else h5_field(handle, self.mask_key)

            for start in starts:
                stop = min(int(start) + self.block_rows, n_rows)
                ids = np.asarray(ids_ds[start:stop]).reshape(-1)
                labels = (
                    np.full(len(ids), "UNKNOWN", dtype="U7")
                    if labels_ds is None
                    else np.char.upper(
                        np.char.strip(
                            np.asarray(labels_ds[start:stop]).reshape(-1).astype(str)
                        )
                    )
                )

                keep = np.ones(len(ids), dtype=bool)
                if self.split != "all":
                    codes = split_codes(
                        ids, self.seed, self.val_fraction, self.n_folds, self.fold
                    )
                    keep &= codes == SPLIT_CODES[self.split]
                if self.exclude_targetids is not None and self.split in ("train", "val"):
                    keep &= ~np.isin(ids, self.exclude_targetids)
                if self.keep_spectypes is not None:
                    keep &= np.isin(labels, self.keep_spectypes)
                if valid_rows_ds is not None:
                    flags = np.asarray(valid_rows_ds[start:stop]).reshape(-1)
                    keep &= flags.astype(bool)
                if not keep.any():
                    continue

                x = _spectra_2d(flux_ds[start:stop])
                w = (
                    np.ones_like(x) if weights_ds is None
                    else _spectra_2d(weights_ds[start:stop])
                )
                valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
                if mask_ds is not None:
                    m = _spectra_2d(mask_ds[start:stop]) != 0
                    valid &= m if self.mask_nonzero_valid else ~m
                keep &= valid.sum(axis=1) >= self.min_valid_pixels

                selected = np.flatnonzero(keep)
                if not len(selected):
                    continue
                if self.shuffle:
                    row_rng.shuffle(selected)

                # Select the kept rows first, then mask: same batches as before, but no
                # work on rows of other splits.
                v = valid[selected]
                x = np.where(v, x[selected], 0.0).astype(np.float32, copy=False)
                w = np.where(v, w[selected], 0.0).astype(np.float32, copy=False)

                for offset in range(0, len(selected), self.batch_size):
                    part = slice(offset, offset + self.batch_size)
                    rows = selected[part]
                    yield (
                        torch.from_numpy(np.ascontiguousarray(x[part])),
                        torch.from_numpy(np.ascontiguousarray(w[part])),
                        labels[rows],
                        ids[rows],
                    )


def passthrough_batch(batch):
    return batch


def make_h5_loader(dataset, device, num_workers=0):
    """Batches come pre-built from the dataset. With num_workers > 0 the workers read
    blocks in parallel; the DataLoader returns their batches in a fixed round-robin
    order, so the sequence is reproducible for a given num_workers."""
    options = dict(
        batch_size=None, num_workers=num_workers, collate_fn=passthrough_batch,
        pin_memory=(torch.device(device).type == "cuda"),
    )
    if num_workers:
        options.update(prefetch_factor=4, persistent_workers=False)
    return DataLoader(dataset, **options)


def shard_files(data_config, split):
    """Files holding `split` of this fold, from a manifest written by repack_folds.py.

    Shard files contain one fold and one split each, so a loader only opens the files it
    needs: train and val read the shards of the other folds, holdout reads fold k's.
    """
    manifest_path = Path(data_config["shard_manifest"]).expanduser()
    manifest = json.loads(manifest_path.read_text())
    made = manifest["config"]
    for key in ("n_folds", "seed", "val_fraction"):
        if made[key] != data_config.get(key):
            raise ValueError(f"{key}={data_config.get(key)} but the shards were made with "
                             f"{made[key]}; repack or change the setting.")
    if data_config.get("min_valid_pixels", 0) < made["min_valid_pixels"]:
        raise ValueError("min_valid_pixels is below the repack cut "
                         f"({made['min_valid_pixels']}); those rows are not in the shards.")
    k = data_config["fold"]
    if split in ("train", "val"):
        chosen = [s for s in manifest["shards"] if s["split"] == split and s["fold"] != k]
    elif split == "holdout":
        chosen = [s for s in manifest["shards"] if s["fold"] == k]
    else:
        chosen = manifest["shards"]
    return [str(manifest_path.parent / s["file"]) for s in chosen]


def make_split_loader(data_config, split, device, num_workers=0, shuffle=None):
    """Loader for one split ("train", "val", "holdout" or "all") of a data_config.

    With data_config["shard_manifest"], only the shard files of that split are read.
    The loader still applies the TARGETID hash split, as a free consistency check.
    """
    params = inspect.signature(H5SpectralBatches).parameters
    cfg = {k: v for k, v in data_config.items() if k in params}
    if data_config.get("shard_manifest"):
        cfg["path"] = shard_files(data_config, split)
    shuffle = (split == "train") if shuffle is None else shuffle
    dataset = H5SpectralBatches(**{**cfg, "split": split, "shuffle": shuffle})
    return make_h5_loader(dataset, device, num_workers)


def make_loaders(data_config, device, num_workers=0):
    """Shuffled train loader and ordered validation loader for one data_config."""
    return (make_split_loader(data_config, "train", device, num_workers),
            make_split_loader(data_config, "val", device, num_workers))


# Training

class TrainingStopped(Exception):
    """Raised after a stop request (Ctrl-C, SIGTERM, SIGUSR1); latest.pt is up to date."""

    def __init__(self, signal_name, message):
        super().__init__(message)
        self.signal_name = signal_name


class StopOnSignal:
    """Turn SIGINT/SIGTERM/SIGUSR1 into a request to stop at the next batch boundary.

    Stopping between batches keeps the checkpoint consistent (never half an optimizer
    step). A second Ctrl-C stops immediately without saving.
    """

    NAMES = ("SIGINT", "SIGTERM", "SIGUSR1")

    def __init__(self):
        self.requested = None
        self._previous = {}
        self._pid = os.getpid()

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for name in self.NAMES:
                sig = getattr(signal, name, None)
                if sig is not None:
                    self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def _handle(self, signum, frame):
        if os.getpid() != self._pid:      # forked DataLoader worker: the main process decides
            return
        if self.requested and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.requested = signal.Signals(signum).name
        print(f"\n{self.requested} received: saving after the current batch "
              "(press Ctrl-C again to stop without saving).", flush=True)

    def __exit__(self, *exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        return False


def _rng_state(device):
    cuda = torch.device(device).type == "cuda"
    return dict(torch_rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state(device) if cuda else None)


def _restore_rng(state, device):
    if state and state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.device(device).type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device=device)


SUM_KEYS = ("loss", "rec", "kl", "nvalid", "clipped")


def run_epoch(model, loader, device, beta, optimizer=None, max_grad_norm=3e7,
              log_every=0, progress=None, save_progress=None, checkpoint_seconds=None,
              stop=None, expected_batches=None, lr_at=None):
    """One pass over loader; trains if an optimizer is given, else evaluates at z = mu.

    progress: saved mid-epoch position ({n, batches, sums, rng}); those batches are
        re-read without training, then the RNG is restored so the run continues exactly.
    save_progress(progress): called every checkpoint_seconds and on a stop request.
    lr_at(batch): learning rate for the batch with this index within the epoch (counting
        re-read batches), set just before each optimizer step. None: leave it as it is.
    Returns per-spectrum means of loss, rec (the WMSE term) and kl, plus rchi2.
    """
    training = optimizer is not None
    model.train(training)
    device = torch.device(device)
    sums = torch.zeros(len(SUM_KEYS), dtype=torch.float64, device=device)
    n = nbatches = 0
    batches = iter(loader)

    if progress:
        n, nbatches = int(progress["n"]), int(progress["batches"])
        sums = torch.tensor([progress["sums"][k] for k in SUM_KEYS],
                            dtype=torch.float64, device=device)
        print(f"Re-reading the {nbatches:,} batches already trained this epoch "
              "(no updates)...", flush=True)
        for _ in range(nbatches):
            if next(batches, None) is None:
                raise ValueError("Saved position is past the end of the data; did the files change?")
            if stop is not None and stop.requested:
                raise TrainingStopped(stop.requested, "Stopped while re-reading; latest.pt unchanged.")
        _restore_rng(progress["rng"], device)

    def snapshot():
        values = sums.tolist()
        return dict(n=n, batches=nbatches, sums=dict(zip(SUM_KEYS, values)),
                    rng=_rng_state(device))

    last_save = started = time.monotonic()
    n_start, b_start = n, nbatches
    norms = []          # gradient norm of every batch trained in this session (on the GPU)
    with torch.set_grad_enabled(training):
        for x, w, _, _ in batches:
            x, w = masked_inputs(x.to(device, non_blocking=True), w.to(device, non_blocking=True))
            recon, mu, logvar, _ = model(x, sample=training)
            total, rec, kl = vae_loss(x, recon, mu, logvar, w, beta)
            clipped = torch.zeros((), device=device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                # Raises on a non-finite gradient, before the weights are touched.
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                                error_if_nonfinite=True)
                clipped = norm > max_grad_norm
                norms.append(norm.detach().float())
                if lr_at is not None:
                    lr = lr_at(nbatches)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                optimizer.step()
            sums += torch.stack([total.detach(), rec.detach(), kl.detach(),
                                 (w > 0).sum(), clipped]).double()
            n += len(x)
            nbatches += 1

            if training and log_every and nbatches % log_every == 0:
                s = dict(zip(SUM_KEYS, sums.tolist()))
                rate = (nbatches - b_start) / max(time.monotonic() - started, 1e-9)
                eta = (f" eta={(expected_batches - nbatches) / rate / 60:.0f}min"
                       if expected_batches and expected_batches > nbatches else "")
                print(f"  batches={nbatches:,} spectra={n:,} rec={s['rec'] / n:.6g} "
                      f"kl={s['kl'] / n:.6g} rchi2={2 * s['rec'] / max(s['nvalid'], 1):.4g} "
                      f"clipped={s['clipped'] / nbatches:.0%} "
                      f"grad_norm p50={_quantiles(norms[-log_every:], (0.5,))[0]:.3g} "
                      f"lr={optimizer.param_groups[0]['lr']:.3g} "
                      f"{(n - n_start) / (time.monotonic() - started):,.0f} spectra/s{eta}",
                      flush=True)

            stopping = stop is not None and stop.requested
            if training and save_progress and (
                    stopping or time.monotonic() - last_save >= checkpoint_seconds):
                save_progress(snapshot())
                last_save = time.monotonic()
            if stopping:
                where = "saved to latest.pt" if training and save_progress else "not saved"
                raise TrainingStopped(stop.requested, f"Stopped at batch {nbatches:,} ({where}). "
                                      "Rerun to resume.")

    if not n:
        raise ValueError("No usable spectra in this split.")
    s = dict(zip(SUM_KEYS, sums.tolist()))
    if not all(np.isfinite(v) for v in s.values()):
        raise FloatingPointError(f"Non-finite loss: {s}")
    return {
        "loss": s["loss"] / n, "rec": s["rec"] / n, "kl": s["kl"] / n,
        # rec = chi^2 / 2, so chi^2 per valid pixel = 2 rec / n_valid (~1 at noise level)
        "rchi2": 2 * s["rec"] / max(s["nvalid"], 1),
        "clip_frac": s["clipped"] / nbatches if training else None,
        "n": n, "batches": nbatches,
        **(dict(zip(GRAD_NORM_KEYS, _quantiles(norms, GRAD_NORM_QUANTILES)))
           if training and norms else {}),
    }


# Gradient-norm quantiles recorded per epoch (as train_grad_norm_p50 ... in history.json).
# They cover the batches trained in the session that finished the epoch.
GRAD_NORM_QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)
GRAD_NORM_KEYS = ("grad_norm_p50", "grad_norm_p90", "grad_norm_p99", "grad_norm_p999",
                  "grad_norm_max")


def _quantiles(values, qs):
    if not values:
        return [float("nan")] * len(qs)
    v = torch.stack(list(values))
    return torch.quantile(v, torch.tensor(qs, device=v.device, dtype=v.dtype)).tolist()


LR_SCHEDULES = ("plateau", "cosine", "wsd")
# Defaults of the schedule settings; checkpoints from before they existed used these.
LR_DEFAULTS = dict(lr_schedule="plateau", lr_warmup_steps=0, lr_final_frac=0.02,
                   lr_decay_frac=0.2)


def lr_factor(schedule, step, total_steps, warmup_steps=0, final_frac=0.02, decay_frac=0.2):
    """Learning rate / peak learning rate at optimizer step `step` (0-based).

    cosine: linear warm-up over warmup_steps, then cosine decay to final_frac at total_steps.
    wsd:    warm-up, constant, then linear decay to final_frac over the last decay_frac
            of the steps ("warmup-stable-decay").
    plateau: 1 (ReduceLROnPlateau decides per epoch instead).
    """
    if schedule == "plateau":
        return 1.0
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "cosine":
        t = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * t))
    if schedule == "wsd":
        decay_start = max(warmup_steps, total_steps * (1 - decay_frac))
        if step < decay_start:
            return 1.0
        t = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
        return 1.0 - (1 - final_frac) * t
    raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, not {schedule!r}")


# Settings that change the result: resuming with different values is refused.
# With a scheduled learning rate, epochs is locked too (it sets the schedule's length).
RESUME_LOCKED = ("learning_rate", "beta_max", "warmup_epochs", "max_grad_norm",
                 *LR_DEFAULTS)


def locked_train_settings(train_config):
    """The training settings that must not change on resume (or between folds)."""
    locked = {k: train_config.get(k, LR_DEFAULTS.get(k)) for k in RESUME_LOCKED}
    if locked["lr_schedule"] != "plateau":
        locked["epochs"] = train_config.get("epochs")
    return locked
PHASES = ("train", "validate", "epoch_complete")


def _load_resume_checkpoint(out_dir):
    latest = load_checkpoint(out_dir / "latest.pt")
    best_path = out_dir / "best.pt"
    # best.pt is written just before latest.pt at the end of an epoch. If the job died
    # in between, best.pt is the newer one.
    if best_path.is_file() and latest["phase"] != "epoch_complete":
        best = load_checkpoint(best_path)
        if best["epoch"] == latest["epoch"]:
            return best
    return latest


def _check_resume(ckpt, model, data_config, train_config, loader_config, train_loader):
    problems = []
    if ckpt.get("loss_config") != LOSS_CONFIG:
        problems.append("loss definition")
    if ckpt["model_config"] != model.config:
        problems.append(f"model_config (saved {ckpt['model_config']})")
    if ckpt["data_config"] != data_config:
        problems.append("data_config")
    if ckpt.get("data_manifest") != train_loader.dataset.file_manifest:
        problems.append("HDF5 files or row counts")
    if ckpt.get("loader_config") != loader_config:
        problems.append(f"num_workers (saved {ckpt.get('loader_config')})")
    saved, now = locked_train_settings(ckpt["train_config"]), locked_train_settings(train_config)
    for key in sorted(set(saved) | set(now)):
        if saved.get(key) != now.get(key):
            problems.append(f"{key} (saved {saved.get(key)})")
    if problems:
        raise ValueError("Cannot resume: these differ from the saved run: "
                         + "; ".join(problems) + ". Use a new run folder to change them.")


STOP_FILE = "STOP_AFTER_EPOCH"   # create this file in a run folder to stop after the epoch


def _stop_at_epoch_end(out_dir, epoch, history, deadline):
    """Raise TrainingStopped after a completed epoch if a stop file or deadline says so."""
    stop_file = out_dir / STOP_FILE
    if stop_file.exists():
        stop_file.unlink(missing_ok=True)
        raise TrainingStopped("stop_file", f"Stopped after epoch {epoch} ({STOP_FILE}); "
                                           "latest.pt saved. Rerun to continue.")
    if deadline is None:
        return
    recent = [r["session_minutes"] for r in history[-2:] if r.get("session_minutes")]
    need = 1.1 * 60 * max(recent) + 120 if recent else None   # seconds, with margin
    left = deadline - time.time()
    if need is not None and left < need:
        raise TrainingStopped("deadline", f"Stopped after epoch {epoch}: {left / 60:.0f} min "
                                          f"left, an epoch takes ~{need / 60:.0f} min. "
                                          "latest.pt saved. Rerun to continue.")


def fit_vae(model, train_loader, val_loader, device, out_dir, data_config,
            epochs=30, learning_rate=3e-4, beta_max=1e-3, warmup_epochs=3, patience=5,
            max_grad_norm=3e7, log_every=2000, checkpoint_minutes=20,
            lr_schedule="plateau", lr_warmup_steps=0, lr_final_frac=0.02, lr_decay_frac=0.2,
            deadline=None):
    """Train with loss = WMSE + beta * KL, resuming from out_dir/latest.pt if present.

    beta ramps linearly to beta_max over warmup_epochs. Validation uses z = mu and
    beta_max every epoch, so val_loss is comparable across epochs; the epoch with the
    lowest val_loss is saved as best.pt and loaded into `model` at the end. Training
    stops after `patience` epochs without improvement (counted after warm-up).

    Learning rate (lr_schedule):
    * "plateau" (default, as before): halves after 3 epochs without improvement.
    * "cosine": linear warm-up over lr_warmup_steps optimizer steps, then cosine decay to
      lr_final_frac * learning_rate at the end of epoch `epochs`.
    * "wsd": warm-up, constant, then linear decay over the last lr_decay_frac of the steps.
    The scheduled rates are a function of (epoch, batch) only, so they resume exactly.
    The length of an epoch in steps is taken from epoch 1 once it is done (estimated from
    the row count during epoch 1). With a scheduled rate, set patience >= epochs so that
    early stopping does not cut the decay short.

    Raises TrainingStopped after Ctrl-C / SIGTERM / SIGUSR1, with latest.pt saved.
    Also stops cleanly at an epoch boundary, so the next resume starts a fresh epoch
    instead of re-reading the batches already trained:
    * when out_dir/STOP_AFTER_EPOCH exists (the file is removed), signal_name "stop_file";
    * when deadline (a time.time() value, e.g. the end of a Jupyter session or Slurm job)
      would arrive before another epoch could finish, signal_name "deadline".
    deadline is not a training setting: it can differ between resumes.
    Returns the per-epoch history (also written to out_dir/history.json).
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)
    if lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, not {lr_schedule!r}")
    train_config = dict(epochs=epochs, learning_rate=learning_rate, beta_max=beta_max,
                        warmup_epochs=warmup_epochs, patience=patience,
                        max_grad_norm=max_grad_norm, log_every=log_every,
                        checkpoint_minutes=checkpoint_minutes, lr_schedule=lr_schedule,
                        lr_warmup_steps=lr_warmup_steps, lr_final_frac=lr_final_frac,
                        lr_decay_frac=lr_decay_frac)
    loader_config = dict(train_num_workers=train_loader.num_workers,
                         val_num_workers=val_loader.num_workers)
    history, best_val, bad_epochs = [], float("inf"), 0

    def save(state, name):
        tmp = out_dir / (name + ".tmp")
        torch.save(state, tmp)
        tmp.replace(out_dir / name)

    def snapshot(epoch, phase, progress=None, train_metrics=None):
        return {
            "checkpoint_version": 5, "phase": phase, "epoch": epoch,
            "loss_config": dict(LOSS_CONFIG),
            "model_state_dict": model.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
            "sch_state_dict": scheduler.state_dict(),
            "best_val": best_val, "bad_epochs": bad_epochs, "history": history,
            "model_config": model.config, "data_config": data_config,
            "data_manifest": train_loader.dataset.file_manifest,
            "loader_config": loader_config, "train_config": train_config,
            "train_progress": progress, "train_metrics": train_metrics,
            **_rng_state(device),
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "selection_metric": "WMSE at z = mu + beta_max * KL, validation split",
        }

    phase, start_epoch, progress, train_metrics = "epoch_complete", 1, None, None
    if (out_dir / "latest.pt").exists():
        ckpt = _load_resume_checkpoint(out_dir)
        _check_resume(ckpt, model, data_config, train_config, loader_config, train_loader)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["opt_state_dict"])
        scheduler.load_state_dict(ckpt["sch_state_dict"])
        history, best_val, bad_epochs = list(ckpt["history"]), ckpt["best_val"], ckpt["bad_epochs"]
        phase = ckpt["phase"]
        start_epoch = ckpt["epoch"] + (phase == "epoch_complete")
        progress = ckpt["train_progress"] if phase == "train" else None
        train_metrics = ckpt["train_metrics"] if phase == "validate" else None
        _restore_rng(ckpt, device)
        where = f", batch {progress['batches']:,}" if progress else ""
        print(f"Resuming {out_dir.name}: epoch {start_epoch}{where} ({phase} phase).")
        del ckpt
    else:
        save(snapshot(0, "epoch_complete"), "latest.pt")
    if (out_dir / STOP_FILE).exists():       # left over from an earlier session
        (out_dir / STOP_FILE).unlink(missing_ok=True)
        print(f"Removed an old {STOP_FILE} file.")
    print("Run folder:", out_dir, flush=True)

    scheduled = lr_schedule != "plateau"
    if scheduled:
        if patience < epochs:
            print(f"note: patience={patience} < epochs={epochs}; early stopping may cut the "
                  "learning-rate decay short.")
        ds = train_loader.dataset
        k, vf = data_config.get("n_folds"), data_config.get("val_fraction", 0.1)
        frac = 1.0 if data_config.get("shard_manifest") else (1 - 1 / k if k else 1.0) * (1 - vf)
        est_batches = max(1, math.ceil(ds.n_rows * frac / ds.batch_size))

        def lr_for(epoch):
            per_epoch = history[0]["train_batches"] if history else est_batches
            total = epochs * per_epoch

            def lr_at(batch):
                step = (epoch - 1) * per_epoch + batch
                return learning_rate * lr_factor(lr_schedule, step, total, lr_warmup_steps,
                                                 lr_final_frac, lr_decay_frac)
            return lr_at

    started = time.monotonic()
    with StopOnSignal() as stop:
        for epoch in range(start_epoch, epochs + 1):
            if bad_epochs >= patience:
                break
            epoch_started = time.monotonic()
            beta = beta_max * min(1.0, epoch / max(1, warmup_epochs))

            if phase != "validate":
                train_loader.dataset.set_epoch(epoch)

                def save_progress(p, epoch=epoch):
                    save(snapshot(epoch, "train", progress=p), "latest.pt")
                    print(f"  checkpoint: epoch {epoch}, batch {p['batches']:,}", flush=True)

                train_metrics = run_epoch(
                    model, train_loader, device, beta, optimizer, max_grad_norm, log_every,
                    progress=progress, save_progress=save_progress,
                    checkpoint_seconds=60 * checkpoint_minutes, stop=stop,
                    expected_batches=history[-1]["train_batches"] if history else None,
                    lr_at=lr_for(epoch) if scheduled else None)
                save(snapshot(epoch, "validate", train_metrics=train_metrics), "latest.pt")
            phase, progress = "train", None

            val_metrics = run_epoch(model, val_loader, device, beta_max, stop=stop)
            history.append({"epoch": epoch, "beta": beta, "lr": optimizer.param_groups[0]["lr"],
                            **{f"train_{k}": v for k, v in train_metrics.items()},
                            **{f"val_{k}": v for k, v in val_metrics.items()},
                            "session_minutes": (time.monotonic() - epoch_started) / 60})

            improved = val_metrics["loss"] < best_val
            if improved:
                best_val, bad_epochs = val_metrics["loss"], 0
            elif epoch > warmup_epochs:
                bad_epochs += 1
            if epoch >= warmup_epochs and not scheduled:
                scheduler.step(val_metrics["loss"])

            ckpt = snapshot(epoch, "epoch_complete")
            if improved:
                save(ckpt, "best.pt")
            save(ckpt, "latest.pt")
            tmp = out_dir / "history.json.tmp"
            tmp.write_text(json.dumps(history, indent=2))
            tmp.replace(out_dir / "history.json")

            print(f"epoch={epoch:03d} beta={beta:.3g} train_rchi2={train_metrics['rchi2']:.4g} "
                  f"val_rchi2={val_metrics['rchi2']:.4g} val_kl={val_metrics['kl']:.5g} "
                  f"val_loss={val_metrics['loss']:.6g}{' *' if improved else ''} "
                  f"clipped={train_metrics['clip_frac'] or 0:.0%} "
                  f"grad_norm p50/p99={train_metrics.get('grad_norm_p50', float('nan')):.3g}"
                  f"/{train_metrics.get('grad_norm_p99', float('nan')):.3g} "
                  f"lr={optimizer.param_groups[0]['lr']:.3g} "
                  f"time={(time.monotonic() - started) / 60:.1f}min", flush=True)
            if bad_epochs >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break
            if epoch < epochs:
                _stop_at_epoch_end(out_dir, epoch, history, deadline)

    if (out_dir / "best.pt").exists():
        best = load_checkpoint(out_dir / "best.pt")
        model.load_state_dict(best["model_state_dict"])
        print(f"Restored best checkpoint (epoch {best['epoch']}).")
    model.eval()
    return history


# Scoring

@torch.inference_mode()
def score_spectra(model, loader, device):
    """Deterministic (z = mu) scores for every spectrum the loader yields.

    chi2  = sum(ivar * residual**2) over valid pixels
    rchi2 = chi2 / n_valid  (~1 for a noise-level fit)
    snr   = median(x * sqrt(ivar)) over valid pixels
    """
    model.eval()
    out = {k: [] for k in ("TARGETID", "SPECTYPE", "mu", "chi2", "n_valid", "snr")}
    for x, w, spectype, targetid in loader:
        x, w = masked_inputs(x.to(device), w.to(device))
        recon, mu, _, _ = model(x, sample=False)
        residual = _tensor_spectra_2d(recon) - x
        valid = w > 0
        out["chi2"].append((w * residual.square()).sum(1).cpu())
        out["n_valid"].append(valid.sum(1).cpu())
        out["snr"].append(torch.where(valid, x * w.sqrt(), torch.nan).nanmedian(dim=1).values.cpu())
        out["mu"].append(mu.float().cpu())
        out["TARGETID"].append(np.asarray(targetid))
        out["SPECTYPE"].append(np.asarray(spectype).astype(str))
    if not out["chi2"]:
        raise ValueError("No spectra to score.")
    scores = {k: (np.concatenate(v) if k in ("TARGETID", "SPECTYPE") else torch.cat(v).numpy())
              for k, v in out.items()}
    scores["rchi2"] = scores["chi2"] / np.maximum(scores["n_valid"], 1)
    return scores


def latent_usage(mu, active_var=0.01):
    """Active units (var(mu) > active_var) and PCA directions for 95% / 99% of variance."""
    mu = np.asarray(mu, dtype=np.float64)
    eig = np.clip(np.linalg.eigvalsh(np.cov(mu.T))[::-1], 0, None)
    cum = np.cumsum(eig) / eig.sum()
    return dict(active_units=int((mu.var(0) > active_var).sum()),
                dirs_95=int(np.argmax(cum >= 0.95)) + 1, dirs_99=int(np.argmax(cum >= 0.99)) + 1)


# Plots

def plot_vae_history(history):
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in history]
    best = min(history, key=lambda r: r["val_loss"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
    for ax, key, label, log in zip(axes, ("rchi2", "kl", "lr"),
                                   ("Reduced χ² (WMSE)", "KL per spectrum", "Learning rate"),
                                   (True, False, True)):
        if key == "lr":
            ax.plot(epochs, [r["lr"] for r in history], ".-")
        else:
            ax.plot(epochs, [r[f"train_{key}"] for r in history], ".-", label="train")
            ax.plot(epochs, [r[f"val_{key}"] for r in history], ".-", label="validation")
            ax.legend()
        ax.axvline(best["epoch"], color="r", ls="--", lw=0.8)
        ax.set(xlabel="epoch", ylabel=label, yscale="log" if log else "linear")
    plt.show()
    return fig


@torch.inference_mode()
def plot_vae_reconstructions(model, loader, device, n=6):
    import matplotlib.pyplot as plt
    model.eval()
    x, w, labels, ids = next(iter(loader))
    x, w = masked_inputs(x.to(device), w.to(device))
    recon = _tensor_spectra_2d(model(x, sample=False)[0])
    score = reconstruction_scores(x, recon, w)
    n = min(n, len(x))
    rows = (n + 1) // 2
    fig = plt.figure(figsize=(12, 2.8 * rows), constrained_layout=True)
    for i in range(n):
        ax = fig.add_subplot(rows, 2, i + 1)
        valid = (w[i] > 0).cpu().numpy()
        ax.plot(np.where(valid, x[i].cpu().numpy(), np.nan), color="0.6", lw=0.7, label="input")
        ax.plot(np.where(valid, recon[i].cpu().numpy(), np.nan), color="#CC6677", lw=1,
                label="reconstruction")
        ax.set(title=f"{labels[i]} | {ids[i]} | WMSE {score[i].item():.4g}",
               xlabel="wavelength bin", ylabel="preprocessed flux")
        ax.legend(fontsize=8)
    plt.show()
    return fig


# Self-test

def self_test(device="cpu"):
    """Shapes, loss value, gradients, config round trip and legacy checkpoint loading."""
    torch.manual_seed(0)
    x = torch.randn(16, 64, device=device)
    w = torch.rand(16, 64, device=device)
    w[:, :5] = 0
    for mlp_hidden in (0, 16):
        model = VAE(nfeat=64, ncode=4, ch1=4, ch2=4, ch3=8, ch4=8,
                    mlp_hidden=mlp_hidden, mlp_channels=3).to(device)
        recon, mu, logvar, z = model(x, sample=True)
        assert recon.shape == (16, 1, 64) and mu.shape == logvar.shape == z.shape == (16, 4)
        total, wmse, kl = vae_loss(x, recon, mu, logvar, w, beta=0.5)
        r = recon[:, 0]
        ref_wmse = 0.5 * (w * (r - x) ** 2).sum()
        ref_kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum()
        assert torch.allclose(wmse, ref_wmse, rtol=1e-5) and torch.allclose(kl, ref_kl, rtol=1e-5)
        assert torch.allclose(total, ref_wmse + 0.5 * ref_kl, rtol=1e-5)
        total.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert VAE(**model.config).config == model.config
        # A v3-style checkpoint with a projection head loads, and gives the same output
        ckpt = dict(model_config={**model.config, "proj_dim": 6, "proj_hidden": 8},
                    model_state_dict={**model.state_dict(),
                                      "proj.0.weight": torch.zeros(8, 4)})
        loaded = VAE.from_checkpoint(ckpt, device)
        model.eval()
        assert torch.allclose(loaded(x, sample=False)[0], model(x, sample=False)[0])
    # Masked pixels contribute nothing, whatever the reconstruction there
    recon2 = recon.detach().clone()
    recon2[..., :5] = 1e6
    assert torch.allclose(vae_loss(x, recon2, mu, logvar, w)[1], vae_loss(x, recon, mu, logvar, w)[1])
    print("self_test passed")
