from pathlib import Path
from os import PathLike
import hashlib
import json
import random
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


LOSS_CONFIG = {
    "name": "direct_ivar_batch_sum_v1",
    "reconstruction": "0.5 * sum_batch_pixels(ivar * residual**2)",
    "kl": "sum_batch_latent",
    "weight_power": 1.0,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Model

def kaiming_init(module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.kaiming_uniform_(module.weight, a=0.2, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VAE(nn.Module):
    def __init__(
        self, nfeat=1000, ncode=16, ch1=32, ch2=64,
        ch3=128, ch4=256, dropout=0.02, use_bn=True,
        proj_dim=0, proj_hidden=64,
    ):
        super().__init__()
        self.nfeat, self.ncode = int(nfeat), int(ncode)
        self.ch1, self.ch2, self.ch3, self.ch4 = ch1, ch2, ch3, ch4
        self.config = dict(
            nfeat=self.nfeat, ncode=self.ncode,
            ch1=ch1, ch2=ch2, ch3=ch3, ch4=ch4,
            dropout=dropout, use_bn=use_bn,
            proj_dim=int(proj_dim), proj_hidden=int(proj_hidden),
        )

        norm = nn.BatchNorm1d if use_bn else lambda _: nn.Identity()

        self.enc = nn.Sequential(
            nn.Conv1d(1, ch1, 9, stride=1, padding=4),
            norm(ch1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch1, ch2, 7, stride=2, padding=3),
            norm(ch2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch2, ch3, 5, stride=2, padding=2),
            norm(ch3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch3, ch4, 3, stride=2, padding=1),
            norm(ch4),
            nn.LeakyReLU(0.2),
        )

        self.L_enc = (self.nfeat + 7) // 8
        enc_dim = ch4 * self.L_enc

        self.enc_ln = nn.LayerNorm([ch4, self.L_enc])
        self.flat = nn.Flatten()
        self.enc_dropout = nn.Dropout(dropout)
        self.mu = nn.Linear(enc_dim, self.ncode)
        self.lv = nn.Linear(enc_dim, self.ncode)

        self.fc = nn.Linear(self.ncode, enc_dim)
        self.dec_preln = nn.LayerNorm([ch4, self.L_enc])
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(
                ch4, ch3, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(
                ch3, ch2, 5, stride=2, padding=2, output_padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(
                ch2, ch1, 7, stride=2, padding=3, output_padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch1, 1, 9, padding=4),
        )

        # Projection head for the contrastive loss (SimCLR/SupCon style). The loss acts on
        # proj(mu), so mu keeps more within-class detail than if it were clustered directly.
        # proj_dim=0: no head (old checkpoints load unchanged); contrast then acts on mu.
        self.proj = (
            nn.Sequential(
                nn.Linear(self.ncode, proj_hidden), nn.ReLU(),
                nn.Linear(proj_hidden, proj_dim),
            )
            if proj_dim else None
        )

        self.apply(kaiming_init)
        nn.init.zeros_(self.lv.weight)
        nn.init.zeros_(self.lv.bias)

    def _as_input(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return x

    def encode(self, x):
        x = self._as_input(x)
        h = self.enc_dropout(self.flat(self.enc_ln(self.enc(x))))
        return self.mu(h), self.lv(h).clamp(-8.0, 8.0)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        h = self.fc(z).reshape(z.shape[0], self.ch4, self.L_enc)
        return self.dec(self.dec_preln(h))[..., :self.nfeat]

    def forward(self, x, sample=None):
        mu, logvar = self.encode(x)
        sample = self.training if sample is None else sample
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar, z

    def project(self, mu):
        return mu if self.proj is None else self.proj(mu)

    def loss(self, x, w=None, beta=1e-3, weight_power=1.0, frac_floor=0.0):
        x, w = masked_inputs(x, w)
        recon, mu, logvar, _ = self(x)
        return vae_loss(x, recon, mu, logvar, w, beta, weight_power, frac_floor)


# Loss

def _tensor_spectra_2d(x, name="x"):
    return x[:, 0].float() if x.ndim == 3 else x.float()


def masked_inputs(x, w=None):
    x = _tensor_spectra_2d(x)
    w = torch.ones_like(x) if w is None else _tensor_spectra_2d(w)
    valid = torch.isfinite(x) & torch.isfinite(w) & (w > 0)
    return torch.where(valid, x, 0.0), torch.where(valid, w, 0.0)


def apply_error_floor(x, w, frac_floor):
    """Effective inverse variance 1 / (1/ivar + (frac_floor * |x|)**2).

    Adds a fractional model/systematic error, so very high-SNR pixels cannot dominate
    the loss. frac_floor=0 returns w unchanged; masked pixels (w == 0) stay 0.
    """
    if not frac_floor:
        return w
    return w / (1 + w * (frac_floor * x).square())


def vae_loss(x, recon, mu, logvar, w=None, beta=1e-3, weight_power=1.0, frac_floor=0.0):
    x, w = masked_inputs(x, w)
    w = apply_error_floor(x, w, frac_floor)
    recon = _tensor_spectra_2d(recon)
    residual = torch.where(w > 0, recon, 0.0) - x
    mu, logvar = mu.float(), logvar.float()

    mse = 0.5 * (w * residual.square()).sum()
    kld = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum()
    return mse + beta * kld, mse, kld


def supcon_loss(z, labels, temperature=0.1, class_balance=0.5):
    """Supervised contrastive loss (Khosla et al. 2020), summed over anchors.

    z: (B, D) embeddings; labels: (B,) int64, -1 = ignored.
    Each anchor's loss is -mean log p(positive) over the other same-class spectra in the
    batch. Anchors with no same-class partner are skipped. Anchors are weighted by
    n_class**(-class_balance), normalized to mean 1, so rare classes are not drowned out
    by galaxies: 0 = no balancing, 1 = every class counts equally.
    Returns (summed loss, number of anchors used, whether the batch had >= 2 classes).
    """
    keep = labels >= 0
    z, labels = nn.functional.normalize(z[keep].float(), dim=1), labels[keep]
    n = len(labels)
    zero = z.sum() * 0.0
    if n < 2:
        return zero, 0, False

    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = (z @ z.T / temperature).masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    npos = pos.sum(1)
    has = npos > 0
    mixed = bool(torch.unique(labels).numel() >= 2)
    if not has.any():
        return zero, 0, mixed

    anchor_loss = -log_prob.masked_fill(~pos, 0.0).sum(1)[has] / npos[has]
    cls = labels[has]
    weight = torch.bincount(cls)[cls].float().pow(-class_balance)
    weight = weight * (len(cls) / weight.sum())
    return (weight * anchor_loss).sum(), int(len(cls)), mixed


def label_indices(spectype, classes):
    """Map SPECTYPE strings to class indices; anything not in classes -> -1."""
    lookup = {c: i for i, c in enumerate(classes)}
    names = np.char.upper(np.char.strip(np.asarray(spectype).astype(str)))
    return torch.as_tensor([lookup.get(s, -1) for s in names], dtype=torch.long)


def reconstruction_scores(x, recon, w=None, weight_power=1.0):
    x, w = masked_inputs(x, w)
    residual = torch.where(w > 0, _tensor_spectra_2d(recon), 0.0) - x
    # Normalized scores for comparing individual spectra.
    weights = (w / w.amax(dim=1, keepdim=True)).pow(weight_power)
    weights = weights / weights.sum(dim=1, keepdim=True)
    mse = (weights * residual.square()).sum(dim=1)
    return mse


# HDF5 data

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

                x = np.where(valid, x, 0.0).astype(np.float32, copy=False)
                w = np.where(valid, w, 0.0).astype(np.float32, copy=False)

                for offset in range(0, len(selected), self.batch_size):
                    rows = selected[offset:offset + self.batch_size]
                    yield (
                        torch.from_numpy(np.ascontiguousarray(x[rows])),
                        torch.from_numpy(np.ascontiguousarray(w[rows])),
                        labels[rows],
                        ids[rows],
                    )


def passthrough_batch(batch):
    return batch


def make_h5_loader(dataset, device, num_workers=0):
    options = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=(torch.device(device).type == "cuda"),
        collate_fn=passthrough_batch,
    )
    if num_workers:
        options.update(prefetch_factor=1, persistent_workers=False)
    return DataLoader(dataset, **options)

# Training

def _restore_torch_rng_state(state, device):
    if state is not None and state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.device(device).type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device=device)


def run_vae_epoch(
    model, loader, device, beta, optimizer=None,
    weight_power=1.0, max_grad_norm=5.0, log_every=200,
    *, resume_progress=None, resume_rng=None, on_batch_end=None, frac_floor=0.0,
    contrast_weight=0.0, contrast_classes=("GALAXY", "QSO", "STAR"),
    temperature=0.1, class_balance=0.5,
):
    training = optimizer is not None
    model.train(training)

    sums = dict(loss=0.0, rec=0.0, kl=0.0, nvalid=0.0, clipped=0.0,
                con=0.0, anchors=0.0, mixed=0.0)
    nseen, nbatches = 0, 0

    if resume_progress is not None:
        sums = {**sums, **resume_progress["sums"]}
        nseen = int(resume_progress["n"])
        nbatches = int(resume_progress["batches"])

    batches = iter(loader)

    # Replay saved batches without updating weights.
    if nbatches:
        print(f"Re-reading {nbatches:,} saved batches.", flush=True)
        for _ in range(nbatches):
            try:
                next(batches)
            except StopIteration as exc:
                raise ValueError("Saved batch position exceeds the current loader.") from exc

    if resume_progress is not None:
        _restore_torch_rng_state(resume_rng, device)

    with torch.set_grad_enabled(training):
        for x, w, spectype, targetid in batches:
            x = x.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            x, w = masked_inputs(x, w)

            if training:
                optimizer.zero_grad(set_to_none=True)

            recon, mu, logvar, _ = model(x, sample=training)
            terms = vae_loss(x, recon, mu, logvar, w, beta, weight_power, frac_floor)
            total = terms[0]
            if contrast_weight:
                labels = label_indices(spectype, contrast_classes).to(x.device)
                con, n_anchor, mixed = supcon_loss(
                    model.project(mu), labels, temperature, class_balance
                )
                total = total + contrast_weight * con
                sums["con"] += con.detach().item()
                sums["anchors"] += n_anchor
                sums["mixed"] += float(mixed)

            if not all(torch.isfinite(term).item() for term in (*terms, total)):
                raise FloatingPointError("Non-finite VAE loss.")

            if training:
                total.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm, error_if_nonfinite=True
                )
                sums["clipped"] += float(grad_norm.item() > max_grad_norm)
                optimizer.step()

            nseen += x.shape[0]
            nbatches += 1

            for key, value in zip(
                ("loss", "rec", "kl"), (total, terms[1], terms[2])
            ):
                sums[key] += value.detach().item()

            sums["nvalid"] += (w > 0).sum().item()

            if on_batch_end is not None:
                on_batch_end({
                    "sums": dict(sums),
                    "n": nseen,
                    "batches": nbatches,
                })

            if training and log_every and nbatches % log_every == 0:
                print(
                    f"  batches={nbatches:,} spectra={nseen:,} "
                    f"rec={sums['rec'] / nseen:.6g} "
                    f"kl={sums['kl'] / nseen:.6g} "
                    f"rchi2={2 * sums['rec'] / max(sums['nvalid'], 1):.4g} "
                    f"clipped={sums['clipped'] / nbatches:.0%}"
                    + (f" con={sums['con'] / max(sums['anchors'], 1):.4g} "
                       f"mixed={sums['mixed'] / nbatches:.0%}" if contrast_weight else "")
                )

    if not nseen:
        raise ValueError("No usable spectra in this split.")

    # Report per-spectrum sums; backpropagation uses full-batch sums.
    return {
        **{key: sums[key] / nseen for key in ("loss", "rec", "kl")},
        # rec = 0.5 * chi^2, so chi^2 per valid pixel = 2 * rec / n_valid (~1 at noise level).
        "rchi2": 2 * sums["rec"] / max(sums["nvalid"], 1),
        "clip_frac": sums["clipped"] / nbatches if training else None,
        # Mean contrastive loss per anchor, and fraction of batches with >= 2 classes.
        "con": sums["con"] / max(sums["anchors"], 1) if contrast_weight else None,
        "mixed_frac": sums["mixed"] / nbatches if contrast_weight else None,
        "n": nseen,
        "batches": nbatches,
    }


def validate_checkpoint_loss(checkpoint, checkpoint_path):
    if checkpoint.get("loss_config") != LOSS_CONFIG:
        raise ValueError(
            f"{checkpoint_path}: checkpoint uses a different loss. "
            "Load its weights into a new OUT_DIR with resume_from=None."
        )


def fit_vae(
    model, train_loader, val_loader, device, out_dir, data_config,
    epochs=30, learning_rate=3e-4, beta_max=1e-3, warmup_epochs=5,
    patience=10, weight_power=1.0, max_grad_norm=5.0, log_every=200,
    resume_from=None, checkpoint_every_batches=200, frac_floor=0.0,
    contrast_weight=0.0, contrast_classes=("GALAXY", "QSO", "STAR"),
    temperature=0.1, class_balance=0.5,
):
    contrast_classes = [str(c).strip().upper() for c in contrast_classes]
    if weight_power != 1.0:
        raise ValueError("Use weight_power=1.0 for direct ivar training.")
    if checkpoint_every_batches and train_loader.num_workers != 0:
        raise ValueError("Use num_workers=0 for batch checkpoints.")

    out_dir = Path(out_dir).expanduser().resolve()
    if resume_from is None and (
        (out_dir / "latest.pt").exists() or (out_dir / "best.pt").exists()
    ):
        raise FileExistsError("Choose a new OUT_DIR or resume the existing run.")
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(state, filename):
        destination = out_dir / filename
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(destination)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    train_config = dict(
        epochs=epochs, learning_rate=learning_rate, beta_max=beta_max,
        warmup_epochs=warmup_epochs, patience=patience,
        weight_power=weight_power, max_grad_norm=max_grad_norm,
        log_every=log_every, checkpoint_every_batches=checkpoint_every_batches,
        frac_floor=frac_floor, contrast_weight=contrast_weight,
        contrast_classes=contrast_classes, temperature=temperature,
        class_balance=class_balance,
    )
    contrast_kw = dict(
        contrast_classes=contrast_classes, temperature=temperature,
        class_balance=class_balance,
    )
    loader_config = dict(
        train_num_workers=train_loader.num_workers,
        val_num_workers=val_loader.num_workers,
    )

    history, best_val, bad_epochs, step = [], float("inf"), 0, 0
    start_epoch = 1
    resume_phase = resume_progress = resume_rng = saved_train_metrics = None

    def checkpoint_state(epoch, phase, progress=None, train_metrics=None):
        return {
            "checkpoint_version": 4,
            "phase": phase,
            "loss_config": dict(LOSS_CONFIG),
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
            "sch_state_dict": scheduler.state_dict(),
            "best_val": best_val,
            "bad_epochs": bad_epochs,
            "model_config": model.config,
            "data_config": data_config,
            "data_manifest": train_loader.dataset.file_manifest,
            "loader_config": loader_config,
            "train_config": train_config,
            "history": history,
            "train_progress": progress,
            "train_metrics": train_metrics,
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(device)
                if torch.device(device).type == "cuda" else None
            ),
            "selection_metric": "deterministic reconstruction at mu + beta_max * KL",
        }

    def checkpoint_position(state):
        phase_order = {"train": 0, "validate": 1, "epoch_complete": 2}
        return (
            int(state["epoch"]),
            phase_order[state.get("phase", "epoch_complete")],
            int(state["step"]),
        )

    if resume_from is not None:
        resume_path = Path(resume_from).expanduser().resolve()
        if resume_path.parent != out_dir:
            raise ValueError("Resume into the checkpoint's existing OUT_DIR.")

        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        validate_checkpoint_loss(checkpoint, resume_path)

        # Recover if interruption occurred between best.pt and latest.pt saves.
        best_path = out_dir / "best.pt"
        if resume_path.name == "latest.pt" and best_path.is_file():
            saved_best = torch.load(best_path, map_location="cpu", weights_only=True)
            validate_checkpoint_loss(saved_best, best_path)
            if checkpoint_position(saved_best) > checkpoint_position(checkpoint):
                checkpoint = saved_best
            del saved_best

        if data_config != checkpoint["data_config"]:
            raise ValueError("Resume with the saved data_config.")

        if checkpoint.get("data_manifest") is not None:
            for loader in (train_loader, val_loader):
                if loader.dataset.file_manifest != checkpoint["data_manifest"]:
                    raise ValueError("HDF5 files or row counts changed.")

        if checkpoint.get("loader_config", loader_config) != loader_config:
            raise ValueError("Resume with the saved num_workers settings.")

        for key in (
            "learning_rate", "beta_max", "warmup_epochs",
            "patience", "weight_power", "max_grad_norm",
        ):
            if train_config[key] != checkpoint["train_config"][key]:
                raise ValueError(f"Restore the saved training setting: {key}.")
        saved = checkpoint["train_config"]
        defaults = dict(frac_floor=0.0, contrast_weight=0.0,
                        contrast_classes=["GALAXY", "QSO", "STAR"],
                        temperature=0.1, class_balance=0.5)
        for key, default in defaults.items():
            if train_config[key] != saved.get(key, default):
                raise ValueError(f"Restore the saved training setting: {key}.")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["opt_state_dict"])
        scheduler.load_state_dict(checkpoint["sch_state_dict"])

        history = list(checkpoint["history"])
        best_val = float(checkpoint["best_val"])
        bad_epochs = int(checkpoint["bad_epochs"])
        step = int(checkpoint["step"])
        resume_phase = checkpoint.get("phase", "epoch_complete")
        start_epoch = int(checkpoint["epoch"]) + (resume_phase == "epoch_complete")

        if resume_phase == "train":
            resume_progress = checkpoint["train_progress"]
            resume_rng = {
                key: checkpoint.get(key)
                for key in ("torch_rng_state", "cuda_rng_state")
            }
        elif resume_phase == "validate":
            saved_train_metrics = checkpoint["train_metrics"]

        _restore_torch_rng_state(checkpoint, device)
        print(f"Resuming epoch {start_epoch}, phase={resume_phase}, steps={step:,}.")
        del checkpoint
    else:
        save_checkpoint(checkpoint_state(0, "epoch_complete"), "latest.pt")

    print("Checkpoint:", out_dir / "latest.pt", flush=True)
    started = time.monotonic()

    for epoch in range(start_epoch, epochs + 1):
        if bad_epochs >= patience:
            break

        train_loader.dataset.set_epoch(epoch)
        ramp = min(1.0, epoch / max(1, warmup_epochs))
        beta = beta_max * ramp

        if resume_phase == "validate":
            train_metrics = saved_train_metrics
        else:
            saved_batches = (
                resume_progress["batches"] if resume_progress is not None else 0
            )
            epoch_start_step = step - saved_batches

            def save_batch(progress):
                nonlocal step
                step = epoch_start_step + progress["batches"]
                if (
                    checkpoint_every_batches
                    and progress["batches"] % checkpoint_every_batches == 0
                ):
                    save_checkpoint(
                        checkpoint_state(epoch, "train", progress=progress),
                        "latest.pt",
                    )
                    print(
                        f"Saved epoch {epoch}, batch {progress['batches']:,}.",
                        flush=True,
                    )

            train_metrics = run_vae_epoch(
                model, train_loader, device, beta, optimizer,
                weight_power, max_grad_norm, log_every,
                resume_progress=resume_progress,
                resume_rng=resume_rng,
                on_batch_end=save_batch,
                frac_floor=frac_floor,
                contrast_weight=contrast_weight * ramp, **contrast_kw,
            )
            save_checkpoint(
                checkpoint_state(epoch, "validate", train_metrics=train_metrics),
                "latest.pt",
            )

        resume_phase = resume_progress = resume_rng = saved_train_metrics = None

        # Use a fixed beta and latent mean for checkpoint selection.
        val_metrics = run_vae_epoch(
            model, val_loader, device, beta_max, weight_power=weight_power,
            frac_floor=frac_floor, contrast_weight=contrast_weight, **contrast_kw,
        )
        history.append({
            "epoch": epoch,
            "beta": beta,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        improved = val_metrics["loss"] < best_val
        if improved:
            best_val, bad_epochs = val_metrics["loss"], 0
        elif epoch > warmup_epochs:
            bad_epochs += 1

        if epoch >= warmup_epochs:
            scheduler.step(val_metrics["loss"])

        checkpoint = checkpoint_state(epoch, "epoch_complete")
        if improved:
            save_checkpoint(checkpoint, "best.pt")
        save_checkpoint(checkpoint, "latest.pt")

        temporary = out_dir / "history.json.tmp"
        temporary.write_text(json.dumps(history, indent=2))
        temporary.replace(out_dir / "history.json")

        print(
            f"epoch={epoch:03d} beta={beta:.3g} "
            f"train_rec={train_metrics['rec']:.6g} "
            f"val_rec={val_metrics['rec']:.6g} "
            f"val_KL={val_metrics['kl']:.6g} "
            f"val_rchi2={val_metrics['rchi2']:.4g} "
            f"clipped={train_metrics.get('clip_frac') or 0:.0%} "
            + (f"val_con={val_metrics['con']:.4g} "
               f"mixed={train_metrics.get('mixed_frac') or 0:.0%} " if contrast_weight else "")
            + f"time={(time.monotonic() - started) / 60:.1f} min"
        )

        if bad_epochs >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    checkpoint = torch.load(
        out_dir / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Restored best checkpoint from epoch {checkpoint['epoch']}.")
    return history


# Self-test

def self_test(device="cpu"):
    """Run once after updating this file: checks supcon_loss against a slow loop
    implementation and backpropagates through the projection head."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(40, 8, generator=g)
    labels = torch.randint(0, 3, (40,), generator=g)
    labels[:3] = -1
    t, cb = 0.2, 0.5
    loss, n_anchor, mixed = supcon_loss(z, labels, t, cb)

    zz = nn.functional.normalize(z[labels >= 0], dim=1)
    ll = labels[labels >= 0].tolist()
    terms, cls = [], []
    for i in range(len(ll)):
        others = [j for j in range(len(ll)) if j != i]
        denom = torch.logsumexp(torch.stack([zz[i] @ zz[j] / t for j in others]), 0)
        pos = [j for j in others if ll[j] == ll[i]]
        if pos:
            terms.append(-torch.stack([zz[i] @ zz[j] / t - denom for j in pos]).mean())
            cls.append(ll[i])
    counts = np.bincount(cls)
    weights = counts[cls].astype(float) ** -cb
    weights *= len(cls) / weights.sum()
    reference = (torch.tensor(weights, dtype=torch.float32) * torch.stack(terms)).sum()
    assert n_anchor == len(cls) and mixed, (n_anchor, len(cls), mixed)
    assert torch.allclose(loss, reference, rtol=1e-4), (loss.item(), reference.item())
    assert not supcon_loss(z[:5], torch.zeros(5, dtype=torch.long))[2]
    assert label_indices(np.array(["galaxy ", "QSO", "XX"]), ["GALAXY", "QSO"]).tolist() == [0, 1, -1]

    model = VAE(nfeat=64, ncode=4, ch1=4, ch2=4, ch3=8, ch4=8, proj_dim=6, proj_hidden=8).to(device)
    x = torch.randn(16, 64, device=device)
    recon, mu, logvar, _ = model(x, sample=True)
    total = vae_loss(x, recon, mu, logvar, torch.ones_like(x))[0]
    total = total + supcon_loss(model.project(mu), torch.arange(16, device=device) % 3)[0]
    total.backward()
    assert torch.isfinite(total) and model.proj[0].weight.grad is not None
    assert model.mu.weight.grad.abs().sum() > 0
    print("self_test passed")


# Scoring

@torch.inference_mode()
def score_spectra(model, loader, device):
    """Deterministic (z = mu) scores for every spectrum the loader yields.

    chi2   = sum(ivar * residual**2) over valid pixels
    rchi2  = chi2 / n_valid  (~1 for a noise-level fit)
    snr    = median(x * sqrt(ivar)) over valid pixels, for SNR-dependent corrections
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
        out["snr"].append(
            torch.where(valid, x * w.sqrt(), torch.nan).nanmedian(dim=1).values.cpu()
        )
        out["mu"].append(mu.float().cpu())
        out["TARGETID"].append(np.asarray(targetid))
        out["SPECTYPE"].append(np.asarray(spectype).astype(str))
    if not out["chi2"]:
        raise ValueError("No spectra to score.")
    scores = {
        k: (np.concatenate(v) if k in ("TARGETID", "SPECTYPE") else torch.cat(v).numpy())
        for k, v in out.items()
    }
    scores["rchi2"] = scores["chi2"] / np.maximum(scores["n_valid"], 1)
    return scores


# Plots

def plot_vae_history(history):
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 3.5), constrained_layout=True
    )

    for ax, key, title in zip(
        axes, ["rec", "kl"], ["Reconstruction per spectrum", "KL per spectrum"]
    ):
        ax.plot(epochs, [row[f"train_{key}"] for row in history], label="Train")
        ax.plot(epochs, [row[f"val_{key}"] for row in history], label="Validation")
        ax.set(xlabel="Epoch", ylabel=title)
        ax.legend()

    plt.show()
    return fig


@torch.inference_mode()
def plot_vae_reconstructions(
    model, loader, device, n=6, weight_power=1.0
):
    import matplotlib.pyplot as plt

    model.eval()
    x, w, labels, ids = next(iter(loader))
    x, w = masked_inputs(x.to(device), w.to(device))

    recon = _tensor_spectra_2d(model(x, sample=False)[0])
    mse = reconstruction_scores(x, recon, w, weight_power)

    n = min(n, len(x))
    fig = plt.figure(
        figsize=(12, 2.8 * ((n + 1) // 2)), constrained_layout=True
    )

    for i in range(n):
        ax = fig.add_subplot((n + 1) // 2, 2, i + 1)
        valid = w[i].cpu().numpy() > 0

        def mask(values):
            return np.where(valid, values[i].cpu().numpy(), np.nan)

        ax.plot(mask(x), color="0.6", linewidth=0.7, label="Input")
  
        ax.plot(mask(recon), color="#CC6677", linewidth=1, label="Reconstruction")

        title = f"{labels[i]} | {ids[i]} | WMSE {mse[i].item():.4g}"

        ax.set(title=title, xlabel="Wavelength bin", ylabel="Preprocessed flux")
        ax.legend(fontsize=8)

    plt.show()
    return fig
