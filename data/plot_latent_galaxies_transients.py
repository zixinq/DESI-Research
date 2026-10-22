"""Run in a notebook with a loaded `model` and prepared galaxy `npz_files`.

Uses posterior means and one shared PCA basis. No anomaly-candidate helpers required.
"""
from pathlib import Path
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

# Configuration: galaxy and transient inputs must use the SAME preprocessing.
TRANSIENT_NPZ = '/global/cfs/cdirs/desi/users/zixinq/data/dr2_catalog/transients_preprocessed_all.npz'
LATENT_SEED = 42
MAX_GALAXY_FILES = 20  # None scans all files; 20 is a quick exploration sample.
MAX_GALAXIES = 10000
MAX_TRANSIENTS = 2000
LATENT_BATCH_SIZE = 256
LATENT_OUTPUT_DIR = Path('latent_comparison')


def sample_latent_inputs(files, limit, seed, galaxies_only=False, exclude_ids=()):
    """Uniform row sample within the supplied files using random priorities.

    Keeps at most `limit` spectra plus one loaded shard in RAM. Repeated target
    observations remain separate rows; source_file/source_row identify them.
    """
    rng = np.random.default_rng(seed)
    selected_x, priorities, records = None, np.empty(0), []
    eligible_count = 0
    for filename in files:
        with np.load(filename, allow_pickle=True) as d:
            x = np.asarray(d['X'], dtype=np.float32)
            w = np.asarray(d['W'], dtype=np.float32)
            if x.ndim == 3 and x.shape[1] == 1: x = x[:, 0, :]
            if w.ndim == 3 and w.shape[1] == 1: w = w[:, 0, :]
            if x.ndim != 2 or x.shape != w.shape:
                raise ValueError(f'Invalid X/W shapes in {filename}')
            ids = np.asarray(d['TARGETID'], dtype=np.int64).reshape(-1)
            if len(ids) != len(x): raise ValueError(f'TARGETID length mismatch: {filename}')
            valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
            keep = (valid.sum(1) >= 50) & (np.where(valid, w, 0).sum(1) > 50)
            if galaxies_only:
                keep &= np.asarray(d['SPECTYPE']).astype(str).reshape(-1) == 'GALAXY'
                keep &= ~np.isin(ids, exclude_ids)
            rows = np.flatnonzero(keep)
            eligible_count += len(rows)
            if not len(rows): continue
            keys = rng.random(len(rows))
            # Only the smallest `limit` keys in this shard can survive globally.
            take = np.argsort(keys)[:limit]
            rows, keys = rows[take], keys[take]
            batch = np.where(valid[rows], x[rows], 0).astype(np.float32)
            selected_x = batch if selected_x is None else np.concatenate([selected_x, batch])
            priorities = np.concatenate([priorities, keys])
            records.extend(dict(TARGETID=int(ids[i]), source_file=str(filename), source_row=int(i)) for i in rows)
            choose = np.argsort(priorities)[:limit]
            selected_x, priorities = selected_x[choose], priorities[choose]
            records = [records[i] for i in choose]
    if selected_x is None:
        raise ValueError('No eligible spectra after class, ID and valid-pixel cuts')
    print(f'Sampled {len(selected_x):,} of {eligible_count:,} eligible rows in {len(files)} files')
    return selected_x, records


@torch.inference_mode()
def encode_latent_means(model, spectra, batch_size):
    model.eval()
    parameter = next(model.parameters())
    result = []
    for start in range(0, len(spectra), batch_size):
        x = torch.as_tensor(spectra[start:start + batch_size], device=parameter.device,
                            dtype=parameter.dtype).unsqueeze(1)
        mu, _ = model.encode(x)
        if not torch.isfinite(mu).all(): raise ValueError('Encoder returned non-finite latent means')
        result.append(mu.cpu().numpy())
    return np.concatenate(result)


def shared_latent_pca(galaxies, transients, seed=42):
    """Fit one centered, unscaled PCA basis to an equal-size sample per class."""
    if galaxies.shape[1] != transients.shape[1] or galaxies.shape[1] < 2:
        raise ValueError('Both latent arrays must have the same dimension >= 2')
    rng = np.random.default_rng(seed)
    count = min(len(galaxies), len(transients), 5000)
    if count < 2: raise ValueError('Need at least two spectra in each group')
    fit = np.concatenate([galaxies[rng.choice(len(galaxies), count, replace=False)],
                          transients[rng.choice(len(transients), count, replace=False)]]).astype(float)
    center = fit.mean(0)
    _, singular, components = np.linalg.svd(fit - center, full_matrices=False)
    variance = singular ** 2
    if variance.sum() <= 0: raise ValueError('Latent vectors have zero variance')
    basis = components[:2]
    fraction = variance[:2] / variance.sum()
    return (galaxies - center) @ basis.T, (transients - center) @ basis.T, center, basis, fraction


def plot_latent_comparison(galaxy_xy, transient_xy, explained):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax in axes[:2]:
        ax.scatter(*galaxy_xy.T, s=7, color='#356A9A', alpha=0.3,
                   linewidths=0, label=f'Galaxies (n={len(galaxy_xy):,})', rasterized=True)
    for ax in (axes[0], axes[2]):
        ax.scatter(*transient_xy.T, s=24, marker='^', facecolors='none',
                   edgecolors='#C45E16', linewidths=0.7, alpha=0.8,
                   label=f'Transients (n={len(transient_xy):,})', rasterized=True)
    for ax, title in zip(axes, ['Overlay', 'Galaxies', 'Transients']):
        ax.set_title(title)
        ax.set_xlabel(f'PC1 ({explained[0]:.1%} of balanced-fit variance)')
        ax.grid(alpha=0.15)
        ax.legend(loc='best', fontsize=8, frameon=False)
    axes[0].set_ylabel(f'PC2 ({explained[1]:.1%} of balanced-fit variance)')
    fig.suptitle('VAE latent means: galaxies and transients · shared PCA coordinates')
    return fig, axes


# No calls to candidate loading/scoring functions are needed below.
if 'model' not in globals() or 'npz_files' not in globals():
    raise RuntimeError('Load the trained VAE as model and define galaxy npz_files first')
if not Path(TRANSIENT_NPZ).is_file():
    raise FileNotFoundError(f'Set TRANSIENT_NPZ to your prepared transient spectra: {TRANSIENT_NPZ}')
if min(MAX_GALAXIES, MAX_TRANSIENTS, LATENT_BATCH_SIZE) < 1:
    raise ValueError('Sample sizes and batch size must be positive')
with np.load(TRANSIENT_NPZ, allow_pickle=True) as transient_data:
    known_transient_ids = np.asarray(transient_data['TARGETID'], dtype=np.int64).reshape(-1)

galaxy_files = sorted(set(map(str, npz_files)))
if MAX_GALAXY_FILES is not None:
    if MAX_GALAXY_FILES < 1: raise ValueError('MAX_GALAXY_FILES must be positive or None')
    selection = np.random.default_rng(LATENT_SEED).choice(
        len(galaxy_files), min(MAX_GALAXY_FILES, len(galaxy_files)), replace=False)
    galaxy_files = [galaxy_files[i] for i in sorted(selection)]

transient_flux, transient_records = sample_latent_inputs([TRANSIENT_NPZ], MAX_TRANSIENTS, LATENT_SEED)
galaxy_flux, galaxy_records = sample_latent_inputs(galaxy_files, MAX_GALAXIES, LATENT_SEED,
                                                galaxies_only=True, exclude_ids=known_transient_ids)
if galaxy_flux.shape[1] != transient_flux.shape[1] or galaxy_flux.shape[1] != model.nfeat:
    raise ValueError('Galaxy/transient pixel counts must match the trained model')
mu_galaxies = encode_latent_means(model, galaxy_flux, LATENT_BATCH_SIZE)
mu_transients = encode_latent_means(model, transient_flux, LATENT_BATCH_SIZE)
del galaxy_flux, transient_flux
galaxy_xy, transient_xy, pca_center, pca_basis, explained_fraction = shared_latent_pca(
    mu_galaxies, mu_transients, LATENT_SEED)
latent_figure, latent_axes = plot_latent_comparison(galaxy_xy, transient_xy, explained_fraction)
LATENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
latent_figure.savefig(LATENT_OUTPUT_DIR / 'galaxies_vs_transients_pca.png', dpi=180)
latent_figure.savefig(LATENT_OUTPUT_DIR / 'galaxies_vs_transients_pca.pdf')
np.savez_compressed(LATENT_OUTPUT_DIR / 'latent_comparison.npz',
                    mu_galaxies=mu_galaxies, mu_transients=mu_transients,
                    galaxy_xy=galaxy_xy, transient_xy=transient_xy,
                    pca_center=pca_center, pca_basis=pca_basis, explained_fraction=explained_fraction)
with (LATENT_OUTPUT_DIR / 'latent_sources.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['group', 'index', 'TARGETID', 'source_file', 'source_row'])
    writer.writeheader()
    for group, records in [('galaxy', galaxy_records), ('transient', transient_records)]:
        writer.writerows(dict(group=group, index=i, **row) for i, row in enumerate(records))
plt.show()
print('Saved latent vectors, source rows and figures to', LATENT_OUTPUT_DIR.resolve())
