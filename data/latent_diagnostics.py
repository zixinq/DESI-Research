"""Plot saved DESI VAE means; imported by VAE_latent_diagnostics.ipynb.

No model training, anomaly selection, or automatic test-split inference.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SOURCES = {
    'galaxy': ('dr2_latents.npz', 'mu_galaxy', 'tid_galaxy'),
    'qso': ('dr2_latents.npz', 'mu_qso', 'tid_qso'),
    'transient': ('transient_latents.npz', 'mu', 'TARGETID'),
    'simulation': ('simulation_latents.npz', 'mu', 'TARGETID'),
}
NAMES = ['Galaxies', 'QSOs', 'Real transients', 'Simulations']
COLORS = ['#356A9A', '#7953A2', '#C45E16', '#27816B']
STYLES = ['-', ':', '--', '-.']
MARKERS = ['o', 's', '^', 'D']


def load_groups(save_dir, max_rows=10000, seed=42, row_masks=None):
    """Sample finite rows uniformly per supplied class, retaining source row IDs.

    row_masks maps class keys to boolean arrays in ORIGINAL NPZ row order.
    Omitted classes use all rows. NPZ members decompress in full, one group at a
    time; the cap limits retained arrays and plotting, not decompression memory.
    """
    if max_rows is not None and max_rows < 2:
        raise ValueError('max_rows must be at least 2 or None')
    row_masks = {} if row_masks is None else row_masks
    if set(row_masks) - set(SOURCES):
        raise ValueError('Unknown class key in row_masks')
    rng = np.random.default_rng(seed)
    groups, ncode = {}, None
    for (key, (file, mu_key, id_key)), name, color, style, marker in zip(
            SOURCES.items(), NAMES, COLORS, STYLES, MARKERS):
        with np.load(Path(save_dir) / file, allow_pickle=False) as data:
            mu = data[mu_key]
            ids = data[id_key].reshape(-1)
        if mu.ndim != 2 or len(ids) != len(mu):
            raise ValueError(f'{key}: expected mu=(N,D) and matching IDs')
        if ncode is None:
            ncode = mu.shape[1]
        if mu.shape[1] != ncode or ncode < 2:
            raise ValueError('Groups must have the same latent dimension >= 2')
        mask = np.asarray(row_masks.get(key, np.ones(len(mu), dtype=bool)))
        if mask.dtype != bool or mask.shape != (len(mu),):
            raise ValueError(f'{key}: mask must be boolean with shape ({len(mu)},)')
        finite = np.isfinite(mu).all(axis=1)
        eligible = np.flatnonzero(mask & finite)
        if len(eligible) < 2:
            raise ValueError(f'{key}: fewer than two eligible finite rows')
        rows = eligible if max_rows is None else np.sort(rng.choice(
            eligible, min(max_rows, len(eligible)), replace=False))
        groups[key] = dict(mu=mu[rows].astype(float), ids=ids[rows], rows=rows,
                           name=name, color=color, style=style, marker=marker,
                           file=file, mu_key=mu_key, total=len(mu),
                           eligible=len(eligible), invalid=int((mask & ~finite).sum()))
        print(f'{name}: {len(rows):,} plotted / {len(eligible):,} eligible; '
              f'{groups[key]["invalid"]:,} non-finite rows removed')
    return groups


def save_selection(groups, output_dir, seed, scope):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'sampled_rows.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['group', 'sample_index', 'source_file', 'source_row', 'TARGETID'])
        for key, g in groups.items():
            writer.writerows((key, i, g['file'], int(row), str(tid))
                             for i, (row, tid) in enumerate(zip(g['rows'], g['ids'])))
    (out / 'selection.json').write_text(json.dumps(dict(
        seed=seed, scope=scope,
        groups={k: {field: g[field] for field in
                    ['file', 'mu_key', 'total', 'eligible', 'invalid']} for k, g in groups.items()}
    ), indent=2))


def grid(n):
    fig, axes = plt.subplots(int(np.ceil(n / 4)), 4,
                             figsize=(15, 3 * int(np.ceil(n / 4))), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()[:n]


def legend(fig, groups):
    fig.legend(handles=[Line2D([], [], color=g['color'], linestyle=g['style'],
                              label=f'{g["name"]} (n={len(g["mu"]):,})')
                        for g in groups.values()],
               loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)


def histograms(groups, bins=50):
    d = next(iter(groups.values()))['mu'].shape[1]
    fig, axes = grid(d)
    for j, ax in enumerate(axes):
        values = np.concatenate([g['mu'][:, j] for g in groups.values()])
        edges = np.histogram_bin_edges(values, bins=bins)
        for g in groups.values():
            ax.hist(g['mu'][:, j], bins=edges, density=True, histtype='step',
                    color=g['color'], linestyle=g['style'], linewidth=1.5)
        ax.set(xlabel=f'μ{j}', ylabel='Probability density')
        ax.grid(alpha=0.12)
    fig.suptitle('Each latent dimension · separate density normalization per class')
    legend(fig, groups)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def boxplots(groups):
    d = next(iter(groups.values()))['mu'].shape[1]
    fig, axes = grid(d)
    for j, ax in enumerate(axes):
        artists = ax.boxplot([g['mu'][:, j] for g in groups.values()],
                            patch_artist=True, showfliers=False, whis=(5, 95))
        for patch, g in zip(artists['boxes'], groups.values()):
            patch.set_facecolor(g['color'])
            patch.set_alpha(0.55)
        ax.set_xticks(np.arange(1, 5), ['Gal', 'QSO', 'Real', 'Sim'])
        ax.set_ylabel(f'μ{j}')
        ax.grid(axis='y', alpha=0.12)
    fig.suptitle('Latent distributions · boxes: 25–75%; center: median; whiskers: 5–95%\n'
                 'Tails beyond whiskers are omitted')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def pairwise_scatter(groups, pairs=((0, 1), (0, 2), (1, 2), (3, 4), (4, 5), (2, 5)),
                     max_points=1000, seed=42):
    d = next(iter(groups.values()))['mu'].shape[1]
    if not pairs or any(a == b or min(a, b) < 0 or max(a, b) >= d for a, b in pairs):
        raise ValueError('Select valid, distinct dimension pairs')
    count = min(max_points, *(len(g['mu']) for g in groups.values()))
    if count < 1:
        raise ValueError('max_points must be positive')
    rng = np.random.default_rng(seed)
    samples = {k: g['mu'][rng.choice(len(g['mu']), count, replace=False)]
               for k, g in groups.items()}
    fig, axes = plt.subplots(int(np.ceil(len(pairs) / 3)), 3,
                             figsize=(15, 4.3 * int(np.ceil(len(pairs) / 3))), squeeze=False)
    for ax, (a, b) in zip(axes.flat, pairs):
        for k, g in groups.items():
            x = samples[k]
            ax.scatter(x[:, a], x[:, b], s=13, marker=g['marker'],
                       color=g['color'], alpha=0.3, linewidths=0, rasterized=True,
                       label=g['name'])
        ax.set(xlabel=f'μ{a}', ylabel=f'μ{b}')
        ax.grid(alpha=0.12)
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f'Pairwise latent means · {count:,} random rows per class; seed={seed}')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def correlations(groups):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#CCCCCC')
    for ax, g in zip(axes.flat, groups.values()):
        x = g['mu']
        sd = x.std(axis=0)
        valid = sd > 0
        corr = np.full((x.shape[1], x.shape[1]), np.nan)
        if valid.sum():
            z = (x[:, valid] - x[:, valid].mean(axis=0)) / sd[valid]
            corr[np.ix_(valid, valid)] = np.clip(z.T @ z / len(z), -1, 1)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap=cmap)
        ax.set(title=f'{g["name"]} · n={len(x):,}', xlabel='Latent dimension',
               ylabel='Latent dimension', xticks=range(x.shape[1]), yticks=range(x.shape[1]))
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Pearson correlation')
    fig.suptitle('Within-class latent correlations · gray = undefined (constant dimension)')
    return fig


def latent_heatmap(groups, max_rows=80, seed=42):
    """One shared scale based on the sampled galaxies; descriptive, not a fitted classifier."""
    reference = groups['galaxy']['mu']
    center, scale = reference.mean(axis=0), reference.std(axis=0)
    scale = np.where(scale > 0, scale, np.nan)
    count = min(max_rows, *(len(g['mu']) for g in groups.values()))
    if count < 1:
        raise ValueError('max_rows must be positive')
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 4, figsize=(15, 8), constrained_layout=True)
    selected = {}
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#CCCCCC')
    for ax, (key, g) in zip(axes, groups.items()):
        idx = rng.choice(len(g['mu']), count, replace=False)
        selected[key] = idx
        z = (g['mu'][idx] - center) / scale
        im = ax.imshow(z, aspect='auto', interpolation='nearest',
                       cmap=cmap, vmin=-3, vmax=3)
        ax.set(title=g['name'], xlabel='Latent dimension',
               xticks=range(z.shape[1]))
    axes[0].set_ylabel('Random sampled row (order is not shared across classes)')
    fig.colorbar(im, ax=axes.tolist(), extend='both', shrink=0.7,
                 label='(μ − galaxy mean) / galaxy SD; colors saturate at ±3')
    fig.suptitle(f'Latent values · {count} rows per class · gray = constant galaxy dimension')
    return fig, selected


def latent_traversals(model, groups, anchor_group='galaxy', anchor_index=0,
                      reference_group='galaxy', wavelength=None):
    """Vary one coordinate of a real saved latent vector at a time.

    Uses reference-group marginal 10/50/90 percentiles. Other coordinates stay
    fixed; combinations may leave the learned data distribution.
    """
    import torch
    anchor = groups[anchor_group]['mu'][anchor_index]
    reference = groups[reference_group]['mu']
    levels = np.quantile(reference, [0.1, 0.5, 0.9], axis=0)
    d = len(anchor)
    if hasattr(model, 'ncode') and model.ncode != d:
        raise ValueError('Model latent dimension does not match saved vectors')
    param = next(model.parameters())
    was_training = model.training
    model.eval()
    reconstructions = []
    try:
        with torch.inference_mode():
            for j in range(d):
                z = np.tile(anchor, (4, 1))
                z[1:, j] = levels[:, j]
                y = model.decode(torch.as_tensor(z, device=param.device, dtype=param.dtype))
                if y.ndim == 3 and y.shape[1] == 1:
                    y = y[:, 0, :]
                if y.ndim != 2 or y.shape[0] != 4 or not torch.isfinite(y).all():
                    raise ValueError('Decoder must return finite (B,L) or (B,1,L) spectra')
                reconstructions.append(y.detach().cpu().numpy())
    finally:
        model.train(was_training)
    reconstructions = np.stack(reconstructions)
    length = reconstructions.shape[-1]
    if wavelength is None:
        wave, label = np.arange(length), 'Input pixel index'
    else:
        wave = np.asarray(wavelength)
        if wave.shape != (length,) or not np.isfinite(wave).all() or not np.all(np.diff(wave) > 0):
            raise ValueError('Wavelength must be a finite increasing 1D decoder input grid')
        label = 'Wavelength (units of supplied grid)'
    fig, axes = grid(d)
    for j, ax in enumerate(axes):
        ax.plot(wave, reconstructions[j, 0], color='#333333', lw=1.3, label='Anchor')
        for k, (color, style, quantile) in enumerate(zip(
                ['#356A9A', '#C45E16', '#7953A2'], ['--', ':', '-.'], [10, 50, 90])):
            ax.plot(wave, reconstructions[j, k + 1], color=color, linestyle=style,
                    lw=1, label=f'Reference p{quantile}')
        ax.set(title=f'Vary μ{j} (anchor={anchor[j]:.2g})', xlabel=label,
               ylabel='Decoded normalized flux')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f'One-coordinate traversals · anchor: {anchor_group}, sample {anchor_index}\n'
                 f'10/50/90 percentiles from {reference_group}; all other coordinates fixed')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, dict(decoded=reconstructions, anchor=anchor, quantile_values=levels,
                     wavelength=wave, anchor_source_row=groups[anchor_group]['rows'][anchor_index],
                     anchor_TARGETID=groups[anchor_group]['ids'][anchor_index],
                     anchor_group=anchor_group, reference_group=reference_group)


def preview_figure(fig, scope='Exploratory sample'):
    """Display inline without writing files; retain the figure for optional export."""
    from IPython.display import display
    fig.text(0.5, -0.01, scope, ha='center', fontsize=10)
    display(fig)
    plt.close(fig)  # Avoid duplicate automatic output; fig remains saveable.
    return fig


def export_figure(fig, output_dir, name):
    """Save a previously previewed figure without replotting or resampling."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
