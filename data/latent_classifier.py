"""Train and evaluate classifiers on saved VAE posterior means, without retraining the VAE."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score)
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def target_partition(targetids, seed=42):
    """Matches the existing DESI VAE training script's TARGETID hash."""
    x = np.asarray(targetids, dtype=np.int64).astype(np.uint64) ^ np.uint64(seed)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def load_classifier_data(save_dir, task='multiclass', seed=42, val_fraction=0.1,
                         test_fraction=0.1, max_train_per_source=100000,
                         split_file=None, sources=None):
    """Read only real labeled populations; preserve original row provenance.

    Optional split_file: NPZ with galaxy/qso/star/transient string arrays containing
    train/val/test/exclude, each aligned with the original corresponding mu array.
    Hash splitting is a fallback; it is not evidence that the VAE never saw a row.
    """
    if task not in {'binary', 'multiclass'}:
        raise ValueError('task must be binary or multiclass')
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError('Invalid validation/test fractions')
    if max_train_per_source is not None and max_train_per_source < 1:
        raise ValueError('Training cap must be positive or None')
    root = Path(save_dir)
    sources = sources or [('galaxy', 'dr2_latents.npz', 'mu_galaxy', 'tid_galaxy'),
               ('qso', 'dr2_latents.npz', 'mu_qso', 'tid_qso'),
               ('star', 'dr2_latents.npz', 'mu_star', 'tid_star'),
               ('transient', 'transient_latents.npz', 'mu', 'TARGETID')]
    if [s[0] for s in sources] != ['galaxy', 'qso', 'star', 'transient']:
        raise ValueError('sources must specify galaxy, qso, star, transient in that order')
    raw, overrides, audit = {}, {}, []
    if split_file is not None:
        with np.load(split_file, allow_pickle=False) as f:
            overrides = {k: f[k].astype(str) for k in ('galaxy', 'qso', 'star', 'transient')}
    for key, filename, mu_key, id_key in sources:
        with np.load(root / filename, allow_pickle=False) as f:
            if mu_key not in f or id_key not in f:
                raise KeyError(f'{filename}: need {mu_key!r} and {id_key!r}; '
                               f'available keys: {f.files}. Edit SOURCES to match your files.')
            mu, ids = f[mu_key], f[id_key].reshape(-1)
        if mu.ndim != 2 or len(mu) != len(ids):
            raise ValueError(f'{key}: expected mu=(N,D) and aligned TARGETIDs')
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f'{key}: IDs must be integer arrays, not floats (precision risk)')
        raw[key] = (mu, ids.astype(np.int64))
    dimensions = {mu.shape[1] for mu, _ in raw.values()}
    if len(dimensions) != 1 or next(iter(dimensions)) < 1:
        raise ValueError('Latent dimensions must match across groups')
    # A target known to host a transient must not also enter as a negative.
    transient_ids = raw['transient'][1]
    ambiguous = np.array([], dtype=np.int64)
    if task == 'multiclass':
        for a, b in [('galaxy', 'qso'), ('galaxy', 'star'), ('qso', 'star')]:
            ambiguous = np.union1d(ambiguous, np.intersect1d(raw[a][1], raw[b][1]))
    rng = np.random.default_rng(seed)
    arrays, frames = [], []
    labels = {'galaxy': 0, 'qso': 0 if task == 'binary' else 1,
              'star': 0 if task == 'binary' else 2,
              'transient': 1 if task == 'binary' else 3}
    for key, filename, _, _ in sources:
        mu, ids = raw[key]
        if split_file is not None:
            split = overrides[key]
            if split.shape != (len(mu),) or not np.isin(split, ['train', 'val', 'test', 'exclude']).all():
                raise ValueError(f'{key}: split array must align with original rows and contain valid labels')
        else:
            h = target_partition(ids, seed)
            split = np.where(h < test_fraction, 'test',
                             np.where(h < test_fraction + val_fraction, 'val', 'train'))
        finite = np.isfinite(mu).all(axis=1)
        conflict = np.zeros(len(mu), dtype=bool)
        if key != 'transient':
            conflict |= np.isin(ids, transient_ids)
            if task == 'multiclass':
                conflict |= np.isin(ids, ambiguous)
        keep = finite & ~conflict & (split != 'exclude')
        # Split before sampling. Validation and test retain their available populations.
        training = np.flatnonzero(keep & (split == 'train'))
        if max_train_per_source is not None and len(training) > max_train_per_source:
            keep[training] = False
            keep[rng.choice(training, max_train_per_source, replace=False)] = True
        rows = np.flatnonzero(keep)
        arrays.append(mu[rows].astype(np.float32))
        frames.append(pd.DataFrame(dict(source=key, source_file=filename, source_row=rows,
                                        TARGETID=ids[rows], label=labels[key], split=split[rows])))
        audit.append(dict(source=key, original=len(mu), nonfinite=int((~finite).sum()),
                          conflicting_reference=int(conflict.sum()), retained=len(rows)))
    x = np.concatenate(arrays)
    metadata = pd.concat(frames, ignore_index=True)
    if (metadata.groupby('TARGETID')['split'].nunique() > 1).any():
        raise ValueError('TARGETID leakage: supplied split assignments cross partitions')
    if (metadata.groupby('TARGETID')['label'].nunique() > 1).any():
        raise ValueError('Conflicting class labels remain for the same TARGETID')
    y = metadata['label'].to_numpy()
    names = ['Galaxy/QSO/Star', 'Transient'] if task == 'binary' else ['Galaxy', 'QSO', 'Star', 'Transient']
    for part in ('train', 'val', 'test'):
        counts = np.bincount(y[metadata['split'].eq(part)], minlength=len(names))
        if (counts == 0).any():
            raise ValueError(f'{part} lacks a class: {counts}. Supply suitable ORIGINAL group splits; '
                             'do not repeatedly search seeds for a better test result.')
    return x, y, metadata, names, pd.DataFrame(audit)


def fit_candidates(x, y, metadata, names, seed=42):
    train = metadata['split'].eq('train').to_numpy()
    val = metadata['split'].eq('val').to_numpy()
    candidates = {
        'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=2000, random_state=seed)),
        'Gradient boosted trees': HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
            l2_regularization=1.0, class_weight='balanced',
            early_stopping=False, random_state=seed),
    }
    rows, scores = [], {}
    for name, estimator in candidates.items():
        # Disable automatic row-level validation; use explicit target-group splits.
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            estimator.fit(x[train], y[train])
            p = estimator.predict_proba(x[val])
        scores[name] = p
        rows.append(dict(model=name, transient_AP=average_precision_score(
            y[val] == len(names) - 1, p[:, -1]),
            macro_F1=f1_score(y[val], estimator.classes_[p.argmax(axis=1)], average='macro')))
    table = pd.DataFrame(rows)
    metric = 'transient_AP' if len(names) == 2 else 'macro_F1'
    winner = table.sort_values(metric, ascending=False, kind='stable').iloc[0]['model']
    return candidates, table, scores, winner


def choose_threshold(y_binary, scores, target_fpr=0.01):
    """Maximize validation recall subject to empirical FPR <= target_fpr; handle ties."""
    if not 0 <= target_fpr < 1:
        raise ValueError('target_fpr must lie in [0,1)')
    fpr, tpr, thresholds = roc_curve(y_binary, scores, drop_intermediate=False)
    # Explicit finite reject-all threshold, also compatible with older sklearn.
    thresholds[0] = np.nextafter(float(np.max(scores)), np.inf)
    allowed = np.flatnonzero(fpr <= target_fpr)
    best = allowed[np.argmax(tpr[allowed])]
    return float(thresholds[best]), dict(validation_FPR=float(fpr[best]),
                                        validation_recall=float(tpr[best]),
                                        negative_rows=int((~np.asarray(y_binary, dtype=bool)).sum()))


def evaluate(estimator, x, y, names, threshold):
    with threadpool_limits(limits=1):
        probabilities = estimator.predict_proba(x)
    scores = probabilities[:, -1]
    truth = y == len(names) - 1
    selected = scores >= threshold
    metrics = dict(transient_AP=float(average_precision_score(truth, scores)),
                   transient_ROC_AUC=float(roc_auc_score(truth, scores)),
                   transient_recall=float(recall_score(truth, selected, zero_division=0)),
                   transient_precision=float(precision_score(truth, selected, zero_division=0)),
                   background_FPR=float(selected[~truth].mean()),
                   selected_rows=int(selected.sum()), rows=len(y),
                   transient_fraction=float(truth.mean()), threshold=threshold)
    prediction = selected.astype(int) if len(names) == 2 else estimator.classes_[probabilities.argmax(1)]
    metrics['macro_F1'] = float(f1_score(y, prediction, average='macro'))
    return metrics, probabilities, prediction


def validation_plot(y_val, scores, names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    truth = y_val == len(names) - 1
    for (name, p), color, style in zip(scores.items(), ['#356A9A', '#C45E16'], ['-', '--']):
        precision, recall, _ = precision_recall_curve(truth, p[:, -1])
        fpr, tpr, _ = roc_curve(truth, p[:, -1])
        axes[0].plot(recall, precision, color=color, linestyle=style, label=name)
        axes[1].plot(fpr, tpr, color=color, linestyle=style, label=name)
    axes[0].axhline(truth.mean(), color='gray', linestyle=':', label='Sample prevalence')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':')
    axes[0].set(xlabel='Transient recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel='Background false-positive rate', ylabel='Transient recall', xlim=(0, 1), ylim=(0, 1.02))
    for ax in axes:
        ax.legend(fontsize=9); ax.grid(alpha=0.15)
    fig.suptitle('Validation only · transient versus rest · precision depends on sample composition')
    return fig


def test_plot(y, probabilities, prediction, names, threshold):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    edges = np.linspace(0, 1, 41)
    for j, (name, color, style) in enumerate(zip(names, ['#356A9A', '#7953A2', '#27816B', '#C45E16'], ['-', ':', '-.', '--'])):
        axes[0].hist(probabilities[y == j, -1], bins=edges, density=True, histtype='step',
                     color=color, linestyle=style, label=f'{name} (n={(y == j).sum():,})')
    axes[0].axvline(threshold, color='gray', linestyle=':', label='Validation threshold')
    axes[0].set(xlabel='Transient score (not calibrated probability)', ylabel='Density')
    axes[0].legend(fontsize=8)
    precision, recall, _ = precision_recall_curve(y == len(names) - 1, probabilities[:, -1])
    axes[1].plot(recall, precision, color='#356A9A')
    axes[1].axhline((y == len(names) - 1).mean(), color='gray', linestyle=':')
    axes[1].set(xlabel='Transient recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
    ConfusionMatrixDisplay(confusion_matrix(y, prediction, labels=range(len(names))),
                           display_labels=names).plot(ax=axes[2], colorbar=False,
                                                     cmap='Blues', values_format='d')
    axes[2].set_title('Fixed threshold' if len(names) == 2 else 'Multiclass argmax')
    fig.suptitle('Held-out classifier test · chosen model and threshold frozen on validation')
    return fig


def preview(fig):
    from IPython.display import display
    display(fig)
    plt.close(fig)
    return fig


def class_pr_plot(y, probabilities, names, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for j, (ax, name, color) in enumerate(zip(axes.flat, names,
                                             ['#356A9A', '#7953A2', '#27816B', '#C45E16'])):
        truth = y == j
        precision, recall, _ = precision_recall_curve(truth, probabilities[:, j])
        ap = average_precision_score(truth, probabilities[:, j])
        ax.plot(recall, precision, color=color, label=f'AP={ap:.3f}')
        ax.axhline(truth.mean(), color='gray', linestyle=':', label='Sample prevalence')
        ax.set(title=f'{name} versus rest', xlabel='Recall', ylabel='Precision',
               xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(); ax.grid(alpha=0.15)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.suptitle(title)
    return fig


def predict_latents(bundle, mu):
    """Score new means from the SAME encoder checkpoint/preprocessing."""
    mu = np.asarray(mu)
    estimator = bundle['classifier']
    if mu.ndim != 2 or mu.shape[1] != estimator.n_features_in_ or not np.isfinite(mu).all():
        raise ValueError('Expected finite (N, trained_latent_dimension) means')
    with threadpool_limits(limits=1):
        p = estimator.predict_proba(mu)
    selected = p[:, -1] >= bundle['threshold']
    labels = selected.astype(int) if len(bundle['class_names']) == 2 else estimator.classes_[p.argmax(1)]
    result = pd.DataFrame(dict(transient_score=p[:, -1], selected_as_transient=selected,
                               predicted_class=np.asarray(bundle['class_names'])[labels]))
    for j, name in enumerate(bundle['class_names']):
        result[f'score_{name}'] = p[:, j]
    return result
