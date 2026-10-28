"""Train one classifier per VAE and predict its held-out fold.

Use fit=True after changing data, checkpoints, sampling, or model settings.
Use fit=False to load existing classifiers and regenerate predictions.
"""
from pathlib import Path
import gzip

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight

import evaluate_cv as ev
import vae_model_cv as vm


def read_batches(paths, run, keys, batch_size=65536, transient_fold_config=None):
    """Read scored spectra in batches."""
    for path in paths:
        with h5py.File(path, "r") as h:
            count = int(h.attrs["scored_rows"])
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                data = {
                    key: (h[key][start:stop] if key not in {"rchi2", "n_valid"} or key in h
                          else np.full(stop - start, np.nan))
                    for key in set(keys) | {"TARGETID", "source_row"}
                }
                fold_config = (transient_fold_config
                               if bool(h.attrs["transient"]) and transient_fold_config is not None
                               else run["data_config"])
                owners = ev.owner_fold(data["TARGETID"], fold_config)
                yield str(path), str(h.attrs["source"]), np.arange(start, stop), data, owners


def select_background(outputs, runs, fold, cap_per_class=None, seed=42):
    """Sample background spectra from this VAE's training split."""
    dc = runs[fold]['data_config']
    loader = vm.make_split_loader(dc, 'train', 'cpu', num_workers=0, shuffle=False)
    train_paths = {str(Path(p).resolve()) for p in loader.dataset.paths}
    del loader
    rng, pools = (np.random.default_rng(seed), {})
    for owner, paths in sorted(outputs.items()):
        if owner == fold:
            continue
        for path in paths:
            with h5py.File(path, 'r') as h:
                if bool(h.attrs['transient']) or str(Path(h.attrs['source']).resolve()) not in train_paths:
                    continue
            for _, source, _, a, _ in read_batches([path], runs[owner], ['SPECTYPE', 'known_transient']):
                ids = a['TARGETID']
                labels = ev.text_array(a['SPECTYPE'])
                eligible = vm.split_codes(ids, dc['seed'], dc['val_fraction'], 5, fold) == 0
                eligible &= ~a['known_transient'].astype(bool)
                if dc.get('exclude_targetids') is not None:
                    eligible &= ~np.isin(ids, np.asarray(dc['exclude_targetids']))
                if dc.get('keep_spectypes') is not None:
                    eligible &= np.isin(labels, [str(s).strip().upper() for s in dc['keep_spectypes']])
                for label in ev.LABELS[:-1]:
                    ii = np.flatnonzero(eligible & (labels == label))
                    if not len(ii):
                        continue
                    frame = pd.DataFrame(dict(SPECTYPE=label, source_file=source, source_row=a['source_row'][ii], priority=rng.random(len(ii))))
                    if label in pools:
                        frame = pd.concat([pools[label], frame], ignore_index=True)
                    if cap_per_class is not None and len(frame) > cap_per_class:
                        chosen = np.argpartition(frame.priority.to_numpy(), cap_per_class - 1)[:cap_per_class]
                        frame = frame.iloc[chosen].reset_index(drop=True)
                    pools[label] = frame
    meta = pd.concat([pools[label] for label in ev.LABELS[:-1]], ignore_index=True)
    meta = meta.drop(columns='priority').sort_values(['source_file', 'source_row']).reset_index(drop=True)
    return meta


@torch.inference_mode()
def encode_background(meta, run, device='cpu', batch_size=512):
    """Encode the selected spectra with the matching VAE."""
    dc = run['data_config']
    model = vm.VAE.from_checkpoint(vm.load_checkpoint(run["checkpoint"]), device)
    model.eval()
    result = np.empty((len(meta), run['model_config']['ncode']), dtype=np.float32)
    block_rows = int(dc.get('block_rows', 4096))
    for source, part in meta.groupby('source_file', sort=False):
        with h5py.File(source, 'r') as h:
            count = len(vm.h5_field(h, dc['targetid_key']))
            for block, group in part.groupby(part.source_row // block_rows, sort=True):
                start, stop = (int(block) * block_rows, min((int(block) + 1) * block_rows, count))
                x, _ = ev.masked_block(h, dc, start, stop)
                local = group.source_row.to_numpy(dtype=int) - start
                for offset in range(0, len(group), batch_size):
                    ii = local[offset:offset + batch_size]
                    mu = model.encode(torch.as_tensor(x[ii], device=device))[0].float().cpu().numpy()
                    result[group.index.to_numpy()[offset:offset + batch_size]] = mu
    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return result


def load_transients(path, fold, run, transient_fold_config):
    """Use transient features from folds outside the held-out fold."""
    features = []
    for _, _, _, data, owners in read_batches([path], run, ["mu"], transient_fold_config=transient_fold_config):
        features.append(data["mu"][owners != fold])
    return np.vstack(features)


def predict_holdout(classifier, paths, fold, run, stream, write_header=False, batch_size=8192, transient_fold_config=None):
    """Predict only the held-out TARGETIDs and held-out transients."""
    count = 0
    for path, source, rows, data, owners in read_batches(paths, run, ['mu', 'SPECTYPE', 'TYPE', 'known_transient', 'rchi2', 'n_valid'], batch_size, transient_fold_config):
        keep = owners == fold
        data = {key: values[keep] for key, values in data.items()}
        rows = rows[keep]
        if not len(rows):
            continue
        ids = ev.text_array(data['TARGETID'])
        probabilities = classifier.predict_proba(data['mu'])
        frame = pd.DataFrame(dict(
            TARGETID=ids, source_file=source, source_row=data['source_row'],
            score_file=path, score_row=rows, fold=fold,
            SPECTYPE=ev.text_array(data['SPECTYPE']), TYPE=ev.text_array(data['TYPE']),
            known_transient=data['known_transient'], role='out_of_fold',
            predicted=np.asarray(ev.LABELS)[classifier.classes_[probabilities.argmax(axis=1)]],
            p_predicted=probabilities.max(axis=1),
            rchi2=data['rchi2'], n_valid=data['n_valid'],
        ))
        for j, class_id in enumerate(classifier.classes_):
            label = ev.LABELS[class_id]
            frame[f'p_{label.lower()}'] = probabilities[:, j]
        frame.to_csv(stream, index=False, header=write_header and count == 0)
        count += len(frame)
    return count


def train_classifier(background, background_labels, transients, *, seed=42, iterations=500):
    """Fit one XGBoost classifier on latent means from one VAE."""
    from xgboost import XGBClassifier

    X = np.vstack([background, transients])
    labels = np.concatenate([background_labels, np.full(len(transients), "TRANSIENT"),])
    y = np.array([ev.LABELS.index(label) for label in labels])
    classifier = XGBClassifier(
        n_estimators=iterations,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=8,
        random_state=seed,
    )
    classifier.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return classifier, labels


def run_fivefold(outputs, shared_outputs, runs, output_dir, *, cap_per_class=None,
                 seed=42, iterations=500, fit=True,
                 device="cpu", batch_size=512, prediction_batch_size=8192):
    """Fit or load five classifiers, then export their held-out predictions.

    Background spectra are sampled from each VAE's training split and encoded
    with that VAE. Transients use the same VAE's shared feature cache, excluding
    its held-out fold. Both background and transient holdouts are predicted.
    TARGETID hashing assigns transients consistently across all five VAEs.
    The classifiers use the latent means (mu) as features.
    cap_per_class=None keeps all eligible background rows present in outputs.
    The original VAE training split is preserved (its validation rows excluded).
    outputs must cover the full background holdouts, without score subsampling.
    Existing feature helpers are required; this module adds no validation checks.
    """
    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "all_spectra_fivefold_probabilities.csv.gz"
    temporary = destination.with_name(destination.name + ".partial")
    transient_fold_config = runs[min(runs)]["data_config"]
    summaries = []
    total = 0

    with gzip.open(temporary, "wt", newline="") as stream:
        for fold, run in sorted(runs.items()):
            model_path = model_dir / f"classifier_fold{fold}.joblib"
            if fit:
                rows = select_background(outputs, runs, fold, cap_per_class, seed + fold)
                background = encode_background(rows, run, device, batch_size)
                transients = load_transients(shared_outputs[fold], fold, run, transient_fold_config)
                classifier, labels = train_classifier(
                    background, rows.SPECTYPE.to_numpy(), transients,
                    seed=seed + fold, iterations=iterations,
                )
                summary = dict(
                    fold=fold, background_train=len(background),
                    transient_train=len(transients),
                    **{f"n_{label.lower()}": int((labels == label).sum())
                       for label in ev.LABELS},
                )
                joblib.dump(dict(classifier=classifier, counts=summary), model_path)
                del rows, background, transients, labels
            else:
                saved = joblib.load(model_path)
                classifier = saved["classifier"]
                summary = saved["counts"].copy()

            holdout_paths = []
            for path in outputs[fold]:
                with h5py.File(path, "r") as h:
                    if not h.attrs["transient"]:
                        holdout_paths.append(path)
            holdout_paths.append(shared_outputs[fold])
            summary["heldout_spectra"] = predict_holdout(
                classifier, holdout_paths, fold, run, stream,
                write_header=(total == 0), batch_size=prediction_batch_size,
                transient_fold_config=transient_fold_config,
            )
            total += summary["heldout_spectra"]
            summaries.append(summary)
            print(
                f"Fold {fold}: {summary['background_train']:,} background + "
                f"{summary['transient_train']:,} transient training spectra; "
                f"{summary['heldout_spectra']:,} holdout predictions",
                flush=True,
            )
            del classifier

    temporary.replace(destination)
    counts = pd.DataFrame(summaries)
    counts.to_csv(output_dir / "classifier_training_counts.csv", index=False)
    return destination, counts
