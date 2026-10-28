"""Encode transient spectra with each pretrained VAE.

Classifier training and held-out predictions live in fold_classifier_training.py.
Existing feature files are reused. After changing data, preprocessing, or VAEs,
pass overwrite_folds=tuple(runs) to regenerate them.
"""
from pathlib import Path
import json

import h5py
import numpy as np
import torch

import evaluate_cv as ev
import vae_model_cv as vm


@torch.inference_mode()
def encode_shared_transients(
    runs, transient_file, transient_config, output_dir,
    device="cpu", batch_size=512, overwrite_folds=(), allow_missing_wave=False,
):
    """Save one transient feature file per VAE; return (paths, None).

    The second return value preserves the notebook's ``outputs, _ = ...`` call.
    ``allow_missing_wave`` is accepted for older calls but is unused: this helper
    no longer validates wavelength grids or other inputs.

    All usable transients are encoded by every VAE. TARGETID assigns each target
    to one of five folds using the first run's saved fold-assignment settings.
    The classifier helper selects the four training folds and the holdout fold.
    """
    source = Path(transient_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_config = runs[min(runs)]["data_config"]
    outputs = {}

    for fold, run in sorted(runs.items()):
        destination = output_dir / f"transients_fold{fold}.h5"
        outputs[fold] = str(destination)
        if destination.exists() and fold not in overwrite_folds:
            print(f"Fold {fold}: reusing {destination.name}", flush=True)
            continue

        config = ev.source_config(run, transient_config)
        checkpoint = vm.load_checkpoint(run["checkpoint"])
        model = vm.VAE.from_checkpoint(checkpoint, device)
        model.eval()
        del checkpoint
        block_rows = int(config.get("block_rows", 4096))
        temporary = destination.with_suffix(".partial.h5")

        with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
            n_rows = len(vm.h5_field(src, config["targetid_key"]))
            dst.attrs.update(
                source=str(source), source_rows=n_rows,
                source_config=json.dumps(config, default=str),
                transient=True, shared_transients=True, fold=fold,
            )
            for start in range(0, n_rows, block_rows):
                stop = min(start + block_rows, n_rows)
                ids = ev.read_ids(vm.h5_field(src, config["targetid_key"])[start:stop])
                flux, weights = ev.masked_block(src, config, start, stop)

                # Keep the same usable-pixel selection as the VAE preprocessing.
                keep = (weights > 0).sum(axis=1) >= config.get("min_valid_pixels", 50)
                if config.get("valid_rows_key"):
                    keep &= np.asarray(
                        vm.h5_field(src, config["valid_rows_key"])[start:stop]
                    ).reshape(-1).astype(bool)
                rows = np.flatnonzero(keep)
                owners = ev.owner_fold(ids, fold_config)
                type_key = config.get("type_key")
                types = (
                    np.char.strip(ev.text_array(vm.h5_field(src, type_key)[start:stop]))
                    if type_key else np.full(len(ids), "UNKNOWN")
                )

                for offset in range(0, len(rows), batch_size):
                    selected = rows[offset:offset + batch_size]
                    x = torch.as_tensor(flux[selected], device=device)
                    mu = model(x, sample=False)[1].float().cpu().numpy()
                    ev.append(dst, dict(
                        mu=mu,
                        TARGETID=ids[selected],
                        SPECTYPE=np.full(len(selected), "TRANSIENT"),
                        TYPE=types[selected],
                        known_transient=np.ones(len(selected), dtype=bool),
                        source_row=start + selected,
                        owner_fold=owners[selected],
                    ))
            count = len(dst["mu"]) if "mu" in dst else 0
            dst.attrs.update(scored_rows=count, complete=True)

        temporary.replace(destination)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"Fold {fold}: encoded {count:,} transient spectra", flush=True)

    return outputs, None
