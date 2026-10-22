#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vae_core import VAE, NPZIterable, collate


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default=os.environ.get('DESI_VAE_DATA_DIR'))
    p.add_argument('--pattern', default='*_prep_1000.npz')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--resume', help='last.pt from this script; resume at next completed epoch')
    p.add_argument('--epochs', type=int, default=30, help='Total epochs, including resumed epochs')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beta', type=float, default=1e-3)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--test-fraction', type=float, default=0.1)
    p.add_argument('--exclude-ids', help='.npy array of confirmed transient TARGETIDs')
    p.add_argument('--nfeat', type=int, default=1000)
    p.add_argument('--latent-dim', type=int, default=16)
    p.add_argument('--channels', type=int, nargs=4, default=[32, 64, 128, 256])
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--evaluate-test', action='store_true', help='Evaluate best.pt after training')
    a = p.parse_args()
    if not a.data_dir:
        p.error('Set --data-dir or DESI_VAE_DATA_DIR')
    if min(a.epochs, a.batch_size, a.threads, a.patience, a.latent_dim, a.log_every, *a.channels) < 1:
        p.error('Epochs, batch size, threads, patience, model sizes and log interval must be positive')
    if a.workers < 0 or a.warmup_epochs < 0 or a.beta < 0 or a.lr <= 0:
        p.error('Invalid workers, warmup, beta or learning rate')
    if a.nfeat < 8 or a.nfeat % 8:
        p.error('nfeat must be a positive multiple of 8 for this decoder')
    if not (0 < a.val_fraction and 0 < a.test_fraction and a.val_fraction + a.test_fraction < 1):
        p.error('Positive validation/test fractions must sum to less than 1')
    return a


def atomic_save(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_epoch(model, loader, device, beta, log_every, optimizer=None):
    training = optimizer is not None
    model.train(training)
    synchronize(device)
    started = time.perf_counter()
    sums = torch.zeros(4, device=device, dtype=torch.float64)
    count = batches = 0
    data_wait = 0.0
    iterator = iter(loader)
    with torch.set_grad_enabled(training):
        while True:
            waiting = time.perf_counter()
            try:
                X, W, _, _ = next(iterator)
            except StopIteration:
                break
            data_wait += time.perf_counter() - waiting
            x = X.to(device, non_blocking=True)
            w = W.to(device, non_blocking=True)
            if x.shape[-1] != model.nfeat:
                raise ValueError(f'Expected {model.nfeat} pixels, received {x.shape[-1]}')
            if training:
                optimizer.zero_grad(set_to_none=True)
            values = model.loss(x, w=w, beta=beta)
            loss = values[0]
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Non-finite objective; inspect flux/weights')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            count += len(X)
            batches += 1
            sums += torch.stack([v.detach() for v in values]).double() * len(X)
            if batches % log_every == 0:
                synchronize(device)
                elapsed = time.perf_counter() - started
                print(f'  {"train" if training else "val/test"} batch={batches} '
                      f'spectra={count:,} elapsed={elapsed/60:.1f}min '
                      f'rate={count/elapsed:.0f} spectra/s', flush=True)
    synchronize(device)
    seconds = time.perf_counter() - started
    if not count:
        raise ValueError('Empty split after quality cuts; check files and TARGETIDs')
    means = (sums / count).cpu().tolist()
    return dict(zip(('loss', 'rec', 'mae', 'kl'), means), n=count, batches=batches,
                seconds=seconds, data_wait_seconds=data_wait, spectra_per_second=count/seconds)


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Request a GPU and load a GPU PyTorch environment.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    files = sorted(str(p.resolve()) for p in data_dir.glob(args.pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f'No {args.pattern} files in {data_dir}')
    # Small manifest catches changed inputs on resume without hashing large spectra files.
    manifest = [[f, Path(f).stat().st_size, Path(f).stat().st_mtime_ns] for f in files]
    excluded = np.load(args.exclude_ids, allow_pickle=False) if args.exclude_ids else np.empty(0, np.int64)
    excluded = np.asarray(excluded, dtype=np.int64).reshape(-1)
    model_kwargs = dict(nfeat=args.nfeat, ncode=args.latent_dim, ch1=args.channels[0],
                        ch2=args.channels[1], ch3=args.channels[2], ch4=args.channels[3],
                        dropout=0.02, use_bn=True)
    config = vars(args).copy()
    config['data_dir'] = str(data_dir)
    for key in ('resume', 'epochs', 'output_dir', 'evaluate_test', 'log_every', 'device', 'threads'):
        config.pop(key)
    out = Path(args.output_dir).expanduser().resolve()
    checkpoint = None
    if args.resume:
        resume = Path(args.resume).expanduser().resolve()
        if resume.parent != out or resume.name != 'last.pt':
            raise ValueError('Resume last.pt in its original --output-dir')
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 1:
            raise ValueError('Resume requires a checkpoint produced by this script')
        if checkpoint['config'] != config or checkpoint['manifest'] != manifest:
            raise ValueError('Resume configuration or inputs changed; use the original training arguments')
        if checkpoint['excluded_targetids'] != excluded.tolist():
            raise ValueError('Exclusion IDs changed since checkpoint')
        if not (out / 'best.pt').is_file():
            raise FileNotFoundError('Keep best.pt beside last.pt when resuming')
    else:
        out.mkdir(parents=True, exist_ok=False)

    datasets = {
        split: NPZIterable(files, split=split, shuffle=split == 'train',
                           seed=args.seed, val_fraction=args.val_fraction,
                           test_fraction=args.test_fraction, exclude_targetids=excluded)
        for split in ('train', 'val', 'test')
    }
    worker_options = dict(multiprocessing_context='spawn', prefetch_factor=2) if args.workers else {}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                num_workers=args.workers, collate_fn=collate,
                                pin_memory=device.type == 'cuda', persistent_workers=False,
                                **worker_options)
               for split, dataset in datasets.items()}
    model = VAE(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    start_epoch, best_val, bad_epochs, history = 1, float('inf'), 0, []
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val, bad_epochs = checkpoint['best_val'], checkpoint['bad_epochs']
        history = checkpoint['history']
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        print(f'Resuming after completed epoch {start_epoch - 1}', flush=True)
    (out / 'run_config.json').write_text(json.dumps(dict(config=config, model_kwargs=model_kwargs,
        input_files=files, torch_version=str(torch.__version__)), indent=2))
    print(f'Files={len(files)} device={device} workers={args.workers} output={out}', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if bad_epochs >= args.patience:
            print('Early-stopping condition already met.', flush=True)
            break
        datasets['train'].epoch = epoch
        beta = args.beta * min(1.0, epoch / max(1, args.warmup_epochs))
        train = run_epoch(model, loaders['train'], device, beta, args.log_every, optimizer)
        val = run_epoch(model, loaders['val'], device, args.beta, args.log_every)
        scheduler.step(val['loss'])
        improved = val['loss'] < best_val
        if improved:
            best_val, bad_epochs = val['loss'], 0
        elif epoch >= args.warmup_epochs:
            bad_epochs += 1
        history.append(dict(epoch=epoch, beta=beta, train=train, val=val))
        state = dict(format_version=1, config=config, manifest=manifest,
                     excluded_targetids=excluded.tolist(), model_kwargs=model_kwargs,
                     model_state_dict=model.state_dict(), opt_state_dict=optimizer.state_dict(),
                     scheduler_state_dict=scheduler.state_dict(), epoch=epoch, beta=beta,
                     beta_max=args.beta, best_val=best_val, bad_epochs=bad_epochs, history=history,
                     loss_version='masked_ivar_mean_per_valid_pixel_v1',
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        saving = time.perf_counter()
        # Best first: last.pt never refers to a best checkpoint not yet written.
        if improved:
            atomic_save(state, out / 'best.pt')
        atomic_save(state, out / 'last.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'Epoch {epoch}/{args.epochs}: train={train["loss"]:.6g} '
              f'val={val["loss"]:.6g} beta={beta:.3g} '
              f'train_min={train["seconds"]/60:.2f} val_min={val["seconds"]/60:.2f} '
              f'save_s={time.perf_counter()-saving:.1f}', flush=True)
    if args.evaluate_test:
        best = torch.load(out / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(best['model_state_dict'])
        metrics = run_epoch(model, loaders['test'], device, args.beta, args.log_every)
        (out / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
        print('Held-out test:', metrics, flush=True)


if __name__ == '__main__':
    main()
