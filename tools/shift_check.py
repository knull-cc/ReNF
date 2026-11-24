#!/usr/bin/env python3
"""Label-free check: does the input centering that generalises better also make the network INPUT shift less
between the training period and later periods? For each centering a (window mean, last value, mean of the last
period, EWMA), the normalised input windows z = (x - a(x)) / std(x) of the train / val / test splits are
projected on the top-k principal directions of the training z, and the Frechet (Gaussian) distance between the
train distribution and the val (or test) distribution is reported relative to the training spread trace(S_train).
Only inputs are used (no labels), so the val column is a legitimate selection criterion.
"""
import argparse
import numpy as np
from anchor_check import DATA, splits, windows, ewma_w


def norm_inputs(X, how, P):
    s = X.std(1, keepdims=True) + 1e-5
    if how == 'mean':
        a = X.mean(1, keepdims=True)
    elif how == 'last':
        a = X[:, -1:]
    elif how == 'lastP':
        a = X[:, -P:].mean(1, keepdims=True)
    else:
        a = (X @ ewma_w(X.shape[1], float(how[4:])))[:, None]
    return (X - a) / s


def psd_sqrt(S):
    w, U = np.linalg.eigh(S)
    return (U * np.sqrt(np.clip(w, 0, None))) @ U.T


def frechet(Z1, Z2):
    m1, m2 = Z1.mean(0), Z2.mean(0)
    S1, S2 = np.cov(Z1, rowvar=False), np.cov(Z2, rowvar=False)
    r = psd_sqrt(S1)
    cross = np.sqrt(np.clip(np.linalg.eigvalsh(r @ S2 @ r), 0, None)).sum()
    return float(((m1 - m2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * cross), float(np.trace(S1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='../dataset')
    ap.add_argument('--datasets', nargs='+', default=['etth1', 'etth2', 'ettm1', 'ettm2', 'weather'])
    ap.add_argument('--H', type=int, default=96)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--max_win', type=int, default=4000)
    a = ap.parse_args()
    for ds in a.datasets:
        _, _, T, P = DATA[ds]
        Xs = [windows(s, T, a.H, a.max_win)[0].astype(np.float64) for s in splits(a.root, ds, T)]
        print(f'\n=== {ds}  T={T}  input shift train->val / train->test, Frechet distance / trace(S_train), '
              f'top-{a.k} PCs of the training inputs')
        out = {}
        for how in ['mean', 'last', 'lastP', 'ewma0.01', 'ewma0.03']:
            Z = [norm_inputs(X, how, P) for X in Xs]
            mu = Z[0].mean(0)
            _, _, Vt = np.linalg.svd(Z[0] - mu, full_matrices=False)
            B = Vt[:a.k].T
            Zp = [(z - mu) @ B for z in Z]
            fv, tr = frechet(Zp[0], Zp[1])
            ft, _ = frechet(Zp[0], Zp[2])
            out[how] = (fv / tr, ft / tr)
            print(f'  {how:9s} train->val {fv / tr:7.4f}   train->test {ft / tr:7.4f}')
        print(f'  smallest shift   val: {min(out, key=lambda h: out[h][0])}   test: {min(out, key=lambda h: out[h][1])}'
              f'   (last vs mean, val: {out["last"][0] / out["mean"][0]:.2f}x, test: {out["last"][1] / out["mean"][1]:.2f}x)')


if __name__ == '__main__':
    main()
