#!/usr/bin/env python3
"""Model-free anchor check (no training). For each split (train / val / test, official borders and scaling)
and each candidate anchor a(x) of the input window, the level error of the naive forecast
    E[(ybar - a(x))^2],   ybar = mean of the next H values,
which is the only part of a forecast's MSE that depends on the anchor (the within-horizon variance is the
same for every anchor). Anchors: window mean, last value, mean of the last seasonal period, EWMA with several
alphas (weights (1-a)^(T-1-t), as in SimpleMLP --revin_ewma), and a constrained linear anchor w.x with
sum(w) = 1 fitted on the TRAINING split only (ridge; a variogram / kriging-style level estimate).

Question it answers: does the training period already prefer the anchor that wins on val / test (then the
training-loss preference for the mean seen with the MLP is a model / optimisation effect), or does the
preference only appear after the training period (then it is drift that the training data cannot show)?
"""
import argparse
import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DATA = {  # name: (file, rule, seq_len (official H=96 config), seasonal period in steps)
    'etth1': ('ETT-small/ETTh1.csv', 'ett_h', 512, 24),
    'etth2': ('ETT-small/ETTh2.csv', 'ett_h', 512, 24),
    'ettm1': ('ETT-small/ETTm1.csv', 'ett_m', 512, 96),
    'ettm2': ('ETT-small/ETTm2.csv', 'ett_m', 512, 96),
    'weather': ('weather/weather.csv', 'custom', 720, 144),
}


def splits(root, name, T):
    rel, rule, _, _ = DATA[name]
    x = pd.read_csv(os.path.join(root, rel)).drop(columns=['date']).values.astype(np.float64)
    if rule in ('ett_h', 'ett_m'):
        m = 1 if rule == 'ett_h' else 4
        b1 = [0, 12 * 30 * 24 * m - T, 12 * 30 * 24 * m + 4 * 30 * 24 * m - T]
        b2 = [12 * 30 * 24 * m, 12 * 30 * 24 * m + 4 * 30 * 24 * m, 12 * 30 * 24 * m + 8 * 30 * 24 * m]
    else:
        n_tr, n_te = int(len(x) * 0.7), int(len(x) * 0.2)
        n_va = len(x) - n_tr - n_te
        b1 = [0, n_tr - T, len(x) - n_te - T]
        b2 = [n_tr, n_tr + n_va, len(x)]
    tr = x[b1[0]:b2[0]]
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    return [((x[a:b] - mu) / sd).astype(np.float32) for a, b in zip(b1, b2)]


def windows(seg, T, H, max_win):
    n = seg.shape[0] - T - H + 1
    stride = max(1, int(np.ceil(n / max_win)))
    w = sliding_window_view(seg, T + H, axis=0)[::stride]      # (W, V, T+H)
    X = w[:, :, :T].reshape(-1, T)                             # (W*V, T)  channel-independent
    yb = w[:, :, T:].mean(axis=2).reshape(-1)
    return X, yb


def ewma_w(T, a):
    w = (1 - a) ** np.arange(T - 1, -1, -1, dtype=np.float64)
    return (w / w.sum()).astype(np.float32)


def fit_linear(X, yb, lam=1e-3):
    A = (X.T.astype(np.float64) @ X) / len(X)
    A += lam * np.trace(A) / len(A) * np.eye(len(A))
    b = (X.T.astype(np.float64) @ yb) / len(X)
    one = np.ones(len(A))
    Ai_b, Ai_1 = np.linalg.solve(A, b), np.linalg.solve(A, one)
    mu = (one @ Ai_b - 1.0) / (one @ Ai_1)
    return (Ai_b - mu * Ai_1).astype(np.float32)               # sum(w) = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='../dataset')
    ap.add_argument('--datasets', nargs='+', default=['etth1', 'etth2', 'ettm1', 'ettm2', 'weather'])
    ap.add_argument('--H', type=int, default=96)
    ap.add_argument('--max_win', type=int, default=6000)
    a = ap.parse_args()
    alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    for ds in a.datasets:
        _, _, T, P = DATA[ds]
        segs = splits(a.root, ds, T)
        data = [windows(s, T, a.H, a.max_win) for s in segs]
        w_lin = fit_linear(*data[0])
        anchors = {'mean': lambda X: X.mean(1), 'last': lambda X: X[:, -1],
                   f'lastP{P}': lambda X: X[:, -P:].mean(1)}
        for al in alphas:
            anchors[f'ewma{al}'] = (lambda w: (lambda X: X @ w))(ewma_w(T, al))
        anchors['linear(train-fit)'] = lambda X: X @ w_lin
        res = {k: [float(np.mean((yb - f(X)) ** 2)) for X, yb in data] for k, f in anchors.items()}
        print(f'\n=== {ds}  T={T} H={a.H} period={P}  level error E[(ybar-a)^2], scaled units;'
              f'  in brackets: % vs window mean (negative = better)')
        print(f'{"anchor":20s} {"train":>18s} {"val":>18s} {"test":>18s}')
        for k, v in res.items():
            rel = [(x / m - 1) * 100 for x, m in zip(v, res['mean'])]
            print(f'{k:20s} ' + ' '.join(f'{x:9.4f} ({r:+6.1f}%)' for x, r in zip(v, rel)))
        best = [min(res, key=lambda k: res[k][i]) for i in range(3)]
        print(f'best anchor   train: {best[0]}   val: {best[1]}   test: {best[2]}')
        w = w_lin
        print(f'linear anchor weight on the last 1 / {P} / {T} steps: {w[-1]:.3f} / {w[-P:].sum():.3f} / {w.sum():.3f}')


if __name__ == '__main__':
    main()
