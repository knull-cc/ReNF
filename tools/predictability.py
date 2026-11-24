#!/usr/bin/env python3
"""Model-free short-term predictability per dataset (direction B: does the value of ReNF's structure grow with
short-term predictability?). On the official splits and scaling, for horizons h:
  persist R2(h) = 1 - E[(x_{t+h} - x_t)^2] / Var(x)                       (validation split)
  linear  R2(h) = out-of-sample R2 of a ridge regression x_{t+h} ~ last T inputs, fitted on train, scored on val
Both channel-independent and pooled over variables. Then the Spearman correlation across datasets between a
predictability summary and the structure gain = MLP+recipe MSE / ReNF MSE - 1 at H=96.
"""
import argparse
import numpy as np
from anchor_check import DATA, splits
from numpy.lib.stride_tricks import sliding_window_view

# MLP + recipe (d1024, p0.5, mean-centred, 3 seeds, ours) vs ReNF (paper Table 11, H=96)
GAIN = {'etth1': (0.3570, 0.350), 'etth2': (0.2676, 0.261), 'ettm1': (0.2841, 0.270),
        'ettm2': (0.1752, 0.157), 'weather': (0.1503, 0.138)}
HS = [1, 4, 12, 24, 48, 96]


def lin_r2(tr, va, T, h, max_win=4000, lam=1e-2):
    def mk(seg):
        w = sliding_window_view(seg, T + h, axis=0)              # (W, V, T+h)
        w = w[::max(1, int(np.ceil(len(w) / max_win)))]
        X = w[:, :, :T].reshape(-1, T).astype(np.float64)
        y = w[:, :, T + h - 1].reshape(-1).astype(np.float64)
        return X - X[:, -1:], y - X[:, -1]                       # predict the change from the last value
    Xt, yt = mk(tr)
    Xv, yv = mk(va)
    A = Xt.T @ Xt / len(Xt)
    A += lam * np.trace(A) / len(A) * np.eye(len(A))
    w = np.linalg.solve(A, Xt.T @ yt / len(Xt))
    resid = yv - Xv @ w
    return 1 - np.mean(resid ** 2) / np.mean((yv - yv.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='../dataset')
    ap.add_argument('--datasets', nargs='+', default=list(GAIN))
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        _, _, T, _ = DATA[ds]
        tr, va, _ = splits(a.root, ds, T)
        var = va.var(0).mean()
        pr = [1 - np.mean((va[h:] - va[:-h]) ** 2) / var for h in HS]
        lr = [lin_r2(tr, va, T, h) for h in HS]
        mlp, renf = GAIN[ds]
        rows.append((ds, pr, lr, (mlp / renf - 1) * 100))
    print('persist R2(h) on val  |  linear R2 of the change x_{t+h}-x_t on val  |  structure gain (MLP/ReNF-1)')
    print(f'{"":8s} ' + ' '.join(f'p{h:<4d}' for h in HS) + ' | ' + ' '.join(f'l{h:<4d}' for h in HS) + ' | gain')
    for ds, pr, lr, g in rows:
        print(f'{ds:8s} ' + ' '.join(f'{x:5.2f}' for x in pr) + ' | ' + ' '.join(f'{x:5.2f}' for x in lr) + f' | {g:+5.1f}%')

    def spear(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return np.corrcoef(rx, ry)[0, 1]
    g = [r[3] for r in rows]
    print('\nSpearman(summary, structure gain) across datasets:')
    for i, h in enumerate(HS):
        print(f'  persist R2 at h={h:<3d}: {spear([r[1][i] for r in rows], g):+.2f}    '
              f'linear R2 of change at h={h:<3d}: {spear([r[2][i] for r in rows], g):+.2f}')
    print('  persist R2(1) - persist R2(96) (near minus far): '
          f'{spear([r[1][0] - r[1][-1] for r in rows], g):+.2f}')


if __name__ == '__main__':
    main()
