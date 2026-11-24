#!/usr/bin/env python3
"""Model-free checks of the two premises of the scale-recursive (DCT coarse-to-fine) plan.
Training split only (same borders as data_provider), z-scored per variable like the loaders.

For every dataset, non-overlapping windows of length L are cut from the training split and split with an
orthonormal DCT-II into low (first f1 coefficients) and high (the rest) components.

1. Cross-frequency coupling: does the local amplitude of the high band, env(t) = moving average of |high| over
   `env_win` steps, depend on the co-located low band? Reported as the multiple correlation R of a least-squares
   fit env ~ [1, low, low^2, |d low/dt|], per variable, averaged over variables.
   The plan's multiplicative modulation only has a physical basis if R is clearly > 0 (>0.3 suggested).
2. Cross-variable correlation per band: mean |corr| between variables of the raw series, the low band and the
   high band (variables subsampled to at most `max_vars`).
3. Same for the low band after removing each variable's mean periodic profile (period P) AND each window's
   mean (per variable): co-movement that a per-variable model cannot get from its own history. Shared daily
   cycles and spurious correlation between independent slow drifts inflate check 2 but not this one.
"""
import argparse
import os
import numpy as np
import pandas as pd

DATASETS = {   # name: (relative path, split rule, period in steps)
    'etth1': ('ETT-small/ETTh1.csv', 'ett_hour', 24),
    'etth2': ('ETT-small/ETTh2.csv', 'ett_hour', 24),
    'weather': ('weather/weather.csv', 'custom', 144),
    'electricity': ('electricity/electricity.csv', 'custom', 24),
    'traffic': ('traffic/traffic.csv', 'custom', 24),
    'solar': ('Solar/solar_AL.txt', 'custom_txt', 144),
}


def load_train(root, name):
    rel, rule, _ = DATASETS[name]
    path = os.path.join(root, rel)
    if rule == 'custom_txt':
        x = np.loadtxt(path, delimiter=',', dtype=np.float64)
    else:
        df = pd.read_csv(path)
        x = df.drop(columns=['date']).values.astype(np.float64)
    n_train = 12 * 30 * 24 if rule == 'ett_hour' else int(len(x) * 0.7)
    x = x[:n_train]
    sd = x.std(axis=0)
    keep = sd > 1e-8                                   # drop constant variables
    return (x[:, keep] - x[:, keep].mean(axis=0)) / sd[keep]


def dct_basis(L):
    n, k = np.arange(L)[:, None], np.arange(L)[None, :]
    C = np.sqrt(2.0 / L) * np.cos(np.pi * (n + 0.5) * k / L)
    C[:, 0] /= np.sqrt(2.0)
    return C                                            # orthonormal: C.T @ C = I


def split_bands(x, L, f1):
    """x: (T, D) -> low, high: (W, L, D) from non-overlapping windows."""
    W = x.shape[0] // L
    win = x[:W * L].reshape(W, L, -1)
    C = dct_basis(L)[:, :f1]
    P = C @ C.T                                         # projection onto the first f1 DCT components
    low = np.einsum('ts,wsd->wtd', P, win)
    return win, low, win - low


def moving_avg(a, k, axis=1):
    if k <= 1:
        return a
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), axis, a)


def coupling(low, high, env_win):
    env = moving_avg(np.abs(high), env_win)
    grad = np.abs(np.gradient(low, axis=1))
    Rs = []
    for d in range(low.shape[2]):
        y = env[:, :, d].reshape(-1)
        l, g = low[:, :, d].reshape(-1), grad[:, :, d].reshape(-1)
        X = np.stack([np.ones_like(l), l, l ** 2, g], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = np.corrcoef(X @ coef, y)[0, 1] if y.std() > 1e-12 else np.nan
        Rs.append(r)
    return float(np.nanmean(Rs)), float(np.nanmedian(Rs))


def mean_abs_corr(a):
    """a: (T, D) -> mean |corr| over distinct variable pairs."""
    a = a[:, a.std(axis=0) > 1e-8]
    c = np.corrcoef(a, rowvar=False)
    iu = np.triu_indices_from(c, k=1)
    return float(np.nanmean(np.abs(c[iu])))


def remove_profile(x, P):
    phase = np.arange(len(x)) % P
    prof = np.stack([x[phase == p].mean(axis=0) for p in range(P)])
    return x - prof[phase]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='../dataset')
    ap.add_argument('--datasets', nargs='+', default=['etth1', 'weather', 'electricity', 'traffic', 'solar'])
    ap.add_argument('--L', type=int, default=96, help='window length = forecast horizon studied')
    ap.add_argument('--f1', type=int, default=0, help='low-band DCT components (0: max(8, L/16))')
    ap.add_argument('--env_win', type=int, default=5)
    ap.add_argument('--max_vars', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    f1 = a.f1 or max(8, a.L // 16)
    rng = np.random.default_rng(a.seed)
    rows = []
    for name in a.datasets:
        x = load_train(a.root, name)
        if x.shape[1] > a.max_vars:
            x = x[:, np.sort(rng.choice(x.shape[1], a.max_vars, replace=False))]
        win, low, high = split_bands(x, a.L, f1)
        R_mean, R_med = coupling(low, high, a.env_win)
        flat = lambda b: b.reshape(-1, b.shape[2])
        P = DATASETS[name][2]
        x_ds = remove_profile(x, P)
        _, low_ds, _ = split_bands(x_ds, a.L, f1)
        low_ds = low_ds - low_ds.mean(axis=1, keepdims=True)      # remove each window's level (drift)
        rows.append(dict(dataset=name, vars=x.shape[1], windows=win.shape[0],
                         low_var_share=float((low ** 2).sum() / (win ** 2).sum()),
                         coupling_R=R_mean, coupling_R_median=R_med,
                         xvar_raw=mean_abs_corr(flat(win)), xvar_low=mean_abs_corr(flat(low)),
                         xvar_high=mean_abs_corr(flat(high)), xvar_low_resid=mean_abs_corr(flat(low_ds))))
        print(f'done {name}', flush=True)
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(f'\nL={a.L}, low band = first {f1} DCT components (periods >= {2 * a.L / f1:.0f} steps), '
          f'envelope window {a.env_win}')
    print(df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\ncoupling_R: how well the high-band envelope is explained by the co-located low band '
          '(premise of multiplicative cross-scale modulation; > 0.3 = meaningful)')
    print('xvar_*: mean |corr| between variables. xvar_low_resid: low band after removing each variable\'s mean '
          'periodic profile and each window\'s level -- co-movement that cross-variable routing could exploit '
          'beyond what each variable gets from its own history')


if __name__ == '__main__':
    main()
