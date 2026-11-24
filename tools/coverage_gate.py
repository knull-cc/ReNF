#!/usr/bin/env python3
"""Pre-training gate for the "coverage" hypothesis (no model is trained).

Hypothesis: supervising an auxiliary short-horizon target helps the far horizon only insofar as the
history directions that predict the auxiliary target also carry the far-horizon signal ("coverage").
Before any training we need two auxiliary targets that are about equally predictable but differ clearly
in coverage; otherwise the decisive experiment cannot be built.

Setup (channel-independent like ReNF, RevIN-style per-window normalisation):
  horizon H (default 96) = prefix (first H/2 steps, auxiliary targets live here) + far segment (last H/2
  steps, never an auxiliary label). Fixed history features Z: last 24 values, lags 48..504, recent means,
  recent slopes; whitened on the fit block.
  Candidates (all 24-dim): DCT low / high half of the prefix, first / last 24 prefix steps, every second
  step. (A deseasoned variant is obtained by the second run, which removes a fixed periodic profile
  estimated on the first half of the training split from the whole series.)

Blocks (by window index, gaps of one window length between blocks so no observation is shared):
  fit = first half of the training split (whitening, ridge fits, auxiliary subspaces U_A);
  S1, S2 = two later, disjoint blocks, each split into halves b1 / b2.
Statistics:
  predictability R2_A : ridge Z -> Y_A fitted on fit, evaluated on S1+S2.
  coverage_A(S)       : <P_U M_F(b1), P_U M_F(b2)> / <M_F(b1), M_F(b2)>, with M_F = Cov(Zw, Y_far) and
                        U = top-k left singular vectors of M_A = Cov(Zw, Y_A) on fit. Using two independent
                        halves keeps only the repeatable part of the far signal.
  readability         : cosine(M_F(b1), M_F(b2)) must exceed the 95th percentile of the same cosine when the
                        far targets are circularly shifted against the history inside each half (random
                        offsets of 25-75 % of the half); otherwise coverage in that block is unreadable.
Gate (pre-registered): a pair (A, B) with |R2_A - R2_B| / max <= 10 %, coverage_A - coverage_B >= 0.20 in
both S1 and S2 (k = 4), and readable far energy in both blocks.
"""
import argparse
import os
import sys
from itertools import permutations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_prior as dp                                       # noqa: E402  (loader + DCT basis)


def make_windows(x, T, H, stride):
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(x, T + H, axis=0)[::stride]      # (W, C, T+H)
    W, C, _ = w.shape
    rows = w.reshape(W * C, T + H).astype(np.float64)
    lb, fut = rows[:, :T], rows[:, T:]
    mu, sd = lb.mean(1, keepdims=True), lb.std(1, keepdims=True) + 1e-5
    return (lb - mu) / sd, (fut - mu) / sd, np.repeat(np.arange(W), C), W


def features(lb):
    T = lb.shape[1]
    cols = [lb[:, -24:]]
    lags = [k for k in (48, 72, 96, 120, 144, 168, 192, 336, 504) if k <= T]
    cols.append(lb[:, [T - k for k in lags]])
    for m in (6, 24, 96, 168, T):
        if m <= T:
            cols.append(lb[:, -m:].mean(1, keepdims=True))
    for m in (24, 96):
        t = np.arange(m) - (m - 1) / 2
        cols.append(((lb[:, -m:] * t).sum(1) / (t ** 2).sum())[:, None])
    return np.concatenate(cols, 1)


def candidates(lb, fut, P):
    pre = fut[:, :P]
    C = dp.dct_basis(P)
    coef, half = pre @ C, P // 2
    return {
        'low (DCT low half)': coef[:, :half],
        'high (DCT high half)': coef[:, half:],
        'first half of prefix': pre[:, :half],
        'second half of prefix': pre[:, half:],
        'every 2nd step': pre[:, 1::2],
    }


def ridge_r2(Zf, Yf, Ze, Ye, lam=1e-3):
    Za = np.hstack([Zf, np.ones((len(Zf), 1))])
    A = Za.T @ Za + lam * len(Zf) * np.eye(Za.shape[1])
    B = np.linalg.solve(A, Za.T @ Yf)
    pred = np.hstack([Ze, np.ones((len(Ze), 1))]) @ B
    return 1 - ((Ye - pred) ** 2).sum() / ((Ye - Ye.mean(0)) ** 2).sum()


def xcov(Z, Y):
    return Z.T @ (Y - Y.mean(0)) / len(Z)


def run(x, T=512, H=96, stride=2, k=4, n_perm=200, seed=0, label=''):
    rng = np.random.default_rng(seed)
    P = H // 2
    lb, fut, widx, W = make_windows(x, T, H, stride)
    Z = features(lb)
    cands = candidates(lb, fut, P)
    far = fut[:, P:]
    gap = (T + H) // stride + 1
    a, b, c = int(0.5 * W), int(0.75 * W), W
    blocks = {'fit': (0, a - gap), 'S1': (a, b - gap), 'S2': (b, c)}
    m = {n: (widx >= lo) & (widx < hi) for n, (lo, hi) in blocks.items()}
    # whiten on fit
    mu = Z[m['fit']].mean(0)
    Zc = Z - mu
    ev, evec = np.linalg.eigh(np.cov(Zc[m['fit']], rowvar=False))
    Zw = Zc @ (evec / np.sqrt(np.maximum(ev, 1e-8 * ev.max())))
    halves = {}
    for s in ('S1', 'S2'):
        lo, hi = blocks[s]
        mid = (lo + hi) // 2
        halves[s] = ((widx >= lo) & (widx < mid - gap // 2), (widx >= mid + gap // 2) & (widx < hi))
    # far repeatable energy; readability via circular-shift null on the cosine between the two halves
    def cosine(A, B):
        return float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))

    C = len(np.unique(widx[widx == 0])) and int((widx == 0).sum())      # rows per window (= variables)
    readable, rep_far, cos_far = {}, {}, {}
    for s_, (h1, h2) in halves.items():
        M1, M2 = xcov(Zw[h1], far[h1]), xcov(Zw[h2], far[h2])
        rep_far[s_], cos_far[s_] = float((M1 * M2).sum()), cosine(M1, M2)
        null = []
        for _ in range(n_perm):
            Ms = []
            for h in (h1, h2):
                F = far[h].reshape(-1, C, far.shape[1])                 # rows are window-major, contiguous
                nw = len(F)
                o = int(rng.integers(nw // 4, 3 * nw // 4 + 1))
                Ms.append(xcov(Zw[h], np.roll(F, o, axis=0).reshape(-1, far.shape[1])))
            null.append(cosine(*Ms))
        readable[s_] = cos_far[s_] > np.quantile(null, 0.95)
    ev_mask = m['S1'] | m['S2']
    far_r2 = ridge_r2(Zw[m['fit']], far[m['fit']], Zw[ev_mask], far[ev_mask])
    print(f'\n=== {label}  (rows {len(Z)}, windows {W}, prefix {P}, far {H - P}, k={k}) ===')
    print(f'far segment: R2 from history features = {far_r2:.3f}; repeatable far energy readable: '
          + ', '.join(f'{s}={"yes" if readable[s] else "NO"} (cos {cos_far[s]:.2f})' for s in readable))
    res = {}
    for name, Y in cands.items():
        r2 = ridge_r2(Zw[m['fit']], Y[m['fit']], Zw[ev_mask], Y[ev_mask])
        U = np.linalg.svd(xcov(Zw[m['fit']], Y[m['fit']]), full_matrices=False)[0]
        covs = {}
        for kk in (2, k, 8):
            Pk = U[:, :kk] @ U[:, :kk].T
            for s, (h1, h2) in halves.items():
                M1, M2 = xcov(Zw[h1], far[h1]), xcov(Zw[h2], far[h2])
                covs[(kk, s)] = float(((Pk @ M1) * (Pk @ M2)).sum() / (M1 * M2).sum())
        res[name] = (r2, covs)
    print(f'{"auxiliary target":24s} {"R2":>6s} ' + ' '.join(f'{"cov_k" + str(kk) + "_" + s:>11s}'
                                                         for kk in (2, k, 8) for s in ('S1', 'S2')))
    for name, (r2, covs) in res.items():
        print(f'{name:24s} {r2:6.3f} ' + ' '.join(f'{covs[(kk, s)]:11.3f}' for kk in (2, k, 8) for s in ('S1', 'S2')))
    passing = []
    for A, B in permutations(res, 2):
        rA, cA = res[A]
        rB, cB = res[B]
        if max(rA, rB) <= 0 or abs(rA - rB) / max(rA, rB) > 0.10:
            continue
        dS1, dS2 = cA[(k, 'S1')] - cB[(k, 'S1')], cA[(k, 'S2')] - cB[(k, 'S2')]
        if dS1 >= 0.20 and dS2 >= 0.20 and all(readable.values()):
            passing.append((A, B, rA, rB, dS1, dS2))
    if passing:
        print('GATE PASSED for:')
        for A, B, rA, rB, d1, d2 in passing:
            print(f'  high = {A} (R2 {rA:.3f})  vs  low = {B} (R2 {rB:.3f}):  coverage gap {d1:+.2f} (S1) {d2:+.2f} (S2)')
    else:
        why = [] if all(readable.values()) else ['far energy not readable in ' + ','.join(s for s in readable if not readable[s])]
        print('GATE NOT PASSED' + (' (' + '; '.join(why) + ')' if why else
                                   ': no pair with R2 within 10 % and a coverage gap >= 0.20 in both blocks'))
    return res, passing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='../dataset')
    ap.add_argument('--dataset', default='etth1')
    ap.add_argument('--seq_len', type=int, default=512)
    ap.add_argument('--pred_len', type=int, default=96)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--max_vars', type=int, default=50)
    ap.add_argument('--n_perm', type=int, default=200)
    a = ap.parse_args()
    x = dp.load_train(a.root, a.dataset)
    if x.shape[1] > a.max_vars:
        x = x[:, np.sort(np.random.default_rng(0).choice(x.shape[1], a.max_vars, replace=False))]
    P = dp.DATASETS[a.dataset][2]
    early = x[: len(x) // 2]
    phase = np.arange(len(x)) % P
    prof = np.stack([early[phase[: len(early)] == p].mean(0) for p in range(P)])
    for label, series in [('raw', x), (f'deseasoned (period {P}, profile from first half)', x - prof[phase])]:
        run(series, a.seq_len, a.pred_len, a.stride, a.k, a.n_perm, label=f'{a.dataset} {label}')


if __name__ == '__main__':
    main()
