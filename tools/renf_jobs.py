#!/usr/bin/env python3
"""Parallel runner for the official ReNF scripts (one GPU, e.g. Colab).

The official scripts run every horizon of a dataset one after another.  This
tool keeps their hyper-parameters byte-for-byte and only changes *how* they
are launched:

  build    Dry-runs scripts/<ds>.sh with a stub `python` command, so bash
           itself expands every variable, loop and per-horizon seq_len change.
           Each captured `run_longExp.py` call becomes one job in
           logs/colab/jobs.json.  Only these flags are overridden:
             --root_path    -> <data_root>/<sub-folder>   (data location)
             --random_seed  -> each seed in --seeds
             --des          -> Exp_s<seed>  (unique checkpoint dir per seed)
             --num_workers  -> small value (default 10 x N jobs kills Colab's
                               CPU/RAM; does not change the sampled batches)
           CUDA_VISIBLE_DEVICES from the scripts (some say 1 or 2, which on a
           1-GPU Colab silently falls back to CPU) is replaced by --gpu.
  run      Runs the jobs in parallel; skips finished ones (safe to re-run after
           a Colab disconnect); waits for free GPU memory before each launch;
           re-queues jobs that died of CUDA OOM.
  status   Done / running / failed / pending per dataset.
  collect  Parses the logs into tables + CSV.  Uses the `mse:` line (the final
           forecast).  The `mse_com:` line printed by exp_main.test() is an
           ORACLE (post_process picks, per point, whichever sub-forecast is
           closer to the ground truth) and must never be reported.
           When a script contains several configs for the same horizon
           (scripts/exchange.sh), the one with the lowest validation loss
           is selected -- test loss is never used for selection.
  stop     Stops the background runner and its training processes.
  diag     Re-tests finished jobs from their saved checkpoints (no training) with
           --diag 1 and summarises the stage diagnostics: how different the stage
           forecasts are (ambiguity), how correlated their errors are, and how much
           later stages improve the segments earlier stages already forecast.

Ablations: `build --tag NAME --extra "--flag value ..."` adds the same official
jobs with extra/overriding flags (e.g. --fb_mode zero, --r_ema 0.0, --d_layers 1).
The tag is appended to model_id, so checkpoints never collide; `collect` then
prints one row per (dataset, variant) with the paired seed-wise change vs base.

Typical Colab use (from the repo root):
  python tools/renf_jobs.py build --datasets etth1 etth2 ettm1 ettm2 weather exchange --seeds 2021
  nohup python tools/renf_jobs.py run --max_parallel 6 > logs/colab/runner.out 2>&1 &
  python tools/renf_jobs.py status
  python tools/renf_jobs.py collect
"""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO / 'logs' / 'colab'
JOBS_FILE = LOG_ROOT / 'jobs.json'
PID_FILE = LOG_ROOT / 'runner.pid'
SEP = '\x1f'
DEFAULT_DATASETS = ['etth1', 'etth2', 'ettm1', 'ettm2', 'weather', 'exchange']

STUB = r'''
python() {
  local a
  for a in "$@"; do printf '%s\037' "$a"; done >> "$RENF_CAPTURE"
  printf '\n' >> "$RENF_CAPTURE"
}
python3() { python "$@"; }
export -f python python3
bash "$1"
'''

MSE_RE = re.compile(r'^mse:\s*([-\d.eE+naif]+),\s*mae:\s*([-\d.eE+naif]+)', re.M)
VALI_RE = re.compile(r'Vali Loss:\s*([-\d.eE+naif]+)')
PROBE_RE = re.compile(r'^probe_(\w+) mse:\s*([-\d.eE+naif]+),\s*mae:\s*([-\d.eE+naif]+)', re.M)
PARAMS_RE = re.compile(r'params:\s*([-\d.eE+]+)M')
PROBES = ['stageavg', 'fbzero', 'fbshuffle']


def now():
    return time.strftime('%H:%M:%S')


# ----------------------------------------------------------------------------- build
def capture_script(script):
    """Return the argv lists of every run_longExp.py call the script makes."""
    text = script.read_text()
    with tempfile.TemporaryDirectory() as tmp:
        for d in set(re.findall(r'logs/[A-Za-z0-9_]+', text)):
            os.makedirs(os.path.join(tmp, d), exist_ok=True)
        cap = os.path.join(tmp, 'capture.txt')
        open(cap, 'w').close()
        env = dict(os.environ, RENF_CAPTURE=cap)
        r = subprocess.run(['bash', '-c', STUB, '_', str(script)], cwd=tmp, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f'  [warn] {script.name} exited {r.returncode}: {r.stderr.strip()[:300]}')
        calls = []
        for line in open(cap).read().splitlines():
            if not line:
                continue
            argv = line.split(SEP)
            if argv and argv[-1] == '':
                argv = argv[:-1]
            if 'run_longExp.py' not in argv:
                continue
            calls.append(argv[argv.index('run_longExp.py') + 1:])
        return calls


def to_pairs(argv):
    pairs, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith('--'):
            if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                pairs.append([tok, argv[i + 1]])
                i += 2
                continue
            pairs.append([tok, None])
        i += 1
    return pairs


def set_arg(pairs, flag, value):
    for p in pairs:
        if p[0] == flag:
            p[1] = value
            return
    pairs.append([flag, value])


def get_arg(pairs, flag, default=None):
    for k, v in pairs:
        if k == flag:
            return v
    return default


def flatten(pairs):
    out = []
    for k, v in pairs:
        out.append(k)
        if v is not None:
            out.append(v)
    return out


def cmd_build(a):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = OrderedDict()
    warned = set()
    horizons = set(a.horizons) if a.horizons else None
    for ds in a.datasets:
        script = REPO / 'scripts' / f'{ds}.sh'
        if not script.exists():
            print(f'[skip] {script} not found')
            continue
        calls = capture_script(script)
        n_ds, summary = 0, []
        for argv in calls:
            pairs = to_pairs(argv)
            pred_len = int(get_arg(pairs, '--pred_len'))
            if horizons and pred_len not in horizons:
                continue
            root = get_arg(pairs, '--root_path', '')
            m = re.search(r'dataset/(.*)$', root)
            if m:
                root = os.path.join(a.data_root, m.group(1))
                set_arg(pairs, '--root_path', root)
            data_file = os.path.join(root if os.path.isabs(root) else str(REPO / root),
                                     get_arg(pairs, '--data_path', ''))
            set_arg(pairs, '--num_workers', str(a.num_workers))
            for k, v in to_pairs(shlex.split(a.extra)):
                set_arg(pairs, k, v)
            if a.tag != 'base':
                set_arg(pairs, '--model_id', get_arg(pairs, '--model_id') + '__' + a.tag)
            for seed in a.seeds:
                p = [list(x) for x in pairs]
                set_arg(p, '--random_seed', str(seed))
                set_arg(p, '--des', f'Exp_s{seed}')
                model, model_id = get_arg(p, '--model'), get_arg(p, '--model_id')
                name = f'{model}_{model_id}_s{seed}'
                if name in jobs:
                    print(f'  [warn] duplicate job {name}: keeping the later one (as bash would)')
                jobs[name] = dict(name=name, script=ds, variant=a.tag, model=model, model_id=model_id,
                                  pred_len=pred_len, seq_len=int(get_arg(p, '--seq_len')),
                                  seed=seed, args=flatten(p),
                                  log=str((LOG_ROOT / ds / f'{name}.log').relative_to(REPO)))
                n_ds += 1
            summary.append(f'H{pred_len}/sl{get_arg(pairs, "--seq_len")}')
            if not os.path.exists(data_file) and data_file not in warned:
                warned.add(data_file)
                print(f'  [warn] data file not found: {data_file}')
        print(f'{ds:12s} {n_ds:3d} jobs  {" ".join(OrderedDict.fromkeys(summary))}')
    if a.append and JOBS_FILE.exists():
        old = json.loads(JOBS_FILE.read_text())
        old.update(jobs)
        jobs = old
    JOBS_FILE.write_text(json.dumps(jobs, indent=1))
    print(f'-> {len(jobs)} jobs in {JOBS_FILE.relative_to(REPO)}' + (f' (variant {a.tag}: {a.extra})' if a.tag != 'base' else ''))


# ----------------------------------------------------------------------------- helpers
def load_jobs():
    if not JOBS_FILE.exists():
        sys.exit('No jobs file. Run `python tools/renf_jobs.py build ...` first.')
    return json.loads(JOBS_FILE.read_text())


def read(path):
    try:
        return (REPO / path).read_text(errors='replace')
    except FileNotFoundError:
        return ''


def is_done(job):
    return bool(MSE_RE.search(read(job['log'])))


def failed_marker(job):
    return REPO / (job['log'] + '.failed')


def runner_alive():
    try:
        pid = int(PID_FILE.read_text())
        os.kill(pid, 0)
        return pid
    except Exception:
        return None


def gpu_free_mb(gpu):
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits',
                              '-i', str(gpu)], capture_output=True, text=True, timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return 10 ** 9


# ----------------------------------------------------------------------------- run
def cmd_run(a):
    jobs = list(load_jobs().values())
    if a.datasets:
        jobs = [j for j in jobs if j['script'] in a.datasets]
    if a.variants:
        jobs = [j for j in jobs if j.get('variant', 'base') in a.variants]
    if a.horizons:
        jobs = [j for j in jobs if j['pred_len'] in a.horizons]
    pid = runner_alive()
    if pid and pid != os.getpid():
        sys.exit(f'A runner is already active (pid {pid}). Use `stop` first.')
    PID_FILE.write_text(str(os.getpid()))
    (REPO / 'forecast_visual').mkdir(exist_ok=True)   # exp_main.test() saves a plot here
    (REPO / 'checkpoints').mkdir(exist_ok=True)

    queue = []
    for j in jobs:
        if is_done(j):
            continue
        if failed_marker(j).exists() and not a.retry_failed:
            continue
        queue.append(j)
    n_done = sum(is_done(j) for j in jobs)
    print(f'[{now()}] {len(jobs)} jobs: {n_done} already done, {len(queue)} to run, '
          f'max_parallel={a.max_parallel}, min_free_mb={a.min_free_mb}', flush=True)

    running, retries, last_launch = {}, Counter(), 0.0

    def shutdown(*_):
        print(f'[{now()}] stopping: terminating {len(running)} running jobs', flush=True)
        for p, _, fh, _ in running.values():
            p.terminate()
        for p, _, fh, _ in running.values():
            try:
                p.wait(timeout=30)
            except Exception:
                p.kill()
            fh.close()
        PID_FILE.unlink(missing_ok=True)
        sys.exit(1)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while queue or running:
        for name in list(running):
            p, job, fh, t0 = running[name]
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            del running[name]
            mins = (time.time() - t0) / 60
            m = MSE_RE.search(read(job['log']))
            if rc == 0 and m:
                print(f'[{now()}] done   {name}  mse={float(m.group(1)):.4f} mae={float(m.group(2)):.4f}  '
                      f'({mins:.1f} min)', flush=True)
                continue
            tail = read(job['log'])[-4000:]
            if 'out of memory' in tail.lower() and retries[name] < a.oom_retries:
                retries[name] += 1
                queue.append(job)
                print(f'[{now()}] OOM    {name} -> re-queued ({retries[name]}/{a.oom_retries})', flush=True)
                continue
            failed_marker(job).write_text(f'returncode={rc}\n' + tail[-2000:])
            print(f'[{now()}] FAILED {name} (rc={rc}); see {job["log"]}', flush=True)

        can_launch = (queue and len(running) < a.max_parallel
                      and time.time() - last_launch >= a.launch_gap
                      and gpu_free_mb(a.gpu) >= a.min_free_mb)
        if can_launch:
            job = queue.pop(0)
            log = REPO / job['log']
            log.parent.mkdir(parents=True, exist_ok=True)
            failed_marker(job).unlink(missing_ok=True)
            fh = open(log, 'w')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONUNBUFFERED='1')
            p = subprocess.Popen([sys.executable, '-u', 'run_longExp.py'] + job['args'],
                                 cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[job['name']] = (p, job, fh, time.time())
            last_launch = time.time()
            print(f'[{now()}] start  {job["name"]}  (running {len(running)}, queued {len(queue)})', flush=True)
            continue
        time.sleep(5)

    PID_FILE.unlink(missing_ok=True)
    print(f'[{now()}] all jobs finished', flush=True)


# ----------------------------------------------------------------------------- status / stop
def cmd_status(a):
    jobs = list(load_jobs().values())
    alive = runner_alive()
    print(f'runner: {"running (pid %d)" % alive if alive else "not running"}')
    per = OrderedDict()
    active, failed = [], []
    for j in jobs:
        v = j.get('variant', 'base')
        c = per.setdefault(j['script'] + ('' if v == 'base' else '/' + v), Counter())
        if is_done(j):
            c['done'] += 1
        elif failed_marker(j).exists():
            c['failed'] += 1
            failed.append(j['name'])
        elif (REPO / j['log']).exists() and alive:
            c['running'] += 1
            ep = len(VALI_RE.findall(read(j['log'])))
            active.append(f'{j["name"]} (epoch {ep})')
        else:
            c['pending'] += 1
    w = max([12] + [len(k) for k in per])
    print(f'{"dataset":{w}s} {"done":>5s} {"run":>5s} {"fail":>5s} {"pend":>5s}')
    for ds, c in per.items():
        print(f'{ds:{w}s} {c["done"]:5d} {c["running"]:5d} {c["failed"]:5d} {c["pending"]:5d}')
    if active:
        print('running:\n  ' + '\n  '.join(active))
    if failed:
        print('failed (re-run with `run --retry_failed`):\n  ' + '\n  '.join(failed))


def cmd_stop(a):
    pid = runner_alive()
    if pid:
        os.kill(pid, signal.SIGTERM)
        print(f'sent SIGTERM to runner {pid}')
    else:
        print('no runner active')


# ----------------------------------------------------------------------------- collect
def cmd_collect(a):
    import numpy as np
    import pandas as pd
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    jobs = list(load_jobs().values())
    ds_order = list(OrderedDict.fromkeys(j['script'] for j in jobs))
    var_order = list(OrderedDict.fromkeys(['base'] + [j.get('variant', 'base') for j in jobs]))
    rows = []
    for j in jobs:
        text = read(j['log'])
        m = MSE_RE.search(text)
        if not m:
            continue
        vali = [float(v) for v in VALI_RE.findall(text)]
        pm = PARAMS_RE.search(text)
        v = j.get('variant', 'base')
        row = dict(variant=v, dataset=j['script'], H=j['pred_len'], seed=j['seed'],
                   config=j['model_id'].replace('__' + v, ''), seq_len=j['seq_len'],
                   mse=float(m.group(1)), mae=float(m.group(2)),
                   best_vali=min(vali) if vali else float('inf'), epochs=len(vali),
                   params_M=float(pm.group(1)) if pm else np.nan)
        for name, mse, _ in PROBE_RE.findall(text):
            row['p_' + name] = float(mse)
        rows.append(row)
    if not rows:
        print('no finished jobs yet')
        return
    df = pd.DataFrame(rows)
    if a.horizons:
        df = df[df.H.isin(a.horizons)]
        if df.empty:
            print('no finished jobs for these horizons yet')
            return
    for p in PROBES:
        if 'p_' + p not in df:
            df['p_' + p] = np.nan
        df['d_' + p] = (df['p_' + p] / df['mse'] - 1) * 100       # % change of test MSE under the probe
    df.to_csv(LOG_ROOT / 'results_all.csv', index=False)

    # one config per (variant, dataset, H): lowest validation loss averaged over seeds (only matters for grids)
    keys = ['variant', 'dataset', 'H']
    mv = df.groupby(keys + ['config'])['best_vali'].mean().reset_index()
    best = mv.loc[mv.groupby(keys)['best_vali'].idxmin(), keys + ['config']]
    sel = df.merge(best, on=keys + ['config'])
    srt = lambda c: c.map({d: i for i, d in enumerate(ds_order)}) if c.name == 'dataset' else \
        (c.map({v: i for i, v in enumerate(var_order)}) if c.name == 'variant' else c)
    fmt = lambda x: f'{x:.4f}' if pd.notna(x) else '-'
    pct = lambda x: f'{x:+.1f}%' if pd.notna(x) else '-'

    # ---- per-horizon table
    agg = sel.groupby(keys, sort=False).agg(
        mse=('mse', 'mean'), mse_std=('mse', 'std'), mae=('mae', 'mean'), mae_std=('mae', 'std'),
        seeds=('seed', 'nunique'), seq_len=('seq_len', lambda x: '/'.join(map(str, sorted(set(x))))),
        config=('config', lambda x: ' '.join(sorted(set(x)))),
        **{p: ('d_' + p, 'mean') for p in PROBES}).reset_index().sort_values(keys, key=srt)
    agg.to_csv(LOG_ROOT / 'results_table.csv', index=False)
    multi = sel['variant'].nunique() > 1
    show = agg if (a.detail or not multi) else agg[agg.variant == 'base']
    if len(show):
        t = show.drop(columns=[] if multi else ['variant']).copy()
        for c in ['mse', 'mse_std', 'mae', 'mae_std']:
            t[c] = t[c].map(fmt)
        for p in PROBES:
            t[p] = t[p].map(pct)
        print(('Per horizon (all variants):' if a.detail and multi else 'Per horizon (base):') if multi else 'Per horizon:')
        print(t.to_string(index=False))

    # ---- summary: horizon-average per seed, then mean/sd over seeds; paired change vs base
    hs = sel.groupby(['variant', 'dataset', 'seed']).agg(
        mse=('mse', 'mean'), mae=('mae', 'mean'), Hs=('H', lambda x: tuple(sorted(x))),
        params_M=('params_M', 'mean'), **{p: ('d_' + p, 'mean') for p in PROBES}).reset_index()
    base = hs[hs.variant == 'base'][['dataset', 'seed', 'Hs', 'mse']].rename(columns={'mse': 'mse_base'})
    hs = hs.merge(base, on=['dataset', 'seed', 'Hs'], how='left')
    hs['dvb'] = (hs['mse'] / hs['mse_base'] - 1) * 100
    summ = hs.groupby(['dataset', 'variant'], sort=False).agg(
        n_H=('Hs', lambda x: len(x.iloc[0])), seeds=('seed', 'nunique'),
        mse=('mse', 'mean'), sd=('mse', 'std'), mae=('mae', 'mean'),
        vs_base=('dvb', 'mean'), vs_base_sd=('dvb', 'std'), paired=('dvb', 'count'),
        params_M=('params_M', 'mean'), **{p: (p, 'mean') for p in PROBES}).reset_index()
    summ = summ.sort_values(['dataset', 'variant'], key=srt)
    summ.to_csv(LOG_ROOT / 'results_summary.csv', index=False)
    t = summ.copy()
    for c in ['mse', 'sd', 'mae']:
        t[c] = t[c].map(fmt)
    t['vs_base'] = [('-' if v == 'base' else f'{m:+.2f}% \u00b1 {s:.2f} (n={n})') if pd.notna(m) else '-'
                    for v, m, s, n in zip(summ.variant, summ.vs_base, summ.vs_base_sd.fillna(0), summ.paired)]
    t['params_M'] = summ.params_M.map(lambda x: f'{x:.2f}' if pd.notna(x) else '-')
    for p in PROBES:
        t[p] = summ[p].map(pct)
    t = t.drop(columns=['vs_base_sd', 'paired'])
    print('\nAverage over horizons (mse/sd: mean/sd over seeds; vs_base: paired seed-wise % change of MSE, '
          'negative = better):')
    print(t.to_string(index=False))
    print('\nprobe columns = % change of test MSE when, at test time only, the final output is replaced by the '
          'average of all stage forecasts (stageavg), or the fed-back forecasts are zeroed (fbzero) / '
          'taken from another sample (fbshuffle). mse_com in the logs is an oracle and is never used.')
    if (df.groupby(keys + ['seed']).size() > 1).any():
        print('Grid datasets: one config per (variant, H), picked by validation loss averaged over seeds.')
    print(f'\nCSV files in {LOG_ROOT.relative_to(REPO)}/  (results_all / results_table / results_summary)')


def cmd_diag(a):
    import numpy as np
    import pandas as pd
    pd.set_option('display.width', 250)
    jobs = [j for j in load_jobs().values() if is_done(j)
            and (not a.datasets or j['script'] in a.datasets)
            and (not a.variants or j.get('variant', 'base') in a.variants)
            and (not a.horizons or j['pred_len'] in a.horizons)]
    if not jobs:
        sys.exit('no finished jobs match')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONUNBUFFERED='1')
    SEG = re.compile(r'^diag_seg (\d+) n_f=(\d+) first=(\S+) final=(\S+) mean_member=(\S+) '
                     r'ambiguity=(\S+) ensemble=(\S+) err_corr=(\S+)', re.M)
    rows = []
    for i, j in enumerate(jobs):
        out = REPO / (j['log'] + '.diag')
        if a.force or 'diag_summary' not in read(str(out.relative_to(REPO))):
            args = to_pairs(j['args'])
            for k, v in [('--is_training', '0'), ('--diag', '1'), ('--probe', '0')]:
                set_arg(args, k, v)
            with open(out, 'w') as fh:
                rc = subprocess.run([sys.executable, '-u', 'run_longExp.py'] + flatten(args), cwd=REPO,
                                    stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
            print(f'[{i + 1}/{len(jobs)}] {j["name"]}' + ('' if rc == 0 else f'  FAILED (rc={rc}, see {out})'),
                  flush=True)
        for m in SEG.findall(read(str(out.relative_to(REPO)))):
            seg, nf, first, final, mem, amb, ens, corr = m
            rows.append(dict(variant=j.get('variant', 'base'), dataset=j['script'], H=j['pred_len'],
                             seed=j['seed'], seg=int(seg), n_f=int(nf), first=float(first), final=float(final),
                             member=float(mem), amb=float(amb), ens=float(ens), corr=float(corr)))
    if not rows:
        print('no diagnostics parsed (single-stage models have none)')
        return
    df = pd.DataFrame(rows)
    df.to_csv(LOG_ROOT / 'diag_all.csv', index=False)

    # final-stage MSE per horizon segment, and its change vs base (paired by seed)
    seg = df.pivot_table(index=['dataset', 'H', 'seed', 'seg'], columns='variant', values='final').reset_index()
    variants = [v for v in df.variant.unique()]
    if 'base' in variants and len(variants) > 1:
        print('Final-stage MSE per horizon segment (seg 1 = earliest steps); % = paired change vs base')
        out = []
        for (ds, H, sg), grp in seg.groupby(['dataset', 'H', 'seg']):
            row = dict(dataset=ds, H=H, seg=sg, base=grp['base'].mean())
            for v in variants:
                if v != 'base' and v in grp:
                    row[v] = f"{grp[v].mean():.4f} ({((grp[v] / grp['base'] - 1) * 100).mean():+.2f}%)"
            out.append(row)
        print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        print()

    df = df[df.n_f > 1]          # stage-agreement statistics need >= 2 stages forecasting the segment
    if df.empty:
        return
    g = df.groupby(['variant', 'dataset', 'H'], sort=False).agg(
        seeds=('seed', 'nunique'), err_corr=('corr', 'mean'),
        amb_ratio=('amb', 'sum'), member=('member', 'sum'),
        refine=('final', 'sum'), first=('first', 'sum'), ens=('ens', 'sum')).reset_index()
    g['diversity_%'] = g.amb_ratio / g.member * 100            # share of member error removable by averaging
    g['refine_%'] = (g.refine / g['first'] - 1) * 100            # later stages vs the stage that first forecast a segment
    g['avg_vs_final_%'] = (g.ens / g.refine - 1) * 100           # averaging stages vs final stage, on shared segments
    g = g[['variant', 'dataset', 'H', 'seeds', 'err_corr', 'diversity_%', 'refine_%', 'avg_vs_final_%']]
    g.to_csv(LOG_ROOT / 'diag_summary.csv', index=False)
    print(g.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\nerr_corr: correlation of the stages\' errors on the segments they share (1 = identical mistakes)')
    print('diversity_%: spread of stage forecasts around their mean, as % of their MSE '
          '(= the most that averaging them could remove)')
    print('refine_%: MSE change from the stage that first forecast a segment to the final stage (negative = later '
          'stages correct earlier ones)')
    print('avg_vs_final_%: averaging all stages vs using the final stage, on the shared segments')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='turn official scripts into jobs')
    b.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                   help='script names under scripts/ (without .sh)')
    b.add_argument('--seeds', nargs='+', type=int, default=[2021])
    b.add_argument('--horizons', nargs='+', type=int, default=None, help='e.g. 96 720 (default: all)')
    b.add_argument('--data_root', default='../dataset',
                   help='folder holding ETT-small/, weather/, exchange_rate/ ... (default: official ../dataset)')
    b.add_argument('--num_workers', type=int, default=2)
    b.add_argument('--append', action='store_true', help='add to the existing jobs file instead of replacing it')
    b.add_argument('--tag', default='base', help='variant name for ablations (default: base = official)')
    b.add_argument('--extra', default='', help='extra/overriding run_longExp.py flags, e.g. "--fb_mode zero"')

    r = sub.add_parser('run', help='run jobs in parallel (resumable)')
    r.add_argument('--max_parallel', type=int, default=4)
    r.add_argument('--min_free_mb', type=int, default=3000, help='free GPU memory needed to launch another job')
    r.add_argument('--launch_gap', type=float, default=20, help='seconds between launches (lets memory settle)')
    r.add_argument('--gpu', type=int, default=0)
    r.add_argument('--oom_retries', type=int, default=2)
    r.add_argument('--retry_failed', action='store_true')
    r.add_argument('--datasets', nargs='+', default=None, help='only run these datasets from the jobs file')
    r.add_argument('--variants', nargs='+', default=None, help='only run these variants from the jobs file')
    r.add_argument('--horizons', nargs='+', type=int, default=None, help='only run these pred_lens, e.g. 96')

    d = sub.add_parser('diag', help='stage diagnostics from saved checkpoints (no training)')
    d.add_argument('--datasets', nargs='+', default=None)
    d.add_argument('--variants', nargs='+', default=['base'])
    d.add_argument('--horizons', nargs='+', type=int, default=None)
    d.add_argument('--gpu', type=int, default=0)
    d.add_argument('--force', action='store_true', help='recompute even if a .diag file exists')

    sub.add_parser('status')
    c = sub.add_parser('collect')
    c.add_argument('--detail', action='store_true', help='also print the per-horizon table for every variant')
    c.add_argument('--horizons', nargs='+', type=int, default=None, help='only use these pred_lens, e.g. 96')
    sub.add_parser('stop')

    a = ap.parse_args()
    {'build': cmd_build, 'run': cmd_run, 'status': cmd_status,
     'collect': cmd_collect, 'stop': cmd_stop, 'diag': cmd_diag}[a.cmd](a)


if __name__ == '__main__':
    main()
