from logging import raiseExceptions
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import ReNF_alpha, ReNF_beta, Model_test, ReNF_cl, SimpleMLP
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, ema_update, mean_filter
from utils.metrics import metric
from thop import profile

import torch
import torch.nn as nn
from torch import optim
# from torch.optim import lr_scheduler 
import pytorch_warmup

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import copy

warnings.filterwarnings('ignore')


class EraBatchSampler(torch.utils.data.Sampler):
    """Every batch holds batch_size // M training windows from each of M groups, group e at positions
    [e*per, (e+1)*per). Real eras: M contiguous time blocks of the (time-ordered) training windows.
    Pseudo eras (control): a seeded random partition into M groups of the same sizes."""
    def __init__(self, n, batch_size, m, pseudo, seed):
        self.m, self.per = m, batch_size // m
        idx = np.random.RandomState(seed + 7).permutation(n) if pseudo else np.arange(n)
        self.groups = [np.sort(g) for g in np.array_split(idx, m)]
        self.n_batches = min(len(g) for g in self.groups) // self.per

    def __iter__(self):
        perms = [np.random.permutation(g) for g in self.groups]
        for b in range(self.n_batches):
            yield np.concatenate([p[b * self.per:(b + 1) * self.per] for p in perms]).tolist()

    def __len__(self):
        return self.n_batches


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        self.train_data, self.train_loader = self._get_data(flag='train')
        if self.args.era_m > 0:
            from torch.utils.data import DataLoader
            sampler = EraBatchSampler(len(self.train_data), self.args.batch_size, self.args.era_m,
                                      bool(self.args.era_pseudo), self.args.random_seed)
            self.train_loader = DataLoader(self.train_data, batch_sampler=sampler, num_workers=self.args.num_workers)
            print('era_consensus: M={} ({} eras) tau={} | {} windows, {} per era per batch, {} batches/epoch'.format(
                self.args.era_m, 'pseudo' if self.args.era_pseudo else 'real time-block', self.args.era_tau,
                len(self.train_data), sampler.per, len(sampler)))
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
        B, T, V = next(iter(self.train_loader))[0].shape
        L = 1
        self.d_inputs = [T, L, V]
        self.model = self._build_model().to(self.device)
        self._set_feedback(self.model, scale=0.0 if self.args.fb_mode == 'zero' else 1.0,
                           detach=None if self.args.fb_detach < 0 else bool(self.args.fb_detach))

        input = torch.zeros(1, T, V).to(self.device)
        flops, params = profile(self.model, inputs=(input, ))
        print('Flops: % .4fG'%(flops / 1e9))
        print('params: % .4fM'% (params / 1e6)) 

    def _build_model(self):
        model_dict = {
            'Model_test': Model_test,
            'ReNF_alpha': ReNF_alpha,
            'ReNF_beta': ReNF_beta,
            'ReNF_cl': ReNF_cl,
            'SimpleMLP': SimpleMLP,
            'RLinear': SimpleMLP,
        }
        if 'ReNF' or 'test' in self.args.model:
            model = model_dict[self.args.model].Model(self.args, d_inputs=self.d_inputs).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _set_feedback(self, model, scale=None, shuffle=None, detach=None):
        for m in model.modules():
            if type(m).__name__ == 'DecoderBlock':
                if scale is not None:
                    m.fb_scale = scale
                if shuffle is not None:
                    m.fb_shuffle = shuffle
                if detach is not None:
                    m.fb_detach = detach

    def _era_backward(self, outputs, batch_y, f_dim, n, k):
        """Per-era gradients of the training loss (equal era sizes, so their mean is the usual batch
        gradient). Element-wise sign agreement a = |mean_e sign(g_e)| (M=4: 1 = 4-0, 0.5 = 3-1, 0 = 2-2);
        the mean gradient is kept where a >= era_tau and zeroed elsewhere (era_tau=0: no masking).
        Returns the kept fraction."""
        m = self.args.era_m
        per = batch_y.shape[0] // m
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = []
        for e in range(m):
            sl = slice(e * per, (e + 1) * per)
            le = 0.
            for j in range(n):
                le = le + (20. / (j + 1)) * self.time_freq_loss(
                    outputs[j][sl], batch_y[sl, :(j + 1) * k, f_dim:], self.args.alpha_freq)
            grads.append(torch.autograd.grad(le, params, retain_graph=e < m - 1, allow_unused=True))
        kept, total = 0., 0
        for i, p in enumerate(params):
            g = [gr[i] for gr in grads if gr[i] is not None]
            if not g:
                continue
            G = torch.stack(g)
            mean = G.mean(0)
            if self.args.era_tau > 0:
                mask = (torch.sign(G).mean(0).abs() >= self.args.era_tau - 1e-6).to(mean.dtype)
                kept += mask.sum().item()
                total += mask.numel()
                mean = mean * mask
            p.grad = mean
        return kept / total if total else 1.0

    def _td_loss(self, pred, batch_x, batch_y, f_dim):
        """Horizon-TD bootstrap target. The window shifted by d steps, x_s = [x[d:], y[:d]], ends d steps
        later; its forecast steps 1..H-d are the same time points as the student's steps d+1..H.
          td_mode=ema : target = EMA model on x_s, no gradient (one-way, teacher sees y[:d])
          td_mode=sym : target = online model on x_s, gradient through both (two-way consistency)
        td_frac=0 gives d=0: plain EMA self-distillation on the same window (no future information)."""
        H = self.args.pred_len
        d = int(round(self.args.td_frac * H))
        T = batch_x.shape[1]          # last T steps of [x, y[:d]]; also valid when d > T (e.g. 720 with lookback 336)
        x_s = torch.cat([batch_x, batch_y[:, :d, :batch_x.shape[-1]]], dim=1)[:, -T:] if d > 0 else batch_x
        if self.args.td_mode == 'ema':
            with torch.no_grad():
                was = self.model_pre.training
                self.model_pre.eval()
                tgt = self.model_pre(x_s)[-1][:, :H - d, f_dim:]
                self.model_pre.train(was)
        else:
            tgt = self.model(x_s)[-1][:, :H - d, f_dim:]
        p = pred[:, d:, f_dim:]
        return (p - tgt).abs().mean() if self.args.td_loss == 'l1' else ((p - tgt) ** 2).mean()

    def _blocks(self, model):
        return [m for m in model.modules() if type(m).__name__ == 'DecoderBlock']

    def _set_fb_train(self, model, active):
        for b in self._blocks(model):
            b.fb_train_mode, b.fb_train_sigma, b.fb_train_active = \
                self.args.fb_train, self.args.fb_train_sigma, active

    def _set_fb_override(self, model, outs):
        blocks = self._blocks(model)
        for b, o in zip(blocks, outs if outs is not None else [None] * len(blocks)):
            b.fb_override = o

    @torch.no_grad()
    def _ema_feedback(self, batch_x):
        """Stage forecasts of the EMA model for this batch (eval mode: no dropout), inside the decoder."""
        was_training = self.model_pre.training
        self.model_pre.eval()
        blocks = self._blocks(self.model_pre)
        for b in blocks:
            b.fb_capture = True
        self.model_pre(batch_x)
        outs = [b.last_out for b in blocks]
        for b in blocks:
            b.fb_capture, b.last_out = False, None
        self.model_pre.train(was_training)
        return outs

    def _feedback_probes(self, trues):
        """Test-time only, same trained weights: last-stage forecast when the fed-back
        forecasts are zeroed, or swapped with another sample's (roll by B/2 inside a
        fixed 256-window batch, i.e. a window 128 steps away, whatever the train batch size)."""
        from torch.utils.data import DataLoader
        blocks = [m for m in self.model.modules() if type(m).__name__ == 'DecoderBlock']
        if self.args.fb_mode == 'zero' or not blocks or max(m.n_c for m in blocks) < 2:
            return
        f_dim = -1 if self.args.features == 'MS' else 0
        self.model.eval()
        for name, kw in [('fbzero', dict(scale=0.0)), ('fbshuffle', dict(shuffle=True))]:
            self._set_feedback(self.model, **kw)
            preds = []
            with torch.no_grad():
                for batch_x, batch_y, _, _ in DataLoader(self.test_data, batch_size=256, shuffle=False):
                    out = self.model(batch_x.float().to(self.device))
                    preds.append(out[-1][:, :, f_dim:].detach().cpu().numpy())
            self._set_feedback(self.model, scale=1.0, shuffle=False)
            mae, mse = metric(np.concatenate(preds, axis=0).flatten(), trues)
            print('probe_{} mse:{}, mae:{}'.format(name, mse, mae))

    def _stage_diagnostics(self, outs, y):
        """outs[j]: (N, h_j, V) forecast of stage j (h_j = (j+1)k); y: (N, L, V).
        On every segment, compare the stages that forecast it:
          ambiguity decomposition  MSE(mean) = mean_j MSE(f_j) - mean_j |f_j - mean|^2
          (the second term = diversity; if ~0, averaging can never help)."""
        n = len(outs)
        if n < 2:
            return
        k = outs[0].shape[1]
        rows = []
        for s in range(n):
            a, b = s * k, (s + 1) * k
            fs = [o[:, a:b] for o in outs[s:]]           # stages s..n-1 all forecast segment s
            yt = y[:, a:b]
            mse = [float(((f - yt) ** 2).mean()) for f in fs]
            if len(fs) > 1:
                avg = np.mean(fs, axis=0)
                amb = float(np.mean([((f - avg) ** 2).mean() for f in fs]))
                e = np.stack([(f - yt).reshape(-1) for f in fs])
                c = np.corrcoef(e)
                corr = float(c[~np.eye(len(fs), dtype=bool)].mean())
                ens = float(((avg - yt) ** 2).mean())
            else:
                amb, corr, ens = 0.0, 1.0, mse[0]
            rows.append((s + 1, len(fs), mse[0], mse[-1], float(np.mean(mse)), amb, ens, corr))
            print('diag_seg {} n_f={} first={:.5f} final={:.5f} mean_member={:.5f} '
                  'ambiguity={:.6f} ensemble={:.5f} err_corr={:.4f}'.format(*rows[-1]))
        tot = lambda i: float(np.mean([r[i] for r in rows if r[1] > 1]))
        print('diag_summary ambiguity_ratio={:.4f} err_corr={:.4f} refine_gain={:+.2f}%'.format(
            tot(5) / tot(4), tot(7), float(np.mean([(r[3] / r[2] - 1) * 100 for r in rows if r[1] > 1]))))

    def _stage_outputs(self, loader):
        """All stage forecasts and targets of a loader, in loader order (eval mode, no dropout)."""
        f_dim = -1 if self.args.features == 'MS' else 0
        outs, ys = None, []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, _, _ in loader:
                o = self.model(batch_x.float().to(self.device))
                if outs is None:
                    outs = [[] for _ in o]
                for j, t in enumerate(o):
                    outs[j].append(t[:, :, f_dim:].cpu().numpy())
                ys.append(batch_y.float()[:, -self.args.pred_len:, f_dim:].numpy())
        return [np.concatenate(x, axis=0) for x in outs], np.concatenate(ys, axis=0)

    def _eta_probe(self):
        """Shrink the last stage towards the previous one on their shared prefix:
            out[:h] = p + eta * (u[:h] - p),  out[h:] = u[h:]   (eta = 1 is the official output)
        Risk identity: R(eta) = R(0) - 2 eta a + eta^2 b, a = sum<y - p, u - p>, b = sum|u - p|^2,
        so eta* = clip(a / b, 0, 1). eta is FITTED ON VALIDATION and frozen before touching test."""
        vo, vy = self._stage_outputs(self.vali_loader)
        if len(vo) < 2:
            return
        to, ty = self._stage_outputs(self.test_loader)
        h = vo[-2].shape[1]

        def ab(o, y):
            p, d = o[-2], o[-1][:, :h] - o[-2]
            return float(((y[:, :h] - p) * d).sum()), float((d * d).sum())

        def fit(a, b):
            return (a / b if b > 0 else 1.0), (min(max(a / b, 0.0), 1.0) if b > 0 else 1.0)

        raw_val, eta_val = fit(*ab(vo, vy))
        m = len(vy) // 2
        _, eta_h1 = fit(*ab([x[:m] for x in vo], vy[:m]))
        _, eta_h2 = fit(*ab([x[m:] for x in vo], vy[m:]))
        _, eta_tr = fit(*ab(*self._stage_outputs(self.train_loader)))
        raw_te, eta_te = fit(*ab(to, ty))          # ORACLE (uses test labels) -- diagnostic only

        p, u = to[-2], to[-1]
        for name, eta in [('eta1', 1.0), ('eta0', 0.0), ('eta05', 0.5), ('etaval', eta_val), ('etatrain', eta_tr),
                          ('etaoracle', eta_te)]:
            out = u.copy()
            out[:, :h] = p + eta * (u[:, :h] - p)
            e = out - ty
            print('eta_probe {} eta={:.4f} mse={:.6f} mae={:.6f} prefix_mse={:.6f}'.format(
                name, eta, float((e ** 2).mean()), float(np.abs(e).mean()), float((e[:, :h] ** 2).mean())))
        print('eta_fit val={:.4f} val_raw={:.4f} val_half1={:.4f} val_half2={:.4f} train={:.4f} '
              'test_oracle={:.4f} test_oracle_raw={:.4f} shared_prefix={}'.format(
                  eta_val, raw_val, eta_h1, eta_h2, eta_tr, eta_te, raw_te, h))

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, pred_len=None)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _update_model_ema(self, model_pre, model, decay=0.998):
        return ema_update(model_pre, model, decay)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # evaluate with the smoothed (EMA) model only
        self.model_pre.eval()
        self.model.eval()
        iter_count = 0
        time_now = time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    raiseExceptions ('No amp Implements')
                else:
                    if 'test' in self.args.model or 'ReNF' in self.args.model or self.args.model in ('SimpleMLP', 'RLinear'):
                        outputs = self.model_pre(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if (i + 1) % 500 == 0:
                    speed = (time.time() - time_now) / iter_count
                    print('\tinference speed: {:.4f}s/iter'.format(speed))
                    iter_count = 0
                    time_now = time.time()

                f_dim = -1 if self.args.features == 'MS' else 0

                pred = outputs[-1][:, -self.args.pred_len:, f_dim:].detach().cpu()
                true = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        # plot loss
    
        total_loss = np.mean(total_loss)
        # self.model.train()
        return total_loss
    
    def time_freq_loss(self, y_hat, y, alpha_freq):
        freq_loss = (torch.fft.rfft(y_hat, dim=1) - torch.fft.rfft(y, dim=1)).abs().mean()
        # weights = (20. / (torch.arange(y_hat.shape[1], dtype=torch.float32, device=y_hat.device) + 1.0)).unsqueeze(0).unsqueeze(-1)
        if self.args.time_loss == 'l2':
            time_loss = ((y_hat - y) ** 2).mean()
        elif self.args.time_loss == 'traj':
            # robust across windows, quadratic within a window: Euclidean norm of each (sample, variable)
            # residual trajectory over the horizon (a spatial-median loss), averaged over samples and variables
            time_loss = torch.sqrt(((y_hat - y) ** 2).mean(dim=1) + 1e-8).mean()
        elif self.args.time_loss == 'huber':
            time_loss = torch.nn.functional.huber_loss(y_hat, y, delta=self.args.huber_delta)
        else:
            time_loss = (y_hat - y).abs().mean()
        return alpha_freq * freq_loss + (1 - alpha_freq) * time_loss
    
    # def time_freq_loss2(self, y_hat, y, alpha_freq):
    #     freq_loss = ((torch.fft.rfft(y_hat, dim=1) - torch.fft.rfft(y, dim=1)).abs()**2).mean()
    #     # weights = (20. / (torch.arange(y_hat.shape[1], dtype=torch.float32, device=y_hat.device) + 1.0)).unsqueeze(0).unsqueeze(-1)
    #     time_loss = ((y_hat - y)**2).mean()
    #     return alpha_freq * freq_loss + (1 - alpha_freq) * time_loss

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_val = nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=2, gamma=self.args.gamma)
        self.model.train()
        self.model_pre = copy.deepcopy(self.model).requires_grad_(False)
        ema_init_w = 1.0      # weight the EMA model still puts on the random initialisation
        use_ema_fb = self.args.fb_source == 'ema'
        if self.args.fb_ema_start >= 0:
            ema_start = self.args.fb_ema_start
        else:
            ema_start = int(round(1.0 / (1.0 - self.args.r_ema))) if self.args.r_ema < 1 else 0
        global_step, ema_fb_steps = 0, 0
        if use_ema_fb:
            print('fb_source=ema: EMA-model feedback from training step {}'.format(ema_start))
        if self.args.fb_train != 'none':
            print('fb_train={} (sigma={}): feedback perturbed in every training forward pass'.format(
                self.args.fb_train, self.args.fb_train_sigma))
        if self.args.td_lambda > 0:
            print('horizon_td: mode={} d={} lambda={} loss={} from epoch {}'.format(
                self.args.td_mode, int(round(self.args.td_frac * self.args.pred_len)), self.args.td_lambda,
                self.args.td_loss, self.args.td_start_epoch + 1))
        valid_losses = []
        test_losses = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            era_keep = []
            if self.args.keep_train_mode:
                self.model.train()
            if epoch < 3:
                print('epoch {}: online model in {} mode'.format(epoch + 1, 'train' if self.model.training else 'EVAL'))

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    pass
                else:
                    if 'test' in self.args.model or 'ReNF' in self.args.model or self.args.model in ('SimpleMLP', 'RLinear'):
                            if use_ema_fb and global_step >= ema_start:
                                self._set_fb_override(self.model, self._ema_feedback(batch_x))
                                ema_fb_steps += 1
                            if self.args.fb_train != 'none':
                                self._set_fb_train(self.model, True)
                            outputs = self.model(batch_x)
                            self._set_fb_override(self.model, None)
                            if self.args.fb_train != 'none':
                                self._set_fb_train(self.model, False)
                            n = len(outputs)
                            k = self.args.pred_len // n
                    else:
                        pass
                    f_dim = -1 if self.args.features == 'MS' else 0

                    # HRL coefficients
                    # lamb = 0.8 # set to 0 to disable
                    # coefs_L = torch.ones_like(batch_y).cuda()
                    # n_c = 4 # better be a factor of pred_len
                    # for j in range(n_c):
                    #     m = self.args.pred_len // n_c
                    #     end = (j + 1) * m
                    #     coefs_L[:, j:end, f_dim:] *= (0.5 + lamb * (n_c - 1 - j))

                    loss = 0.
                    gamma = 20.
                    total_w = sum(gamma / (i + 1) for i in range(n))
                    for j in range(n):
                        if self.args.stage_weight == 'harmonic':      # official
                            w = gamma / (j + 1)
                        elif self.args.stage_weight == 'uniform':
                            w = total_w / n
                        else:                                         # 'last'
                            if j < n - 1:
                                continue
                            w = total_w
                        loss += w * self.time_freq_loss(outputs[j], batch_y[:, :(j + 1) * k, f_dim:], self.args.alpha_freq)
                    # loss = self.time_freq_loss(outputs[-1], batch_y[:, :, f_dim:], self.args.alpha_freq)
                    if self.args.td_lambda > 0 and epoch >= self.args.td_start_epoch:
                        loss = loss + total_w * self.args.td_lambda * self._td_loss(outputs[-1], batch_x, batch_y, f_dim)
                    train_loss.append(loss.item())
  
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    pass
                else:
                    if self.args.era_m > 0:
                        era_keep.append(self._era_backward(outputs, batch_y, f_dim, n, k))
                    else:
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    model_optim.step()
                    with torch.no_grad():
                        decay = self.args.r_ema
                        if self.args.ema_warmup:        # running mean until the window is full: no init weight
                            decay = min(decay, global_step / (global_step + 1.0))
                        ema_init_w *= decay
                        self.model_pre = self._update_model_ema(self.model_pre, self.model, decay=decay)
                    global_step += 1
                # store the weight of each layer
                # if (epoch == self.args.train_epochs - 1) and (i == train_steps - 5):
                #     weights = []
                #     import matplotlib.pyplot as plt
                #     for name, param in self.model.named_parameters():
                #         if 'weight' in name:
                #             weights.append(torch.norm(param.data.cpu(), p=2))
                #     plt.plot(weights)
                #     plt.savefig('weights.png')
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print('ema_init_weight epoch {}: {:.4f}'.format(epoch + 1, ema_init_w))
            if era_keep:
                print('era_keep epoch {}: {:.3f} of gradient entries kept'.format(epoch + 1, float(np.mean(era_keep))))
            train_loss = np.mean(train_loss)
            vali_loss = self.vali(self.vali_data, self.vali_loader, criterion_val)  # use the same indicator as in training phase
            valid_losses.append(vali_loss)
            test_loss = self.vali(self.test_data, self.test_loader, criterion_val)   # use test indicator
            test_losses.append(test_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model_pre, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            scheduler.step()
            print('Updating learning rate to {:.6f}'.format(scheduler.get_last_lr()[0]))
        # import matplotlib.pyplot as plt
        # np.save('./toy_exp/valid_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.npy', valid_losses)
        # np.save('./toy_exp/test_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.npy', test_losses)
        # plt.plot()
        # plt.savefig('./toy_exp/valid_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.pdf')
        # plt.clf()
        # plt.plot()
        # plt.savefig('./toy_exp/test_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.pdf')
        if use_ema_fb:
            print('fb_source=ema: used on {} of {} training steps'.format(ema_fb_steps, global_step))
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def post_process(self, outputs, batch_y, iter, strategy='optimal', plot=True):
        # Keep tensors on GPU for computation, store original device
        device = batch_y.device
        
        n = len(outputs)
        k = self.args.pred_len // n
    
        if strategy == 'average':
            pred_com = torch.zeros_like(batch_y)
            weight_sum = torch.zeros_like(batch_y)
            
            for i, output in enumerate(outputs):
                # Each output has length (i+1) * k where k = pred_len // n
                output_len = output.shape[1]
                target_len = min(output_len, batch_y.shape[1])
                
                # Add this output's contribution to the overlapping region
                pred_com[:, :target_len, :] += output[:, :target_len, :]
                weight_sum[:, :target_len, :] += 1.0
            
            # Normalize by the number of outputs that contributed to each position
            pred_com = torch.where(weight_sum > 0, pred_com / weight_sum, pred_com)
        else:
            # Optimal combination
            if n == 0:
                pred_com = torch.zeros_like(batch_y)
            else:
                pred_com = torch.zeros_like(batch_y)
                seq_len = min(outputs[0].shape[1], batch_y.shape[1])
                pred_com[:, :seq_len, :] = outputs[0][:, :seq_len, :]
                
                # Compare with subsequent outputs one by one
                for i in range(1, n):
                    output = outputs[i]
                    seq_len = min(output.shape[1], batch_y.shape[1])
                    
                    # Calculate errors only for the relevant sequence length
                    pred_error = torch.abs(batch_y[:, :seq_len, :] - pred_com[:, :seq_len, :])
                    output_error = torch.abs(batch_y[:, :seq_len, :] - output[:, :seq_len, :])
                    
                    # Update where output is better
                    mask = output_error < pred_error
                    pred_com[:, :seq_len, :] = torch.where(mask, output[:, :seq_len, :], pred_com[:, :seq_len, :])
        
        # Only create plot when actually needed - move to CPU only for plotting
        if plot and iter == 10:
            # Move to CPU only for plotting
            batch_y_cpu = batch_y.detach().cpu()
            pred_com_cpu = pred_com.detach().cpu()
            outputs_cpu = [output.detach().cpu() for output in outputs]
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot forecasts
            for i, output_cpu in enumerate(outputs_cpu):
                ax.plot(output_cpu[0, :, 0].numpy(), label=f'forecast_{i}')
            
            # Plot ground truth and combined prediction
            ax.plot(batch_y_cpu[0, :, 0].numpy(), label='true observation')
            ax.plot(pred_com_cpu[0, :, 0].numpy(), label=strategy)
            
            # Optimized styling
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True,
                     framealpha=0.9, loc='upper left')
            
            # Set spine properties in batch
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
            
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
            plt.tight_layout()
            plt.savefig(f'./forecast_visual/result_{self.args.model_id}_{iter}_{strategy}.pdf')
            plt.close()
            
        return pred_com

    def test(self, setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # # Pre-calculate total samples for efficient array allocation
        # total_samples = len(self.test_loader.dataset)
        # pred_len = self.args.pred_len
        # n_features = self.test_loader.dataset[0][1].shape[-1] if hasattr(self.test_loader.dataset, '__getitem__') else 7
        
        # Pre-allocate arrays for better performance
        all_preds = []
        all_trues = []
        all_pred_coms = []
        all_avgs = []
        stage_outs = []
        all_drift = []
        # all_reps = []
        
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                    
                batch_x = batch_x.float().to(self.device)  # (B, T, V, L)
                batch_y = batch_y.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        raise Exception('amp not supported')
                else:
                    if 'test' in self.args.model or 'ReNF' in self.args.model or self.args.model in ('SimpleMLP', 'RLinear'):
                        outputs, reps = self.model(batch_x, return_rep=True)
                    else:
                        raise Exception('Model not supported')

                f_dim = -1 if self.args.features == 'MS' else 0
                
                pred = outputs[-1][:, :, f_dim:].detach().cpu().numpy()
                true = batch_y[:, :, f_dim:].detach().cpu().numpy()
                pred_com = self.post_process(outputs, batch_y, i).detach().cpu().numpy()
                # non-oracle combination: average of all stage forecasts where they overlap
                avg = torch.zeros_like(batch_y)
                cnt = torch.zeros_like(batch_y)
                for o in outputs:
                    L = min(o.shape[1], batch_y.shape[1])
                    avg[:, :L] += o[:, :L]
                    cnt[:, :L] += 1
                all_avgs.append((avg / cnt.clamp(min=1))[:, :, f_dim:].detach().cpu().numpy())
                if self.args.diag:
                    stage_outs.append([o[:, :, f_dim:].detach().cpu().numpy() for o in outputs])
                if self.args.save_sample:
                    # drift score (analysis only, uses the label): |mean of the horizon - mean of the last
                    # 96 inputs| / std of the input window, averaged over variables
                    xw = batch_x[:, :, f_dim:]
                    tail = xw[:, -min(96, xw.shape[1]):]
                    drift = ((batch_y[:, :, f_dim:].mean(dim=1) - tail.mean(dim=1)).abs()
                             / (xw.std(dim=1) + 1e-5)).mean(dim=1)
                    all_drift.append(drift.cpu().numpy())
                # Collect batch results
                all_preds.append(pred)
                all_trues.append(true)
                all_pred_coms.append(pred_com)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        rep_store = {}
        for n in range(len(reps)):
            rep_store['rep'+str(n)] = reps[n].detach().cpu().numpy()
        np.save('./rep_store.npy', rep_store)
        if self.args.eta_probe:
            self._eta_probe()
        if self.args.diag:
            self._stage_diagnostics([np.concatenate([b[j] for b in stage_outs], axis=0)
                                     for j in range(len(stage_outs[0]))],
                                    np.concatenate(all_trues, axis=0))
        # Efficient concatenation and flattening
        preds = np.concatenate(all_preds, axis=0).flatten()
        trues = np.concatenate(all_trues, axis=0).flatten()
        pred_coms = np.concatenate(all_pred_coms, axis=0).flatten()
        
        print(f"Results shape: {trues.shape}")
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse = metric(preds, trues)
        mae_com, mse_com = metric(pred_coms, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('mse_com:{}, mae_com:{}'.format(mse_com, mae_com))   # ORACLE (uses ground truth) -- never report
        if self.args.probe_biascorr:
            # mean residual per (horizon step, variable) of the SAME model on the validation set, added to the
            # test forecasts: tests whether training for the median (L1) leaves a mean bias that costs MSE
            f_dim = -1 if self.args.features == 'MS' else 0
            res = []
            with torch.no_grad():
                for bx, by, _, _ in self.vali_loader:
                    out = self.model(bx.float().to(self.device))
                    res.append((by.float().to(self.device)[:, :, f_dim:] - out[-1][:, :, f_dim:]).cpu().numpy())
            bias = np.concatenate(res, axis=0).mean(axis=0, keepdims=True)
            P = np.concatenate(all_preds, axis=0)
            mae_b, mse_b = metric((P + bias).flatten(), trues)
            print('probe_biascorr mse:{}, mae:{}'.format(mse_b, mae_b))
            print('val_mean_residual_abs: {:.5f}'.format(float(np.abs(bias).mean())))
        if self.args.probe:
            mae_a, mse_a = metric(np.concatenate(all_avgs, axis=0).flatten(), trues)
            print('probe_stageavg mse:{}, mae:{}'.format(mse_a, mae_a))
            self._feedback_probes(trues)
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        if self.args.save_sample:
            P, Y = np.concatenate(all_preds, axis=0), np.concatenate(all_trues, axis=0)
            np.save(folder_path + 'sample_mse.npy', ((P - Y) ** 2).mean(axis=(1, 2)))
            np.save(folder_path + 'drift.npy', np.concatenate(all_drift, axis=0))
        # The following savings are for model analyses
        # torch.save(store, folder_path + 'store.pt')
        # torch.save(Ys, folder_path + 'Ys.pt')
        return
