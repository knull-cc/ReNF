"""ReNF-CL: ReNF-alpha with a closed-loop *observable* error signal.

Stage 1 becomes a sliding block predictor g (input window of length T - s):
  * forecast:  g(X[s+1:T])   -> next steps after T            (stage-1 forecast, first k steps)
  * backtest:  g(X[1:T-s])   -> steps T-s+1..T, already observed in the input
  r_obs = X[T-s+1:T] - backtest   (no future information is used; available at deployment)
Because g is trained to forecast "the steps after a window", the backtest is the same task,
so no extra loss is needed. r_obs (detached) is concatenated to the input of every later stage,
next to the official forecast feedback.

--cl_use_r 0 gives the matched control: identical architecture and parameter count, but r_obs = 0.
Everything else (RevIN, positional embedding, Fore/MidBlocks, feedback switches) follows ReNF_alpha.
"""
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from models.ReNF_alpha import ForeBlock, MidBlock


class Model(nn.Module):
    def __init__(self, configs, d_inputs):
        super(Model, self).__init__()
        self.rev_in = configs.revin
        self.pred_len = configs.pred_len
        if self.rev_in:
            self.revin_layer = RevIN(d_inputs[-1], affine=True, subtract_last=False)
        self.decoder = Decoder(d_inputs, self.pred_len, n_c=configs.d_layers, d_ff=configs.d_ff,
                               dropout=configs.dropout, pe=configs.pe, s=configs.cl_s, use_r=configs.cl_use_r)

    def forward(self, x, return_rep=False):
        if self.rev_in:
            x = self.revin_layer(x, 'norm')
        result = self.decoder(x, return_rep=return_rep)
        ys, reps = result if return_rep else (result, None)
        if self.rev_in:
            ys = [self.revin_layer(y, 'denorm') for y in ys]
        return (ys, reps) if return_rep else ys


class Decoder(nn.Module):
    def __init__(self, d_inputs, d_pred, n_c=3, d_ff=256, dropout=0., pe=False, s=48, use_r=1):
        super(Decoder, self).__init__()
        T, V = d_inputs[0], d_inputs[-1]
        self.receiver = DecoderBlock(d_ff, T, d_pred, n_c, drop_ff=dropout, s=s, use_r=use_r)
        self.pe = pe
        W_pos = torch.empty((T, V))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.embedding = nn.Parameter(W_pos)

    def forward(self, x, return_rep=False):
        x_raw = x                                   # normalised input WITHOUT positional embedding (for g)
        if self.pe:
            x = x + self.embedding
        result = self.receiver(x.transpose(1, 2), x_raw.transpose(1, 2), return_rep)
        ys, reps = result if return_rep else (result, None)
        ys = [y.transpose(1, 2) for y in ys]
        return (ys, reps) if return_rep else ys


class DecoderBlock(nn.Module):
    def __init__(self, d_ff, d_in, d_out, n_c=3, drop_ff=None, drop_in=0.3, s=48, use_r=1):
        super(DecoderBlock, self).__init__()
        assert 0 < s < d_in, 'cl_s must be between 1 and seq_len - 1'
        self.pred_base = k = d_out // n_c
        self.n_c, self.s, self.use_r = n_c, s, use_r
        # stage 1: sliding block predictor g, window length d_in - s -> next max(s, k) steps
        self.cascade = nn.ModuleList([ForeBlock(d_in - s, max(s, k), d_ff, drop_ff)])
        self.fb_scale, self.fb_shuffle, self.fb_detach = 1.0, False, True   # as ReNF_alpha
        self.fb_override, self.fb_capture, self.last_out = None, False, None
        for i in range(1, n_c):
            n_acc = i * (i + 1) // 2
            d_cat = d_in + k * n_acc + s                                     # + s for r_obs
            act = 'nonlinear' if i < n_c - 1 else 'linear'
            drop_in_layer = min(drop_in * (i + 1), 0.6)
            self.cascade.append(MidBlock(d_cat, k * (i + 1), d_ff, drop_ff, drop_in_layer, act))

    def _feedback(self, out):
        if self.fb_override is not None:
            y_cat = torch.cat(self.fb_override[:len(out)], dim=-1)
        else:
            y_cat = torch.cat(out, dim=-1)
            if self.fb_detach:
                y_cat = y_cat.detach()
        if self.fb_shuffle and not self.training and y_cat.shape[0] > 1:
            y_cat = torch.roll(y_cat, shifts=y_cat.shape[0] // 2, dims=0)
        return y_cat * self.fb_scale

    def forward(self, x0, x_raw, return_rep=False):
        s, k = self.s, self.pred_base
        g = self.cascade[0]
        y_fut, rep = g(x_raw[..., s:])               # most recent window -> future
        out, reps = [y_fut[..., :k]], [rep]
        if self.n_c > 1:
            if self.use_r:
                y_bt, _ = g(x_raw[..., :-s])          # same predictor, window ending s steps ago
                r = (x_raw[..., -s:] - y_bt[..., :s]).detach()
            else:
                r = torch.zeros_like(x_raw[..., -s:])
            for layer in self.cascade[1:]:
                x_k = torch.cat([x0, self._feedback(out), r], dim=-1)
                y_k, rep = layer(x_k)
                out.append(y_k)
                reps.append(rep)
        if self.fb_capture:
            self.last_out = [o.detach() for o in out]
        return (out, reps) if return_rep else out
