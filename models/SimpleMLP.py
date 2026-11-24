"""Plain baselines that plug into the ReNF pipeline (same data, lookback, optimiser, EMA, early stopping, loss).

SimpleMLP : RevIN -> Linear(T, d_ff) -> GELU -> Dropout -> Linear(d_ff, H), channel-independent (shared
            weights across variables). No stages, no positional embedding, no feedback.
RLinear   : RevIN -> Linear(T, H)  (selected with --model RLinear).
Both return a one-element list so that the ReNF training / test code (which expects a list of stage
outputs) runs unchanged; a single "stage" predicts the whole horizon.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs, d_inputs):
        super(Model, self).__init__()
        T, V, H = d_inputs[0], d_inputs[-1], configs.pred_len
        self.rev_in = configs.revin
        if self.rev_in:
            # revin_last=1: centre on the last input value, so a zero/shrunk output means persistence
            self.revin_layer = RevIN(V, affine=True, subtract_last=bool(getattr(configs, 'revin_last', 0)))
        # revin_ewma=1: learnable anchor between the window mean and the last value. Per variable,
        # anchor = sum_t w_t x_t with w_t ∝ (1 - a)^(T-1-t): a -> 0 gives the mean, a -> 1 the last value.
        # a = sigmoid(10 * theta) (x10 so that Adam can move it within a few hundred steps); init a = 0.01.
        self.ewma = bool(getattr(configs, 'revin_ewma', 0)) and self.rev_in
        if self.ewma:
            self.theta = nn.Parameter(torch.full((V,), math.log(0.01 / 0.99) / 10.0))
            self.aw = nn.Parameter(torch.ones(V))
            self.ab = nn.Parameter(torch.zeros(V))
            self.register_buffer('dist', torch.arange(T - 1, -1, -1, dtype=torch.float32))
        # anchor_out (input always centred on the window mean, output anchored elsewhere; decoupled):
        #   last   : y = x_T + s * f((x - mean) / s)
        #   linear : y = w.x + s * f(...), w (sum 1) fitted in closed form on the training split by the experiment
        self.anchor_out = getattr(configs, 'anchor_out', 'none') if self.rev_in else 'none'
        # anchor_in=last: input centred on the last value, output still anchored on the window mean
        # center_k=K   : input AND output centred on the mean of the last K steps (K = whole periods: phase-neutral)
        self.anchor_in = getattr(configs, 'anchor_in', 'none') if self.rev_in else 'none'
        self.center_k = int(getattr(configs, 'center_k', 0)) if self.rev_in else 0
        self.custom = self.anchor_out != 'none' or self.anchor_in != 'none' or self.center_k > 0
        if self.custom:
            self.oaw = nn.Parameter(torch.ones(V))
            self.oab = nn.Parameter(torch.zeros(V))
            self.register_buffer('anchor_w', torch.zeros(T))
        if configs.model == 'RLinear':
            self.net = nn.Linear(T, H)
        else:
            self.net = nn.Sequential(nn.Linear(T, configs.d_ff), nn.GELU(), nn.Dropout(configs.dropout),
                                     nn.Linear(configs.d_ff, H))

    @property
    def anchor_alpha(self):
        return torch.sigmoid(10.0 * self.theta) if self.ewma else None

    def forward(self, x, return_rep=False):
        if self.custom:
            m = (x[:, -self.center_k:] if self.center_k > 0 else x).mean(dim=1, keepdim=True).detach()
            s = x.std(dim=1, keepdim=True, unbiased=False).detach() + 1e-5
            if self.anchor_out == 'last':
                c = x[:, -1:, :]
            elif self.anchor_out == 'linear':
                c = (x * self.anchor_w[None, :, None]).sum(dim=1, keepdim=True)
            else:
                c = m
            ci = x[:, -1:, :] if self.anchor_in == 'last' else m
            xn = (x - ci) / s * self.oaw + self.oab
            y = self.net(xn.transpose(1, 2)).transpose(1, 2)
            y = (y - self.oab) / (self.oaw + 1e-10) * s + c
            return ([y], [y]) if return_rep else [y]
        if self.ewma:
            w = torch.softmax(self.dist[:, None] * F.logsigmoid(-10.0 * self.theta)[None, :], dim=0)   # (T, V)
            c = (x * w).sum(dim=1, keepdim=True)
            s = x.std(dim=1, keepdim=True, unbiased=False).detach() + 1e-5
            xn = (x - c) / s * self.aw + self.ab
            y = self.net(xn.transpose(1, 2)).transpose(1, 2)
            y = (y - self.ab) / (self.aw + 1e-10) * s + c
            return ([y], [y]) if return_rep else [y]
        if self.rev_in:
            x = self.revin_layer(x, 'norm')
        y = self.net(x.transpose(1, 2)).transpose(1, 2)      # (B, H, V)
        if self.rev_in:
            y = self.revin_layer(y, 'denorm')
        return ([y], [y]) if return_rep else [y]
