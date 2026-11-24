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
        # Horizon-gated head (hypothesis: nonlinear capacity belongs where the horizon has signal):
        #   y = g * MLP(z) + A z  [+ C * stopgrad(MLP(z)[:K]) on the far steps when prop=1]
        #   gate: none (plain MLP; with skip=1 the gate is 1 everywhere), hard (g=1 for the first K steps, 0
        #   after), revhard (g=1 only on the last K steps; control), exp (g=exp(-h/K)), rev (mirror of exp).
        #   The gate also scales the gradient each horizon step sends into the shared nonlinear layer.
        gate = getattr(configs, 'gate', 'none')
        self.skip = bool(getattr(configs, 'skip', 0)) or gate != 'none'
        self.prop = bool(getattr(configs, 'prop', 0)) and gate == 'hard'
        self.gk = int(getattr(configs, 'gate_k', 0)) or H // 2
        self.gated = gate != 'none'
        if self.gated:
            h = torch.arange(H, dtype=torch.float32)
            g = {'hard': (h < self.gk).float(), 'revhard': (h >= H - self.gk).float(),
                 'exp': torch.exp(-h / self.gk), 'rev': torch.exp(-(H - 1 - h) / self.gk)}[gate]
            self.register_buffer('g', g)
        if self.skip and configs.model != 'RLinear':
            self.lin = nn.Linear(T, H)
        if self.prop:
            self.C = nn.Linear(self.gk, H - self.gk, bias=False)
            nn.init.zeros_(self.C.weight)
        # Gradient routing (HAGR): the forward pass is unchanged; horizon step t sends only omega_t of its gradient
        # into the shared hidden layer, while its own output weights still get the full gradient.
        #   route: hard (omega = 1 for the first route_k steps, 0 after), revhard (mirror),
        #          exp (omega_t = exp(-lam * t / (H - 1))), revexp (mirror).
        # hw: the whole output gradient of step t is scaled by omega_t (= per-step loss weighting; control for route).
        steps = torch.arange(H, dtype=torch.float32)
        rk = int(getattr(configs, 'route_k', 0)) or H // 2

        def omega(kind, lam):
            return {'hard': (steps < rk).float(), 'revhard': (steps >= H - rk).float(),
                    'exp': torch.exp(-lam * steps / max(H - 1, 1)),
                    'revexp': torch.exp(-lam * (H - 1 - steps) / max(H - 1, 1))}[kind]
        route = getattr(configs, 'route', 'none')
        self.routed = route != 'none' and configs.model != 'RLinear'
        if self.routed:
            self.register_buffer('omega', omega(route, float(getattr(configs, 'route_lam', 2.0))))
        hw = getattr(configs, 'hw', 'none')
        self.hw = hw != 'none'
        if self.hw:
            self.register_buffer('hw_w', omega(hw, float(getattr(configs, 'hw_lam', 2.0))))
        self.is_lin = configs.model == 'RLinear'
        if self.is_lin:
            self.skip = self.gated = self.prop = False
            self.net = nn.Linear(T, H)
        else:
            self.net = nn.Sequential(nn.Linear(T, configs.d_ff), nn.GELU(), nn.Dropout(configs.dropout),
                                     nn.Linear(configs.d_ff, H))
        # F-CMP (foldable causal multi-scale preconditioning; hypothesis: under Adam the recent level is learned
        # too slowly, because it sits in very few input coordinates). First layer: u = W z + alpha * W_rec (P z),
        # where P holds M fixed suffix means (the last 1, 2, 4, ... steps). W_rec starts at zero and is built
        # without consuming the global RNG, so the function class, the initial function and the seed-paired
        # initialisation / data order are unchanged; only Adam's step along the recent-level directions is scaled
        # by alpha (default sqrt(T / M)). At inference W + alpha * W_rec P folds into one matrix.
        # fcmp_rev=1: prefix means of the first 1, 2, 4, ... steps instead (control).
        self.fcmp = bool(getattr(configs, 'fcmp', 0)) and not self.is_lin
        if self.fcmp:
            M = int(getattr(configs, 'fcmp_m', 8))
            rev = bool(getattr(configs, 'fcmp_rev', 0))
            P = torch.zeros(M, T)
            for m in range(M):
                L = min(2 ** m, T)
                if rev:
                    P[m, :L] = 1.0 / L
                else:
                    P[m, T - L:] = 1.0 / L
            self.register_buffer('P', P)
            a = float(getattr(configs, 'fcmp_alpha', -1.0))
            self.fa = a if a > 0 else math.sqrt(T / M)
            rng = torch.random.get_rng_state()
            self.rec = nn.Linear(M, configs.d_ff, bias=False)
            torch.random.set_rng_state(rng)
            nn.init.zeros_(self.rec.weight)
            print('F-CMP: M={} alpha={:.2f} {}'.format(M, self.fa, 'prefix means (control)' if rev else 'suffix means'))

    def _head(self, z):
        """z: (B, V, T) normalised input -> (B, V, H) in the same space."""
        if self.is_lin:
            y = self.net(z)
        else:
            u = self.net[0](z)
            if self.fcmp:
                u = u + self.fa * self.rec(F.linear(z, self.P))
            h = self.net[2](self.net[1](u))
            n = self.net[3](h)
            if self.routed:
                n = self.omega * n + (1.0 - self.omega) * self.net[3](h.detach())
            if self.skip or self.gated:
                y = n * self.g if self.gated else n
                y = y + self.lin(z)
                if self.prop:
                    near = n[..., :self.gk].detach()
                    y = torch.cat([y[..., :self.gk], y[..., self.gk:] + self.C(near)], dim=-1)
            else:
                y = n
        if self.hw:
            y = self.hw_w * y + (1.0 - self.hw_w) * y.detach()
        return y

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
            y = self._head(xn.transpose(1, 2)).transpose(1, 2)
            y = (y - self.oab) / (self.oaw + 1e-10) * s + c
            return ([y], [y]) if return_rep else [y]
        if self.ewma:
            w = torch.softmax(self.dist[:, None] * F.logsigmoid(-10.0 * self.theta)[None, :], dim=0)   # (T, V)
            c = (x * w).sum(dim=1, keepdim=True)
            s = x.std(dim=1, keepdim=True, unbiased=False).detach() + 1e-5
            xn = (x - c) / s * self.aw + self.ab
            y = self._head(xn.transpose(1, 2)).transpose(1, 2)
            y = (y - self.ab) / (self.aw + 1e-10) * s + c
            return ([y], [y]) if return_rep else [y]
        if self.rev_in:
            x = self.revin_layer(x, 'norm')
        y = self._head(x.transpose(1, 2)).transpose(1, 2)      # (B, H, V)
        if self.rev_in:
            y = self.revin_layer(y, 'denorm')
        return ([y], [y]) if return_rep else [y]
