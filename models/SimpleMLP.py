"""Plain baselines that plug into the ReNF pipeline (same data, lookback, optimiser, EMA, early stopping, loss).

SimpleMLP : RevIN -> Linear(T, d_ff) -> GELU -> Dropout -> Linear(d_ff, H), channel-independent (shared
            weights across variables). No stages, no positional embedding, no feedback.
RLinear   : RevIN -> Linear(T, H)  (selected with --model RLinear).
Both return a one-element list so that the ReNF training / test code (which expects a list of stage
outputs) runs unchanged; a single "stage" predicts the whole horizon.
"""
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs, d_inputs):
        super(Model, self).__init__()
        T, V, H = d_inputs[0], d_inputs[-1], configs.pred_len
        self.rev_in = configs.revin
        if self.rev_in:
            # revin_last=1: centre on the last input value, so a zero/shrunk output means persistence
            self.revin_layer = RevIN(V, affine=True, subtract_last=bool(getattr(configs, 'revin_last', 0)))
        if configs.model == 'RLinear':
            self.net = nn.Linear(T, H)
        else:
            self.net = nn.Sequential(nn.Linear(T, configs.d_ff), nn.GELU(), nn.Dropout(configs.dropout),
                                     nn.Linear(configs.d_ff, H))

    def forward(self, x, return_rep=False):
        if self.rev_in:
            x = self.revin_layer(x, 'norm')
        y = self.net(x.transpose(1, 2)).transpose(1, 2)      # (B, H, V)
        if self.rev_in:
            y = self.revin_layer(y, 'denorm')
        return ([y], [y]) if return_rep else [y]
