__all__ = ['GoodModel']

# from ast import Return
import torch
import torch.nn as nn
import torch.nn.init as init
import copy
import torch.nn.functional as F
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs, d_inputs):
        """

        """
        super(Model, self).__init__()
        self.rev_in = configs.revin
        self.anchor_in = getattr(configs, 'anchor_in', 'none')
        self.pred_len = configs.pred_len
        self.L = configs.level_dim
        if self.rev_in:
            self.revin_layer = RevIN(d_inputs[-1], affine=False, subtract_last=False)

        dropout = configs.dropout
        d_ff = configs.d_ff
        n_c = configs.d_layers
        n_b = configs.n_block
        pe = configs.pe

        self.decoder = Decoder(d_inputs, self.pred_len,
                                  n_c, n_b=n_b, d_ff=d_ff, dropout=dropout, pe=pe, norm=configs.norm_name)
            
        # self.apply(self._init_weights)

    # # Add to Model.__init__:
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             torch.nn.init.zeros_(m.bias)
    #     elif isinstance(m, nn.Parameter):
    #         torch.nn.init.xavier_uniform_(m.data)

    def forward(self, x, return_rep=False):
        # B, T, V = x.shape  # (B, T, V)
        # Y_tilde = None
        if self.rev_in:
            x = self.revin_layer(x, 'norm')
            if self.anchor_in == 'last':   # input centred on the last value; output still de-normalised on the mean
                x = x - x[:, -1:, :]          # beta's RevIN has no affine: plain shift by the normalised last value
        result = self.decoder(x, return_rep=return_rep) # (B, L, V, T)  # fix: was passed positionally into x_mark
        if return_rep:
            ys, reps = result
        else:
            ys = result
        if self.rev_in:
            ys = [self.revin_layer(y, 'denorm') for y in ys]
        if return_rep:
            return ys, reps
        else:
            return ys

class Decoder(nn.Module):
    def __init__(self, d_inputs, d_pred, n_c=3, n_b=1, d_ff=256, dropout=0., pe=False, exv_use=0, norm='instance'):
        super(Decoder, self).__init__()
        self.receiver = DecoderBlock(d_ff, d_inputs[0], d_pred, n_c, n_b=n_b, drop_ff=dropout, norm=norm)  # fix: official Model passes n_b / norm that Decoder did not accept
        self.d_inputs = d_inputs
        self.pred_len = d_pred
        self.pe = pe
        W_pos = torch.empty((d_inputs[0], d_inputs[-1]))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.embedding = nn.Parameter(W_pos)
        self.exv_use = exv_use
        if exv_use:
            self.fea_proj = nn.Sequential(
                                            nn.Linear(4, 128, bias=False),
                                            nn.GELU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(128, d_inputs[-1]),
                                        )

    def forward(self, x, x_mark=None, return_rep=False):
        if self.pe:
            x = x + self.embedding
        if x_mark is not None and self.exv_use:
            x = x + self.fea_proj(x_mark)
        result = self.receiver(x.transpose(1, 2), return_rep)
        if return_rep:
            ys, reps = result
        else:
            ys = result
        ys = [y.transpose(1, 2) for y in ys]
        if return_rep:
            return ys, reps
        else:
            return ys

class ForeBlock(nn.Module):
    def __init__(self, d_in, d_out, d_ff, drop_ff=0.):
        super(ForeBlock, self).__init__()
        self.bottom = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(d_in, d_ff),  # proj from data space into representation space
        )
        self.MLP = nn.Sequential(   
                                    nn.Linear(d_ff, d_ff, bias=False),
                                    nn.GELU(),
                                    nn.Dropout(drop_ff),
                                 )
        self.head =  nn.Linear(d_ff, d_out)

    def forward(self, input, rep_prev=None):
        # breakpoint()
        rep = self.bottom(input)
        mid = self.MLP(rep)
        output = self.head(mid)

        return output, mid
        
class Activation(nn.Module):
    def __init__(self, act='nonlinear') -> None:
        super().__init__()
        self.act = nn.GELU() if act == 'nonlinear' else nn.Identity()

    def forward(self, x):
        return self.act(x)

class InstanceNorm1dWrapper(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features)
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, seq_len, variates)
        x = self.norm(x)
        x = x.transpose(1, 2)  # Back to (batch_size, variates, seq_len)
        return x

class MLP(nn.Module):
    def __init__(self, d_ff, drop_ff=0.):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d_ff, d_ff),
            # nn.GELU(),
            nn.Dropout(drop_ff), 
        )

    def forward(self, x):
        y = self.block(x)
        return y

class MidBlock(nn.Module):
    def __init__(self, d_in, d_out, d_ff, n_b=1, drop_ff=0., drop_in=0.1, act='nonlinear', norm='instance'):
        super(MidBlock, self).__init__()
        self.bottom = nn.Sequential(
            InstanceNorm1dWrapper(d_in) if norm.lower() == 'instance' else nn.LayerNorm(d_in),
            nn.Dropout(drop_in),
            nn.Linear(d_in, d_ff),
        )
        # Convert MLP sequential to ModuleList
        self.body = nn.ModuleList([MLP(d_ff, drop_ff) for _ in range(n_b)])
        self.activation = Activation(act)
        # self.activation = nn.GLU()
        self.head = nn.Linear(d_ff, d_out)
        
        self.drop_skip = nn.Dropout(drop_ff)    
        self.drop_rep_pre = nn.Dropout(drop_ff)
        self.drop_skip2 = nn.Dropout(drop_ff)

    def forward(self, input, rep_prev=None):
        # project to representation space
        rep = x = self.bottom(input) + self.drop_rep_pre(rep_prev)
        # high-level transforms
        for i in range(len(self.body)):
            rep = self.body[i](rep)

        mid = self.activation(rep) + self.drop_skip(x) + self.drop_skip2(rep_prev)
        # project to real space
        output = self.head(mid)

        return output, mid

class DecoderBlock(nn.Module):
    def __init__(self, d_ff, d_in, d_out, n_c=3, n_b=1, drop_ff=None, drop_in=0.1, norm='instance'):
        super(DecoderBlock, self).__init__()
        self.pred_base = d_out // n_c 
        self.cascade = nn.ModuleList([ForeBlock(d_in, self.pred_base, d_ff, drop_ff)])
        self.n_c = n_c
        self.fb_scale, self.fb_shuffle, self.fb_detach = 1.0, False, False  # official: not detached
        self.fb_override, self.fb_capture, self.last_out = None, False, None
        self.fb_train_mode, self.fb_train_sigma, self.fb_train_active = 'none', 0.3, False

        for i in range(1, n_c):
            n_acc = i * (i + 1) // 2
            d_cat = d_in + self.pred_base * n_acc
            act='nonlinear'
            # drop_in = min(drop_in * (i + 1), 0.6)
            drop_in_layer = max(drop_in * (0.8 ** i), 0.05)
            # drop_in_layer = 0.
            # drop_in_layer = min(drop_in * (i + 1), 0.6)
            self.cascade.append(MidBlock(d_cat, self.pred_base * (i + 1), d_ff, n_b, drop_ff, drop_in_layer, act, norm))

    def _feedback(self, out):
        """Earlier-stage forecasts fed to the next stage (ablation switches; defaults = official)."""
        if self.fb_override is not None:
            # --fb_source ema: earlier-stage forecasts of the EMA model (no grad). Exp_Main sets this only
            # around the training forward pass (NOT conditioned on self.training: official ReNF trains in
            # eval mode from epoch 2 on, because vali() calls model.eval() and never switches back)
            y_cat = torch.cat(self.fb_override[:len(out)], dim=-1)
        else:
            y_cat = torch.cat(out, dim=-1)
            if self.fb_detach:
                y_cat = y_cat.detach()
        if self.fb_train_active and self.fb_train_mode != 'none':
            # --fb_train: perturb the feedback during the TRAINING forward pass only (switched on by Exp_Main,
            # not via self.training, because official ReNF trains in eval mode from epoch 2 on)
            if self.fb_train_mode == 'shuffle' and y_cat.shape[0] > 1:
                y_cat = torch.roll(y_cat, shifts=y_cat.shape[0] // 2, dims=0)   # train batches are shuffled
            elif self.fb_train_mode == 'gauss':
                y_cat = y_cat + self.fb_train_sigma * y_cat.detach().std() * torch.randn_like(y_cat)
        if self.fb_shuffle and not self.training and y_cat.shape[0] > 1:
            # test-time probe: give each sample the feedback of another sample in the batch
            y_cat = torch.roll(y_cat, shifts=y_cat.shape[0] // 2, dims=0)
        return y_cat * self.fb_scale

    def forward(self, x0, return_rep=False):
        out = []
        reps = []
        x_k = x0
        rep = None
        for k, layer in enumerate(self.cascade):
            y_k, rep = layer(x_k, rep_prev=rep)
            # TODO: save the representation for visualization
            if return_rep:
                reps.append(rep)
            out.append(y_k)
            x_k = torch.cat([x0, self._feedback(out)], dim=-1)  # official: cat(out) * 1.0
        if self.fb_capture:
            self.last_out = [o.detach() for o in out]
        if return_rep:
            return out, reps
        else:
            return out

    # @torch.no_grad()
    # def predict(self, x):
    #     return self.forward(x)
