"""Wanda用の入力統計。ミニバッチ分割によらない統計を蓄積する。"""
import torch
import torch.nn as nn


class WrappedLayer:
    def __init__(self, layer, layer_id=0, layer_name='none', p_norm=2):
        if p_norm not in (1, 2):
            raise ValueError('p_norm must be 1 or 2')
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.scaler_row = torch.zeros(self.columns, device=self.dev, dtype=torch.float32)
        self.input_mean = torch.zeros_like(self.scaler_row)
        self.nsamples = 0
        self.layer_id, self.layer_name, self.p_norm = layer_id, layer_name, p_norm
        # 修正: import時にTF32等のグローバル設定を上書きしない。

    @torch.no_grad()
    def add_batch(self, inp, out):
        if inp.shape[-1] != self.columns:
            raise ValueError('Input width does not match the layer')
        inp = inp.detach().reshape(-1, self.columns).float()
        count = len(inp)
        if not count:
            return
        total = self.nsamples + count
        # 修正: バッチ平均の単純加算をtoken数で重み付けした平均に変更。
        moment = inp.square().sum(0) if self.p_norm == 2 else inp.abs().sum(0)
        self.scaler_row.mul_(self.nsamples / total).add_(moment / total)
        self.input_mean.mul_(self.nsamples / total).add_(inp.sum(0) / total)
        self.nsamples = total
        if not torch.isfinite(self.scaler_row).all():
            raise FloatingPointError('Non-finite activation statistics')

    @torch.no_grad()
    def prune(self, W_mask):
        if not self.nsamples:
            raise ValueError('Call add_batch before bias-corrected pruning')
        # 修正: 未定義のinp1/out1ではなく入力平均から線形層の平均誤差を補正。
        correction = (self.layer.weight.float() * W_mask) @ self.input_mean
        self.layer.weight.masked_fill_(W_mask, 0)
        if self.layer.bias is None:
            self.layer.bias = nn.Parameter(correction.to(self.layer.weight.dtype))
        else:
            self.layer.bias.add_(correction.to(self.layer.bias.dtype))

    def free(self):
        # 修正: 未定義のDEBUG参照を削除。global CUDA cacheは毎回解放しない。
        self.scaler_row = None
        self.input_mean = None
