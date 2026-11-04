"""Small Transformer forecasters for the three core experiments.

Same Forecaster interface as the baselines. The training pool (--pool) decides
what the model sees, so one class covers two experiments:

    --models transformer     --pool target_only           Taiwan-only
    --models transformer     --pool overseas_plus_target  joint training
    --models transformer_ft  --pool overseas_plus_target  pretrain on overseas, fine-tune on Taiwan
    --models transformer     --pool overseas              zero-shot transfer (optional)

Fine-tuning variants (same pool as transformer_ft, one setting changed each):
    transformer_ft_head    freeze embedding + encoder, fine-tune only the output head
    transformer_ft_short   at most 20 fine-tuning epochs instead of 100
    transformer_ft_lowlr   fine-tuning learning rate 1/100 of pretraining instead of 1/10

Inputs are the runner's windows X (samples, lookback, features), one token per
quarter, already standardized inside the training fold. The model predicts all
horizons at once from the last token.

Early stopping uses a time-ordered split inside the training fold: the last
`val_quarters` origins are the inner validation set, and the inner training set
ends `gap_quarters` earlier so its labels do not overlap the validation labels.
After the best epoch count is found, the model is retrained from the same start
on all training rows for that many epochs.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from housing_forecast.backtesting.infoset import SampleSet
from housing_forecast.models.base import Forecaster


class TransformerNet(nn.Module):
    """Linear embedding + learned positions -> Transformer encoder -> last token -> horizons."""

    def __init__(self, n_features: int, n_horizons: int, lookback: int, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + self.pos[:, -x.shape[1]:]
        h = self.encoder(h)
        return self.head(self.norm(h[:, -1]))


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over labels that exist; hidden (unpublished) labels have mask 0."""
    return ((pred - y) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def origin_ordinal(s: SampleSet) -> np.ndarray:
    return pd.PeriodIndex(s.meta["origin"], freq="Q").asi8


class TransformerForecaster(Forecaster):
    name = "transformer"
    stochastic = True
    needs_scaling = True

    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 2, d_ff: int = 64,
                 dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 64, max_epochs: int = 200, patience: int = 20,
                 default_epochs: int = 50, val_quarters: int = 8, gap_quarters: int = 4,
                 min_rows: int = 8):
        self.arch = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout)
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.default_epochs = max_epochs, patience, default_epochs
        self.val_quarters, self.gap_quarters, self.min_rows = val_quarters, gap_quarters, min_rows
        self.net: TransformerNet | None = None
        self.epochs_used_: list[int] = []        # best epoch count of each stage, for checking

    # ---- data -------------------------------------------------------------
    def _set_target_scale(self, train: SampleSet) -> None:
        # y is small (a few % per year); scale it so the loss is well conditioned.
        self.y_mean_ = np.nanmean(train.y, axis=0)
        std = np.nanstd(train.y, axis=0)
        self.y_std_ = np.where(std < 1e-8, 1.0, std)

    def _tensors(self, s: SampleSet):
        y = (s.y - self.y_mean_) / self.y_std_
        return (torch.tensor(np.nan_to_num(s.X), dtype=torch.float32),
                torch.tensor(np.nan_to_num(y), dtype=torch.float32),
                torch.tensor(~np.isnan(y), dtype=torch.float32))

    def _inner_split(self, s: SampleSet):
        o = origin_ordinal(s)
        last = o.max()
        val = o > last - self.val_quarters
        tr = o <= last - self.val_quarters - self.gap_quarters
        if val.sum() < self.min_rows or tr.sum() < self.min_rows:
            return None
        return s.subset(tr), s.subset(val)

    def _new_net(self, s: SampleSet) -> TransformerNet:
        return TransformerNet(n_features=s.X.shape[2], n_horizons=s.y.shape[1],
                              lookback=s.X.shape[1], **self.arch)

    # ---- training ---------------------------------------------------------
    def _train(self, net: nn.Module, data, epochs: int, lr: float, gen: torch.Generator,
               val=None) -> int:
        """Train for up to `epochs`. With `val`, stop early and return the best epoch count."""
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        X, y, m = data
        best, best_ep, bad = float("inf"), epochs, 0
        for ep in range(1, epochs + 1):
            net.train()
            perm = torch.randperm(len(X), generator=gen)
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                loss = masked_mse(net(X[idx]), y[idx], m[idx])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            if val is not None:
                net.eval()
                with torch.no_grad():
                    v = masked_mse(net(val[0]), val[1], val[2]).item()
                if v < best - 1e-6:
                    best, best_ep, bad = v, ep, 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        return best_ep

    def _fit_stage(self, make_net: Callable[[], nn.Module], s: SampleSet, lr: float,
                   max_epochs: int, seed: int) -> nn.Module:
        """Pick the epoch count on the inner time split, then retrain on all rows of `s`."""
        split = self._inner_split(s)
        if split is None:
            epochs = self.default_epochs
        else:
            torch.manual_seed(seed)
            epochs = self._train(make_net(), self._tensors(split[0]), max_epochs, lr,
                                 torch.Generator().manual_seed(seed), val=self._tensors(split[1]))
        self.epochs_used_.append(epochs)
        torch.manual_seed(seed)
        net = make_net()
        self._train(net, self._tensors(s), epochs, lr, torch.Generator().manual_seed(seed))
        return net

    @staticmethod
    def _labeled(train: SampleSet) -> SampleSet:
        return train.subset(~np.isnan(train.y).all(axis=1))   # rows whose labels are all hidden teach nothing

    def fit(self, train: SampleSet, seed: int) -> None:
        train = self._labeled(train)
        self._set_target_scale(train)
        self.net = self._fit_stage(lambda: self._new_net(train), train, self.lr, self.max_epochs, seed)

    def predict(self, test: SampleSet) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(np.nan_to_num(test.X), dtype=torch.float32)).numpy()
        return out * self.y_std_ + self.y_mean_


class TransformerFineTune(TransformerForecaster):
    """Pretrain on the overseas rows of the pool, then fine-tune on the home-country rows."""

    name = "transformer_ft"

    def __init__(self, ft_lr_scale: float = 0.1, ft_max_epochs: int = 100,
                 ft_trainable: str = "all", home_country: str = "Taiwan", **kw):
        super().__init__(**kw)
        if ft_trainable not in ("all", "head"):
            raise ValueError("ft_trainable must be 'all' or 'head'")
        self.ft_lr_scale, self.ft_max_epochs, self.home_country = ft_lr_scale, ft_max_epochs, home_country
        self.ft_trainable = ft_trainable

    def _start_finetune(self, pre: TransformerNet) -> TransformerNet:
        """Fresh copy of the pretrained net; with 'head', only the final norm + head can change."""
        net = copy.deepcopy(pre)
        if self.ft_trainable == "head":
            for p in net.parameters():
                p.requires_grad_(False)
            for p in list(net.norm.parameters()) + list(net.head.parameters()):
                p.requires_grad_(True)
        return net

    def fit(self, train: SampleSet, seed: int) -> None:
        train = self._labeled(train)
        self._set_target_scale(train)
        home = (train.meta["country"] == self.home_country).to_numpy()
        src, tgt = train.subset(~home), train.subset(home)
        if len(src) == 0:
            raise ValueError("transformer_ft needs overseas rows: use --pool overseas_plus_target")
        pre = self._fit_stage(lambda: self._new_net(src), src, self.lr, self.max_epochs, seed)
        if len(tgt) == 0:           # --pool overseas: no fine-tuning possible
            self.net = pre
            return
        self.net = self._fit_stage(lambda: self._start_finetune(pre), tgt, self.lr * self.ft_lr_scale,
                                   self.ft_max_epochs, seed)


class TransformerFTHead(TransformerFineTune):
    name = "transformer_ft_head"

    def __init__(self, **kw):
        super().__init__(ft_trainable="head", **kw)


class TransformerFTShort(TransformerFineTune):
    name = "transformer_ft_short"

    def __init__(self, **kw):
        kw.setdefault("ft_max_epochs", 20)
        super().__init__(**kw)


class TransformerFTLowLR(TransformerFineTune):
    name = "transformer_ft_lowlr"

    def __init__(self, **kw):
        super().__init__(ft_lr_scale=0.01, **kw)
