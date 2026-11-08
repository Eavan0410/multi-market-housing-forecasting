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

Market embedding:
    transformer_market     joint model plus a learned vector per market (series_id),
                           added to every token, so the model knows which market a
                           window comes from. During training each sample's market is
                           replaced by an "unknown" slot with probability 0.1, so the
                           model also learns a shared fallback for markets it has not seen.

    transformer_per_market fits and scales each market separately.

Inputs are the runner's windows X (samples, lookback, features), one token per
quarter, already standardized inside the training fold. The model predicts all
horizons at once from the last token.

Early stopping uses a time-ordered split inside the training fold: the last
`val_quarters` origins are the inner validation set, and the inner training set
ends `gap_quarters` (or the largest horizon, if longer) earlier so its labels do not
overlap the validation labels.
After the best epoch count is found, the model is retrained from the same start
on all training rows for that many epochs.
"""

from __future__ import annotations

import copy
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from housing_forecast.backtesting.infoset import SampleSet
from housing_forecast.backtesting.scaling import WindowStandardizer
from housing_forecast.models.base import Forecaster


class TransformerNet(nn.Module):
    """Linear embedding + learned positions -> Transformer encoder -> last token -> horizons."""

    def __init__(self, n_features: int, n_horizons: int, lookback: int, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 64, dropout: float = 0.1,
                 n_markets: int = 0):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_horizons)
        # Index 0 = unknown market. Created last so models without it keep the same initialization.
        self.market = nn.Embedding(n_markets + 1, d_model) if n_markets > 0 else None
        if self.market is not None:
            nn.init.normal_(self.market.weight, std=0.02)

    def forward(self, x: torch.Tensor, market: torch.Tensor | None = None) -> torch.Tensor:
        h = self.embed(x) + self.pos[:, -x.shape[1]:]
        if self.market is not None and market is not None:
            h = h + self.market(market)[:, None, :]
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
                 min_rows: int = 8, use_market: bool = False, market_dropout: float = 0.1):
        self.arch = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout)
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.default_epochs = max_epochs, patience, default_epochs
        self.val_quarters, self.gap_quarters, self.min_rows = val_quarters, gap_quarters, min_rows
        self.use_market, self.market_dropout = use_market, market_dropout
        self.market_index_: dict[str, int] = {}
        self.net: TransformerNet | None = None
        self.epochs_used_: list[int] = []

    def _set_target_scale(self, train: SampleSet) -> None:
        # y is small (a few % per year); scale it so the loss is well conditioned.
        has_labels = ~np.isnan(train.y).all(axis=0)
        self.y_mean_ = np.full(train.y.shape[1], np.nan)
        self.y_std_ = np.full(train.y.shape[1], np.nan)
        self.y_mean_[has_labels] = np.nanmean(train.y[:, has_labels], axis=0)
        std = np.nanstd(train.y[:, has_labels], axis=0)
        self.y_std_[has_labels] = np.where(std < 1e-8, 1.0, std)

    def _set_markets(self, train: SampleSet) -> None:
        names = sorted(train.meta["series_id"].unique()) if self.use_market else []
        self.market_index_ = {name: i + 1 for i, name in enumerate(names)}   # 0 = unknown

    def _market_ids(self, s: SampleSet) -> torch.Tensor:
        ids = [self.market_index_.get(name, 0) for name in s.meta["series_id"]]
        return torch.tensor(ids, dtype=torch.long)

    def _tensors(self, s: SampleSet):
        y = (s.y - self.y_mean_) / self.y_std_
        return (torch.tensor(np.nan_to_num(s.X), dtype=torch.float32),
                torch.tensor(np.nan_to_num(y), dtype=torch.float32),
                torch.tensor(~np.isnan(y), dtype=torch.float32),
                self._market_ids(s))

    def _inner_split(self, s: SampleSet):
        o = origin_ordinal(s)
        last = o.max()
        val = o > last - self.val_quarters
        gap = max(self.gap_quarters, max(s.horizons))   # inner labels must not reach the inner validation
        tr = o <= last - self.val_quarters - gap
        if val.sum() < self.min_rows or tr.sum() < self.min_rows:
            return None
        return s.subset(tr), s.subset(val)

    def _new_net(self, s: SampleSet) -> TransformerNet:
        return TransformerNet(n_features=s.X.shape[2], n_horizons=s.y.shape[1],
                              lookback=s.X.shape[1], n_markets=len(self.market_index_), **self.arch)

    def _train(self, net: nn.Module, data, epochs: int, lr: float, gen: torch.Generator,
               val=None) -> int:
        """Train for up to `epochs`. With `val`, stop early and return the best epoch count."""
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        X, y, m, mk = data
        best, best_ep, bad = float("inf"), epochs, 0
        for ep in range(1, epochs + 1):
            net.train()
            perm = torch.randperm(len(X), generator=gen)
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                mk_b = mk[idx]
                if self.market_index_ and self.market_dropout > 0:
                    drop = torch.rand(len(idx), generator=gen) < self.market_dropout
                    mk_b = mk_b.masked_fill(drop, 0)
                loss = masked_mse(net(X[idx], mk_b), y[idx], m[idx])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            if val is not None:
                net.eval()
                with torch.no_grad():
                    v = masked_mse(net(val[0], val[3]), val[1], val[2]).item()
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
        return train.subset(~np.isnan(train.y).all(axis=1))

    def fit(self, train: SampleSet, seed: int) -> None:
        train = self._labeled(train)
        self._set_target_scale(train)
        self._set_markets(train)
        self.net = self._fit_stage(lambda: self._new_net(train), train, self.lr, self.max_epochs, seed)

    def predict(self, test: SampleSet) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(np.nan_to_num(test.X), dtype=torch.float32),
                           self._market_ids(test)).numpy()
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
        self._set_markets(train)
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


class TransformerMarket(TransformerForecaster):
    name = "transformer_market"

    def __init__(self, **kw):
        super().__init__(use_market=True, **kw)


class TransformerPerMarket(Forecaster):
    """Fit one Transformer per market. Use no-change where training labels are unavailable."""

    name = "transformer_per_market"
    stochastic = True
    needs_scaling = False

    DIAG_COLUMNS = ["series_id", "n_train_rows", "fit_status", "reason", "epochs", "epochs_from",
                    "n_forecasts", "n_fallback_forecasts", "fallback_horizons"]

    def __init__(self, min_market_rows: int = 24, **kw):
        self.min_market_rows, self.kw = min_market_rows, kw
        self.models_: dict[str, TransformerForecaster] = {}
        self.scalers_: dict[str, WindowStandardizer] = {}
        self.diag_: dict[str, dict] = {}

    def _diag(self, sid: str, n: int, status: str, reason: str = "") -> dict:
        d = {"n_train_rows": n, "fit_status": status, "reason": reason, "epochs": np.nan,
             "epochs_from": "", "n_forecasts": 0, "n_fallback_forecasts": 0, "fallback_horizons": ""}
        self.diag_[sid] = d
        return d

    def fit(self, train: SampleSet, seed: int) -> None:
        train = TransformerForecaster._labeled(train)
        self.models_, self.scalers_, self.diag_ = {}, {}, {}
        sids = train.meta["series_id"].to_numpy()
        for sid in sorted(set(sids)):
            own = train.subset(sids == sid)
            if len(own) < self.min_market_rows:
                self._diag(sid, len(own), "too_short",
                           f"{len(own)} < {self.min_market_rows} labelled training rows; no-change forecast")
                continue
            scaler = WindowStandardizer().fit(own.X)
            own = own.replace(X=scaler.transform(own.X))
            model = TransformerForecaster(**self.kw)
            inner = model._inner_split(own)
            model.fit(own, seed)
            d = self._diag(sid, len(own), "fitted")
            d["epochs"] = model.epochs_used_[-1]
            d["epochs_from"] = "early_stopping" if inner is not None else "default_epochs"
            self.models_[sid], self.scalers_[sid] = model, scaler

    def diagnostics(self) -> pd.DataFrame:
        """Report fit status and the number of origins with any fallback."""
        return pd.DataFrame([{"series_id": sid, **d} for sid, d in self.diag_.items()],
                            columns=self.DIAG_COLUMNS)

    def predict(self, test: SampleSet) -> np.ndarray:
        out = np.zeros((len(test), len(test.horizons)))
        sids = test.meta["series_id"].to_numpy()
        for sid in sorted(set(sids)):
            rows = sids == sid
            n_rows = int(rows.sum())
            if sid not in self.diag_:
                self._diag(sid, 0, "no_training_history",
                           "market has no labelled training rows in this fold; no-change forecast")
            d = self.diag_[sid]
            d["n_forecasts"] += n_rows
            if sid not in self.models_:
                d["n_fallback_forecasts"] += n_rows
                continue
            part = test.subset(rows)
            model = self.models_[sid]
            pred = model.predict(part.replace(X=self.scalers_[sid].transform(part.X)))
            missing = np.isnan(model.y_mean_)
            if missing.any():
                pred[:, missing] = 0.0
                d["fallback_horizons"] = ",".join(str(h) for h, m in zip(test.horizons, missing) if m)
                d["n_fallback_forecasts"] += int((~np.isnan(part.y[:, missing])).any(axis=1).sum())
            out[rows] = pred
        return out


class SharedTransformerFineTune(Forecaster):
    """Fit a shared model, then adapt independent copies on each market's past rows."""

    name = "transformer_shared_ft"
    stochastic = True
    needs_scaling = True

    def __init__(self, ft_epochs: int = 10, ft_lr_scale: float = 0.1, min_market_rows: int = 24,
                 adapt_markets: list[str] | None = None, **kwargs):
        if ft_epochs < 1 or ft_lr_scale <= 0 or min_market_rows < 1:
            raise ValueError("Fine-tuning epochs, learning-rate scale and minimum rows must be positive.")
        self.ft_epochs, self.ft_lr_scale = ft_epochs, ft_lr_scale
        self.min_market_rows, self.adapt_markets = min_market_rows, adapt_markets
        self.kwargs = kwargs

    def fit(self, train: SampleSet, seed: int) -> None:
        train = TransformerForecaster._labeled(train)
        self.shared_ = TransformerForecaster(**self.kwargs)
        started = time.perf_counter()
        self.shared_.fit(train, seed)
        self.models_, self.missing_, self.diag_ = {}, {}, {}
        self.diag_["__shared__"] = {
            "stage": "shared", "n_train_rows": len(train), "fit_status": "fitted",
            "epochs": self.shared_.epochs_used_[-1], "fit_seconds": time.perf_counter() - started,
            "n_forecasts": 0, "n_fallback_forecasts": 0, "fallback_horizons": "",
        }
        names = sorted(train.meta["series_id"].unique()) if self.adapt_markets is None else sorted(self.adapt_markets)
        for sid in names:
            own = train.subset(train.meta["series_id"].eq(sid).to_numpy())
            self.diag_[sid] = {
                "stage": "adaptation", "n_train_rows": len(own), "fit_status": "too_short",
                "epochs": 0, "fit_seconds": 0.0, "n_forecasts": 0,
                "n_fallback_forecasts": 0, "fallback_horizons": "",
            }
            if len(own) < self.min_market_rows:
                continue
            # Keep the pretrained input and output units during local adaptation.
            local = copy.deepcopy(self.shared_)
            torch.manual_seed(seed)
            started = time.perf_counter()
            local._train(local.net, local._tensors(own), self.ft_epochs,
                         local.lr * self.ft_lr_scale, torch.Generator().manual_seed(seed))
            self.models_[sid] = local
            missing = np.isnan(own.y).all(axis=0)
            self.missing_[sid] = missing
            self.diag_[sid].update({
                "fit_status": "fine_tuned", "epochs": self.ft_epochs,
                "fit_seconds": time.perf_counter() - started,
                "fallback_horizons": ",".join(str(h) for h, absent in zip(own.horizons, missing) if absent),
            })

    def predict_shared(self, test: SampleSet) -> np.ndarray:
        return self.shared_.predict(test)

    def predict(self, test: SampleSet) -> np.ndarray:
        shared = self.predict_shared(test)
        out = shared.copy()
        self.diag_["__shared__"]["n_forecasts"] += len(test)
        for sid in sorted(test.meta["series_id"].unique()):
            rows = test.meta["series_id"].eq(sid).to_numpy()
            n = int(rows.sum())
            if sid not in self.diag_:
                self.diag_[sid] = {
                    "stage": "adaptation", "n_train_rows": 0, "fit_status": "no_adaptation",
                    "epochs": 0, "fit_seconds": 0.0, "n_forecasts": 0,
                    "n_fallback_forecasts": 0, "fallback_horizons": "",
                }
            self.diag_[sid]["n_forecasts"] += n
            if sid not in self.models_:
                self.diag_[sid]["n_fallback_forecasts"] += n
                continue
            local = self.models_[sid].predict(test.subset(rows))
            missing = self.missing_[sid]
            local[:, missing] = shared[rows][:, missing]
            if missing.any():
                self.diag_[sid]["n_fallback_forecasts"] += n
            out[rows] = local
        return out

    def diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame([{"series_id": sid, **row} for sid, row in self.diag_.items()])


class SharedTransformerFineTuneSelect(SharedTransformerFineTune):
    """Shared pretraining plus local fine-tuning, with each market's epoch count chosen inside training.

    Added after the fixed 10-epoch run had been seen. For every fold and seed:

    1. Split the fold's training rows by forecast date with the same rule as the shared model's
       early stopping: the last `val_quarters` origins form an inner validation set, and the inner
       training set ends `max(gap_quarters, largest horizon)` quarters earlier, so none of its labels
       reach the inner validation period. Inner validation labels were already masked to the fold's
       cutoff by the runner.
    2. Refit input scaling on the inner training rows, pretrain a shared model on them, then fine-tune
       a copy on each market's inner training rows and record that market's inner validation loss
       after 0, 1, ..., `ft_max_epochs` epochs. Zero epochs means keeping the shared model.
    3. Pick the epoch count with the lowest loss (ties go to fewer epochs). Markets without enough
       inner training rows or inner validation windows get zero.
    4. Pretrain again on all of the fold's training rows and fine-tune each market for its chosen
       number of epochs. Predict the validation period with that model.
    """

    name = "transformer_shared_ftsel"

    def __init__(self, ft_max_epochs: int = 30, ft_lr_scale: float = 0.1, min_market_rows: int = 24,
                 min_val_windows: int = 4, adapt_markets: list[str] | None = None, **kwargs):
        if ft_max_epochs < 1 or ft_lr_scale <= 0 or min_market_rows < 1 or min_val_windows < 1:
            raise ValueError("Maximum epochs, learning-rate scale and minimum rows must be positive.")
        self.ft_max_epochs, self.ft_lr_scale = ft_max_epochs, ft_lr_scale
        self.min_market_rows, self.min_val_windows = min_market_rows, min_val_windows
        self.adapt_markets, self.kwargs = adapt_markets, kwargs

    def _loss_curve(self, model: TransformerForecaster, own: SampleSet, val: SampleSet, seed: int) -> list[float]:
        """Inner validation loss after 0..ft_max_epochs epochs of the same loop `_train` runs."""
        net = copy.deepcopy(model.net)
        X, y, m, mk = model._tensors(own)
        vX, vy, vm, vmk = model._tensors(val)

        def loss() -> float:
            net.eval()
            with torch.no_grad():
                return masked_mse(net(vX, vmk), vy, vm).item()

        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=model.lr * self.ft_lr_scale, weight_decay=model.weight_decay)
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        curve = [loss()]
        for _ in range(self.ft_max_epochs):
            net.train()
            perm = torch.randperm(len(X), generator=gen)
            for i in range(0, len(X), model.batch_size):
                idx = perm[i:i + model.batch_size]
                batch_loss = masked_mse(net(X[idx], mk[idx]), y[idx], m[idx])
                opt.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            curve.append(loss())
        return curve

    def _select_epochs(self, train: SampleSet, names: list[str], seed: int) -> dict[str, dict]:
        probe = TransformerForecaster(**self.kwargs)
        split = probe._inner_split(train)
        choice = {sid: {"selected_epochs": 0, "select_reason": "no_inner_split", "inner_train_rows": 0,
                        "inner_val_windows": 0, "inner_loss_shared": np.nan, "inner_loss_best": np.nan}
                  for sid in names}
        if split is None:
            return choice
        inner_tr, inner_val = split
        scaler = WindowStandardizer().fit(inner_tr.X)        # inner training rows only
        inner_tr = inner_tr.replace(X=scaler.transform(inner_tr.X))
        inner_val = inner_val.replace(X=scaler.transform(inner_val.X))
        shared = TransformerForecaster(**self.kwargs)
        shared.fit(inner_tr, seed)
        tr_ids, val_ids = inner_tr.meta["series_id"].to_numpy(), inner_val.meta["series_id"].to_numpy()
        for sid in names:
            own = inner_tr.subset(tr_ids == sid)
            val = inner_val.subset(val_ids == sid)
            usable = ~np.isnan((val.y - shared.y_mean_) / shared.y_std_)
            val = val.subset(usable.any(axis=1))
            c = choice[sid]
            c["inner_train_rows"], c["inner_val_windows"] = len(own), len(val)
            if len(own) < self.min_market_rows:
                c["select_reason"] = "inner_train_too_short"
            elif len(val) < self.min_val_windows:
                c["select_reason"] = "inner_validation_too_short"
            else:
                curve = self._loss_curve(shared, own, val, seed)
                best = int(np.argmin(curve))                  # first minimum, so ties keep fewer epochs
                c.update(selected_epochs=best, select_reason="selected",
                         inner_loss_shared=curve[0], inner_loss_best=curve[best])
        return choice

    def fit(self, train: SampleSet, seed: int) -> None:
        train = TransformerForecaster._labeled(train)
        names = sorted(train.meta["series_id"].unique()) if self.adapt_markets is None else sorted(self.adapt_markets)
        started = time.perf_counter()
        choice = self._select_epochs(train, names, seed)
        select_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.shared_ = TransformerForecaster(**self.kwargs)
        self.shared_.fit(train, seed)
        self.models_, self.missing_, self.diag_ = {}, {}, {}
        self.diag_["__shared__"] = {
            "stage": "shared", "n_train_rows": len(train), "fit_status": "fitted",
            "epochs": self.shared_.epochs_used_[-1], "fit_seconds": time.perf_counter() - started,
            "select_seconds": select_seconds, "n_forecasts": 0, "n_fallback_forecasts": 0, "fallback_horizons": "",
        }
        ids = train.meta["series_id"].to_numpy()
        for sid in names:
            own = train.subset(ids == sid)
            epochs = choice[sid]["selected_epochs"]
            d = {"stage": "adaptation", "n_train_rows": len(own), "fit_status": "kept_shared",
                 "epochs": 0, "fit_seconds": 0.0, "n_forecasts": 0, "n_fallback_forecasts": 0,
                 "fallback_horizons": "", **choice[sid]}
            self.diag_[sid] = d
            if epochs == 0:
                continue
            if len(own) < self.min_market_rows:
                d["fit_status"] = "too_short"
                continue
            local = copy.deepcopy(self.shared_)
            torch.manual_seed(seed)
            started = time.perf_counter()
            local._train(local.net, local._tensors(own), epochs, local.lr * self.ft_lr_scale,
                         torch.Generator().manual_seed(seed))
            self.models_[sid] = local
            missing = np.isnan(own.y).all(axis=0)
            self.missing_[sid] = missing
            d.update(fit_status="fine_tuned", epochs=epochs, fit_seconds=time.perf_counter() - started,
                     fallback_horizons=",".join(str(h) for h, absent in zip(own.horizons, missing) if absent))
