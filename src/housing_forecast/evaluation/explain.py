"""Calculate input gradients and Integrated Gradients, with lag 1 most recent.

Use the training mean as the attribution baseline. Rescale gradients to raw input
units before comparing with AR coefficients. Gradients and attribution shares
measure different aspects of input influence."""

from __future__ import annotations

import numpy as np

from housing_forecast.backtesting.scaling import WindowStandardizer
from housing_forecast.models.baselines import ARDirect


def ar_lag_weights(model: ARDirect, n_lags: int) -> np.ndarray:
    """Pooled AR coefficients as (horizons, n_lags), newest lag first; lags beyond p are 0."""
    B = model.coefs_["__pooled__"]                        # (1 + p, H), row 0 = intercept
    W = np.zeros((B.shape[1], n_lags))
    W[:, : model.p] = B[1:].T
    return W


def ar_attributions(model: ARDirect, X_raw: np.ndarray, feature: int, baseline: float) -> np.ndarray:
    """(n, H, L) contributions: coefficient x (input - baseline), lag 1 = most recent quarter."""
    L = X_raw.shape[1]
    W = ar_lag_weights(model, L)
    lags = X_raw[:, ::-1, feature] - baseline             # (n, L), newest first
    return lags[:, None, :] * W[None, :, :]


def transformer_attributions(model, scaler: WindowStandardizer, X_raw: np.ndarray, feature: int = 0,
                             steps: int = 32, chunk: int = 8192) -> dict:
    """Local gradients and Integrated Gradients of a fitted TransformerForecaster.

    Returns arrays with lag 1 = most recent quarter:
      grad (n, H, L)  d forecast / d raw input
      ig   (n, H, L)  contribution of each input quarter (forecast units)
      pred (n, H)     forecast at the actual input (equals model.predict)
      pred_baseline (H,)  forecast when every input quarter equals the training mean
      completeness_error  max |sum(ig) - (pred - pred_baseline)|
    The baseline is the training mean, i.e. 0 after scaling. The integral uses the midpoint
    rule with `steps` points.
    """
    import torch

    net = model.net
    net.eval()
    Xs = np.nan_to_num(scaler.transform(X_raw)).astype(np.float32)
    n, L, F = Xs.shape
    y_std = np.asarray(model.y_std_, dtype=np.float64)
    y_mean = np.asarray(model.y_mean_, dtype=np.float64)
    H = len(y_std)
    to_raw = y_std[:, None] / scaler.std_[feature]        # d forecast / d raw input = d out / d scaled * this

    def grads(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Network output (scaled) and d out_h / d x[..., feature] for every horizon."""
        out_all = np.empty((len(x), H))
        g_all = np.empty((len(x), H, L))
        for s in range(0, len(x), chunk):
            xt = torch.tensor(x[s:s + chunk], requires_grad=True)
            out = net(xt)
            out_all[s:s + chunk] = out.detach().numpy()
            for h in range(H):
                g, = torch.autograd.grad(out[:, h].sum(), xt, retain_graph=h < H - 1)
                g_all[s:s + chunk, h] = g[:, :, feature].numpy()
        return out_all, g_all

    out, g = grads(Xs)
    alphas = (np.arange(steps) + 0.5) / steps
    path = (alphas[:, None, None, None] * Xs[None]).reshape(steps * n, L, F).astype(np.float32)  # baseline 0
    _, g_path = grads(path)
    avg_g = g_path.reshape(steps, n, H, L).mean(axis=0)
    ig = Xs[:, None, :, feature] * avg_g * y_std[None, :, None]
    out_base, _ = grads(np.zeros((1, L, F), dtype=np.float32))

    pred = out * y_std + y_mean
    pred_base = out_base[0] * y_std + y_mean
    err = np.nanmax(np.abs(ig.sum(axis=2) - (pred - pred_base[None, :])))
    rev = slice(None, None, -1)                                           # window order -> lag order
    return {"grad": (g * to_raw[None])[:, :, rev], "ig": ig[:, :, rev], "pred": pred,
            "pred_baseline": pred_base, "completeness_error": float(err)}
