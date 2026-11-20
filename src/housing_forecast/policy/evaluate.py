"""Check extracted policies against manual document labels and event timelines.

Report relevance, policy type, direction and event matching within a date window."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPOT_COLUMNS = ["human_relevant", "human_policy_type", "human_direction"]


def quarter_agreement(llm_index: pd.DataFrame, manual_index: pd.DataFrame, col: str = "policy_flow",
                      atol: float = 0.05) -> dict:
    m = llm_index[["quarter", col]].merge(manual_index[["quarter", col]], on="quarter", suffixes=("_llm", "_manual"))
    a, b = m[f"{col}_llm"].to_numpy(), m[f"{col}_manual"].to_numpy()
    a, b = np.where(np.abs(a) < atol, 0.0, a), np.where(np.abs(b) < atol, 0.0, b)   # ignore decayed tails
    active = (a != 0) | (b != 0)
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "series": col,
        "n_quarters": int(len(m)),
        "n_active_quarters": int(active.sum()),
        "pearson": corr(a, b),
        "spearman": corr(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()),
        "sign_agreement_active": float((np.sign(a[active]) == np.sign(b[active])).mean()) if active.any() else float("nan"),
    }


def event_match(llm_events: pd.DataFrame, manual_events: pd.DataFrame, window_days: int = 14) -> dict:
    llm = llm_events[llm_events["relevant"].astype(bool) & (llm_events["direction"] != 0)].copy()
    llm["date"] = pd.to_datetime(llm["date"])
    man = manual_events[manual_events["direction"] != 0].copy()
    man["date"] = pd.to_datetime(man["date"])
    win = pd.Timedelta(days=window_days)

    def near(row, pool):
        same = pool[np.sign(pool["direction"]) == np.sign(row["direction"])]
        return bool(((same["date"] - row["date"]).abs() <= win).any())

    found = man.apply(lambda r: near(r, llm), axis=1) if len(man) else pd.Series(dtype=bool)
    confirmed = llm.apply(lambda r: near(r, man), axis=1) if len(llm) else pd.Series(dtype=bool)
    return {
        "window_days": window_days,
        "n_manual_events": int(len(man)),
        "n_llm_signals": int(len(llm)),
        "recall": float(found.mean()) if len(found) else float("nan"),
        "precision": float(confirmed.mean()) if len(confirmed) else float("nan"),
        "missed_event_ids": man.loc[~found.astype(bool), "event_id"].tolist() if len(man) else [],
    }


def spotcheck_sample(ex: pd.DataFrame, docs: pd.DataFrame, n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Random sample for hand labeling: half from documents the model called relevant, half from the rest."""
    ok = ex[ex["parse_ok"]].merge(docs[["doc_id", "title", "text"]], on="doc_id")
    rel, irr = ok[ok["relevant"]], ok[~ok["relevant"]]
    k_irr = min(len(irr), n - min(len(rel), n // 2))
    k_rel = min(len(rel), n - k_irr)                               # top up from the other group if one is short
    sample = pd.concat([rel.sample(k_rel, random_state=seed), irr.sample(k_irr, random_state=seed)])
    sample = sample.sample(frac=1, random_state=seed)              # shuffle so labelers can't tell
    out = sample[["doc_id", "published_date", "title"]].assign(text_excerpt=sample["text"].str[:500])
    out = out.assign(llm_relevant=sample["relevant"], llm_policy_type=sample["policy_type"],
                     llm_direction=sample["direction"], llm_evidence=sample["evidence"])
    for c in SPOT_COLUMNS:
        out[c] = ""
    return out.reset_index(drop=True)


def score_spotcheck(sheet: pd.DataFrame) -> dict:
    s = sheet.dropna(subset=["human_relevant"]).copy()
    s = s[s["human_relevant"].astype(str).str.strip() != ""]
    truthy = lambda x: x.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
    s["human_relevant"], s["llm_relevant"] = truthy(s["human_relevant"]), truthy(s["llm_relevant"])
    both = s[s["human_relevant"] & s["llm_relevant"]].dropna(subset=["human_direction"])
    hd = both["human_direction"].astype(int)
    ld = both["llm_direction"].astype(int)
    tp = int((s["human_relevant"] & s["llm_relevant"]).sum())
    return {
        "n_labeled": int(len(s)),
        "relevance_accuracy": float((s["human_relevant"] == s["llm_relevant"]).mean()) if len(s) else float("nan"),
        "relevance_precision": tp / max(int(s["llm_relevant"].sum()), 1),
        "relevance_recall": tp / max(int(s["human_relevant"].sum()), 1),
        "n_both_relevant": int(len(both)),
        "direction_sign_accuracy": float((np.sign(hd) == np.sign(ld)).mean()) if len(both) else float("nan"),
        "direction_exact_accuracy": float((hd == ld).mean()) if len(both) else float("nan"),
        "policy_type_accuracy": float((both["human_policy_type"] == both["llm_policy_type"]).mean()) if len(both) else float("nan"),
    }
