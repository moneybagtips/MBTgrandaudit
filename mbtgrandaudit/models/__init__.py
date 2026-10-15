"""Model registry and ensemble."""
from __future__ import annotations

import json

import numpy as np

from ..config import WEIGHTS_FILE
from .base import MatchContext, Model
from .form import FormModel, H2HModel
from .ratings import Elo, PiRating
from .strength import BivariatePoisson, DixonColes, NegativeBinomial, StrengthModel, XGPoisson


def build_models(settings) -> list[Model]:
    return [
        DixonColes(settings.xi),
        StrengthModel(settings.xi),
        BivariatePoisson(settings.xi),
        NegativeBinomial(settings.xi),
        XGPoisson(settings.xi),
        Elo(settings.xi),
        PiRating(settings.xi),
        FormModel(settings.form_matches),
        H2HModel(),
    ]


def load_weights(settings, league_key: str | None = None) -> dict:
    """Backtest-tuned weights (per league if available), else the defaults."""
    if WEIGHTS_FILE.exists():
        try:
            saved = json.loads(WEIGHTS_FILE.read_text())
            w = saved.get(league_key) if league_key else None
            w = w or saved.get("default")
            if w:
                return _with_market(w, settings)
        except json.JSONDecodeError:
            pass
    return dict(settings.default_weights)


def _with_market(w: dict, settings) -> dict:
    """Weights tuned before the market view existed: give it its default share."""
    if "market_view" in w or "market_view" not in settings.default_weights:
        return dict(w)
    d = settings.default_weights
    share = d["market_view"] / sum(d.values())
    total = sum(w.values())
    out = {k: v / total * (1 - share) for k, v in w.items()}
    out["market_view"] = share
    return out


def ensemble(matrices: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    avail = {k: m for k, m in matrices.items() if m is not None and weights.get(k, 0) > 0}
    if not avail:
        raise ValueError("no model produced a prediction")
    total = sum(weights[k] for k in avail)
    out = sum(weights[k] / total * (m / m.sum()) for k, m in avail.items())
    return out / out.sum()


__all__ = ["MatchContext", "Model", "build_models", "ensemble", "load_weights"]
