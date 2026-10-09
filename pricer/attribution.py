"""Attribution du P&L d'un livre couvert, pas par pas.

Sur [t_i, t_{i+1}], la variation de la valeur (marquée au modèle) du livre
    B = position * V + h * S + cash
se décompose, via un développement de Taylor de V (formule d'Itô discrète) :

    dB ≈ (h + position*Δ) dS          -> delta   : erreur de couverture au 1er ordre
       + position * ½ Γ dS²            -> gamma   : convexité
       + position * Θ dt               -> theta   : passage du temps
       + cash * (e^{r dt} - 1)         -> carry   : intérêts sur le cash
       - coûts                         -> costs
       + reste                         -> residual (ordres supérieurs)

Pour une couverture en delta continue, l'équation de Black–Scholes donne
    gamma + theta + carry ≈ position * ½ Γ S² ( (dS/S)² - σ_mark² dt ),
d'où la formule clé (vendeur, position = -1) :
    P&L ≈ ½ ∫ Γ S² (σ_impl² - σ_réal²) dt.
La colonne `vol` est ce terme théorique, à comparer à gamma + theta + carry.
"""

from __future__ import annotations

import numpy as np

from . import black_scholes as bs
from .hedging import HedgeResult

COMPONENTS = ("delta", "gamma", "theta", "carry", "costs", "residual")


def attribute(res: HedgeResult) -> dict[str, np.ndarray]:
    """Renvoie un dict {composante: tableau (n_paths, n_steps)} des P&L par pas,
    plus 'total' (variation exacte du livre) et 'vol' (terme théorique)."""
    S, t = res.S, res.t
    o, pos, r, sig = res.option, res.position, res.r, res.sigma_mark
    dt = np.diff(t)[None, :]
    tau = (o.T - t)[None, :-1]
    Si, dS = S[:, :-1], np.diff(S, axis=1)

    D = bs.delta(Si, o.K, tau, r, sig, o.option)
    G = bs.gamma(Si, o.K, tau, r, sig)
    Th = bs.theta(Si, o.K, tau, r, sig, o.option)

    h = res.holdings[:, :-1]
    cash = res.cash[:, :-1]

    out = {
        "delta": (h + pos * D) * dS,
        "gamma": pos * 0.5 * G * dS**2,
        "theta": pos * Th * dt,
        "carry": cash * (np.exp(r * dt) - 1.0),
        "costs": -res.costs[:, 1:],
    }
    total = np.diff(res.book_value, axis=1)
    out["residual"] = total - sum(out[k] for k in COMPONENTS[:-1])
    out["total"] = total
    out["vol"] = pos * 0.5 * G * Si**2 * ((dS / Si) ** 2 - sig**2 * dt)
    return out


def summarize(attr: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Moyenne et écart-type (sur les trajectoires) de chaque composante cumulée."""
    out = {}
    for k, v in attr.items():
        cum = v.sum(axis=1)
        out[k] = (float(cum.mean()), float(cum.std(ddof=1)))
    return out
