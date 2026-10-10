"""Calibration : estimer les paramètres des modèles à partir de données réelles.

On estime sur l'historique (monde réel, probabilité P). Pour le PRICING, on garde
les paramètres de VOLATILITÉ (sigma, sauts) et on remplace le drift par celui de
la probabilité risque-neutre (0 pour un future). Girsanov dit justement que le
changement de probabilité modifie le drift, pas la volatilité.

Toutes les fonctions prennent des rendements log quotidiens (ou des log-prix) et
dt = 1/252 (un jour de bourse en années).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT = 1 / 252


@dataclass
class GBMParams:
    mu: float
    sigma: float


@dataclass
class MertonParams:
    mu: float
    sigma: float       # volatilité de la partie continue (hors sauts)
    lam: float         # nombre moyen de sauts par an
    jump_mean: float   # taille moyenne d'un saut (en log)
    jump_std: float    # dispersion des sauts (en log)

    @property
    def total_vol(self) -> float:
        """Volatilité totale : sqrt(sigma² + lam (jump_mean² + jump_std²))."""
        return float(np.sqrt(self.sigma**2 + self.lam * (self.jump_mean**2 + self.jump_std**2)))


@dataclass
class SchwartzParams:
    kappa: float       # force de rappel (par an)
    theta: float       # niveau de long terme du log-prix
    sigma: float

    @property
    def half_life_years(self) -> float:
        return float(np.log(2) / self.kappa)

    @property
    def long_run_price(self) -> float:
        return float(np.exp(self.theta))


def fit_gbm(returns, dt: float = DT) -> GBMParams:
    """MBG : les rendements log sont i.i.d. N((mu - sigma²/2) dt, sigma² dt).
    On en déduit sigma par l'écart-type et mu par la moyenne."""
    r = np.asarray(returns, float)
    sigma = r.std(ddof=1) / np.sqrt(dt)
    mu = r.mean() / dt + 0.5 * sigma**2
    return GBMParams(float(mu), float(sigma))


def fit_merton(returns, dt: float = DT, k: float = 4.0) -> MertonParams:
    """Merton par SEUILLAGE (méthode simple et transparente) :

    1. on repère les sauts : rendements à plus de k écarts-types ROBUSTES de la médiane ;
    2. la partie continue (sigma) est estimée sur les jours SANS saut ;
    3. lam = nombre de sauts / durée en années ;
    4. jump_mean, jump_std = moyenne et écart-type des rendements des jours de saut.

    Limite : un petit saut ne se distingue pas d'un gros jour « normal ». La méthode
    sous-estime donc un peu lam, mais elle capture l'essentiel : les queues épaisses.
    """
    r = np.asarray(returns, float)
    med = np.median(r)
    robust_sd = 1.4826 * np.median(np.abs(r - med))
    is_jump = np.abs(r - med) > k * robust_sd
    normal = r[~is_jump]
    jumps = r[is_jump]
    sigma = normal.std(ddof=1) / np.sqrt(dt)
    lam = jumps.size / (r.size * dt)
    if jumps.size >= 2:
        jump_mean, jump_std = jumps.mean(), jumps.std(ddof=1)
    elif jumps.size == 1:
        jump_mean, jump_std = jumps[0], 0.0
    else:
        jump_mean, jump_std = 0.0, 0.0
    mu = r.mean() / dt + 0.5 * sigma**2
    return MertonParams(float(mu), float(sigma), float(lam), float(jump_mean), float(jump_std))


def fit_schwartz(prices, dt: float = DT) -> SchwartzParams:
    """Schwartz (OU sur ln S) par régression linéaire.

    Le schéma exact s'écrit comme un AR(1) :
        x_{t+1} = a + b x_t + e,   b = e^{-kappa dt},  a = theta (1 - b),
        Var(e) = sigma² (1 - b²) / (2 kappa).
    On estime a et b par moindres carrés, puis on inverse ces relations.
    Si b ≥ 1, il n'y a pas de retour à la moyenne détectable : kappa ≈ 0.
    """
    x = np.log(np.asarray(prices, float))
    X, Y = x[:-1], x[1:]
    b, a = np.polyfit(X, Y, 1)
    resid = Y - (a + b * X)
    b = min(b, 1 - 1e-8)
    kappa = -np.log(b) / dt
    theta = a / (1 - b)
    sigma = resid.std(ddof=2) * np.sqrt(2 * kappa / (1 - b**2))
    return SchwartzParams(float(kappa), float(theta), float(sigma))
