"""Moteur de couverture : portefeuille autofinancé en temps discret.

Le « livre » du teneur de marché contient, à chaque date t_i :
    - `position` options (position = -1 : il a vendu une option),
    - h_i actions (la couverture),
    - du cash, rémunéré au taux sans risque r.

À t_0, il encaisse la prime (prix Black–Scholes à la vol `sigma_price`),
achète la couverture initiale et place le reste. Ensuite, à chaque date de
rééquilibrage, il ajuste h en finançant l'achat par le cash (autofinancement).
Les coûts de transaction sont proportionnels au montant échangé :
    coût = cost_rate * |Δh| * S.

À maturité, le P&L final vaut
    cash_T + h_T S_T + position * payoff(S_T).
Avec une couverture parfaite (temps continu, bon modèle, pas de coûts), il vaut 0.

Stratégies
----------
Une stratégie est un objet appelable
    strategy(i, ctx) -> nouvelle quantité d'actions (tableau de taille n_paths)
où `ctx` est un `HedgeContext` donnant accès au passé uniquement (S[:, :i+1],
holdings précédentes, ...). C'est la notion de processus *adapté* du cours :
la stratégie ne doit jamais lire S[:, i+1:].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs
from .market import Paths


@dataclass
class OptionSpec:
    K: float
    T: float
    option: str = "call"  # "call" ou "put"

    def payoff(self, S):
        return bs.payoff(S, self.K, self.option)


@dataclass
class HedgeContext:
    """Information disponible pour la stratégie à la date t_i."""
    i: int
    t: float
    tau: float               # temps restant jusqu'à maturité
    S: np.ndarray            # prix courant S_i, forme (n_paths,)
    S_hist: np.ndarray       # S[:, :i+1] (passé et présent uniquement)
    holdings: np.ndarray     # quantité d'actions détenue avant rééquilibrage
    option: OptionSpec
    position: float
    r: float
    dt: float


# --------------------------------------------------------------------------
# Stratégies de base
# --------------------------------------------------------------------------

class DeltaHedge:
    """Couverture en delta Black–Scholes, rééquilibrée tous les `every` pas.

    sigma : volatilité utilisée pour calculer le delta. Un scalaire, ou une
            fonction ctx -> sigma (permet d'utiliser une estimation en ligne).
    """

    def __init__(self, sigma, every: int = 1):
        self.sigma = sigma
        self.every = int(every)

    def _sigma(self, ctx):
        return self.sigma(ctx) if callable(self.sigma) else self.sigma

    def target(self, ctx):
        d = bs.delta(ctx.S, ctx.option.K, ctx.tau, ctx.r, self._sigma(ctx), ctx.option.option)
        return -ctx.position * d

    def __call__(self, i, ctx):
        if i % self.every == 0:
            return self.target(ctx)
        return ctx.holdings


class NoHedge:
    """Aucune couverture : le vendeur garde la prime et attend."""

    def __call__(self, i, ctx):
        return np.zeros_like(ctx.S)


# --------------------------------------------------------------------------
# Résultat et moteur
# --------------------------------------------------------------------------

@dataclass
class HedgeResult:
    t: np.ndarray
    S: np.ndarray             # (n_paths, n+1)
    holdings: np.ndarray      # (n_paths, n+1) actions détenues APRÈS rééquilibrage en t_i
    cash: np.ndarray          # (n_paths, n+1) cash APRÈS rééquilibrage en t_i
    costs: np.ndarray         # (n_paths, n+1) coûts payés en t_i
    option_value: np.ndarray  # (n_paths, n+1) valeur de l'option marquée au modèle (sigma_mark)
    premium: float            # prime encaissée (par option)
    pnl: np.ndarray           # (n_paths,) P&L final
    option: OptionSpec
    position: float
    r: float
    sigma_mark: float

    @property
    def book_value(self) -> np.ndarray:
        """Valeur marquée au modèle du livre (options + actions + cash), par date."""
        return self.cash + self.holdings * self.S + self.position * self.option_value

    @property
    def n_trades(self) -> np.ndarray:
        """Nombre de transactions (achat initial compris), par trajectoire."""
        return (np.abs(np.diff(self.holdings, axis=1, prepend=0.0)) > 1e-12).sum(axis=1)

    def summary(self) -> dict:
        p = self.pnl
        q05 = np.quantile(p, 0.05)
        return {
            "premium": self.premium,
            "mean": p.mean(),
            "std": p.std(ddof=1),
            "q05": q05,
            "cvar05": p[p <= q05].mean(),
            "min": p.min(),
            "mean_costs": self.costs.sum(axis=1).mean(),
            "mean_trades": self.n_trades.mean(),
        }


def run_hedge(paths: Paths, option: OptionSpec, r: float, sigma_price: float,
              strategy=None, position: float = -1.0, cost_rate: float = 0.0,
              sigma_mark: float | None = None, premium: float | None = None) -> HedgeResult:
    """Simule le livre d'un teneur de marché qui a vendu (position=-1) ou acheté
    (position=+1) une option, et la couvre avec `strategy`.

    sigma_price : vol utilisée pour fixer la prime (la vol implicite de vente).
    strategy    : stratégie de couverture, par défaut DeltaHedge(sigma_price).
    sigma_mark  : vol pour marquer l'option au modèle (attribution), par défaut sigma_price.
    premium     : pour imposer une prime (ex. prix de marché), sinon prix BS(sigma_price).
    """
    if not np.isclose(paths.t[-1], option.T):
        raise ValueError("Les trajectoires doivent s'arrêter à la maturité de l'option.")
    strategy = strategy or DeltaHedge(sigma_price)
    sigma_mark = sigma_price if sigma_mark is None else sigma_mark

    t, S = paths.t, paths.S
    n_paths, n1 = S.shape
    dt = paths.dt
    growth = np.exp(r * dt)
    tau = option.T - t

    V = bs.price(S, option.K, tau[None, :], r, sigma_mark, option.option)
    prem = bs.price(S[0, 0], option.K, option.T, r, sigma_price, option.option) if premium is None else premium

    holdings = np.zeros((n_paths, n1))
    cash = np.zeros((n_paths, n1))
    costs = np.zeros((n_paths, n1))

    # État du livre, un élément par trajectoire (tout est vectorisé sur les trajectoires) :
    h = np.zeros(n_paths)                    # actions détenues
    c = np.full(n_paths, -position * prem)   # cash : vendre (position=-1) rapporte la prime

    # Boucle sur le TEMPS (pas sur les trajectoires : elles avancent toutes ensemble)
    for i in range(n1 - 1):  # pas de rééquilibrage à maturité
        # 1) le cash a rapporté des intérêts depuis la date précédente
        if i > 0:
            c = c * growth
        # 2) la stratégie décide de la nouvelle quantité d'actions,
        #    en ne voyant que le passé S[:, :i+1]  (processus ADAPTÉ)
        ctx = HedgeContext(i=i, t=t[i], tau=tau[i], S=S[:, i], S_hist=S[:, : i + 1],
                           holdings=h, option=option, position=position, r=r, dt=dt)
        h_new = np.asarray(strategy(i, ctx), float)
        # 3) AUTOFINANCEMENT : l'achat d'actions est payé par le cash (et inversement)
        trade = h_new - h
        cost = cost_rate * np.abs(trade) * S[:, i]   # coût proportionnel au montant échangé
        c = c - trade * S[:, i] - cost
        h = h_new
        holdings[:, i], cash[:, i], costs[:, i] = h, c, cost

    # Maturité : le cash court un dernier pas, on garde les actions, on paie le payoff.
    # P&L = cash + actions + (position x payoff). Couverture parfaite => P&L = 0.
    c = c * growth
    holdings[:, -1], cash[:, -1] = h, c
    pnl = c + h * S[:, -1] + position * option.payoff(S[:, -1])

    return HedgeResult(t=t, S=S, holdings=holdings, cash=cash, costs=costs,
                       option_value=V, premium=float(prem), pnl=pnl, option=option,
                       position=position, r=r, sigma_mark=sigma_mark)
