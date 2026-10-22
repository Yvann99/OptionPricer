"""Phase A : anatomie du P&L d'un vendeur couvert.

Des expériences dont on CONNAÎT le résultat théorique. Chacune renvoie des
nombres (pas de graphique) pour être testée et réutilisée en phase C.

1. error_vs_n            erreur de couverture ∝ 1/√N (et plateau si le modèle est faux)
2. drift_independence    la couverture élimine le drift μ
3. vol_bet               P&L ≈ ½ ∫ Γ S² (σ_impl² - σ_réal²) dt, trajectoire par trajectoire
4. implied_vs_realized   couvrir à la vol implicite ou à la vol réalisée
5. cost_tradeoff         fréquence de rééquilibrage : précision contre coûts
6. model_comparison      même couverture BS dans des mondes différents (Heston, sauts)

Un « simulateur » est une fonction simulate(n_steps, seed) -> Paths, pour que
la même expérience tourne dans n'importe quel modèle de market.py.
"""

from __future__ import annotations

import numpy as np

from . import attribution
from . import black_scholes as bs
from .hedging import DeltaHedge, NoHedge, OptionSpec, run_hedge
from .market import Paths


def _skew_kurt(x):
    z = (x - x.mean()) / x.std()
    return float((z**3).mean()), float((z**4).mean() - 3.0)


def describe(pnl: np.ndarray) -> dict:
    """Statistiques d'une distribution de P&L. On regarde la queue gauche (CVaR),
    pas seulement la moyenne : vendre des options, c'est gagner souvent un peu
    et perdre rarement beaucoup."""
    q05 = np.quantile(pnl, 0.05)
    skew, kurt = _skew_kurt(pnl)
    return {"mean": float(pnl.mean()), "std": float(pnl.std(ddof=1)), "q05": float(q05),
            "cvar05": float(pnl[pnl <= q05].mean()), "min": float(pnl.min()),
            "skew": skew, "kurt": kurt}


# --------------------------------------------------------------------------
# 1. Erreur de couverture en fonction du nombre de rééquilibrages
# --------------------------------------------------------------------------

def error_vs_n(simulate, option: OptionSpec, r: float, sigma: float, Ns, seed: int = 0) -> np.ndarray:
    """Écart-type du P&L d'un vendeur couvert en delta à chaque pas, pour chaque N.

    Dans le monde de Black–Scholes (bonne vol), l'erreur décroît en 1/√N : la
    somme des N erreurs locales ½ Γ (dS² - σ² S² dt), indépendantes et de
    variance O(dt²), a une variance O(N dt²) = O(1/N).
    Si le modèle est faux (vol stochastique, sauts), une partie du risque ne
    dépend pas de N : l'erreur atteint un PLATEAU.
    """
    return np.array([run_hedge(simulate(n, seed + k), option, r, sigma).pnl.std(ddof=1)
                     for k, n in enumerate(Ns)])


def slope_loglog(Ns, errors) -> float:
    """Pente de log(erreur) contre log(N) : -1/2 attendu dans le monde de Black–Scholes."""
    return float(np.polyfit(np.log(Ns), np.log(errors), 1)[0])


# --------------------------------------------------------------------------
# 2. Indépendance au drift
# --------------------------------------------------------------------------

def drift_independence(simulate_mu, option: OptionSpec, r: float, sigma: float, mus,
                       n_steps: int = 250, seed: int = 0) -> dict:
    """Moyenne du P&L du vendeur, couvert ou non, pour plusieurs drifts μ.

    simulate_mu(mu, n_steps, seed) -> Paths.
    Couvert : la moyenne ne dépend pas de μ (la réplication élimine les anticipations).
    Non couvert : c'est un pari directionnel, la moyenne dépend fortement de μ.
    """
    out = {"mu": np.asarray(mus, float), "hedged": [], "naked": []}
    for mu in mus:
        p = simulate_mu(mu, n_steps, seed)
        out["hedged"].append(describe(run_hedge(p, option, r, sigma).pnl))
        out["naked"].append(describe(run_hedge(p, option, r, sigma, strategy=NoHedge()).pnl))
    return out


# --------------------------------------------------------------------------
# 3. Le pari sur la variation quadratique
# --------------------------------------------------------------------------

def vol_bet(paths: Paths, option: OptionSpec, r: float, sigma_impl: float) -> dict:
    """Vendre à `sigma_impl`, couvrir en delta à `sigma_impl`.

    Renvoie le P&L final et sa prédiction théorique, trajectoire par trajectoire :
        ½ Σ Γ_i S_i² (σ_impl² dt - (dS_i/S_i)²)
    C'est la variation quadratique RÉALISÉE, pondérée par le gamma dollar, qui
    décide du résultat : le vendeur gagne si le marché bouge moins que σ_impl.
    """
    res = run_hedge(paths, option, r, sigma_impl)
    theory = attribution.attribute(res)["vol"].sum(axis=1)
    return {"result": res, "pnl": res.pnl, "theory": theory,
            "realized_vol": paths.realized_vol()}


# --------------------------------------------------------------------------
# 4. Couvrir à la vol implicite ou à la vol réalisée
# --------------------------------------------------------------------------

def implied_vs_realized(paths: Paths, option: OptionSpec, r: float,
                        sigma_impl: float, sigma_real: float) -> dict:
    """Le vendeur encaisse la prime BS(sigma_impl), le marché réalise sigma_real.

    - Couverture à la vol IMPLICITE : le livre (marqué à σ_impl) gagne
      régulièrement ½ Γ S² (σ_impl² - σ_réal²) dt, mais le total dépend du
      chemin (du gamma rencontré) : P&L final ALÉATOIRE.
    - Couverture à la vol RÉALISÉE (supposée connue) : le P&L final est
      CERTAIN, égal à (BS(σ_impl) - BS(σ_réal)) e^{rT}, mais le livre marqué à
      σ_impl est bruité en cours de route.
    Les deux livres sont marqués à σ_impl (c'est le prix « de marché »).
    """
    implied = run_hedge(paths, option, r, sigma_impl)
    realized = run_hedge(paths, option, r, sigma_impl,
                         strategy=DeltaHedge(sigma_real), sigma_mark=sigma_impl)
    S0 = paths.S[0, 0]
    locked = float((bs.price(S0, option.K, option.T, r, sigma_impl, option.option)
                    - bs.price(S0, option.K, option.T, r, sigma_real, option.option))
                   * np.exp(r * option.T))
    return {"implied": implied, "realized": realized, "locked_in": locked}


# --------------------------------------------------------------------------
# 5. Coûts de transaction : précision contre coût
# --------------------------------------------------------------------------

def cost_tradeoff(paths: Paths, option: OptionSpec, r: float, sigma: float,
                  cost_rate: float, everys) -> list[dict]:
    """Couverture en delta tous les `every` pas, avec coûts proportionnels.

    Rééquilibrer souvent réduit l'erreur de couverture (écart-type) mais
    multiplie les coûts (moyenne qui baisse) : il n'y a pas de fréquence
    « parfaite », seulement un compromis risque / coût (sujet de la phase C).
    """
    out = []
    for every in everys:
        res = run_hedge(paths, option, r, sigma, strategy=DeltaHedge(sigma, every=every),
                        cost_rate=cost_rate)
        d = describe(res.pnl)
        d.update(every=int(every), n_rebal=paths.n_steps // every,
                 mean_costs=float(res.costs.sum(axis=1).mean()))
        out.append(d)
    return out


# --------------------------------------------------------------------------
# 6. Modèle faux
# --------------------------------------------------------------------------

def model_comparison(paths_by_model: dict, option: OptionSpec, r: float, sigma: float) -> dict:
    """La même couverture Black–Scholes (vol `sigma`) appliquée dans plusieurs mondes.

    Dans un monde Heston, la vol réalisée est aléatoire : le P&L hérite de
    l'écart σ² - σ_réal² et s'étale. Dans un monde à sauts, un saut ne peut pas
    être couvert par le delta : la perte vaut environ ½ Γ (ΔS)², d'où une queue
    gauche épaisse (skew négatif, kurtosis élevée).
    """
    return {name: describe(run_hedge(p, option, r, sigma).pnl)
            for name, p in paths_by_model.items()}
