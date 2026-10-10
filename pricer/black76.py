"""Modèle de Black (1976) : options sur FUTURES.

Les options sur café, cacao, sucre... ne portent pas sur la marchandise elle-même
mais sur un contrat future. Or un prix future F se comporte différemment d'une
action :

    - acheter un future ne coûte rien aujourd'hui (pas de financement à payer) ;
    - donc, sous la probabilité risque-neutre Q, F est une MARTINGALE :
          dF = sigma F dW^Q      (pas de terme en r dt !)

C'est la martingale exponentielle du cours : F_T = F_0 exp(sigma W_T - sigma² T / 2).

Le prix d'un call de maturité T, de strike K, est alors
    C = e^{-rT} [ F N(d1) - K N(d2) ],
    d1 = [ln(F/K) + sigma² T / 2] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T).

Remarque : c'est exactement Black–Scholes avec r = 0 (le future ne « croît » pas
sous Q), puis actualisé par e^{-rT} (le payoff est payé en T). On réutilise donc
le module black_scholes au lieu de tout réécrire.
"""

from __future__ import annotations

import numpy as np

from . import black_scholes as bs


def price(F, K, T, r, sigma, option="call"):
    """Prix de Black-76 d'un call ou d'un put européen sur future."""
    return np.exp(-r * np.asarray(T, float)) * bs.price(F, K, T, 0.0, sigma, option)


def delta(F, K, T, r, sigma, option="call"):
    """dV/dF : nombre de contrats futures à détenir pour couvrir l'option."""
    return np.exp(-r * np.asarray(T, float)) * bs.delta(F, K, T, 0.0, sigma, option)


def gamma(F, K, T, r, sigma):
    """d²V/dF²."""
    return np.exp(-r * np.asarray(T, float)) * bs.gamma(F, K, T, 0.0, sigma)


def vega(F, K, T, r, sigma):
    """dV/dsigma (pour une variation de sigma de 1)."""
    return np.exp(-r * np.asarray(T, float)) * bs.vega(F, K, T, 0.0, sigma)


def implied_vol(market_price, F, K, T, r, option="call"):
    """Volatilité implicite de Black-76 : on « dés-actualise » le prix, puis on
    inverse la formule de Black–Scholes avec r = 0."""
    undiscounted = np.asarray(market_price, float) * np.exp(r * np.asarray(T, float))
    return bs.implied_vol(undiscounted, F, K, T, 0.0, option)
