"""Formules fermées qui servent de RÉFÉRENCE pour valider le Monte-Carlo.

Principe : avant de faire confiance à un Monte-Carlo sur un produit sans formule
(asiatique arithmétique, barrière, modèle à sauts...), on le teste sur des cas
voisins où la réponse exacte est connue.

COÛT DE PORTAGE b
-----------------
Toutes les formules prennent un paramètre b = taux de croissance du sous-jacent
sous Q :
    - action sans dividende : b = r
    - future (café, cacao...) : b = 0   (le future est une martingale sous Q)
Le prix d'un call européen devient
    C = e^{-rT} [ S e^{bT} N(d1) - K N(d2) ],
    d1 = [ln(S/K) + (b + sigma²/2) T] / (sigma sqrt(T)).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, poisson


def bs_carry(S, K, T, r, b, sigma, option="call"):
    """Black–Scholes généralisé avec coût de portage b."""
    w = 1 if option == "call" else -1
    vol = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / vol
    d2 = d1 - vol
    return np.exp(-r * T) * w * (S * np.exp(b * T) * norm.cdf(w * d1) - K * norm.cdf(w * d2))


def asian_geometric(S, K, T, r, b, sigma, n_fixings, option="call"):
    """Option asiatique à moyenne GÉOMÉTRIQUE discrète (Kemna–Vorst).

    Moyenne sur les dates t_i = i T / n, i = 1..n :  G = (Π S_{t_i})^{1/n}.
    Comme ln G est une moyenne de gaussiennes, ln G est gaussien :
        E[ln G]   = ln S + (b - sigma²/2) T (n+1) / (2n)
        Var[ln G] = sigma² T (n+1)(2n+1) / (6 n²)
    et on obtient une formule à la Black–Scholes.

    La moyenne géométrique est toujours ≤ la moyenne arithmétique : ce prix est
    donc proche, mais inférieur, de celui de l'asiatique arithmétique. C'est la
    variable de contrôle idéale pour le Monte-Carlo de l'asiatique arithmétique.
    """
    n = n_fixings
    m = np.log(S) + (b - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    sv = np.sqrt(v)
    d1 = (m - np.log(K) + v) / sv
    d2 = d1 - sv
    w = 1 if option == "call" else -1
    return np.exp(-r * T) * w * (np.exp(m + 0.5 * v) * norm.cdf(w * d1) - K * norm.cdf(w * d2))


def merton(S, K, T, r, b, sigma, lam, jump_mean, jump_std, option="call", n_max=100):
    """Option européenne dans le modèle de Merton (diffusion + sauts log-normaux).

    Conditionnellement au nombre de sauts N = n, ln S_T est gaussien : on retombe
    sur Black–Scholes avec
        sigma_n² = sigma² + n jump_std² / T
        b_n      = b - lam k + n ln(1 + k) / T,    k = E[e^Y - 1]
    Le prix est la moyenne de ces prix pondérée par P(N = n) (loi de Poisson) :
        C = Σ_n P(N = n) C_BS(sigma_n, b_n).
    """
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    total = 0.0
    for n in range(n_max + 1):
        p = poisson.pmf(n, lam * T)
        if p < 1e-16 and n > lam * T:
            break
        sig_n = np.sqrt(sigma**2 + n * jump_std**2 / T)
        b_n = b - lam * k + n * np.log1p(k) / T
        total = total + p * bs_carry(S, K, T, r, b_n, sig_n, option)
    return total
