"""Formules de Black–Scholes pour les options européennes.

Conventions :
- T est le temps RESTANT avant maturité (en années), pas la date de maturité.
- r est le taux sans risque, composé en continu.
- sigma est la volatilité annuelle (0.2 = 20 %).
- option vaut "call" ou "put".
- vega est donné pour une variation de sigma de 1 (pas de 1 %).
- theta est la dérivée par rapport au temps calendaire t, par an (theta = -dV/dT).

VECTORISATION
-------------
Toutes les fonctions acceptent des nombres OU des tableaux numpy, et appliquent
les règles de « broadcasting » de numpy. Par exemple :
    price(S=np.array([90, 100, 110]), K=100, T=0.5, r=0.02, sigma=0.2)
renvoie les trois prix d'un coup. C'est indispensable pour la couverture, où l'on
traite des milliers de trajectoires en même temps.

Le seul cas délicat est T = 0 (maturité) : la formule divise par sigma*sqrt(T).
On utilise donc un MASQUE `live = T > 0` : la formule n'est appliquée qu'aux
éléments encore vivants, les autres reçoivent la valeur à maturité.
"""

import numpy as np
from scipy.stats import norm  # loi normale centrée réduite : norm.cdf = N, norm.pdf = densité

_EPS_T = 1e-12  # en dessous de ce temps restant, on considère l'option arrivée à maturité


# ---------------------------------------------------------------------------
# Outils internes
# ---------------------------------------------------------------------------

def _sign(option):
    """+1 pour un call, -1 pour un put.

    Astuce : les formules du call et du put ne diffèrent que par des signes.
    Avec w = +1 (call) ou -1 (put) on écrit une seule formule pour les deux :
        prix = w * ( S N(w d1) - K e^{-rT} N(w d2) )
    """
    if option == "call":
        return 1
    if option == "put":
        return -1
    raise ValueError(f"option doit être 'call' ou 'put', reçu : {option!r}")


def _as_arrays(*xs):
    """Convertit les arguments en tableaux float de même forme (broadcasting).

    np.broadcast_arrays renvoie des « vues » en lecture seule : on ne les modifie
    jamais, on écrit toujours dans un nouveau tableau `out`.
    """
    return np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in xs))


def _ret(out):
    """Renvoie un vrai nombre si le résultat est de dimension 0, sinon le tableau."""
    return out[()] if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Payoff, d1, d2, prix
# ---------------------------------------------------------------------------

def payoff(S, K, option="call"):
    """Payoff à maturité : (S-K)^+ pour un call, (K-S)^+ pour un put."""
    w = _sign(option)
    return np.maximum(w * (np.asarray(S, dtype=float) - K), 0.0)


def d1_d2(S, K, T, r, sigma):
    """Calcule d1 et d2. Suppose T > 0 et sigma > 0.

    d1 = [ln(S/K) + (r + sigma²/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    N(d2) = Q(S_T > K) : probabilité risque-neutre de finir dans la monnaie.
    N(d1) = delta du call.
    """
    vol_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


def price(S, K, T, r, sigma, option="call"):
    """Prix Black–Scholes d'un call ou d'un put européen (vectorisé).

    Call : C = S N(d1) - K e^{-rT} N(d2)
    Put  : P = K e^{-rT} N(-d2) - S N(-d1)
    À maturité (T = 0), le prix est le payoff.
    """
    w = _sign(option)
    S, K, T, sigma = _as_arrays(S, K, T, sigma)

    # 1) on remplit tout avec la valeur à maturité (le payoff)...
    out = np.array(payoff(S, K, option), dtype=float)

    # 2) ...puis on écrase avec la formule là où l'option est encore vivante
    live = T > _EPS_T
    if np.any(live):
        Sl, Kl, Tl, sl = S[live], K[live], T[live], sigma[live]
        d1, d2 = d1_d2(Sl, Kl, Tl, r, sl)
        out[live] = w * (Sl * norm.cdf(w * d1) - Kl * np.exp(-r * Tl) * norm.cdf(w * d2))
    return _ret(out)


# ---------------------------------------------------------------------------
# Grecques : dérivées du prix
# ---------------------------------------------------------------------------
# Elles se calculent en dérivant la formule du prix. Le calcul se simplifie grâce
# à l'identité  S N'(d1) = K e^{-rT} N'(d2)  (à vérifier à la main !).

def delta(S, K, T, r, sigma, option="call"):
    """Delta = dV/dS : nombre d'actions à détenir pour couvrir l'option.

    Call : N(d1) (entre 0 et 1)      Put : N(d1) - 1 (entre -1 et 0)
    À maturité : 1 (call) ou -1 (put) si l'option est dans la monnaie, 0 sinon.
    """
    w = _sign(option)
    S, K, T, sigma = _as_arrays(S, K, T, sigma)
    out = np.where(w * (S - K) > 0, float(w), 0.0)
    live = T > _EPS_T
    if np.any(live):
        d1, _ = d1_d2(S[live], K[live], T[live], r, sigma[live])
        out[live] = w * norm.cdf(w * d1)
    return _ret(out)


def gamma(S, K, T, r, sigma):
    """Gamma = d²V/dS² = N'(d1) / (S sigma sqrt(T)). Identique pour call et put.

    Mesure la convexité : de combien le delta bouge quand S bouge.
    C'est le gamma que le vendeur « paie » sur les mouvements réalisés.
    """
    S, K, T, sigma = _as_arrays(S, K, T, sigma)
    out = np.zeros_like(S)
    live = T > _EPS_T
    if np.any(live):
        Sl, Tl, sl = S[live], T[live], sigma[live]
        d1, _ = d1_d2(Sl, K[live], Tl, r, sl)
        out[live] = norm.pdf(d1) / (Sl * sl * np.sqrt(Tl))
    return _ret(out)


def vega(S, K, T, r, sigma):
    """Vega = dV/dsigma = S N'(d1) sqrt(T). Identique pour call et put.

    Sensibilité à la volatilité : c'est l'exposition d'un vendeur d'options.
    """
    S, K, T, sigma = _as_arrays(S, K, T, sigma)
    out = np.zeros_like(S)
    live = T > _EPS_T
    if np.any(live):
        Sl, Tl = S[live], T[live]
        d1, _ = d1_d2(Sl, K[live], Tl, r, sigma[live])
        out[live] = Sl * norm.pdf(d1) * np.sqrt(Tl)
    return _ret(out)


def theta(S, K, T, r, sigma, option="call"):
    """Theta = dV/dt (temps calendaire), par an.

    Theta = - S N'(d1) sigma / (2 sqrt(T)) - w r K e^{-rT} N(w d2)
    Le plus souvent négatif : l'option perd de la valeur avec le temps.
    Pour le vendeur, c'est l'inverse : il « encaisse » le theta.
    """
    w = _sign(option)
    S, K, T, sigma = _as_arrays(S, K, T, sigma)
    out = np.zeros_like(S)
    live = T > _EPS_T
    if np.any(live):
        Sl, Kl, Tl, sl = S[live], K[live], T[live], sigma[live]
        d1, d2 = d1_d2(Sl, Kl, Tl, r, sl)
        out[live] = (-Sl * norm.pdf(d1) * sl / (2 * np.sqrt(Tl))
                     - w * r * Kl * np.exp(-r * Tl) * norm.cdf(w * d2))
    return _ret(out)


# ---------------------------------------------------------------------------
# Volatilité implicite
# ---------------------------------------------------------------------------

def implied_vol(market_price, S, K, T, r, option="call",
                lo=1e-6, hi=5.0, tol=1e-10, max_iter=200):
    """Volatilité implicite : le sigma tel que price(sigma) = market_price.

    Méthode : DICHOTOMIE. Le prix est croissant en sigma (vega > 0), donc :
      - on part de l'intervalle [lo, hi] ;
      - on teste le milieu m : si price(m) < prix de marché, la solution est à droite,
        sinon à gauche ;
      - on recommence jusqu'à ce que l'intervalle soit minuscule.
    Plus lent que Newton, mais ne diverge jamais. Vectorisé : tous les éléments
    sont traités en parallèle.

    Renvoie NaN si le prix est hors des bornes d'absence d'arbitrage
    (aucune volatilité ne peut le produire).
    """
    p, S, K, T = _as_arrays(market_price, S, K, T)
    a = np.full(p.shape, lo)
    b = np.full(p.shape, hi)

    p_lo = price(S, K, T, r, a, option)
    p_hi = price(S, K, T, r, b, option)
    valid = (p >= p_lo - tol) & (p <= p_hi + tol) & (T > _EPS_T)

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        too_low = price(S, K, T, r, m, option) < p   # prix trop bas -> il faut plus de vol
        a = np.where(too_low, m, a)
        b = np.where(too_low, b, m)
        if np.max(b - a) < tol:
            break

    out = np.where(valid, 0.5 * (a + b), np.nan)
    return _ret(out)
