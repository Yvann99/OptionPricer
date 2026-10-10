"""Simulateurs de trajectoires de prix.

Chaque simulateur renvoie un objet `Paths` :
    t : grille de temps, forme (n_steps + 1,)
    S : prix, forme (n_paths, n_steps + 1)
    extra : dict d'autres processus (variance de Heston, régime, ...)

Les modèles
-----------
- gbm      : mouvement brownien géométrique (monde de Black–Scholes), schéma exact.
- heston   : volatilité stochastique, Euler « full truncation ».
- regime   : MBG dont la volatilité suit une chaîne de Markov (calme / stress).
- merton   : MBG + sauts log-normaux (Poisson composé).

`mu` est toujours le rendement moyen instantané : E[S_t] = S0 e^{mu t}.
Pour simuler sous la probabilité risque-neutre, prendre mu = r (- q).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Paths:
    t: np.ndarray
    S: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def n_paths(self) -> int:
        return self.S.shape[0]

    @property
    def n_steps(self) -> int:
        return self.S.shape[1] - 1

    def log_returns(self) -> np.ndarray:
        return np.diff(np.log(self.S), axis=1)

    def realized_vol(self) -> np.ndarray:
        """Volatilité réalisée annualisée de chaque trajectoire (sqrt de [log S]_T / T)."""
        T = self.t[-1] - self.t[0]
        return np.sqrt(np.sum(self.log_returns() ** 2, axis=1) / T)


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _grid(T, n_steps):
    return np.linspace(0.0, T, n_steps + 1)


def gbm(S0, mu, sigma, T, n_steps, n_paths, seed=None, antithetic=False) -> Paths:
    """dS = mu S dt + sigma S dW, simulé EXACTEMENT via log S.

    Par la formule d'Itô appliquée à ln S :
        d(ln S) = (mu - sigma²/2) dt + sigma dW
    Le -sigma²/2 est la correction d'Itô. Donc sur un pas dt :
        ln S_{t+dt} - ln S_t = (mu - sigma²/2) dt + sigma sqrt(dt) Z,   Z ~ N(0, 1)
    Pas d'erreur de discrétisation : la loi est exacte quel que soit dt.

    antithetic=True : on utilise Z et -Z (réduction de variance pour Monte-Carlo).
    """
    rng = _rng(seed)
    dt = T / n_steps
    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.vstack([half, -half])
        if n_paths % 2:
            Z = np.vstack([Z, rng.standard_normal((1, n_steps))])
    else:
        Z = rng.standard_normal((n_paths, n_steps))       # une ligne = une trajectoire
    incr = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z   # accroissements de ln S
    # ln(S_t / S0) = somme cumulée des accroissements ; on ajoute 0 au départ (t = 0)
    logS = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return Paths(_grid(T, n_steps), S0 * np.exp(logS))


def heston(S0, v0, mu, kappa, theta, xi, rho, T, n_steps, n_paths, seed=None) -> Paths:
    """Modèle de Heston :
        dS = mu S dt + sqrt(v) S dW1
        dv = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1, W2> = rho dt
    Schéma d'Euler « full truncation » (v^+ dans les coefficients).
    """
    rng = _rng(seed)
    dt = T / n_steps
    sq = np.sqrt(dt)
    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0], v[:, 0] = S0, v0
    logS = np.full(n_paths, np.log(S0))
    for i in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        vp = np.maximum(v[:, i], 0.0)
        logS = logS + (mu - 0.5 * vp) * dt + np.sqrt(vp) * sq * z1
        v[:, i + 1] = v[:, i] + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sq * z2
        S[:, i + 1] = np.exp(logS)
    return Paths(_grid(T, n_steps), S, {"v": np.maximum(v, 0.0)})


def regime_switching(S0, mu, sigmas, transition, T, n_steps, n_paths,
                     seed=None, initial_state=0) -> Paths:
    """MBG dont la volatilité dépend d'un régime markovien caché.

    sigmas     : volatilités par régime, ex. [0.12, 0.40]
    transition : matrice de transition PAR PAS de temps (lignes sommant à 1),
                 ou matrice génératrice Q (lignes sommant à 0) convertie via expm(Q dt).
    extra["state"] contient le régime au début de chaque pas, forme (n_paths, n_steps).
    """
    from scipy.linalg import expm

    rng = _rng(seed)
    dt = T / n_steps
    sigmas = np.asarray(sigmas, float)
    P = np.asarray(transition, float)
    if np.allclose(P.sum(axis=1), 0.0):
        P = expm(P * dt)
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("transition doit être stochastique (lignes = 1) ou génératrice (lignes = 0)")
    cumP = np.cumsum(P, axis=1)

    state = np.empty((n_paths, n_steps), dtype=int)
    s = np.full(n_paths, initial_state)
    for i in range(n_steps):
        state[:, i] = s
        u = rng.random(n_paths)
        s = (u[:, None] > cumP[s]).sum(axis=1)

    sig = sigmas[state]
    Z = rng.standard_normal((n_paths, n_steps))
    incr = (mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z
    logS = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return Paths(_grid(T, n_steps), S0 * np.exp(logS), {"state": state, "sigma": sig})


def merton_jump(S0, mu, sigma, lam, jump_mean, jump_std, T, n_steps, n_paths, seed=None) -> Paths:
    """MBG avec sauts : à chaque saut, log S bouge de Y ~ N(jump_mean, jump_std²).
    lam : intensité (nombre moyen de sauts par an).
    Le drift est compensé pour que E[S_t] = S0 e^{mu t}.
    """
    rng = _rng(seed)
    dt = T / n_steps
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0  # E[e^Y - 1]
    Z = rng.standard_normal((n_paths, n_steps))
    N = rng.poisson(lam * dt, (n_paths, n_steps))
    J = jump_mean * N + jump_std * np.sqrt(N) * rng.standard_normal((n_paths, n_steps))
    incr = (mu - lam * k - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + J
    logS = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return Paths(_grid(T, n_steps), S0 * np.exp(logS), {"n_jumps": N})


def schwartz(S0, kappa, theta, sigma, T, n_steps, n_paths, seed=None) -> Paths:
    """Modèle de Schwartz à un facteur : RETOUR À LA MOYENNE du log-prix.

        d(ln S) = kappa (theta - ln S) dt + sigma dW

    ln S est un processus d'Ornstein–Uhlenbeck : il est tiré vers le niveau theta
    avec la force kappa. Intuition matières premières : quand le café est très cher,
    les producteurs plantent plus et les consommateurs substituent, donc l'offre et
    la demande ramènent le prix vers un niveau « normal ».
    Demi-vie d'un écart : ln(2) / kappa.

    Schéma EXACT (la loi de transition d'un OU est gaussienne connue) :
        x_{t+dt} = x_t e^{-kappa dt} + theta (1 - e^{-kappa dt})
                   + sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa)) Z
    Modèle de monde RÉEL (sous P), pour des scénarios de prix, pas pour le pricing.
    """
    rng = _rng(seed)
    dt = T / n_steps
    a = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1 - a**2) / (2 * kappa))
    x = np.empty((n_paths, n_steps + 1))
    x[:, 0] = np.log(S0)
    Z = rng.standard_normal((n_paths, n_steps))
    for i in range(n_steps):
        x[:, i + 1] = x[:, i] * a + theta * (1 - a) + sd * Z[:, i]
    return Paths(_grid(T, n_steps), np.exp(x))
