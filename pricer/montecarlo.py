"""Pricing par Monte-Carlo.

L'IDÉE
------
Le prix d'une option est une espérance sous la probabilité risque-neutre :
    V_0 = E^Q[ e^{-rT} h(trajectoire) ].
La loi des grands nombres dit qu'une moyenne de N tirages indépendants converge
vers l'espérance. On simule donc N trajectoires sous Q, on calcule le payoff de
chacune, on actualise, et on fait la moyenne.

L'ERREUR
--------
Le théorème central limite donne la précision : l'estimateur est approximativement
gaussien, d'écart-type  s / sqrt(N)  (s = écart-type des payoffs actualisés).
D'où l'intervalle de confiance à 95 % :  prix ± 1.96 s / sqrt(N).
Pour gagner un chiffre de précision (÷10), il faut 100 fois plus de trajectoires.
C'est lent : d'où les techniques de RÉDUCTION DE VARIANCE, qui diminuent s au
lieu d'augmenter N.

POURQUOI LE MONTE-CARLO
-----------------------
Black–Scholes ne donne une formule que pour les cas simples. Dès que le payoff
dépend de toute la trajectoire (asiatique, barrière, lookback) ou que le modèle
est plus riche (sauts, volatilité stochastique), le Monte-Carlo marche toujours :
il suffit de savoir SIMULER.

Les payoffs ci-dessous prennent un tableau S de forme (n_paths, n_steps + 1)
(une ligne = une trajectoire, colonne 0 = date initiale) et renvoient un tableau
(n_paths,) : un payoff par trajectoire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Résultat d'un Monte-Carlo
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    price: float
    stderr: float      # erreur type = s / sqrt(N)
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        """Intervalle de confiance à 95 % (théorème central limite)."""
        return self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr

    def __repr__(self):
        lo, hi = self.ci95
        return f"{self.price:.4f} ± {1.96 * self.stderr:.4f}  (IC 95 % [{lo:.4f}, {hi:.4f}], N={self.n_paths})"


def mc_price(payoffs, r: float, T: float) -> MCResult:
    """Prix Monte-Carlo standard : moyenne des payoffs actualisés."""
    x = np.exp(-r * T) * np.asarray(payoffs, float)
    return MCResult(float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)), x.size)


def mc_price_control_variate(payoffs, control, control_mean: float, r: float, T: float) -> MCResult:
    """Monte-Carlo avec VARIABLE DE CONTRÔLE.

    On connaît exactement l'espérance d'une autre quantité Y (le « contrôle »),
    très corrélée au payoff X. On corrige l'estimateur :
        X_cv = X - beta (Y - E[Y]),    beta = Cov(X, Y) / Var(Y).
    E[X_cv] = E[X] (on retire une quantité d'espérance nulle), mais
    Var(X_cv) = Var(X) (1 - corr(X, Y)²) : si la corrélation vaut 0.99,
    la variance est divisée par 50, comme si on avait 50 fois plus de trajectoires.

    payoffs, control : payoffs NON actualisés (même actualisation pour les deux)
    control_mean     : E^Q[e^{-rT} Y], valeur exacte (formule fermée)
    """
    disc = np.exp(-r * T)
    X = disc * np.asarray(payoffs, float)
    Y = disc * np.asarray(control, float)
    C = np.cov(X, Y)
    beta = C[0, 1] / C[1, 1]
    Z = X - beta * (Y - control_mean)
    return MCResult(float(Z.mean()), float(Z.std(ddof=1) / np.sqrt(Z.size)), Z.size)


# ---------------------------------------------------------------------------
# Payoffs
# ---------------------------------------------------------------------------

def _w(option):
    if option == "call":
        return 1.0
    if option == "put":
        return -1.0
    raise ValueError(f"option doit être 'call' ou 'put', reçu : {option!r}")


def european(S, K, option="call"):
    """Payoff européen : ne dépend que du prix final S_T."""
    return np.maximum(_w(option) * (S[:, -1] - K), 0.0)


def digital(S, K, option="call"):
    """Option binaire : paie 1 si S_T > K (call) ou S_T < K (put)."""
    return (_w(option) * (S[:, -1] - K) > 0).astype(float)


def asian(S, K, option="call", average="arithmetic"):
    """Option ASIATIQUE : le payoff porte sur la MOYENNE des prix, pas le prix final.

    Très utilisée en matières premières : un torréfacteur qui achète du café toute
    l'année est exposé au prix moyen, pas au prix d'un jour précis. La moyenne lisse
    les mouvements, donc l'option est moins chère qu'une européenne.
    La moyenne porte sur les dates 1..n (on exclut la date 0, déjà connue).
    """
    path = S[:, 1:]
    if average == "arithmetic":
        A = path.mean(axis=1)
    elif average == "geometric":
        A = np.exp(np.log(path).mean(axis=1))
    else:
        raise ValueError("average doit être 'arithmetic' ou 'geometric'")
    return np.maximum(_w(option) * (A - K), 0.0)


def barrier(S, K, B, option="call", kind="up-and-out"):
    """Option à BARRIÈRE : s'active (« in ») ou se désactive (« out ») si le prix
    touche le niveau B pendant la vie de l'option.

    kind : "up-and-out", "up-and-in", "down-and-out", "down-and-in".
    Exemple : un call up-and-out à barrière 130 protège contre la hausse, mais
    seulement jusqu'à 130 ; il est donc moins cher qu'un call classique.
    Relation de parité : in + out = européenne (exactement une des deux est active).

    Surveillance DISCRÈTE (aux dates de la grille) : une trajectoire continue
    pourrait toucher B entre deux dates sans qu'on le voie. Plus la grille est
    fine, plus on se rapproche de la barrière continue.
    """
    direction, _, knock = kind.split("-")
    if direction == "up":
        touched = S.max(axis=1) >= B
    elif direction == "down":
        touched = S.min(axis=1) <= B
    else:
        raise ValueError("kind doit commencer par 'up' ou 'down'")
    alive = touched if knock == "in" else ~touched
    return np.where(alive, european(S, K, option), 0.0)


def lookback(S, option="call"):
    """Option LOOKBACK à strike flottant : « acheter au plus bas, vendre au plus haut ».
        call : S_T - min(S)       put : max(S) - S_T
    Toujours positive : c'est le rêve du trader, donc elle coûte cher.
    """
    if option == "call":
        return S[:, -1] - S.min(axis=1)
    if option == "put":
        return S.max(axis=1) - S[:, -1]
    raise ValueError(f"option doit être 'call' ou 'put', reçu : {option!r}")


# ---------------------------------------------------------------------------
# Convergence et grecques
# ---------------------------------------------------------------------------

def convergence(simulate, payoff, r, T, n_list, seed=0):
    """Prix Monte-Carlo pour des nombres de trajectoires croissants.

    simulate(n_paths, seed) -> tableau S (n_paths, n_steps+1)
    payoff(S)               -> payoffs (n_paths,)
    Sert à visualiser la décroissance de l'erreur en 1/sqrt(N).
    """
    S = simulate(max(n_list), seed)
    return [mc_price(payoff(S[:n]), r, T) for n in n_list]


def mc_greeks(simulate, payoff, r, T, S0, rel_bump=0.01, seed=0):
    """Delta et gamma par différences finies (« bump and revalue »).

    On re-simule avec S0 (1 ± h) en gardant les MÊMES nombres aléatoires (même seed,
    « common random numbers »). Sinon le bruit Monte-Carlo de chaque prix serait bien
    plus grand que la différence qu'on cherche à mesurer, et le delta serait inutilisable.

    simulate(S0, seed) -> tableau S
    """
    h = rel_bump * S0
    disc = np.exp(-r * T)
    up = disc * payoff(simulate(S0 + h, seed)).mean()
    mid = disc * payoff(simulate(S0, seed)).mean()
    down = disc * payoff(simulate(S0 - h, seed)).mean()
    return {"price": mid, "delta": (up - down) / (2 * h), "gamma": (up - 2 * mid + down) / h**2}
