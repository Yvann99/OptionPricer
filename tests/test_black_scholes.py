"""Tests du pricer Black–Scholes.

Règle d'or : chaque test vérifie une vérité INDÉPENDANTE du code
(valeur publiée, relation d'absence d'arbitrage, définition mathématique).
"""

import numpy as np
import pytest

from pricer import black_scholes as bs
from pricer import market

# Exemple classique de Hull
S, K, T, r, sigma = 42.0, 40.0, 0.5, 0.10, 0.20


# --- Étape 1 : le prix -----------------------------------------------------

def test_call_valeur_de_reference():
    assert bs.price(S, K, T, r, sigma, "call") == pytest.approx(4.7594, abs=1e-4)


def test_put_valeur_de_reference():
    assert bs.price(S, K, T, r, sigma, "put") == pytest.approx(0.8086, abs=1e-4)


def test_parite_call_put():
    for S_test in [80.0, 100.0, 120.0]:
        c = bs.price(S_test, 100.0, 1.0, 0.03, 0.25, "call")
        p = bs.price(S_test, 100.0, 1.0, 0.03, 0.25, "put")
        assert c - p == pytest.approx(S_test - 100.0 * np.exp(-0.03 * 1.0))


def test_a_maturite_le_prix_est_le_payoff():
    assert bs.price(110.0, 100.0, 0.0, r, sigma, "call") == pytest.approx(10.0)
    assert bs.price(110.0, 100.0, 0.0, r, sigma, "put") == pytest.approx(0.0)


def test_option_invalide():
    with pytest.raises(ValueError):
        bs.price(S, K, T, r, sigma, "Call")


# --- Étape 2 : les grecques ------------------------------------------------

@pytest.mark.parametrize("option", ["call", "put"])
def test_grecques_egales_aux_differences_finies(option):
    """Une dérivée ≈ [f(x+h) - f(x-h)] / 2h : on vérifie chaque grecque ainsi."""
    h = 1e-3

    def V(S=S, T=T, sigma=sigma):
        return bs.price(S, K, T, r, sigma, option)

    assert bs.delta(S, K, T, r, sigma, option) == pytest.approx((V(S=S + h) - V(S=S - h)) / (2 * h), abs=1e-6)
    assert bs.gamma(S, K, T, r, sigma) == pytest.approx((V(S=S + h) - 2 * V() + V(S=S - h)) / h**2, abs=1e-4)
    assert bs.vega(S, K, T, r, sigma) == pytest.approx((V(sigma=sigma + h) - V(sigma=sigma - h)) / (2 * h), rel=1e-5)
    # theta = dV/dt = -dV/dT (quand le temps passe, le temps restant diminue)
    assert bs.theta(S, K, T, r, sigma, option) == pytest.approx(-(V(T=T + h) - V(T=T - h)) / (2 * h), rel=1e-5)


def test_bornes_du_delta():
    Ss = np.linspace(50, 150, 11)
    assert np.all((bs.delta(Ss, 100, 1.0, r, sigma, "call") >= 0) & (bs.delta(Ss, 100, 1.0, r, sigma, "call") <= 1))
    assert np.all((bs.delta(Ss, 100, 1.0, r, sigma, "put") >= -1) & (bs.delta(Ss, 100, 1.0, r, sigma, "put") <= 0))


@pytest.mark.parametrize("option", ["call", "put"])
def test_edp_de_black_scholes(option):
    """theta + r S delta + ½ sigma² S² gamma - r V = 0 (formule d'Itô + absence d'arbitrage)."""
    V = bs.price(S, K, T, r, sigma, option)
    lhs = (bs.theta(S, K, T, r, sigma, option)
           + r * S * bs.delta(S, K, T, r, sigma, option)
           + 0.5 * sigma**2 * S**2 * bs.gamma(S, K, T, r, sigma)
           - r * V)
    assert lhs == pytest.approx(0.0, abs=1e-10)


def test_volatilite_implicite_aller_retour():
    vols = np.array([0.05, 0.2, 0.6, 1.5])
    prix = bs.price(S, K, T, r, vols)
    assert np.allclose(bs.implied_vol(prix, S, K, T, r), vols, atol=1e-7)


def test_volatilite_implicite_prix_impossible():
    # un call ne peut pas valoir plus que l'action : aucune vol ne donne ce prix
    assert np.isnan(bs.implied_vol(S + 1, S, K, T, r))


# --- Étape 3 : la vectorisation --------------------------------------------

def test_vectorisation():
    Ss = np.array([90.0, 100.0, 110.0])
    prix = bs.price(Ss, 100.0, 1.0, r, sigma)
    assert prix.shape == (3,)
    for i in range(3):  # même résultat que trois appels séparés
        assert prix[i] == pytest.approx(bs.price(Ss[i], 100.0, 1.0, r, sigma))
    # maturités mélangées : certaines options mortes, d'autres vivantes
    Ts = np.array([0.0, 0.5, 0.0])
    assert np.allclose(bs.price(Ss, 100.0, Ts, r, sigma)[[0, 2]], [0.0, 10.0])
    assert np.allclose(bs.delta(Ss, 100.0, 0.0, r, sigma, "put"), [-1, 0, 0])


# --- Lien avec les martingales ---------------------------------------------

def test_monte_carlo_egal_formule_fermee():
    """Sous Q (mu = r), le prix est l'espérance du payoff actualisé."""
    paths = market.gbm(S, r, sigma, T, 1, 400_000, seed=0, antithetic=True)
    disc = np.exp(-r * T) * bs.payoff(paths.S[:, -1], K)
    erreur_type = disc.std() / np.sqrt(disc.size)
    assert abs(disc.mean() - bs.price(S, K, T, r, sigma)) < 4 * erreur_type
