"""Tests du Monte-Carlo, de Black-76 et des formules fermées.

Chaque test compare le Monte-Carlo à une valeur EXACTE, en tolérant l'erreur
statistique : |MC - exact| < 4 erreurs types (faux échec ≈ 1 fois sur 16 000).
"""

import numpy as np
import pytest

from pricer import black76, black_scholes as bs, closed_forms as cf, market, montecarlo as mc

F0, K, T, r, sig = 300.0, 300.0, 0.25, 0.04, 0.35   # café : ~300 USc/lb, vol ~35 %


def _close(res, exact, k=4):
    assert abs(res.price - exact) < k * res.stderr, f"{res} vs exact {exact:.4f}"


# --- Black-76 ---------------------------------------------------------------

def test_black76_est_black_scholes_avec_r_nul_actualise():
    assert black76.price(F0, K, T, r, sig) == pytest.approx(np.exp(-r * T) * bs.price(F0, K, T, 0, sig))


def test_black76_parite_call_put():
    # pour un future : C - P = e^{-rT} (F - K)
    for k in (250.0, 300.0, 350.0):
        c = black76.price(F0, k, T, r, sig, "call")
        p = black76.price(F0, k, T, r, sig, "put")
        assert c - p == pytest.approx(np.exp(-r * T) * (F0 - k))


def test_black76_vol_implicite_aller_retour():
    p = black76.price(F0, 330.0, T, r, 0.42)
    assert black76.implied_vol(p, F0, 330.0, T, r) == pytest.approx(0.42, abs=1e-7)


def test_future_est_martingale_sous_Q():
    paths = market.gbm(F0, 0.0, sig, T, 10, 200_000, seed=1)
    for j in range(1, 11):
        x = paths.S[:, j]
        assert abs(x.mean() - F0) < 4 * x.std() / np.sqrt(x.size)


# --- Monte-Carlo européen ---------------------------------------------------

def test_mc_europeen_egal_black76():
    S = market.gbm(F0, 0.0, sig, T, 1, 200_000, seed=2).S
    for option in ("call", "put"):
        _close(mc.mc_price(mc.european(S, K, option), r, T), black76.price(F0, K, T, r, sig, option))


def test_erreur_en_un_sur_racine_de_N():
    sim = lambda n, seed: market.gbm(F0, 0.0, sig, T, 1, n, seed=seed).S
    res = mc.convergence(sim, lambda S: mc.european(S, K), r, T, [10_000, 160_000])
    assert res[0].stderr / res[1].stderr == pytest.approx(4.0, rel=0.05)   # sqrt(16) = 4


def test_ic95_contient_le_vrai_prix_environ_95_pourcent_du_temps():
    exact = black76.price(F0, K, T, r, sig)
    hits = 0
    for seed in range(200):
        S = market.gbm(F0, 0.0, sig, T, 1, 2_000, seed=seed).S
        lo, hi = mc.mc_price(mc.european(S, K), r, T).ci95
        hits += lo <= exact <= hi
    assert 0.90 <= hits / 200 <= 0.99


def test_antithetique_reduit_la_variance():
    plain = mc.mc_price(mc.european(market.gbm(F0, 0, sig, T, 1, 100_000, seed=3).S, K), r, T)
    # l'erreur type naïve sous-estime le gain ; on compare la dispersion sur plusieurs graines
    est = [mc.mc_price(mc.european(market.gbm(F0, 0, sig, T, 1, 20_000, seed=s, antithetic=True).S, K), r, T).price
           for s in range(30)]
    naive = [mc.mc_price(mc.european(market.gbm(F0, 0, sig, T, 1, 20_000, seed=100 + s).S, K), r, T).price
             for s in range(30)]
    assert np.std(est) < np.std(naive)
    assert plain.stderr > 0


# --- Exotiques ---------------------------------------------------------------

def test_asiatique_geometrique_egal_formule_fermee():
    n = 63
    S = market.gbm(F0, 0.0, sig, T, n, 200_000, seed=4).S
    exact = cf.asian_geometric(F0, K, T, r, 0.0, sig, n)
    _close(mc.mc_price(mc.asian(S, K, average="geometric"), r, T), exact)


def test_variable_de_controle_reduit_fortement_l_erreur():
    n = 63
    S = market.gbm(F0, 0.0, sig, T, n, 50_000, seed=5).S
    arith = mc.asian(S, K)
    plain = mc.mc_price(arith, r, T)
    cv = mc.mc_price_control_variate(arith, mc.asian(S, K, average="geometric"),
                                     cf.asian_geometric(F0, K, T, r, 0.0, sig, n), r, T)
    assert cv.stderr < plain.stderr / 10
    _close(plain, cv.price, k=4)


def test_asiatique_moins_chere_que_europeenne():
    S = market.gbm(F0, 0.0, sig, T, 63, 50_000, seed=6).S
    assert mc.mc_price(mc.asian(S, K), r, T).price < mc.mc_price(mc.european(S, K), r, T).price


def test_parite_barriere_in_plus_out_egal_europeenne():
    S = market.gbm(F0, 0.0, sig, T, 63, 20_000, seed=7).S
    for B, d in ((360.0, "up"), (250.0, "down")):
        tot = mc.barrier(S, K, B, kind=f"{d}-and-in") + mc.barrier(S, K, B, kind=f"{d}-and-out")
        assert np.allclose(tot, mc.european(S, K))


def test_lookback_toujours_positif_et_plus_cher():
    S = market.gbm(F0, 0.0, sig, T, 63, 20_000, seed=8).S
    lb = mc.lookback(S, "call")
    assert (lb >= 0).all()
    assert lb.mean() > mc.european(S, K).mean()


# --- Sauts --------------------------------------------------------------------

def test_mc_merton_egal_serie_de_merton():
    lam, jm, js = 3.0, -0.05, 0.10
    S = market.merton_jump(F0, 0.0, 0.25, lam, jm, js, T, 20, 300_000, seed=9).S
    for k in (240.0, 300.0, 360.0):
        _close(mc.mc_price(mc.european(S, k), r, T), cf.merton(F0, k, T, r, 0.0, 0.25, lam, jm, js))


def test_merton_sans_saut_redonne_black76():
    assert cf.merton(F0, K, T, r, 0.0, sig, 0.0, 0.0, 0.0) == pytest.approx(black76.price(F0, K, T, r, sig))


def test_les_sauts_creent_un_smile():
    """Vol implicite de Black-76 des prix de Merton : plus haute loin de la monnaie."""
    strikes = np.array([210.0, 300.0, 390.0])
    prices = np.array([cf.merton(F0, k, T, r, 0.0, 0.25, 3.0, 0.0, 0.12, "put" if k < F0 else "call")
                       for k in strikes])
    iv = np.array([black76.implied_vol(p, F0, k, T, r, "put" if k < F0 else "call")
                   for p, k in zip(prices, strikes)])
    assert iv[0] > iv[1] and iv[2] > iv[1]


# --- Grecques Monte-Carlo -------------------------------------------------------

def test_delta_monte_carlo_egal_delta_black76():
    sim = lambda s0, seed: market.gbm(s0, 0.0, sig, T, 1, 400_000, seed=seed).S
    g = mc.mc_greeks(sim, lambda S: mc.european(S, K), r, T, F0)
    assert g["delta"] == pytest.approx(black76.delta(F0, K, T, r, sig), abs=0.01)
