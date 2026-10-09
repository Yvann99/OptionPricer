import numpy as np
import pytest

from pricer import market

S0, mu, T = 100.0, 0.05, 1.0


def _check_mean(ST, target, k=4):
    se = ST.std() / np.sqrt(ST.size)
    assert abs(ST.mean() - target) < k * se


def test_gbm_moments_and_martingale():
    p = market.gbm(S0, mu, 0.2, T, 50, 200_000, seed=1)
    assert p.S.shape == (200_000, 51)
    _check_mean(p.S[:, -1], S0 * np.exp(mu * T))
    # variance du log : sigma² T
    assert np.var(np.log(p.S[:, -1] / S0)) == pytest.approx(0.04, rel=0.02)
    # sous mu = r, le prix actualisé est une martingale : E[e^{-rt} S_t] = S0 pour tout t
    q = market.gbm(S0, 0.03, 0.2, T, 4, 200_000, seed=2)
    disc = np.exp(-0.03 * q.t) * q.S
    for j in range(1, 5):
        _check_mean(disc[:, j], S0)


def test_realized_vol_is_quadratic_variation():
    p = market.gbm(S0, mu, 0.3, T, 10_000, 200, seed=3)
    assert np.mean(p.realized_vol()) == pytest.approx(0.3, rel=0.01)


def test_heston_reduces_to_gbm_without_vol_of_vol():
    p = market.heston(S0, 0.04, mu, kappa=2.0, theta=0.04, xi=0.0, rho=-0.7,
                      T=T, n_steps=50, n_paths=100_000, seed=4)
    assert np.allclose(p.extra["v"], 0.04)
    assert np.var(np.log(p.S[:, -1] / S0)) == pytest.approx(0.04, rel=0.03)
    _check_mean(p.S[:, -1], S0 * np.exp(mu * T))


def test_heston_mean():
    p = market.heston(S0, 0.09, mu, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
                      T=T, n_steps=200, n_paths=100_000, seed=5)
    _check_mean(p.S[:, -1], S0 * np.exp(mu * T))
    assert (p.extra["v"] >= 0).all()


def test_regime_switching():
    P = np.array([[0.99, 0.01], [0.05, 0.95]])
    p = market.regime_switching(S0, mu, [0.1, 0.4], P, T, 2520, 2000, seed=6)
    st = p.extra["state"]
    assert set(np.unique(st)) <= {0, 1}
    # fréquence stationnaire du régime 1 : 0.01 / (0.01 + 0.05) = 1/6
    assert st[:, 500:].mean() == pytest.approx(1 / 6, abs=0.02)
    _check_mean(p.S[:, -1], S0 * np.exp(mu * T))
    # génératrice acceptée aussi
    Q = np.array([[-2.0, 2.0], [10.0, -10.0]])
    market.regime_switching(S0, mu, [0.1, 0.4], Q, T, 252, 10, seed=7)


def test_merton_mean_is_compensated():
    p = market.merton_jump(S0, mu, 0.15, lam=3.0, jump_mean=-0.1, jump_std=0.15,
                           T=T, n_steps=100, n_paths=300_000, seed=8)
    _check_mean(p.S[:, -1], S0 * np.exp(mu * T))
    assert p.extra["n_jumps"].sum(axis=1).mean() == pytest.approx(3.0, rel=0.02)
