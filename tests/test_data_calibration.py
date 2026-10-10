"""Tests des statistiques et de la calibration.

On ne télécharge rien ici (les tests doivent marcher hors ligne) : on fabrique
des séries dont on CONNAÎT les paramètres, et on vérifie qu'on les retrouve.
"""

import numpy as np
import pandas as pd
import pytest

from pricer import calibration as cal, market

data = pytest.importorskip("pricer.data")  # nécessite pandas


def _series(paths):
    idx = pd.bdate_range("2010-01-01", periods=paths.S.shape[1])
    return pd.Series(paths.S[0], index=idx)


def test_vol_realisee_retrouve_sigma():
    s = _series(market.gbm(300, 0.0, 0.3, 20, 5040, 1, seed=1))
    rv = data.realized_vol(data.log_returns(s), window=252).dropna()
    assert rv.mean() == pytest.approx(0.3, rel=0.05)


def test_stats_loi_normale_sans_queues_epaisses():
    s = _series(market.gbm(300, 0.0, 0.3, 40, 10_080, 1, seed=2))
    st = data.return_stats(data.log_returns(s))
    assert st["vol_annual"] == pytest.approx(0.3, rel=0.03)
    assert abs(st["excess_kurtosis"]) < 0.3


def test_sauts_donnent_des_queues_epaisses_et_sont_detectes():
    s = _series(market.merton_jump(300, 0.0, 0.2, 5.0, 0.0, 0.08, 40, 10_080, 1, seed=3))
    r = data.log_returns(s)
    assert data.return_stats(r)["excess_kurtosis"] > 2
    assert data.detect_jumps(r).sum() > 50


def test_vol_par_mois_a_12_valeurs():
    s = _series(market.gbm(300, 0.0, 0.3, 10, 2520, 1, seed=4))
    v = data.vol_by_month(data.log_returns(s))
    assert list(v.index) == list(range(1, 13))


def test_fit_gbm():
    p = market.gbm(300, 0.05, 0.3, 40, 10_080, 1, seed=5)
    est = cal.fit_gbm(np.diff(np.log(p.S[0])))
    assert est.sigma == pytest.approx(0.3, rel=0.03)


def test_fit_merton_retrouve_les_sauts():
    p = market.merton_jump(300, 0.0, 0.2, 4.0, -0.05, 0.10, 100, 25_200, 1, seed=6)
    est = cal.fit_merton(np.diff(np.log(p.S[0])))
    assert est.sigma == pytest.approx(0.2, rel=0.1)
    assert est.lam == pytest.approx(4.0, rel=0.5)          # seuillage : ordre de grandeur
    assert est.total_vol == pytest.approx(np.sqrt(0.04 + 4 * (0.05**2 + 0.1**2)), rel=0.1)


def test_fit_schwartz_retrouve_le_retour_a_la_moyenne():
    p = market.schwartz(300, kappa=2.0, theta=np.log(200), sigma=0.3, T=100, n_steps=25_200,
                        n_paths=1, seed=7)
    est = cal.fit_schwartz(p.S[0])
    assert est.kappa == pytest.approx(2.0, rel=0.3)
    assert est.long_run_price == pytest.approx(200, rel=0.1)
    assert est.sigma == pytest.approx(0.3, rel=0.05)
