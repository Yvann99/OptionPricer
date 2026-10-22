import numpy as np
import pytest

from pricer import OptionSpec, experiments as ex, market

S0, K, T, r, sig = 100.0, 100.0, 0.5, 0.02, 0.2
opt = OptionSpec(K, T, "call")
N_PATHS = 4_000
Ns = [25, 100, 400]


def _gbm(n, seed):
    return market.gbm(S0, 0.05, sig, T, n, N_PATHS, seed=seed)


def _merton(n, seed):
    # vol totale = 20 % : sigma diffusion² + lam (m² + s²) = sig²
    lam, m, s = 2.0, -0.08, 0.05
    return market.merton_jump(S0, 0.05, np.sqrt(sig**2 - lam * (m**2 + s**2)), lam, m, s,
                              T, n, N_PATHS, seed=seed)


def test_error_slope_is_minus_half_in_black_scholes_world():
    errors = ex.error_vs_n(_gbm, opt, r, sig, Ns)
    assert ex.slope_loglog(Ns, errors) == pytest.approx(-0.5, abs=0.07)


def test_error_plateaus_when_model_is_wrong():
    errors = ex.error_vs_n(_merton, opt, r, sig, Ns)
    # 16x plus de rééquilibrages, mais l'erreur ne baisse presque pas (au lieu de /4)
    assert errors[-1] / errors[0] > 0.75


def test_hedged_seller_ignores_drift_naked_does_not():
    d = ex.drift_independence(lambda mu, n, s: market.gbm(S0, mu, sig, T, n, N_PATHS, seed=s),
                              opt, r, sig, [-0.2, 0.4], n_steps=100)
    hedged = [x["mean"] for x in d["hedged"]]
    naked = [x["mean"] for x in d["naked"]]
    assert abs(hedged[1] - hedged[0]) < 0.1
    assert naked[0] - naked[1] > 10


def test_pnl_is_a_bet_on_quadratic_variation():
    p = market.gbm(S0, 0.05, 0.15, T, 250, N_PATHS, seed=3)
    v = ex.vol_bet(p, opt, r, sigma_impl=0.25)
    assert np.corrcoef(v["pnl"], v["theory"])[0, 1] > 0.99
    assert v["pnl"].mean() == pytest.approx(v["theory"].mean(), rel=0.05)


def test_hedging_at_realized_vol_locks_in_the_pnl():
    p = market.gbm(S0, 0.05, 0.15, T, 250, N_PATHS, seed=4)
    out = ex.implied_vs_realized(p, opt, r, sigma_impl=0.25, sigma_real=0.15)
    implied, realized = out["implied"].pnl, out["realized"].pnl
    # même gain moyen, mais certain (au bruit de discrétisation près) à la vol réalisée
    assert realized.mean() == pytest.approx(out["locked_in"], rel=0.02)
    assert implied.mean() == pytest.approx(out["locked_in"], rel=0.05)
    assert realized.std() < 0.4 * implied.std()
    # ... au prix d'un livre bruité en cours de route
    mid = p.n_steps // 2
    assert out["realized"].book_value[:, mid].std() > out["implied"].book_value[:, mid].std()


def test_costs_trade_precision_for_cost():
    p = market.gbm(S0, 0.05, sig, T, 250, N_PATHS, seed=5)
    rows = ex.cost_tradeoff(p, opt, r, sig, cost_rate=0.002, everys=[1, 5, 25])
    costs = [x["mean_costs"] for x in rows]
    stds = [x["std"] for x in rows]
    assert costs[0] > costs[1] > costs[2]   # rééquilibrer souvent coûte cher
    assert stds[0] < stds[1] < stds[2]      # mais couvre mieux


def test_jumps_give_a_fat_left_tail():
    out = ex.model_comparison({"gbm": _gbm(250, 6), "merton": _merton(250, 6)}, opt, r, sig)
    assert out["merton"]["skew"] < out["gbm"]["skew"] - 1
    assert out["merton"]["cvar05"] < 3 * out["gbm"]["cvar05"]
