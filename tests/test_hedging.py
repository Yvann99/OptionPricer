import numpy as np
import pytest

from pricer import DeltaHedge, NoHedge, OptionSpec, attribution, market, run_hedge

S0, K, T, r, sig = 100.0, 100.0, 0.5, 0.02, 0.2
opt = OptionSpec(K, T, "call")


def _std(n_steps, seed=0):
    p = market.gbm(S0, 0.08, sig, T, n_steps, 20_000, seed=seed)
    return run_hedge(p, opt, r, sig).pnl


def test_hedging_error_scales_like_one_over_sqrt_n():
    s1, s4 = _std(25).std(), _std(100).std()
    # 4x plus de rééquilibrages -> erreur divisée par environ 2
    assert s1 / s4 == pytest.approx(2.0, rel=0.1)


def test_hedge_is_fair_and_independent_of_drift():
    for mu in (-0.2, 0.3):
        p = market.gbm(S0, mu, sig, T, 200, 20_000, seed=1)
        res = run_hedge(p, opt, r, sig)
        assert abs(res.pnl.mean()) < 0.05          # pas de gain espéré
        assert res.pnl.std() < 0.1 * res.premium     # le risque a quasi disparu


def test_unhedged_seller_is_much_riskier():
    p = market.gbm(S0, 0.08, sig, T, 100, 20_000, seed=2)
    hedged = run_hedge(p, opt, r, sig)
    naked = run_hedge(p, opt, r, sig, strategy=NoHedge())
    assert naked.pnl.std() > 5 * hedged.pnl.std()


def test_selling_vol_above_realized_is_profitable():
    p = market.gbm(S0, 0.05, 0.15, T, 250, 10_000, seed=3)
    res = run_hedge(p, opt, r, sigma_price=0.25)
    assert res.pnl.mean() > 0
    # ordre de grandeur : ½ ∫ Γ S² (σi² - σr²) dt ≈ vega (σi - σr)
    from pricer import black_scholes as bs
    approx = bs.vega(S0, K, T, r, 0.2) * (0.25 - 0.15)
    assert res.pnl.mean() == pytest.approx(approx, rel=0.25)


def test_costs_reduce_pnl_and_rebalancing_every():
    p = market.gbm(S0, 0.05, sig, T, 200, 5_000, seed=4)
    free = run_hedge(p, opt, r, sig)
    costly = run_hedge(p, opt, r, sig, cost_rate=0.001)
    assert costly.pnl.mean() < free.pnl.mean()
    assert np.allclose(free.pnl - costly.pnl, costly.costs.sum(axis=1)
                       * 1.0, rtol=0.05)  # au portage près
    weekly = run_hedge(p, opt, r, sig, strategy=DeltaHedge(sig, every=5), cost_rate=0.001)
    assert weekly.n_trades.mean() == pytest.approx(40, abs=2)
    assert weekly.costs.sum(axis=1).mean() < costly.costs.sum(axis=1).mean()


def test_attribution_adds_up():
    p = market.gbm(S0, 0.05, 0.3, T, 100, 2_000, seed=5)
    res = run_hedge(p, opt, r, sigma_price=0.2, cost_rate=0.0005)
    a = attribution.attribute(res)
    # la somme des composantes redonne exactement la variation du livre
    parts = sum(a[k] for k in attribution.COMPONENTS)
    assert np.allclose(parts, a["total"])
    # et le P&L final = valeur initiale du livre + somme des variations
    assert np.allclose(res.book_value[:, 0] + a["total"].sum(axis=1), res.pnl)
    # couverture en delta tous les pas : composante delta nulle
    assert np.allclose(a["delta"], 0.0)
    # gamma + theta + carry ≈ terme de vol théorique
    gtc = (a["gamma"] + a["theta"] + a["carry"]).sum(axis=1)
    assert np.corrcoef(gtc, a["vol"].sum(axis=1))[0, 1] > 0.95


def test_strategy_cannot_see_future():
    seen = []

    class Spy:
        def __call__(self, i, ctx):
            seen.append(ctx.S_hist.shape[1])
            return np.zeros_like(ctx.S)

    p = market.gbm(S0, 0.05, sig, T, 10, 3, seed=6)
    run_hedge(p, opt, r, sig, strategy=Spy())
    assert seen == list(range(1, 11))
