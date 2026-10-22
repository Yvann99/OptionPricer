"""Point d'entrée : appelle toutes les fonctions du package `pricer`.

    python main.py                  # menu interactif
    python main.py --all            # toutes les sections (sauf données réelles)
    python main.py bs mc hedge      # sections choisies
    python main.py data --ticker CC=F
    python main.py --S0 100 --K 110 --T 1 --sigma 0.3 bs b76

Sections :
    bs      Black–Scholes : payoff, d1/d2, prix, grecques, vol implicite
    b76     Black-76 (options sur futures) : prix, grecques, vol implicite
    closed  Formules fermées : BS avec portage, asiatique géométrique, Merton
    sim     Simulateurs : MBG, Heston, régimes, Merton (sauts), Schwartz
    mc      Monte-Carlo : payoffs (européen, digital, asiatique, barrière, lookback),
            variable de contrôle, convergence, grecques par bump
    hedge   Couverture : DeltaHedge, NoHedge, run_hedge, résumé du P&L
    attrib  Attribution du P&L par grecque
    phasea  Phase A : 1/√N, drift, pari sur la vol, vol implicite/réalisée, coûts, modèle faux
    calib   Calibration MBG / Merton / Schwartz sur trajectoires simulées
    data    Données réelles Yahoo Finance (internet au premier lancement)
"""

import argparse
import sys

import numpy as np

from pricer import (DeltaHedge, NoHedge, OptionSpec, attribution, black76, black_scholes as bs,
                    calibration as cal, closed_forms as cf, experiments as ex, market,
                    montecarlo as mc, run_hedge)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _f(x):
    return float(np.asarray(x))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def run_bs(a):
    section(f"BLACK–SCHOLES  S0={a.S0}, K={a.K}, T={a.T}, r={a.r}, sigma={a.sigma}")
    d1, d2 = bs.d1_d2(a.S0, a.K, a.T, a.r, a.sigma)
    print(f"d1, d2                  : {_f(d1):.4f}, {_f(d2):.4f}")
    print(f"Payoff immédiat         : call {_f(bs.payoff(a.S0, a.K, 'call')):.4f}, "
          f"put {_f(bs.payoff(a.S0, a.K, 'put')):.4f}")
    c = _f(bs.price(a.S0, a.K, a.T, a.r, a.sigma, "call"))
    p = _f(bs.price(a.S0, a.K, a.T, a.r, a.sigma, "put"))
    print(f"Prix                    : call {c:.4f}, put {p:.4f}")
    print(f"Parité C - P            : {c - p:.4f}  vs  S - K e^(-rT) = "
          f"{a.S0 - a.K * np.exp(-a.r * a.T):.4f}")
    for opt in ("call", "put"):
        print(f"Grecques {opt:4s}           : "
              f"delta {_f(bs.delta(a.S0, a.K, a.T, a.r, a.sigma, opt)):+.4f}, "
              f"gamma {_f(bs.gamma(a.S0, a.K, a.T, a.r, a.sigma)):.4f}, "
              f"vega {_f(bs.vega(a.S0, a.K, a.T, a.r, a.sigma)):.4f}, "
              f"theta {_f(bs.theta(a.S0, a.K, a.T, a.r, a.sigma, opt)):+.4f}")
    print(f"Vol implicite (du call) : {_f(bs.implied_vol(c, a.S0, a.K, a.T, a.r, 'call')):.4%}")
    strikes = np.linspace(0.8, 1.2, 5) * a.S0
    print("Vectorisé (K -> prix)   :",
          dict(zip(np.round(strikes, 1).tolist(),
                   np.round(bs.price(a.S0, strikes, a.T, a.r, a.sigma, a.option), 4).tolist())))


def run_b76(a):
    F = a.S0
    section(f"BLACK-76  F={F}, K={a.K}, T={a.T}, r={a.r}, sigma={a.sigma}")
    c = _f(black76.price(F, a.K, a.T, a.r, a.sigma, "call"))
    p = _f(black76.price(F, a.K, a.T, a.r, a.sigma, "put"))
    print(f"Prix                    : call {c:.4f}, put {p:.4f}")
    print(f"Delta call / put        : {_f(black76.delta(F, a.K, a.T, a.r, a.sigma, 'call')):+.4f} / "
          f"{_f(black76.delta(F, a.K, a.T, a.r, a.sigma, 'put')):+.4f}")
    print(f"Gamma, vega             : {_f(black76.gamma(F, a.K, a.T, a.r, a.sigma)):.4f}, "
          f"{_f(black76.vega(F, a.K, a.T, a.r, a.sigma)):.4f}")
    print(f"Vol implicite (du call) : {_f(black76.implied_vol(c, F, a.K, a.T, a.r, 'call')):.4%}")


def run_closed(a):
    section("FORMULES FERMÉES")
    print(f"BS avec portage b=r     : {_f(cf.bs_carry(a.S0, a.K, a.T, a.r, a.r, a.sigma, a.option)):.4f}"
          f"  (= Black–Scholes {_f(bs.price(a.S0, a.K, a.T, a.r, a.sigma, a.option)):.4f})")
    print(f"BS avec portage b=0     : {_f(cf.bs_carry(a.S0, a.K, a.T, a.r, 0.0, a.sigma, a.option)):.4f}"
          f"  (= Black-76 {_f(black76.price(a.S0, a.K, a.T, a.r, a.sigma, a.option)):.4f})")
    print(f"Asiatique géom. (252 f.) : {_f(cf.asian_geometric(a.S0, a.K, a.T, a.r, a.r, a.sigma, 252, a.option)):.4f}")
    print(f"Merton (lam=1, -10%/15%) : "
          f"{_f(cf.merton(a.S0, a.K, a.T, a.r, a.r, a.sigma, 1.0, -0.10, 0.15, a.option)):.4f}")


def _simulators(a, n_paths):
    n = int(round(a.T * 252))
    return {
        "MBG": market.gbm(a.S0, a.r, a.sigma, a.T, n, n_paths, seed=1),
        "MBG antithétique": market.gbm(a.S0, a.r, a.sigma, a.T, n, n_paths, seed=1, antithetic=True),
        "Heston": market.heston(a.S0, a.sigma**2, a.r, 2.0, a.sigma**2, 0.5, -0.7, a.T, n, n_paths, seed=2),
        "Régimes": market.regime_switching(a.S0, a.r, [0.12, 0.40], [[0.99, 0.01], [0.05, 0.95]],
                                           a.T, n, n_paths, seed=3),
        "Merton (sauts)": market.merton_jump(a.S0, a.r, a.sigma, 1.0, -0.10, 0.15, a.T, n, n_paths, seed=4),
        "Schwartz": market.schwartz(a.S0, 1.0, np.log(a.S0), a.sigma, a.T, n, n_paths, seed=5),
    }


def run_sim(a):
    section(f"SIMULATEURS  ({a.paths} trajectoires, mu = r)")
    print(f"{'modèle':18s} {'E[e^-rT S_T]':>13s} {'vol réalisée':>13s} {'q05 S_T':>9s} {'q95 S_T':>9s}")
    for name, p in _simulators(a, a.paths).items():
        ST = p.S[:, -1]
        print(f"{name:18s} {np.exp(-a.r * a.T) * ST.mean():13.4f} {p.realized_vol().mean():13.2%} "
              f"{np.quantile(ST, 0.05):9.2f} {np.quantile(ST, 0.95):9.2f}")
    print(f"(martingale : E[e^-rT S_T] doit valoir S0 = {a.S0} sauf pour Schwartz, modèle sous P)")


def run_mc(a):
    n = int(round(a.T * 252))
    section(f"MONTE-CARLO  ({a.paths} trajectoires MBG sous Q)")
    S = market.gbm(a.S0, a.r, a.sigma, a.T, n, a.paths, seed=0).S
    exact = _f(bs.price(a.S0, a.K, a.T, a.r, a.sigma, a.option))
    euro = mc.mc_price(mc.european(S, a.K, a.option), a.r, a.T)
    print(f"Européen                : {euro}   (formule {exact:.4f})")
    print(f"Digital                 : {mc.mc_price(mc.digital(S, a.K, a.option), a.r, a.T)}")
    geo_exact = _f(cf.asian_geometric(a.S0, a.K, a.T, a.r, a.r, a.sigma, n, a.option))
    asian = mc.asian(S, a.K, a.option)
    print(f"Asiatique               : {mc.mc_price(asian, a.r, a.T)}")
    print(f"Asiatique (var. contrôle): "
          f"{mc.mc_price_control_variate(asian, mc.asian(S, a.K, a.option, 'geometric'), geo_exact, a.r, a.T)}")
    B_up, B_down = 1.2 * a.S0, 0.8 * a.S0
    for kind, B in [("up-and-out", B_up), ("up-and-in", B_up), ("down-and-out", B_down), ("down-and-in", B_down)]:
        print(f"Barrière {kind:12s} B={B:6.1f}: {mc.mc_price(mc.barrier(S, a.K, B, a.option, kind), a.r, a.T)}")
    print(f"Lookback                : {mc.mc_price(mc.lookback(S, a.option), a.r, a.T)}")

    n_list = [1_000, 4_000, 16_000, 64_000]
    conv = mc.convergence(lambda m, seed: market.gbm(a.S0, a.r, a.sigma, a.T, n, m, seed=seed).S,
                          lambda X: mc.european(X, a.K, a.option), a.r, a.T, n_list)
    print("Convergence (N -> err.) :", {m: round(c.stderr, 4) for m, c in zip(n_list, conv)})

    g = mc.mc_greeks(lambda s0, seed: market.gbm(s0, a.r, a.sigma, a.T, n, a.paths, seed=seed).S,
                     lambda X: mc.european(X, a.K, a.option), a.r, a.T, a.S0)
    print(f"Grecques MC (bump)      : delta {g['delta']:+.4f} (BS {_f(bs.delta(a.S0, a.K, a.T, a.r, a.sigma, a.option)):+.4f}), "
          f"gamma {g['gamma']:.4f} (BS {_f(bs.gamma(a.S0, a.K, a.T, a.r, a.sigma)):.4f})")


def _hedge_setup(a):
    n = int(round(a.T * 252))
    paths = market.gbm(a.S0, 0.08, a.sigma, a.T, n, min(a.paths, 20_000), seed=0)
    return paths, OptionSpec(a.K, a.T, a.option)


def run_hedge_section(a):
    paths, opt = _hedge_setup(a)
    section(f"COUVERTURE  (vendeur d'un {a.option}, vol réalisée = vol de vente = {a.sigma:.0%})")
    strategies = {
        "Sans couverture": dict(strategy=NoHedge()),
        "Delta quotidien": dict(strategy=DeltaHedge(a.sigma)),
        "Delta hebdo": dict(strategy=DeltaHedge(a.sigma, every=5)),
        "Delta quotidien + coûts": dict(strategy=DeltaHedge(a.sigma), cost_rate=0.001),
        "Vendu à vol +5 pts": dict(sigma_price=a.sigma + 0.05),
    }
    print(f"{'stratégie':26s} {'prime':>7s} {'moyenne':>8s} {'écart-t':>8s} {'CVaR5%':>8s} {'coûts':>7s} {'trades':>7s}")
    for name, kw in strategies.items():
        kw = {"sigma_price": a.sigma, **kw}
        s = run_hedge(paths, opt, a.r, **kw).summary()
        print(f"{name:26s} {s['premium']:7.3f} {s['mean']:+8.3f} {s['std']:8.3f} {s['cvar05']:+8.3f} "
              f"{s['mean_costs']:7.3f} {s['mean_trades']:7.1f}")


def run_attrib(a):
    paths, opt = _hedge_setup(a)
    section(f"ATTRIBUTION  (vendu à {a.sigma + 0.05:.0%}, vol réalisée {a.sigma:.0%}, coûts 5 pb)")
    res = run_hedge(paths, opt, a.r, sigma_price=a.sigma + 0.05, cost_rate=0.0005)
    for k, (m, s) in attribution.summarize(attribution.attribute(res)).items():
        print(f"   {k:9s} {m:+8.4f}  (écart-type {s:.4f})")


def run_phase_a(a):
    n, m = int(round(a.T * 252)), min(a.paths, 10_000)
    opt = OptionSpec(a.K, a.T, a.option)
    section(f"PHASE A : ANATOMIE DU P&L  ({m} trajectoires, vol {a.sigma:.0%})")
    lam, jm, js = 2.0, -0.08, 0.05
    sims = {
        "MBG (BS)": lambda k, s: market.gbm(a.S0, 0.05, a.sigma, a.T, k, m, seed=s),
        "Heston": lambda k, s: market.heston(a.S0, a.sigma**2, 0.05, 2.0, a.sigma**2, 0.6, -0.7,
                                             a.T, k, m, seed=s),
        "Merton (sauts)": lambda k, s: market.merton_jump(
            a.S0, 0.05, np.sqrt(max(a.sigma**2 - lam * (jm**2 + js**2), 1e-4)), lam, jm, js, a.T, k, m, seed=s),
    }
    Ns = [10, 50, 250]
    print("1. Écart-type du P&L pour N =", Ns)
    for name, sim in sims.items():
        err = ex.error_vs_n(sim, opt, a.r, a.sigma, Ns)
        print(f"   {name:15s} {np.round(err, 3)}  pente {ex.slope_loglog(Ns, err):+.2f}")

    d = ex.drift_independence(lambda mu, k, s: market.gbm(a.S0, mu, a.sigma, a.T, k, m, seed=s),
                              opt, a.r, a.sigma, [-0.2, 0.2], n_steps=n)
    print(f"2. Drift -20 % / +20 %   : couvert {d['hedged'][0]['mean']:+.3f} / {d['hedged'][1]['mean']:+.3f}, "
          f"non couvert {d['naked'][0]['mean']:+.3f} / {d['naked'][1]['mean']:+.3f}")

    s_impl, s_real = a.sigma + 0.05, a.sigma - 0.05
    p = market.gbm(a.S0, 0.05, s_real, a.T, n, m, seed=3)
    v = ex.vol_bet(p, opt, a.r, s_impl)
    print(f"3. Vendu {s_impl:.0%}, réalisé {s_real:.0%} : P&L moyen {v['pnl'].mean():+.3f}, "
          f"théorie {v['theory'].mean():+.3f}, corrélation {np.corrcoef(v['pnl'], v['theory'])[0, 1]:.4f}")

    iv = ex.implied_vs_realized(p, opt, a.r, s_impl, s_real)
    print(f"4. Gain verrouillé {iv['locked_in']:.3f} | couvert à σ_impl : écart-t {iv['implied'].pnl.std():.3f}"
          f" | couvert à σ_réal : écart-t {iv['realized'].pnl.std():.3f}")

    print("5. Coûts 20 pb, rééquilibrage tous les … jours :")
    for x in ex.cost_tradeoff(sims["MBG (BS)"](n, 5), opt, a.r, a.sigma, 0.002, [1, 5, 25]):
        print(f"   {x['every']:3d} j  moyenne {x['mean']:+.3f}  écart-t {x['std']:.3f}")

    print("6. Modèle faux (couverture BS à vol constante) :")
    for name, s in ex.model_comparison({k: f(n, 6) for k, f in sims.items()}, opt, a.r, a.sigma).items():
        print(f"   {name:15s} écart-t {s['std']:.3f}  CVaR5% {s['cvar05']:+.3f}  skew {s['skew']:+.2f}")


def run_calib(a):
    section("CALIBRATION  (sur une trajectoire simulée de 20 ans : on doit retrouver les paramètres)")
    T, n = 20.0, 20 * 252
    g = cal.fit_gbm(market.gbm(100, 0.05, 0.25, T, n, 1, seed=0).log_returns()[0])
    print(f"MBG      vrai mu=5%, sigma=25%                 -> mu={g.mu:.1%}, sigma={g.sigma:.1%}")
    m = cal.fit_merton(market.merton_jump(100, 0.05, 0.20, 3.0, -0.08, 0.04, T, n, 1, seed=1).log_returns()[0])
    print(f"Merton   vrai sigma=20%, lam=3, saut -8%±4%   -> sigma={m.sigma:.1%}, lam={m.lam:.1f}, "
          f"saut {m.jump_mean:+.1%}±{m.jump_std:.1%}, vol totale {m.total_vol:.1%}")
    s = cal.fit_schwartz(market.schwartz(100, 1.5, np.log(120), 0.30, T, n, 1, seed=2).S[0])
    print(f"Schwartz vrai kappa=1.5, niveau 120, sigma=30% -> kappa={s.kappa:.2f}, "
          f"niveau {s.long_run_price:.0f}, sigma={s.sigma:.1%}, demi-vie {s.half_life_years:.2f} ans")


def run_data(a):
    try:
        from pricer import data  # nécessite pandas et yfinance
    except ImportError as e:
        print(f"\nSection 'data' indisponible ({e}). Installer : pip install -r requirements.txt")
        return

    df = data.load(a.ticker, start=a.start, refresh=a.refresh)
    close = df["Close"]
    ret = data.log_returns(close)
    section(f"DONNÉES RÉELLES  {a.ticker}, {close.index[0].date()} -> {close.index[-1].date()}")
    print(f"Dernier prix            : {close.iloc[-1]:.2f}")
    print(f"Taux sans risque        : {data.risk_free_rate():.2%}")
    print(f"Vol réalisée 1 mois     : {data.realized_vol(ret).iloc[-1]:.1%}")
    if {"High", "Low"} <= set(df.columns):
        print(f"Vol de Parkinson 1 mois : {data.parkinson_vol(df).iloc[-1]:.1%}")
    for k, v in data.return_stats(ret).items():
        print(f"   {k:18s} {v:.4f}" if isinstance(v, float) else f"   {k:18s} {v}")
    print(f"Jours de saut (> 4 σ)   : {int(data.detect_jumps(ret).sum())}")
    print("Vol par mois            :", {int(k): round(v, 3) for k, v in data.vol_by_month(ret).items()})
    g, m, s = cal.fit_gbm(ret), cal.fit_merton(ret), cal.fit_schwartz(close)
    print(f"Calibration MBG         : sigma {g.sigma:.1%}")
    print(f"Calibration Merton      : sigma {m.sigma:.1%}, {m.lam:.1f} sauts/an, vol totale {m.total_vol:.1%}")
    print(f"Calibration Schwartz    : demi-vie {s.half_life_years:.1f} ans, niveau {s.long_run_price:.0f}")


SECTIONS = {
    "bs": ("Black–Scholes", run_bs),
    "b76": ("Black-76 (futures)", run_b76),
    "closed": ("Formules fermées", run_closed),
    "sim": ("Simulateurs de marché", run_sim),
    "mc": ("Monte-Carlo et exotiques", run_mc),
    "hedge": ("Couverture dynamique", run_hedge_section),
    "attrib": ("Attribution du P&L", run_attrib),
    "phasea": ("Phase A : anatomie du P&L", run_phase_a),
    "calib": ("Calibration", run_calib),
    "data": ("Données réelles (internet)", run_data),
}


# ---------------------------------------------------------------------------
# Menu et ligne de commande
# ---------------------------------------------------------------------------

def menu(a):
    keys = list(SECTIONS)
    while True:
        print("\n=== PricerOption ===")
        for i, k in enumerate(keys, 1):
            print(f"  {i}. {SECTIONS[k][0]}")
        print("  a. Tout (sauf données réelles)\n  q. Quitter")
        choice = input("Choix : ").strip().lower()
        if choice in ("q", ""):
            return
        if choice == "a":
            selected = [k for k in keys if k != "data"]
        elif choice.isdigit() and 1 <= int(choice) <= len(keys):
            selected = [keys[int(choice) - 1]]
        elif choice in SECTIONS:
            selected = [choice]
        else:
            print("Choix invalide.")
            continue
        for k in selected:
            SECTIONS[k][1](a)


def main():
    ap = argparse.ArgumentParser(description="PricerOption : appelle toutes les fonctions du package.")
    ap.add_argument("sections", nargs="*", metavar="SECTION",
                    help=f"parmi : {', '.join(SECTIONS)} (aucune = menu interactif)")
    ap.add_argument("--all", action="store_true", help="toutes les sections sauf 'data'")
    ap.add_argument("--S0", type=float, default=100.0, help="spot (ou future pour Black-76)")
    ap.add_argument("--K", type=float, default=100.0, help="strike")
    ap.add_argument("--T", type=float, default=0.5, help="maturité en années")
    ap.add_argument("--r", type=float, default=0.02, help="taux sans risque")
    ap.add_argument("--sigma", type=float, default=0.2, help="volatilité")
    ap.add_argument("--option", choices=["call", "put"], default="call")
    ap.add_argument("--paths", type=int, default=50_000, help="trajectoires Monte-Carlo")
    ap.add_argument("--ticker", default="KC=F", help="ticker Yahoo pour 'data'")
    ap.add_argument("--start", default="2010-01-01", help="début de l'historique pour 'data'")
    ap.add_argument("--refresh", action="store_true", help="retélécharger les données")
    a = ap.parse_args()
    unknown = [k for k in a.sections if k not in SECTIONS]
    if unknown:
        ap.error(f"section(s) inconnue(s) : {', '.join(unknown)} (choix : {', '.join(SECTIONS)})")

    if a.all:
        selected = [k for k in SECTIONS if k != "data"]
    elif a.sections:
        selected = a.sections
    else:
        menu(a)
        return
    for k in selected:
        SECTIONS[k][1](a)


if __name__ == "__main__":
    sys.exit(main())
