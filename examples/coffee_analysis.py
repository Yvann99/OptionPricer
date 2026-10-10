"""Analyse d'options sur le CAFÉ (future Coffee C, ICE) avec données réelles.

    python examples/coffee_analysis.py                  # café depuis 2010
    python examples/coffee_analysis.py --ticker CC=F    # cacao
    python examples/coffee_analysis.py --refresh        # retélécharger les données

Nécessite internet au premier lancement (ensuite, cache dans data/).

Déroulé
-------
1. Le marché : prix, volatilité réalisée, saisonnalité, queues épaisses.
2. Calibration : MBG, Merton (sauts), Schwartz (retour à la moyenne).
3. Pricing d'un call 3 mois à la monnaie :
   Black-76 (formule) vs Monte-Carlo, et convergence en 1/sqrt(N).
4. Les sauts créent un SMILE de volatilité.
5. Options exotiques pour un torréfacteur : asiatique, barrière, lookback.
6. Scénarios de prix à 1 an (monde réel) : MBG vs retour à la moyenne.

Les figures sont enregistrées dans figures/<nom>_market.png et figures/<nom>_pricing.png.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pricer import black76, calibration as cal, closed_forms as cf, data, market, montecarlo as mc  # noqa: E402

NAMES = {"KC=F": "Café", "CC=F": "Cacao", "SB=F": "Sucre", "ZW=F": "Blé", "GC=F": "Or"}


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="KC=F")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--maturity", type=float, default=0.25, help="maturité en années")
    ap.add_argument("--paths", type=int, default=200_000)
    args = ap.parse_args()

    name = NAMES.get(args.ticker, args.ticker)
    tag = args.ticker.replace("=", "").lower()
    figdir = ROOT / "figures"
    figdir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Le marché
    # ------------------------------------------------------------------
    df = data.load(args.ticker, start=args.start, refresh=args.refresh)
    close = df["Close"]
    ret = data.log_returns(close)
    r = data.risk_free_rate()
    F0 = float(close.iloc[-1])
    rv21 = data.realized_vol(ret, 21)
    rv63 = data.realized_vol(ret, 63)
    vol_1y = float(np.sqrt((ret.iloc[-252:] ** 2).mean() * data.TRADING_DAYS))

    section(f"1. MARCHÉ — {name} ({args.ticker}), {close.index[0].date()} → {close.index[-1].date()}")
    print(f"Dernier prix F0         : {F0:.2f}")
    print(f"Taux sans risque r      : {r:.2%}")
    print(f"Vol réalisée 1 mois     : {rv21.iloc[-1]:.1%}")
    print(f"Vol réalisée 1 an       : {vol_1y:.1%}")
    st = data.return_stats(ret)
    print(f"Vol historique totale   : {st['vol_annual']:.1%}")
    print(f"Asymétrie (skew)        : {st['skew']:+.2f}   (0 pour une loi normale)")
    print(f"Kurtosis en excès       : {st['excess_kurtosis']:+.2f}   (0 pour une loi normale)")
    print(f"Pire / meilleur jour    : {st['worst_day']:+.1%} / {st['best_day']:+.1%}")
    print(f"Jours au-delà de 4 σ    : {st['share_beyond_4sd']:.3%}   (loi normale : 0,006 %)")
    vm = data.vol_by_month(ret)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].plot(close.index, close, lw=0.8)
    ax[0, 0].set(title=f"{name} : prix du future continu", ylabel="prix")
    ax[0, 1].plot(rv21.index, rv21, lw=0.6, alpha=0.6, label="1 mois")
    ax[0, 1].plot(rv63.index, rv63, lw=1.2, label="3 mois")
    ax[0, 1].axhline(st["vol_annual"], color="k", ls="--", lw=0.8, label="moyenne")
    ax[0, 1].set(title="Volatilité réalisée (variation quadratique glissante)", ylabel="vol annualisée")
    ax[0, 1].legend()
    months = ["jan", "fév", "mar", "avr", "mai", "juin", "juil", "août", "sep", "oct", "nov", "déc"]
    ax[1, 0].bar(range(12), vm.values, color="C2")
    ax[1, 0].set_xticks(range(12))
    ax[1, 0].set_xticklabels(months)
    ax[1, 0].axhline(st["vol_annual"], color="k", ls="--", lw=0.8)
    ax[1, 0].set(title="Saisonnalité : volatilité par mois", ylabel="vol annualisée")
    z = (ret - ret.mean()) / ret.std()
    bins = np.linspace(-8, 8, 121)
    ax[1, 1].hist(z, bins=bins, density=True, alpha=0.6, label="rendements réels")
    ax[1, 1].plot(bins, norm.pdf(bins), "k", lw=1, label="loi normale (Black–Scholes)")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylim(1e-5, 1)
    ax[1, 1].set(title="Queues épaisses (échelle log)", xlabel="rendement en écarts-types")
    ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(figdir / f"{tag}_market.png", dpi=120)

    # ------------------------------------------------------------------
    # 2. Calibration
    # ------------------------------------------------------------------
    gbm_p = cal.fit_gbm(ret)
    mer = cal.fit_merton(ret)
    sch = cal.fit_schwartz(close)
    section("2. CALIBRATION (sur tout l'historique)")
    print(f"MBG      : sigma = {gbm_p.sigma:.1%}")
    print(f"Merton   : sigma continue = {mer.sigma:.1%}, {mer.lam:.1f} sauts/an, "
          f"taille moyenne {mer.jump_mean:+.1%}, dispersion {mer.jump_std:.1%}")
    print(f"           vol totale = {mer.total_vol:.1%}  "
          f"(part due aux sauts : {1 - mer.sigma**2 / mer.total_vol**2:.0%} de la variance)")
    print(f"Schwartz : demi-vie d'un écart = {sch.half_life_years:.1f} ans, "
          f"niveau de long terme = {sch.long_run_price:.0f}, sigma = {sch.sigma:.1%}")

    # Pour comparer à armes égales, le MBG reçoit la MÊME variance totale que Merton :
    # la seule différence entre les deux est la forme de la distribution (les queues).
    sigma = mer.total_vol
    T = args.maturity
    n_steps = int(round(T * data.TRADING_DAYS))
    K = F0

    # ------------------------------------------------------------------
    # 3. Pricing du call à la monnaie : formule vs Monte-Carlo
    # ------------------------------------------------------------------
    section(f"3. CALL À LA MONNAIE, maturité {T:g} an, K = {K:.2f}, sigma = {sigma:.1%}")
    exact = float(black76.price(F0, K, T, r, sigma))
    S_gbm = market.gbm(F0, 0.0, sigma, T, n_steps, args.paths, seed=1).S   # mu = 0 : future sous Q
    print(f"Black-76 (formule)      : {exact:.4f}")
    print(f"Monte-Carlo MBG         : {mc.mc_price(mc.european(S_gbm, K), r, T)}")
    S_mer = market.merton_jump(F0, 0.0, mer.sigma, mer.lam, mer.jump_mean, mer.jump_std,
                               T, n_steps, args.paths, seed=2).S
    mer_exact = float(cf.merton(F0, K, T, r, 0.0, mer.sigma, mer.lam, mer.jump_mean, mer.jump_std))
    print(f"Merton (série exacte)   : {mer_exact:.4f}")
    print(f"Monte-Carlo Merton      : {mc.mc_price(mc.european(S_mer, K), r, T)}")
    print(f"Delta Black-76          : {float(black76.delta(F0, K, T, r, sigma)):.3f} contrat future par option")

    n_list = np.unique(np.logspace(2, np.log10(args.paths), 25).astype(int))
    conv = mc.convergence(lambda n, s: S_gbm[:n], lambda S: mc.european(S, K), r, T, n_list)

    # ------------------------------------------------------------------
    # 4. Smile créé par les sauts
    # ------------------------------------------------------------------
    moneyness = np.linspace(0.7, 1.3, 25)
    strikes = moneyness * F0
    iv = []
    for k in strikes:
        opt = "put" if k < F0 else "call"   # on utilise l'option hors de la monnaie, plus stable
        p = cf.merton(F0, k, T, r, 0.0, mer.sigma, mer.lam, mer.jump_mean, mer.jump_std, opt)
        iv.append(float(black76.implied_vol(p, F0, k, T, r, opt)))
    iv = np.array(iv)
    section("4. SMILE IMPLICITE DU MODÈLE DE MERTON")
    for m_, v in zip(moneyness[::6], iv[::6]):
        print(f"K = {m_:.0%} de F0  ->  vol implicite {v:.1%}")

    # ------------------------------------------------------------------
    # 5. Exotiques pour un torréfacteur
    # ------------------------------------------------------------------
    section("5. OPTIONS EXOTIQUES (MBG, mêmes trajectoires)")
    S_ex = S_gbm[:100_000]
    geo_exact = float(cf.asian_geometric(F0, K, T, r, 0.0, sigma, n_steps))
    asian_plain = mc.mc_price(mc.asian(S_ex, K), r, T)
    asian_cv = mc.mc_price_control_variate(mc.asian(S_ex, K), mc.asian(S_ex, K, average="geometric"),
                                           geo_exact, r, T)
    B = 1.25 * F0
    products = {
        "Européen": mc.mc_price(mc.european(S_ex, K), r, T),
        "Asiatique": asian_cv,
        f"Up-and-out B={B:.0f}": mc.mc_price(mc.barrier(S_ex, K, B, kind="up-and-out"), r, T),
        "Lookback": mc.mc_price(mc.lookback(S_ex, "call"), r, T),
    }
    for k_, v in products.items():
        print(f"{k_:24s}: {v}")
    print(f"Variable de contrôle (asiatique) : erreur type {asian_plain.stderr:.4f} -> {asian_cv.stderr:.4f}, "
          f"soit {(asian_plain.stderr / asian_cv.stderr) ** 2:.0f}x moins de trajectoires nécessaires")

    # ------------------------------------------------------------------
    # 6. Scénarios à 1 an (monde réel)
    # ------------------------------------------------------------------
    H = 1.0
    sc_gbm = market.gbm(F0, 0.0, gbm_p.sigma, H, 252, 20_000, seed=3)
    sc_sch = market.schwartz(F0, max(sch.kappa, 1e-6), sch.theta, sch.sigma, H, 252, 20_000, seed=3)

    # ------------------------------------------------------------------
    # Figure pricing
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    p_ = np.array([c.price for c in conv])
    e_ = np.array([1.96 * c.stderr for c in conv])
    a = ax[0, 0]
    a.semilogx(n_list, p_, "o-", ms=3, label="Monte-Carlo")
    a.fill_between(n_list, p_ - e_, p_ + e_, alpha=0.25, label="IC 95 %")
    a.axhline(exact, color="k", ls="--", label="Black-76")
    a.set(title="3. Convergence du Monte-Carlo", xlabel="nombre de trajectoires N", ylabel="prix")
    a.legend()

    a = ax[0, 1]
    a.loglog(n_list, [c.stderr for c in conv], "o", ms=3, label="erreur type")
    a.loglog(n_list, conv[0].stderr * np.sqrt(n_list[0] / n_list), "--", label="pente -1/2")
    a.set(title="Erreur en 1/√N", xlabel="N")
    a.legend()

    a = ax[0, 2]
    a.plot(moneyness * 100, iv * 100, "o-", ms=3, label="Merton (sauts calibrés)")
    a.axhline(sigma * 100, color="k", ls="--", label="MBG (vol constante)")
    a.set(title="4. Les sauts créent un smile", xlabel="strike (% de F0)", ylabel="vol implicite (%)")
    a.legend()

    a = ax[1, 0]
    lr_g = np.log(S_gbm[:, -1] / F0)
    lr_m = np.log(S_mer[:, -1] / F0)
    b_ = np.linspace(min(lr_g.min(), lr_m.min()), max(lr_g.max(), lr_m.max()), 150)
    a.hist(lr_g, bins=b_, density=True, alpha=0.5, label="MBG")
    a.hist(lr_m, bins=b_, density=True, alpha=0.5, label="Merton")
    a.set_yscale("log")
    a.set(title=f"Distribution de ln(F_T/F0) sous Q, T={T:g}", xlabel="rendement log")
    a.legend()

    a = ax[1, 1]
    labels = list(products)
    vals = [v.price for v in products.values()]
    errs = [1.96 * v.stderr for v in products.values()]
    a.bar(range(len(labels)), vals, yerr=errs, capsize=4, color=["C0", "C2", "C1", "C3"])
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels, rotation=15)
    a.set(title="5. Prix des exotiques (call, K = F0)", ylabel="prix")

    a = ax[1, 2]
    t = sc_gbm.t * 12
    for paths, c, lab in ((sc_gbm, "C0", "MBG"), (sc_sch, "C3", "Schwartz (retour à la moyenne)")):
        q = np.percentile(paths.S, [5, 50, 95], axis=0)
        a.plot(t, q[1], color=c, label=f"{lab} : médiane")
        a.fill_between(t, q[0], q[2], color=c, alpha=0.15, label=f"{lab} : 5 %–95 %")
    a.axhline(sch.long_run_price, color="C3", ls=":", lw=1, label="niveau de long terme")
    a.set(title="6. Scénarios de prix à 1 an (monde réel)", xlabel="mois", ylabel="prix")
    a.legend(fontsize=8)

    fig.suptitle(f"{name} — F0 = {F0:.2f}, r = {r:.2%}, sigma = {sigma:.1%}", fontsize=13)
    fig.tight_layout()
    fig.savefig(figdir / f"{tag}_pricing.png", dpi=120)
    print(f"\nFigures : figures/{tag}_market.png, figures/{tag}_pricing.png")


if __name__ == "__main__":
    main()
