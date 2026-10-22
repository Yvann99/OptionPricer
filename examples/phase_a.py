"""Phase A : anatomie du P&L d'un vendeur couvert, en six expériences.

    python examples/phase_a.py

1. Erreur de couverture en 1/√N... sauf quand le modèle est faux (plateau).
2. Couvert, le vendeur ignore le drift μ ; non couvert, il parie dessus.
3. P&L ≈ ½ ∫ Γ S² (σ_impl² - σ_réal²) dt, trajectoire par trajectoire.
4. Couvrir à la vol implicite (livre lisse, P&L aléatoire) ou réalisée (P&L certain).
5. Coûts de transaction : précision contre coût.
6. Modèle faux : Heston et sauts épaississent la queue gauche.
La figure est enregistrée dans figures/phase_a.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# permet de lancer le script depuis n'importe où : on ajoute la racine du projet au chemin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pricer import OptionSpec, experiments as ex, market

S0, K, T, r, sig = 100.0, 100.0, 0.5, 0.02, 0.2
opt = OptionSpec(K, T, "call")
N_PATHS = 10_000
N_DAYS = 125  # 6 mois de jours ouvrés

# Trois mondes de même vol moyenne (20 %) : seul le premier est celui de Black–Scholes
LAM, JM, JS = 2.0, -0.08, 0.05   # 2 sauts/an de -8 % ± 5 %
SIMS = {
    "MBG (BS)": lambda n, s: market.gbm(S0, 0.05, sig, T, n, N_PATHS, seed=s),
    "Heston": lambda n, s: market.heston(S0, sig**2, 0.05, 2.0, sig**2, 0.6, -0.7, T, n, N_PATHS, seed=s),
    "Merton (sauts)": lambda n, s: market.merton_jump(S0, 0.05, np.sqrt(sig**2 - LAM * (JM**2 + JS**2)),
                                                      LAM, JM, JS, T, n, N_PATHS, seed=s),
}

fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
axes = axes.ravel()

# 1. Erreur en 1/√N, plateau si le modèle est faux
Ns = np.array([10, 25, 50, 125, 250, 500])
ax = axes[0]
print("1. Écart-type du P&L selon N (pente log-log) :")
for name, sim in SIMS.items():
    err = ex.error_vs_n(sim, opt, r, sig, Ns)
    ax.loglog(Ns, err, "o-", label=f"{name} (pente {ex.slope_loglog(Ns, err):+.2f})")
    print(f"   {name:15s}", np.round(err, 3), f"pente {ex.slope_loglog(Ns, err):+.2f}")
ax.loglog(Ns, 1.0 * np.sqrt(Ns[0] / Ns), "k--", lw=0.8, label="pente -1/2")
ax.set(xlabel="nombre de rééquilibrages N", ylabel="écart-type du P&L",
       title="1. Erreur ∝ 1/√N… sauf si le modèle est faux")
ax.legend(fontsize=8)

# 2. Indépendance au drift
mus = np.linspace(-0.3, 0.5, 9)
d = ex.drift_independence(lambda mu, n, s: market.gbm(S0, mu, sig, T, n, N_PATHS, seed=s),
                          opt, r, sig, mus, n_steps=N_DAYS)
ax = axes[1]
for key, label in [("hedged", "couvert en delta"), ("naked", "non couvert")]:
    m = np.array([x["mean"] for x in d[key]])
    s = np.array([x["std"] for x in d[key]])
    ax.plot(mus, m, "o-", label=label)
    ax.fill_between(mus, m - s, m + s, alpha=0.15)
ax.axhline(0, color="k", lw=0.8)
ax.set(xlabel="drift réel μ", ylabel="P&L moyen ± 1 écart-type",
       title="2. Couvert, le vendeur ne parie plus sur μ")
ax.legend()
print("2. P&L moyen couvert   :", np.round([x["mean"] for x in d["hedged"]], 3))
print("   P&L moyen non couvert :", np.round([x["mean"] for x in d["naked"]], 3))

# 3. Le pari sur la variation quadratique : vendu à 20 %, vol réalisée aléatoire (Heston)
p_heston = SIMS["Heston"](N_DAYS, 3)
v = ex.vol_bet(p_heston, opt, r, sigma_impl=sig)
ax = axes[2]
sc = ax.scatter(v["theory"], v["pnl"], c=v["realized_vol"], s=3, cmap="coolwarm")
lim = np.array([v["pnl"].min(), v["pnl"].max()])
ax.plot(lim, lim, "k--", lw=0.8)
fig.colorbar(sc, ax=ax, label="vol réalisée")
corr = np.corrcoef(v["pnl"], v["theory"])[0, 1]
ax.set(xlabel="½ Σ Γ S² (σ_impl² dt - (dS/S)²)", ylabel="P&L final",
       title=f"3. Un pari sur la variation quadratique (corr. {corr:.3f})")
print(f"3. Corrélation P&L / formule théorique : {corr:.4f}")

# 4. Couvrir à la vol implicite ou réalisée : vendu à 25 %, marché à 15 %
p = market.gbm(S0, 0.05, 0.15, T, N_DAYS, N_PATHS, seed=4)
iv = ex.implied_vs_realized(p, opt, r, sigma_impl=0.25, sigma_real=0.15)
ax = axes[3]
for key, color, label in [("implied", "C0", "couvert à σ_impl"), ("realized", "C1", "couvert à σ_réal")]:
    res = iv[key]
    pnl_path = res.book_value - res.book_value[:, :1]   # gain marqué au modèle depuis t=0
    for j in range(15):
        ax.plot(res.t, pnl_path[j], color=color, lw=0.6, alpha=0.6,
                label=f"{label} (écart-t final {res.pnl.std():.2f})" if j == 0 else None)
ax.axhline(iv["locked_in"], color="k", ls="--", lw=0.8, label=f"gain verrouillé {iv['locked_in']:.2f}")
ax.set(xlabel="temps (années)", ylabel="P&L marqué à σ_impl",
       title="4. Vol implicite (lisse) ou réalisée (P&L certain)")
ax.legend(fontsize=8)
print(f"4. Gain verrouillé {iv['locked_in']:.3f} | couvert à σ_impl : moy. {iv['implied'].pnl.mean():.3f}, "
      f"écart-t {iv['implied'].pnl.std():.3f} | couvert à σ_réal : moy. {iv['realized'].pnl.mean():.3f}, "
      f"écart-t {iv['realized'].pnl.std():.3f}")

# 5. Coûts : précision contre coût (couverture tous les `every` jours, 20 pb par transaction)
p = market.gbm(S0, 0.05, sig, T, N_DAYS, N_PATHS, seed=5)
rows = ex.cost_tradeoff(p, opt, r, sig, cost_rate=0.002, everys=[1, 2, 3, 5, 10, 25, 125])
ax = axes[4]
ax.plot([x["std"] for x in rows], [x["mean"] for x in rows], "o-")
for x in rows:
    ax.annotate(f"{x['every']} j", (x["std"], x["mean"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set(xlabel="risque (écart-type du P&L)", ylabel="P&L moyen (= - coûts)",
       title="5. Coûts : rééquilibrer tous les … jours")
print("5. every, moyenne, écart-type, coûts :")
for x in rows:
    print(f"   {x['every']:4d} j  {x['mean']:+7.3f}  {x['std']:6.3f}  {x['mean_costs']:6.3f}")

# 6. Modèle faux : distributions du P&L
paths = {name: sim(N_DAYS, 6) for name, sim in SIMS.items()}
stats = ex.model_comparison(paths, opt, r, sig)
ax = axes[5]
from pricer import run_hedge  # noqa: E402
bins = np.linspace(-15, 5, 150)
for name, pth in paths.items():
    pnl = run_hedge(pth, opt, r, sig).pnl
    s = stats[name]
    ax.hist(pnl, bins=bins, alpha=0.5, density=True,
            label=f"{name} : CVaR5% {s['cvar05']:+.2f}, skew {s['skew']:+.1f}")
ax.set_yscale("log")
ax.set(xlabel="P&L final du vendeur couvert", title="6. Modèle faux : la queue gauche s'épaissit")
ax.legend(fontsize=8)
print("6. Modèle faux :")
for name, s in stats.items():
    print(f"   {name:15s} moy. {s['mean']:+.3f}  écart-t {s['std']:.3f}  CVaR5% {s['cvar05']:+.3f}  "
          f"skew {s['skew']:+.2f}  kurt {s['kurt']:+.2f}")

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "figures" / "phase_a.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130)
print("Figure :", out)
