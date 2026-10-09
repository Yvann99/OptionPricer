"""Démo du socle : trois expériences dont on connaît le résultat théorique.

    python examples/demo_phase0.py

1. Erreur de couverture en 1/sqrt(N) dans le monde de Black–Scholes.
2. Vendre à une vol implicite > vol réalisée : P&L ≈ ½ ∫ Γ S² (σi² - σr²) dt.
3. Attribution du P&L par grecque.
La figure est enregistrée dans figures/phase0.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# permet de lancer le script depuis n'importe où : on ajoute la racine du projet au chemin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pricer import OptionSpec, attribution, market, run_hedge

S0, K, T, r = 100.0, 100.0, 0.5, 0.02
opt = OptionSpec(K, T, "call")
N_PATHS = 10_000

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 1. Convergence de la couverture
Ns = np.array([10, 25, 50, 100, 250, 500])
stds = []
for n in Ns:
    p = market.gbm(S0, 0.08, 0.2, T, n, N_PATHS, seed=n)
    stds.append(run_hedge(p, opt, r, 0.2).pnl.std())
stds = np.array(stds)
ax = axes[0]
ax.loglog(Ns, stds, "o-", label="écart-type du P&L")
ax.loglog(Ns, stds[0] * np.sqrt(Ns[0] / Ns), "--", label="pente -1/2")
ax.set(xlabel="nombre de rééquilibrages N", ylabel="écart-type du P&L",
       title="1. Erreur de couverture ∝ 1/√N")
ax.legend()
print("1. Écart-type du P&L :", dict(zip(Ns.tolist(), np.round(stds, 4).tolist())))

# 2. Vol implicite vs vol réalisée
p = market.gbm(S0, 0.05, 0.15, T, 250, N_PATHS, seed=0)
for s_impl, color in [(0.15, "C0"), (0.20, "C1"), (0.25, "C2")]:
    res = run_hedge(p, opt, r, sigma_price=s_impl)
    axes[1].hist(res.pnl, bins=80, alpha=0.6, color=color,
                 label=f"σ impl = {s_impl:.0%} (moy. {res.pnl.mean():+.2f})")
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set(xlabel="P&L final du vendeur", title="2. Vol réalisée = 15 %")
axes[1].legend()

# 3. Attribution
res = run_hedge(p, opt, r, sigma_price=0.25, cost_rate=0.0005)
summ = attribution.summarize(attribution.attribute(res))
keys = list(attribution.COMPONENTS)
means = [summ[k][0] for k in keys]
axes[2].bar(keys, means, color=["C7" if m == 0 else ("C2" if m > 0 else "C3") for m in means])
axes[2].axhline(0, color="k", lw=0.8)
axes[2].set(title=f"3. Attribution moyenne (P&L total {res.pnl.mean():+.2f})")
print("3. Attribution (moyenne, écart-type) :")
for k, (m, s) in summ.items():
    print(f"   {k:9s} {m:+8.4f}  ({s:.4f})")

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "figures" / "phase0.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130)
print("Figure :", out)
