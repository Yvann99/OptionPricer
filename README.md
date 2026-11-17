# OptionPricer : le café

**Prix, volatilité et options sur le café (arabica et robusta), reconstruits pas à pas en Python.**

> **Question du projet :** sur les 20 dernières années, comment le prix du café, sa volatilité et le P&L d'un vendeur d'options couvert sont-ils liés ?

---

## Le fil rouge

> **Le prix d'une option est le coût de sa réplication. Un vendeur couvert ne parie donc pas sur la direction du marché, mais sur sa volatilité.**

Le café se prête bien à l'exercice : un marché étroit, une offre concentrée (Brésil, Vietnam), des chocs climatiques (gel, sécheresse) et donc des **sauts** et des **régimes de volatilité** bien marqués.

## Méthode de travail

L'objectif est d'**apprendre**, pas seulement d'obtenir des résultats. Chaque étape suit la même boucle :

1. **Le concept** : ce qu'on calcule, et pourquoi.
2. **Le code**, écrit par moi (Yvann). Claude relit, pose des questions, pointe les erreurs, sans donner la solution tout de suite.
3. **Un test** qui vérifie une vérité indépendante du code.
4. **Une conclusion** sur les données, notée dans ce README.

Seul l'outillage sans intérêt pédagogique (cache, mise en forme) est écrit par Claude.

---

## Le socle conservé

Ces quatre modules existent déjà. Il faut se les **réapproprier** : les relire, savoir les expliquer, réécrire certains morceaux.

| Maîtrisé | Module | Contenu | Lien avec le cours |
|---|---|---|---|
| [ ] | `pricer/black_scholes.py` | Payoff, $d_1$, $d_2$, prix, grecques, vol implicite (vectorisé) | Prix $= \mathbb E^{\mathbb Q}[e^{-rT}h(S_T)]$, EDP de Black–Scholes |
| [ ] | `pricer/market.py` | Simulateurs : MBG, Heston, MBG à régimes, Merton (sauts) | EDS, mouvement brownien géométrique |
| [ ] | `pricer/hedging.py` | Portefeuille autofinancé : prime, delta, rééquilibrage, coûts | Intégrale d'Itô $\int \varphi\,dS$, processus adaptés |
| [ ] | `pricer/attribution.py` | P&L décomposé en delta, gamma, theta, carry, coûts | Formule d'Itô discrète |

**Tests du socle** (`pytest -v`, 27 tests) : valeurs de référence de Hull, parité call-put, grecques ≈ différences finies, EDP de Black–Scholes, Monte-Carlo ≈ formule fermée, martingale sous $\mathbb Q$, vol réalisée ≈ variation quadratique, erreur de couverture en $1/\sqrt N$, indépendance au drift, attribution qui somme au P&L total, stratégie qui ne voit pas le futur.

---

## Feuille de route

### Étape 1 : les prix

- [ ] Télécharger l'historique quotidien de l'**arabica** (ICE US, `KC=F` sur Yahoo Finance, en cents US par livre)
- [ ] Charger le **robusta** (ICE Londres, en USD par tonne) depuis le CSV d'Investing.com
- [ ] Mettre en cache localement (`data/`), pour des résultats reproductibles et sans internet
- [ ] Nettoyer : jours manquants, valeurs aberrantes, sauts artificiels liés au **roll** des contrats futures
- [ ] Rendre les deux séries comparables (unités, devises)

### Étape 2 : la volatilité historique

- [ ] Rendements logarithmiques et volatilité réalisée annualisée (fenêtres de 20, 60 et 252 jours)
- [ ] Lien avec la variation quadratique : $\sigma_{\text{réal}}^2 \approx [\log S]_T / T$
- [ ] Estimateurs plus fins si on a les prix haut/bas : Parkinson, Garman–Klass
- [ ] Queues épaisses, sauts, saisonnalité (récoltes, gel brésilien)

### Étape 3 : la volatilité implicite

- [ ] Récupérer la chaîne d'options du jour sur l'arabica (Barchart ou prix de règlement ICE)
- [ ] Options sur futures : passer de Black–Scholes à **Black-76** (le future est une martingale sous $\mathbb Q$)
- [ ] Inverser la formule pour obtenir la **vol implicite**, puis le smile par strike
- [ ] Comparer vol implicite et vol réalisée : la **prime de variance**

### Étape 4 : l'analyse sur 20 ans

- [ ] Tendances de long terme du prix et de la volatilité, arabica contre robusta
- [ ] Régimes de volatilité : calme, tension, crise (gel de 2021, flambée de 2024…)
- [ ] Relation prix/vol : la vol monte-t-elle quand le prix monte (effet inverse de celui des actions) ?
- [ ] **P&L d'un vendeur couvert** rejoué sur l'historique réel avec `hedging.py`, et décomposé avec `attribution.py` : quand gagne-t-il, quand perd-il ?

### Étape 5 : le site interactif

- [ ] Page HTML interactive pour naviguer dans les résultats : choisir la période, le café (arabica ou robusta), la fenêtre de vol, l'option vendue, et voir prix, vol et P&L se mettre à jour

---

## Données : la question ouverte

Vérifié le 17/11/2026. Pas d'accès à Bloomberg ou Refinitiv : uniquement des sources gratuites.

| Donnée | Source | Statut |
|---|---|---|
| Arabica (futures) | Yahoo Finance, `KC=F`, quotidien depuis le 03/01/2000 | ✅ Disponible |
| Robusta (futures) | Absent de Yahoo (`RC=F`, `RM=F` introuvables). CSV quotidien à télécharger à la main sur Investing.com (« London Robusta Coffee Futures »), ou série mensuelle du *Pink Sheet* de la Banque mondiale | ⏳ Téléchargement manuel |
| Taux sans risque | T-bills 13 semaines, `^IRX` sur Yahoo Finance | ✅ Disponible |
| Options sur le café | Absentes de Yahoo (pas d'options sur les futures, ETN café `JO` et `CAFE` retirés de la cote). Chaîne du jour sur Barchart ou dans les prix de règlement ICE | ⚠️ Instantané du jour seulement |

**Conséquence :** l'analyse sur 20 ans (prix, vol, P&L) se fait avec la **vol réalisée**. La **vol implicite** est calculée sur la chaîne d'options du jour, puis comparée à la vol réalisée actuelle : c'est la prime de variance d'aujourd'hui.

---

## Structure du projet

```
PricerOption/
├── pricer/
│   ├── black_scholes.py   prix, grecques, volatilité implicite
│   ├── market.py          simulateurs de trajectoires
│   ├── hedging.py         portefeuille autofinancé, couverture
│   └── attribution.py     décomposition du P&L
├── tests/                 un fichier de test par module
├── data/                  cache local des données (non versionné)
├── pyproject.toml
└── requirements.txt
```

Les nouveaux modules (données, volatilité, Black-76, analyse, site) seront ajoutés au fil des étapes.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -v
```

## Conventions

- `T` dans les fonctions Black–Scholes = temps **restant** jusqu'à la maturité, en années.
- `r` = taux sans risque continu ; `sigma` = volatilité annuelle (0.2 = 20 %).
- `position = -1` : on a **vendu** l'option.
- Une stratégie de couverture ne lit que le **passé** des prix : elle est *adaptée* à la filtration. C'est la règle d'or contre le biais d'anticipation.

## Pièges à surveiller

- **Biais d'anticipation** : un signal à la date $t$ n'utilise que les données jusqu'à $t$.
- **Future continu** : la série colle bout à bout des contrats d'échéances différentes. Au roll, le prix peut sauter sans que le marché ait bougé, ce qui gonfle la vol et les queues.
- **Queues de distribution** : vendre des options, c'est gagner souvent un peu et perdre rarement beaucoup. On regarde la **CVaR**, pas seulement la moyenne.
- **Unités** : l'arabica est coté en cents US par livre, le robusta en USD par tonne.
