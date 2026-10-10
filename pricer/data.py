"""Données de marché réelles : téléchargement, cache local et statistiques.

SOURCE
------
Yahoo Finance, via la librairie `yfinance` (gratuite, sans clé). Les matières
premières y sont disponibles sous forme de FUTURES « continus » :

    café (Coffee C, ICE)   KC=F   en cents US par livre (USc/lb)
    cacao (ICE)            CC=F   en USD par tonne
    sucre n°11 (ICE)       SB=F   en cents US par livre
    blé (CBOT)             ZW=F   en cents US par boisseau
    or (COMEX)             GC=F   en USD par once

Le taux sans risque est approché par le rendement des T-bills à 13 semaines (^IRX).

CACHE
-----
Chaque série téléchargée est enregistrée dans `data/<ticker>.csv`. Les appels
suivants relisent le fichier : pas besoin d'internet, et les résultats restent
reproductibles. `refresh=True` force un nouveau téléchargement.

ATTENTION : LE FUTURE « CONTINU »
---------------------------------
Un contrat future a une échéance. La série KC=F colle bout à bout les contrats
successifs (« roll »). Au moment du roll, le prix peut sauter sans que le marché
ait bougé (écart entre deux échéances). Ces sauts artificiels gonflent un peu
la volatilité et les queues. On les garde, mais il faut le savoir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRADING_DAYS = 252  # nombre de jours de bourse par an, pour annualiser

COMMODITIES = {
    "coffee": "KC=F",
    "cocoa": "CC=F",
    "sugar": "SB=F",
    "wheat": "ZW=F",
    "gold": "GC=F",
}
RISK_FREE_TICKER = "^IRX"


# ---------------------------------------------------------------------------
# Téléchargement et cache
# ---------------------------------------------------------------------------

def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "").replace("=", "_").replace("/", "_")
    return DATA_DIR / f"{safe}.csv"


def download(ticker: str, start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """Télécharge l'historique quotidien (Open, High, Low, Close, Volume) depuis Yahoo."""
    try:
        import yfinance as yf
    except ImportError as e:  # message clair plutôt qu'une erreur obscure
        raise ImportError("Installe yfinance : pip install yfinance") from e

    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"Aucune donnée reçue pour {ticker} (connexion internet ? ticker valide ?)")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load(ticker: str, start: str = "2000-01-01", refresh: bool = False) -> pd.DataFrame:
    """Charge une série depuis le cache local, ou la télécharge si besoin.

    `ticker` peut être un code Yahoo ("KC=F") ou un nom de COMMODITIES ("coffee").
    """
    ticker = COMMODITIES.get(ticker, ticker)
    path = _cache_path(ticker)
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        df = download(ticker, start=start)
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(path)
    df = df[df.index >= pd.Timestamp(start)]
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage minimal : trie, retire les doublons et les prix nuls ou manquants."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Close"])
    return df[df["Close"] > 0]


def risk_free_rate(default: float = 0.04) -> float:
    """Dernier taux sans risque (T-bill 13 semaines), en décimal continu.

    ^IRX est coté en pourcentage annuel (ex. 4.3 -> 4,3 %). On le convertit en
    taux continu : r = ln(1 + taux). Si le téléchargement échoue, on renvoie `default`.
    """
    try:
        irx = load(RISK_FREE_TICKER, start="2015-01-01")["Close"].iloc[-1] / 100
        return float(np.log1p(irx))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Rendements et volatilité
# ---------------------------------------------------------------------------

def log_returns(close: pd.Series) -> pd.Series:
    """Rendements logarithmiques quotidiens : ln(S_t / S_{t-1})."""
    return np.log(close).diff().dropna()


def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Volatilité réalisée glissante, annualisée.

    C'est la variation quadratique empirique du cours :
        sigma² ≈ (1/T) Σ (ln S_{i+1} - ln S_i)²  sur la fenêtre.
    window=21 ≈ un mois de bourse. On ne retire pas la moyenne (elle est négligeable
    à l'échelle d'une journée), comme dans la définition de [ln S]_t.
    """
    return np.sqrt((returns**2).rolling(window).mean() * TRADING_DAYS)


def parkinson_vol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Volatilité de Parkinson : utilise l'écart haut-bas de la journée.

    sigma² = (1 / 4 ln 2) E[(ln H/L)²]. Environ 5 fois plus efficace que
    l'estimateur close-to-close, car il exploite l'information intra-journalière.
    """
    hl = np.log(df["High"] / df["Low"]) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4 * np.log(2)) * TRADING_DAYS)


def vol_by_month(returns: pd.Series) -> pd.Series:
    """Volatilité annualisée par mois calendaire (1 = janvier ... 12 = décembre).

    Pour le café, on s'attend à une saisonnalité : l'hiver austral (juin-août)
    est la période des risques de gel au Brésil, premier producteur mondial.
    """
    v = (returns**2).groupby(returns.index.month).mean()
    return np.sqrt(v * TRADING_DAYS).rename("vol")


def return_stats(returns: pd.Series) -> dict:
    """Statistiques descriptives des rendements quotidiens.

    - skew : asymétrie (0 pour une loi normale) ;
    - excess_kurtosis : aplatissement en excès (0 pour une normale). Positif =
      queues épaisses : les gros mouvements sont plus fréquents que le modèle
      de Black–Scholes ne le prévoit.
    """
    r = returns.dropna()
    return {
        "n_days": int(r.size),
        "mean_annual": float(r.mean() * TRADING_DAYS),
        "vol_annual": float(r.std() * np.sqrt(TRADING_DAYS)),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurt()),
        "worst_day": float(r.min()),
        "best_day": float(r.max()),
        "share_beyond_4sd": float((np.abs(r - r.mean()) > 4 * r.std()).mean()),
    }


def detect_jumps(returns: pd.Series, k: float = 4.0) -> pd.Series:
    """Repère les jours de « saut » : |r - médiane| > k × (écart-type robuste).

    L'écart-type robuste est 1.4826 × MAD (médiane des écarts absolus) : il n'est
    pas gonflé par les sauts eux-mêmes, contrairement à l'écart-type classique.
    Sous une loi normale, k = 4 n'arrive qu'environ une fois tous les 60 ans
    de bourse : ce qui dépasse est très probablement un saut.
    """
    med = returns.median()
    robust_sd = 1.4826 * (returns - med).abs().median()
    return (returns - med).abs() > k * robust_sd
