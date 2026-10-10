"""PricerOption : pricing, couverture dynamique et analyse de volatilité.

Modules :
    black_scholes  prix et grecques (actions)
    black76        prix et grecques (options sur futures : café, cacao...)
    closed_forms   formules fermées de référence (asiatique géométrique, Merton)
    market         simulateurs : MBG, Heston, régimes, Merton, Schwartz
    montecarlo     pricing Monte-Carlo, réduction de variance, options exotiques
    hedging        portefeuille autofinancé et stratégies de couverture
    attribution    décomposition du P&L par grecque
    calibration    estimation des modèles sur données réelles
    data           données Yahoo Finance (à importer explicitement : `from pricer import data`,
                   car il nécessite pandas et yfinance)
"""

from . import (attribution, black76, black_scholes, calibration, closed_forms, hedging,
               market, montecarlo)
from .hedging import DeltaHedge, NoHedge, OptionSpec, run_hedge

__all__ = ["attribution", "black76", "black_scholes", "calibration", "closed_forms",
           "hedging", "market", "montecarlo",
           "DeltaHedge", "NoHedge", "OptionSpec", "run_hedge"]
__version__ = "0.2.0"
