"""PricerOption : pricing, simulation, couverture dynamique et attribution du P&L.

Modules :
    black_scholes  prix, grecques et volatilité implicite
    market         simulateurs de trajectoires : MBG, Heston, régimes, Merton
    hedging        portefeuille autofinancé et stratégies de couverture
    attribution    décomposition du P&L par grecque
"""

from . import attribution, black_scholes, hedging, market
from .hedging import DeltaHedge, NoHedge, OptionSpec, run_hedge

__all__ = ["attribution", "black_scholes", "hedging", "market",
           "DeltaHedge", "NoHedge", "OptionSpec", "run_hedge"]
__version__ = "0.4.0"
