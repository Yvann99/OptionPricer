import yfinance as yf
import pandas as pd
from pathlib import Path
import numpy as np
#On configure mtn le cache pour optimiser la recherche de data

DATA_DIR = Path(__file__).resolve().parents[1]/"data" #on créer un lien du fichier contenant la data, resolve en fait un chemin absolu
#Fonction de récupération de la donnée 

def load_prices(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Historique quotidien d'un future (ex. "KC=F" pour l'arabica)."""
    path = DATA_DIR/f"{ticker}.csv" #avec path le / ne divise pas mais rajoute le data derriere le lien
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col = 0, parse_dates = True) #Parse_date convertit la colonne en date pour faciliter les calculs dessus
    
    df = yf.Ticker(ticker).history(period="max") #Ticker créer un objet qui représente le contrat KC=F et history récupère tout l'historique disponible
    df = df[["Open", "High", "Low", "Close"]]
    #attention au fuseau horaire -5heures (celui de NY)
    df.index = df.index.tz_localize(None) #tz pour time zone, l'index est la colonne de gauche, donc les dates, qu'on transforme en enlevant le time zone
    DATA_DIR.mkdir(exist_ok=True) #on sauvegarde pour la prochaine fois et si le dossier existe déjà, il ne se plaint pas avec exist_ok=true
    df.to_csv(path) #écrit le tableau dans le fichier
    return df

# On calcule les log rendements
def log_returns(prices: pd.Series) -> pd.Series:
    """Rendements logarithmiques quotidiens : r_t = log(S_t / S_{t-1})."""
    r = np.log(prices / prices.shift(1))
    return r.dropna()



