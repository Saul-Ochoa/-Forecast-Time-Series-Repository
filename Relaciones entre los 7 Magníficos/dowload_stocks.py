import pandas as pd
import numpy as np
import seaborn as sns
import yfinance as yf
import time
import warnings
warnings.filterwarnings('ignore')


#####################################################################
######################## Stock finance ##############################
#####################################################################
# El periodo como minimo debe ser de un año
def obtener_datos(ticker, periodo):
    """Descarga datos y calcula indicadores técnicos básicos."""
    df = yf.download(ticker, period=periodo, interval="1d")
    
    # Manejo de MultiIndex en versiones recientes de yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df['Returns'] = df['Close'].pct_change()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Returns"].rolling(20).std()
    
    # Bandas de Bollinger
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]
    
    return df

def agregar_indicadores(df):
    """Añade RSI y MACD al DataFrame existente."""
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    return df
