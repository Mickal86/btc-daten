import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

# ============================
# 1. BTC Preise von CoinGecko laden
# ============================

def fetch_btc_history():
    print("→ Lade Bitcoin-Preis-Daten von CoinGecko...")

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "max",
        "interval": "daily"
    }

    r = requests.get(url, params=params)
    data = r.json()

    prices = data["prices"]  # [timestamp, price]

    dates = []
    daily_price = []
    days_since_genesis = []

    genesis = datetime(2009, 1, 3)

    for ts, price in prices:
        dt = datetime.utcfromtimestamp(ts / 1000)
        dates.append(dt.strftime("%Y-%m-%d"))
        daily_price.append(price)
        days_since_genesis.append((dt - genesis).days)

    df = pd.DataFrame({
        "date": dates,
        "day": days_since_genesis,
        "price": daily_price
    })

    return df


# ============================
# 2. Power-Law berechnen
# ============================

def compute_powerlaw(df):
    print("→ Berechne Power-Law-Regression...")

    # Entferne Nullwerte
    df = df[df["day"] > 0]
    df = df[df["price"] > 0]

    logday = np.log10(df["day"].values)
    logprice = np.log10(df["price"].values)

    # lineare Regression in log-log Raum
    p = np.polyfit(logday, logprice, 1)
    slope = p[0]
    intercept = p[1]

    # individual intercept wie in MATLAB
    individual = logprice - slope * logday
    mean_intercept = np.mean(individual)

    # Modell
    logPL = slope * logday + mean_intercept
    modelprice = 10 ** logPL

    # Standardabweichung
    diff = np.abs(logprice - logPL)
    mean_diff = np.mean(diff)
    squared = (diff - mean_diff) ** 2
    std = np.sqrt(np.sum(squared) / (len(squared) - 1))

    # Modell-Levels
    df["pl"] = modelprice
    df["dev_down_1"] = modelprice * 10 ** (-1 * std)
    df["dev_down_1_5"] = modelprice * 10 ** (-1.5 * std)
    df["dev_down_2"] = modelprice * 10 ** (-2 * std)
    df["dev_up_1"] = modelprice * 10 ** (1 * std)
    df["dev_up_1_5"] = modelprice * 10 ** (1.5 * std)
    df["dev_up_2"] = modelprice * 10 ** (2 * std)
    df["dev_up_2_5"] = modelprice * 10 ** (2.5 * std)
    df["dev_up_3"] = modelprice * 10 ** (3 * std)

    result = {
        "slope": float(slope),
        "intercept": float(mean_intercept),
        "std": float(std),
        "last_update": datetime.utcnow().isoformat(),
        "data": df.to_dict(orient="records")
    }

    return result


# ============================
# 3. JSON speichern
# ============================

def save_json(result):
    os.makedirs("generated", exist_ok=True)

    filepath = "generated/powerlaw.json"

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✓ JSON gespeichert: {filepath}")


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    print("=== Power-Law Update gestartet ===")

    df = fetch_btc_history()
    result = compute_powerlaw(df)
    save_json(result)

    print("=== Fertig ===")
