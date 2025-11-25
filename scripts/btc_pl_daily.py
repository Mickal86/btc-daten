import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

# ============================================
# 0. Einlesen CMC API Key
# ============================================

CMC_API_KEY = os.getenv("CMC_API_KEY")
if CMC_API_KEY is None:
    raise ValueError("!! Kein API Key gefunden. Bitte GitHub Secret 'CMC_API_KEY' setzen.")


# ============================================
# 1. Preise von CoinMarketCap laden
# ============================================

def fetch_latest_btc_price():
    print("→ Lade aktuellen BTC-Preis von CoinMarketCap...")

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    params = {"symbol": "BTC", "convert": "USD"}
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}

    r = requests.get(url, params=params, headers=headers)
    data = r.json()

    price = data["data"]["BTC"]["quote"]["USD"]["price"]
    timestamp = data["data"]["BTC"]["quote"]["USD"]["last_updated"]

    date = timestamp[:10]   # yyyy-mm-dd

    return date, float(price)


# ============================================
# 2. Historische Preisdaten laden + erweitern
# ============================================

def load_historic_csv():
    print("→ Lade historische CSV...")

    df = pd.read_csv("BTC_PL_Daily_Data.csv", sep=";")

    # Spaltennamen anpassen (Groß/Kleinschreibung vereinheitlichen)
    df = df.rename(columns={
        "Date": "date",
        "DaysGB": "daysGB",
        "Price": "price"
    })

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

# Preiswerte sicher in float umwandeln
    df["price"] = (
        df["price"]
        .astype(str)            # falls mixed types
        .str.replace(",", ".", regex=False)  # deutsches Komma zu Punkt
        .astype(float)
    )

    # daysGB sicher als int/float
    df["daysGB"] = df["daysGB"].astype(int)

    return df



def append_latest_price(df, latest_date, latest_price):
    print("→ Ergänze CSV um heutigen Preis...")

    # Prüfen ob Datum bereits existiert
    if latest_date in df["date"].dt.strftime("%Y-%m-%d").values:
        print("✓ Aktuelles Datum existiert schon – kein Update nötig.")
        return df

    genesis = datetime(2009, 1, 3)
    days_gb = (datetime.fromisoformat(latest_date) - genesis).days

    new_row = {
        "date": latest_date,
        "daysGB": days_gb,
        "price": latest_price
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


# ============================================
# 3. Power-Law berechnen
# ============================================

def compute_powerlaw(df):
    print("→ Berechne Power-Law...")

    df = df[df["daysGB"] > 0]
    df = df[df["price"] > 0]

    logday = np.log10(df["daysGB"].values)
    logprice = np.log10(df["price"].values)

    # Regression
    slope, intercept = np.polyfit(logday, logprice, 1)

    # Individueller intercept
    individual = logprice - slope * logday
    mean_intercept = np.mean(individual)

    # Modell
    logPL = slope * logday + mean_intercept
    modelprice = 10 ** logPL

    df["pl"] = modelprice

    # Standardabweichung
    diff = np.abs(logprice - logPL)
    mean_diff = np.mean(diff)
    squared = (diff - mean_diff) ** 2
    std = np.sqrt(np.sum(squared) / (len(squared) - 1))

    # Levels
    df["dev_down_1"] = modelprice * 10 ** (-1 * std)
    df["dev_up_1"] = modelprice * 10 ** (1 * std)
    df["dev_up_2"] = modelprice * 10 ** (2 * std)

    # Datum für JSON serialisierbar machen
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    result = {
        "slope": float(slope),
        "intercept": float(mean_intercept),
        "std": float(std),
        "last_update": datetime.utcnow().isoformat(),
        "data": df.to_dict(orient="records")
    }

    return result


# ============================================
# 4. JSON speichern für Webflow
# ============================================

def save_json(result):
    os.makedirs("data", exist_ok=True)

    filepath = "data/powerlaw.json"

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✓ exportiert nach {filepath}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=== BTC Power-Law Update gestartet ===")

    # CSV laden
    df = load_historic_csv()

    # Aktuellen Preis abrufen
    latest_date, latest_price = fetch_latest_btc_price()

    # ergänzen
    df = append_latest_price(df, latest_date, latest_price)

    # Power-Law berechnen
    result = compute_powerlaw(df)

    # JSON speichern
    save_json(result)

    print("=== Fertig ===")
