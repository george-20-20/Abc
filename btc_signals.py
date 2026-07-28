"""
Bot wykrywający zbliżające się dołki i górki dla BTC/USDT (Binance),
jednocześnie na 3 interwałach czasowych: 15m (mikro), 1h (średnie), 4h (duże).
"""

import os
import requests
import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"

TIMEFRAMES = [
    {"label": "MIKRO", "interval": "15m", "lookback": 40},
    {"label": "ŚREDNI", "interval": "1h",  "lookback": 40},
    {"label": "DUŻY",   "interval": "4h",  "lookback": 40},
]

ATR_PROXIMITY_MULTIPLIER = 0.5
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
STOCH_OVERSOLD = 25
STOCH_OVERBOUGHT = 75

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol, interval, limit=200):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    response.raise_for_status()
    raw = response.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic_k(df, period=14):
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    return 100 * (df["close"] - low_min) / (high_max - low_min)


def atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def analyze_timeframe(label, interval, lookback):
    df = fetch_klines(SYMBOL, interval, limit=max(lookback + 20, 100))
    close = df["close"]
    last_close = close.iloc[-1]

    window = df.iloc[-(lookback + 1):-1]
    range_high = window["high"].max()
    range_low = window["low"].min()
    range_size = range_high - range_low

    position_in_range = (last_close - range_low) / range_size if range_size > 0 else 0.5

    atr_val = atr(df).iloc[-1]
    rsi_val = rsi(close).iloc[-1]
    stoch_val = stochastic_k(df).iloc[-1]

    distance_to_low_atr = (last_close - range_low) / atr_val if atr_val > 0 else 999
    distance_to_high_atr = (range_high - last_close) / atr_val if atr_val > 0 else 999

    near_bottom = distance_to_low_atr <= ATR_PROXIMITY_MULTIPLIER
    near_top = distance_to_high_atr <= ATR_PROXIMITY_MULTIPLIER

    signal = None
    if near_bottom and rsi_val < RSI_OVERSOLD and stoch_val < STOCH_OVERSOLD:
        signal = "DOŁEK"
    elif near_top and rsi_val > RSI_OVERBOUGHT and stoch_val > STOCH_OVERBOUGHT:
        signal = "SZCZYT"

    return {
        "label": label, "interval": interval, "price": last_close,
        "range_low": range_low, "range_high": range_high,
        "position_in_range_pct": round(position_in_range * 100, 1),
        "rsi": round(rsi_val, 1), "stoch": round(stoch_val, 1),
        "atr": round(atr_val, 2), "signal": signal,
    }


def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Brak danych Telegram - pomijam wysyłkę.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        if not r.ok:
            print(f"Telegram błąd: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Błąd wysyłki na Telegram: {e}")


def format_alert(result):
    icon = "🟢" if result["signal"] == "DOŁEK" else "🔴"
    return (
        f"{icon} BTC — możliwy {result['signal']} ({result['label']}, {result['interval']})\n"
        f"Cena: {result['price']:.1f}\n"
        f"Zakres: {result['range_low']:.1f} – {result['range_high']:.1f} "
        f"(pozycja: {result['position_in_range_pct']}%)\n"
        f"RSI: {result['rsi']}  |  Stochastic: {result['stoch']}  |  ATR: {result['atr']}"
    )


def run_once():
    print(f"--- Sprawdzanie {SYMBOL} na {len(TIMEFRAMES)} interwałach ---")
    any_signal = False
    for tf in TIMEFRAMES:
        try:
            result = analyze_timeframe(tf["label"], tf["interval"], tf["lookback"])
            print(
                f"[{result['label']} / {result['interval']}] cena={result['price']:.1f} "
                f"zakres=({result['range_low']:.1f}-{result['range_high']:.1f}) "
                f"pozycja={result['position_in_range_pct']}% "
                f"RSI={result['rsi']} Stoch={result['stoch']} sygnał={result['signal']}"
            )
            if result["signal"]:
                send_telegram_message(format_alert(result))
                any_signal = True
        except Exception as e:
            print(f"Błąd przy analizie interwału {tf['interval']}: {e}")
    if not any_signal:
        print("Brak sygnałów w tym przebiegu.")


if __name__ == "__main__":
    run_once()
