"""
Backtest logiki z btc_signals.py na danych historycznych z Kraken (domyślnie ~60 dni).

Dla każdego interwału (15m/1h/4h):
1. Pobiera historyczne świece BTC/USD z Kraken (z paginacją przez parametr "since")
2. Przechodzi świeca po świecy (walk-forward - używa TYLKO danych sprzed danego
   momentu, tak jak robiłby to bot na żywo) i sprawdza, czy pojawiłby się sygnał
3. Dla każdego sygnału sprawdza, czy cena po N świecach poszła w oczekiwaną stronę
4. Wypisuje podsumowanie: liczba sygnałów, % trafień, średni ruch ceny

UWAGA: To uproszczony backtest bez kosztów transakcji, poślizgu ceny, i bez
realnego zarządzania pozycją (stop-loss/take-profit). Służy do orientacyjnej
oceny, czy logika ma sens - nie jest gwarancją wyników na przyszłość.
"""

import time
import requests
import numpy as np
import pandas as pd

PAIR = "XBTUSD"
DAYS_BACK = 60

TIMEFRAMES = [
    {"label": "MIKRO", "minutes": 15, "lookback": 40, "forward_candles": 8},
    {"label": "ŚREDNI", "minutes": 60, "lookback": 40, "forward_candles": 8},
    {"label": "DUŻY",   "minutes": 240, "lookback": 40, "forward_candles": 8},
]

ATR_PROXIMITY_MULTIPLIER = 0.5
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
STOCH_OVERSOLD = 25
STOCH_OVERBOUGHT = 75

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"


def fetch_historical_klines(interval_minutes: int, days_back: int) -> pd.DataFrame:
    since = int(time.time()) - days_back * 24 * 60 * 60
    all_rows = []
    seen_lasts = set()

    while True:
        params = {"pair": PAIR, "interval": interval_minutes, "since": since}
        response = requests.get(KRAKEN_OHLC_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            raise Exception(f"Kraken API error: {data['error']}")

        result = data["result"]
        pair_key = next(k for k in result.keys() if k != "last")
        rows = result[pair_key]

        if not rows:
            break

        all_rows.extend(rows)
        new_since = result["last"]

        if new_since in seen_lasts or new_since == since:
            break
        seen_lasts.add(new_since)
        since = new_since

        if since >= int(time.time()) - interval_minutes * 60:
            break

        time.sleep(0.5)

    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = df[col].astype(float)
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    return 100 * (df["close"] - low_min) / (high_max - low_min)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def backtest_timeframe(label: str, minutes: int, lookback: int, forward_candles: int) -> dict:
    df = fetch_historical_klines(minutes, DAYS_BACK)
    print(f"[{label}/{minutes}min] pobrano {len(df)} świec")

    close = df["close"]
    rsi_series = rsi(close)
    stoch_series = stochastic_k(df)
    atr_series = atr(df)

    signals = []

    start_idx = lookback + 20
    end_idx = len(df) - forward_candles - 1

    for i in range(start_idx, end_idx):
        window = df.iloc[i - lookback:i]
        range_high = window["high"].max()
        range_low = window["low"].min()
        range_size = range_high - range_low
        if range_size <= 0:
            continue

        last_close = close.iloc[i]
        atr_val = atr_series.iloc[i]
        rsi_val = rsi_series.iloc[i]
        stoch_val = stoch_series.iloc[i]

        if pd.isna(atr_val) or pd.isna(rsi_val) or pd.isna(stoch_val) or atr_val <= 0:
            continue

        distance_to_low_atr = (last_close - range_low) / atr_val
        distance_to_high_atr = (range_high - last_close) / atr_val

        near_bottom = distance_to_low_atr <= ATR_PROXIMITY_MULTIPLIER
        near_top = distance_to_high_atr <= ATR_PROXIMITY_MULTIPLIER

        if near_bottom and rsi_val < RSI_OVERSOLD and stoch_val < STOCH_OVERSOLD:
            signals.append((i, "DOŁEK", last_close))
        elif near_top and rsi_val > RSI_OVERBOUGHT and stoch_val > STOCH_OVERBOUGHT:
            signals.append((i, "SZCZYT", last_close))

    wins = 0
    losses = 0
    moves_pct = []

    for idx, signal_type, entry_price in signals:
        future_price = close.iloc[idx + forward_candles]
        move_pct = (future_price - entry_price) / entry_price * 100

        if signal_type == "DOŁEK":
            correct = future_price > entry_price
            moves_pct.append(move_pct)
        else:
            correct = future_price < entry_price
            moves_pct.append(-move_pct)

        if correct:
            wins += 1
        else:
            losses += 1

    total = wins + losses
    win_rate = round(wins / total * 100, 1) if total > 0 else None
    avg_move = round(np.mean(moves_pct), 3) if moves_pct else None

    return {
        "label": label,
        "minutes": minutes,
        "total_signals": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "avg_move_pct": avg_move,
        "candles_tested": len(df),
    }


def run_backtest():
    print(f"=== BACKTEST na ostatnich {DAYS_BACK} dniach dla {PAIR} (Kraken) ===\n")

    results = []
    for tf in TIMEFRAMES:
        try:
            result = backtest_timeframe(tf["label"], tf["minutes"], tf["lookback"], tf["forward_candles"])
            results.append(result)
        except Exception as e:
            print(f"Błąd backtestu dla {tf['minutes']}min: {e}")

    print("\n=== PODSUMOWANIE ===")
    for r in results:
        print(
            f"\n[{r['label']} / {r['minutes']}min] "
            f"(przetestowano {r['candles_tested']} świec)\n"
            f"  Liczba sygnałów: {r['total_signals']}\n"
            f"  Trafione: {r['wins']}  |  Nietrafione: {r['losses']}\n"
            f"  Skuteczność: {r['win_rate_pct']}%\n"
            f"  Średni ruch ceny po sygnale: {r['avg_move_pct']}% "
            f"(dodatnie = zgodnie z oczekiwaniem)"
        )

    print(
        "\nUWAGA: to uproszczony test - bez kosztów transakcji, poślizgu ceny "
        "i bez realnego stop-loss/take-profit. Traktuj to jako orientacyjny "
        "wskaźnik, nie gwarancję wyników na przyszłość."
    )


if __name__ == "__main__":
    run_backtest()
