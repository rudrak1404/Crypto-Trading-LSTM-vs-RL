"""Download and cache hourly BTCUSDT candles from Binance's public data archive.

Source: https://data.binance.vision (monthly kline zips, no API key needed).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "BTCUSDT_1h.csv"
URL = "https://data.binance.vision/data/spot/monthly/klines/{sym}/{iv}/{sym}-{iv}-{y}-{m:02d}.zip"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _months(start: str, end: str):
    for p in pd.period_range(start, end, freq="M"):
        yield p.year, p.month


def _read_month(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, names=COLS)
    if not str(df.iloc[0, 0]).isdigit():  # some archives ship a header row
        df = df.iloc[1:]
    ts = df["open_time"].astype("int64")
    # Binance switched spot timestamps from ms to us in 2025.
    unit = "us" if ts.iloc[0] > 10**14 else "ms"
    df.index = pd.to_datetime(ts, unit=unit, utc=True)
    return df[["open", "high", "low", "close", "volume", "trades"]].astype(float)


def download(symbol="BTCUSDT", interval="1h", start="2019-01", end="2026-10") -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for y, m in _months(start, end):
        path = RAW_DIR / f"{symbol}-{interval}-{y}-{m:02d}.zip"
        if not path.exists():
            with urllib.request.urlopen(URL.format(sym=symbol, iv=interval, y=y, m=m), timeout=60) as r:
                path.write_bytes(r.read())
        frames.append(_read_month(path.read_bytes()))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "time"
    return df


def load(refresh: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE, index_col="time", parse_dates=["time"])
    df = download()
    df.to_csv(CACHE)
    return df


def report_gaps(df: pd.DataFrame) -> dict:
    full = pd.date_range(df.index[0], df.index[-1], freq="1h")
    return {"rows": len(df), "start": str(df.index[0]), "end": str(df.index[-1]),
            "missing_hours": len(full.difference(df.index))}


if __name__ == "__main__":
    d = load(refresh=True)
    print(report_gaps(d))
