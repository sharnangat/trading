import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

from stock_config import load_tickers

NSE_TICKERS = load_tickers()


def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """Download recent 1-year daily data for a stock."""
    try:
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})

        if "Close" in df.columns:
            df["close"] = df["Close"]
        if "Volume" in df.columns:
            df["volume"] = df["Volume"]

        return df[["date", "close", "volume"]].dropna().reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving average, RSI, and volume-based metrics."""
    if df.empty:
        return df

    df = df.copy()
    df["sma20"] = SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma50"] = SMAIndicator(df["close"], window=50).sma_indicator()
    df["rsi14"] = RSIIndicator(df["close"], window=14).rsi()
    df["avg_vol_20"] = df["volume"].rolling(window=20).mean()
    df["year_high"] = df["close"].rolling(window=252, min_periods=1).max()
    return df


def is_bullish_signal(row: pd.Series) -> bool:
    """Return True if the latest stock data matches a bullish setup."""
    price = row["close"]
    sma20 = row["sma20"]
    sma50 = row["sma50"]
    rsi = row["rsi14"]
    volume = row["volume"]
    avg_vol = row["avg_vol_20"]
    year_high = row["year_high"]

    if pd.isna(price) or pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi):
        return False

    return (
        price > sma20
        and sma20 > sma50
        and 50 <= rsi <= 75
        and volume > 1.5 * avg_vol
        and price >= 0.9 * year_high
    )


def scan_stocks() -> pd.DataFrame:
    results = []

    for ticker in NSE_TICKERS:
        df = fetch_stock_data(ticker)
        if df.empty:
            continue

        df = add_indicators(df)
        latest = df.dropna(subset=["sma20", "sma50", "rsi14", "avg_vol_20"]).tail(1)
        if latest.empty:
            continue

        row = latest.iloc[0]
        if is_bullish_signal(row):
            results.append(
                {
                    "ticker": ticker,
                    "price": round(float(row["close"]), 2),
                    "sma20": round(float(row["sma20"]), 2),
                    "sma50": round(float(row["sma50"]), 2),
                    "rsi14": round(float(row["rsi14"]), 2),
                    "volume": int(row["volume"]),
                    "avg_vol_20": round(float(row["avg_vol_20"]), 2),
                    "year_high": round(float(row["year_high"]), 2),
                }
            )

    if not results:
        print("No stocks match the current scanner conditions.")
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("rsi14", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    results = scan_stocks()
    if not results.empty:
        print(results.to_string(index=False))
        # Optional: save to CSV
        # results.to_csv("indian_stock_scan.csv", index=False)
