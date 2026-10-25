import pandas as pd
import numpy as np
import yfinance as yf

# Indian stocks to screen
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "ITC.NS", "SBIN.NS", "LTIM.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "WIPRO.NS",
    "TATACONSUM.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS"
]


def safe_float(value):
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def get_latest_financials(ticker):
    t = yf.Ticker(ticker)
    info = t.info

    pe = safe_float(info.get("trailingPE"))
    roe = safe_float(info.get("returnOnEquity")) / 100.0
    profit_margin = safe_float(info.get("profitMargins"))
    debt_to_equity = safe_float(info.get("debtToEquity"))
    current_ratio = safe_float(info.get("currentRatio"))
    market_cap = safe_float(info.get("marketCap"))

    rev_df = t.financials.loc["Total Revenue"] if "Total Revenue" in t.financials.index else pd.Series(dtype=float)
    net_income_df = t.financials.loc["Net Income"] if "Net Income" in t.financials.index else pd.Series(dtype=float)

    if not rev_df.empty:
        rev_df = rev_df.dropna().sort_index()
    if not net_income_df.empty:
        net_income_df = net_income_df.dropna().sort_index()

    revenue_cagr = np.nan
    if len(rev_df) >= 2:
        years = list(rev_df.index)
        try:
            latest_rev = float(rev_df.iloc[-1])
            old_rev = float(rev_df.iloc[-3]) if len(rev_df) >= 3 else float(rev_df.iloc[0])
            if old_rev > 0 and latest_rev > 0:
                revenue_cagr = (latest_rev / old_rev) ** (1 / max(1, min(3, len(rev_df) - 1))) - 1
        except Exception:
            pass

    avg_net_margin = np.nan
    if not net_income_df.empty and not rev_df.empty:
        try:
            avg_margin = 0
            count = 0
            for idx in range(len(net_income_df)):
                if idx < len(rev_df):
                    ni = float(net_income_df.iloc[idx])
                    rev = float(rev_df.iloc[idx])
                    if rev > 0:
                        avg_margin += ni / rev
                        count += 1
            if count > 0:
                avg_net_margin = avg_margin / count
        except Exception:
            pass

    return {
        "ticker": ticker,
        "pe": pe,
        "roe": roe,
        "profit_margin": profit_margin,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "revenue_cagr_3y": revenue_cagr,
        "avg_net_margin": avg_net_margin,
        "market_cap": market_cap,
    }


def is_fundamentally_good(data):
    if pd.isna(data["pe"]) or pd.isna(data["roe"]) or pd.isna(data["profit_margin"]):
        return False

    return (
        (data["pe"] > 0 and data["pe"] < 25)
        and (data["roe"] > 0.15)
        and (data["profit_margin"] > 0.10)
        and (data["current_ratio"] > 1.2)
        and (data["debt_to_equity"] < 1.0 or pd.isna(data["debt_to_equity"]))
        and (pd.isna(data["revenue_cagr_3y"]) or data["revenue_cagr_3y"] > 0.08)
        and (pd.isna(data["avg_net_margin"]) or data["avg_net_margin"] > 0.08)
    )


def screen_fundamental_stocks():
    results = []
    for ticker in TICKERS:
        try:
            data = get_latest_financials(ticker)
            if is_fundamentally_good(data):
                results.append(data)
        except Exception:
            pass

    df = pd.DataFrame(results)
    if df.empty:
        print("No fundamentally strong stocks found with the current criteria.")
        return

    df = df.sort_values(["roe", "pe"], ascending=[False, True]).reset_index(drop=True)
    print(
        df[
            [
                "ticker",
                "pe",
                "roe",
                "profit_margin",
                "current_ratio",
                "debt_to_equity",
                "revenue_cagr_3y",
                "market_cap",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    screen_fundamental_stocks()
