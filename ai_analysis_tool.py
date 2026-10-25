#!/usr/bin/env python3
"""AI-powered stock analysis helper for the trading repo.

Usage examples:
    python ai_analysis_tool.py RELIANCE.NS
    python ai_analysis_tool.py --all
    OPENAI_API_KEY=... python ai_analysis_tool.py --ticker TCS.NS
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

from stock_config import load_tickers

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fetch_market_snapshot(ticker: str) -> dict[str, Any]:
    """Build a compact market snapshot for AI analysis."""
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    hist = stock.history(period="6mo", interval="1d", auto_adjust=True)

    if hist.empty:
        raise ValueError(f"No market data available for {ticker}.")

    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna()
    if close.empty:
        raise ValueError(f"No close-price data available for {ticker}.")

    price = _safe_float(info.get("regularMarketPrice"), default=float(close.iloc[-1]))
    previous_close = _safe_float(info.get("regularMarketPreviousClose"), default=float(close.iloc[-2]) if len(close) > 1 else price)
    change_pct = ((price - previous_close) / previous_close * 100.0) if previous_close else 0.0
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    rsi14 = RSIIndicator(close, window=14).rsi().iloc[-1]
    volume_avg_20 = volume.rolling(20).mean().iloc[-1] if not volume.empty else 0.0
    volume_ratio = (float(volume.iloc[-1]) / float(volume_avg_20)) if volume_avg_20 else 0.0

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "change_pct": round(change_pct, 2),
        "sma20": round(float(sma20), 2),
        "sma50": round(float(sma50), 2),
        "rsi14": round(float(rsi14), 2),
        "volume_ratio": round(volume_ratio, 2),
        "market_cap": _safe_float(info.get("marketCap")),
        "pe_ratio": _safe_float(info.get("trailingPE")),
        "roe": _safe_float(info.get("returnOnEquity")),
        "profit_margin": _safe_float(info.get("profitMargins")),
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
        "current_ratio": _safe_float(info.get("currentRatio")),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


def build_analysis_prompt(snapshot: dict[str, Any]) -> str:
    return f"""
You are an experienced stock analyst. Evaluate the following ticker using technical and fundamental indicators.

Ticker snapshot:
{json.dumps(snapshot, indent=2, ensure_ascii=False)}

Return valid JSON only with the following keys:
- overall_signal: one of "bullish", "neutral", or "bearish"
- summary: a 2-4 sentence view of the setup
- key_reasons: a list of 3 concise reasons supporting the signal
- risk_factors: a list of 3 concise risk points
- action: a one-sentence actionable idea such as "Watch for breakout", "Scale in gradually", or "Avoid for now"

Keep the analysis practical and balanced. Do not claim certainty about future price movement.
"""


def call_openai_analysis(snapshot: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running the AI analysis tool."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The openai package is not installed. Pip install -r requirements.txt") from exc

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a disciplined stock analyst focused on risk-aware investment commentary."},
            {"role": "user", "content": build_analysis_prompt(snapshot)},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned no content for the stock analysis.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI returned malformed JSON: {content}") from exc


def analyze_ticker(ticker: str) -> dict[str, Any]:
    snapshot = fetch_market_snapshot(ticker)
    ai_result = call_openai_analysis(snapshot)
    return {
        "ticker": ticker,
        "snapshot": snapshot,
        "analysis": ai_result,
    }


def analyze_tickers(tickers: list[str]) -> list[dict[str, Any]]:
    return [analyze_ticker(ticker) for ticker in tickers]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI stock analysis tool")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol to analyze, e.g. RELIANCE.NS")
    parser.add_argument("--all", action="store_true", help="Analyze all configured tickers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        tickers = load_tickers() if args.all else [args.ticker or "RELIANCE.NS"]
        results = analyze_tickers(tickers)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as exc:  # pragma: no cover - CLI safety
        raise SystemExit(str(exc))
