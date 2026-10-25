import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("tickers.json")


def load_tickers() -> list[str]:
    """Load tickers from the environment or the repo config file."""
    env_value = os.getenv("NSE_TICKERS")
    if env_value:
        return [ticker.strip() for ticker in env_value.split(",") if ticker.strip()]

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            payload = json.load(config_file)

        tickers = payload.get("tickers", []) if isinstance(payload, dict) else payload
        if isinstance(tickers, list) and tickers:
            return [str(ticker).strip() for ticker in tickers if str(ticker).strip()]

    raise ValueError(
        "No stock tickers configured. Set the NSE_TICKERS environment variable or add "
        "tickers to tickers.json."
    )
