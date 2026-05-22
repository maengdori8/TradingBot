import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def load_config(path: str = None) -> dict:
    if path is None:
        path = BASE_DIR / "config" / "settings.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_api_credentials() -> dict:
    return {
        "api_key": os.getenv("BYBIT_API_KEY", ""),
        "api_secret": os.getenv("BYBIT_API_SECRET", ""),
        "testnet": os.getenv("BYBIT_TESTNET", "true").lower() == "true",
    }
