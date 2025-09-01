from __future__ import annotations
import requests

# Minimal map from common names/symbols → CoinGecko IDs
COIN_MAP = {
    "btc": "bitcoin", "bitcoin": "bitcoin",
    "eth": "ethereum", "ethereum": "ethereum",
    "sol": "solana", "solana": "solana",
    "xrp": "ripple", "ripple": "ripple",
}

def resolve_coin_id(text_or_symbol: str, default: str = "bitcoin") -> str:
    t = (text_or_symbol or "").lower().strip()
    # try exact-in-text matches first (longest keys first)
    for key in sorted(COIN_MAP.keys(), key=len, reverse=True):
        if key in t.split() or key in t:
            return COIN_MAP[key]
    return default

def get_price(coin_id: str = "bitcoin", vs: str = "usd"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies={vs}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get(coin_id, {}).get(vs)

def get_price_any(coin_or_query: str, vs: str = "usd"):
    coin_id = resolve_coin_id(coin_or_query)
    return coin_id, get_price(coin_id, vs)

def get_price_multi(coin_ids: list[str], vs: str = "usd") -> dict:
    # Efficient batch call (one request) e.g. ["bitcoin","ethereum","solana","ripple"]
    unique = ",".join(sorted(set(coin_ids)))
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={unique}&vs_currencies={vs}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def get_fear_greed():
    url = "https://api.alternative.me/fng/"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("data"):
        return data["data"][0]
    return None
