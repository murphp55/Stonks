# yfinance_api.py
# Minimal pulls for: TTM EPS, ~5y equity start/end (+CAGR span),
# Total Liabilities + Equity (latest), and 5y min/max P/E.
from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from stock import StockBasics


# ---------- label helpers ----------

# Based on labels you printed from yfinance for MSFT/AAPL
_EQUITY_ROWS = [
    "Common Stock Equity",
    "Stockholders Equity",
    "Total Stockholder Equity",
    "Total Stockholders Equity",
    "Total Shareholders' Equity",
    "Total Equity",
    "Total Equity Gross Minority Interest",
]
_LIAB_ROWS = [
    "Total Liabilities Net Minority Interest",
    "Total Liabilities",
    "Total Liab",
]
_LIAB_EQTY_COMBINED_ROWS = [
    "Total Liabilities and Stockholders' Equity",
    "Total Liabilities & Stockholders' Equity",
    "Liabilities and Stockholders Equity",
]
_TOTAL_ASSETS_ROWS = ["Total Assets", "Total Assets Net Minority Interest"]
_EPS_ROWS = [
    "Diluted EPS", "EPS (Diluted)",
    "Basic EPS", "EPS (Basic)",
    "Earnings Per Share",
]


def _norm_key(s: str) -> str:
    """Lowercase, replace '&' with 'and', strip non-alnum."""
    s = str(s).lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", s)


def _first_match_row(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    """Find first matching row (index label) in df for any candidate name."""
    if df is None or df.empty:
        return None
    idx_map = {_norm_key(i): i for i in df.index}
    # exact normalized match
    for c in candidates:
        key = _norm_key(c)
        if key in idx_map:
            return df.loc[idx_map[key]]
    # contains match (lenient)
    for c in candidates:
        key = _norm_key(c)
        for k, orig in idx_map.items():
            if key in k or k in key:
                return df.loc[orig]
    return None


# ---------- core data pulls ----------

def get_ttm_eps(ticker: str) -> Optional[float]:
    """Prefer quarterly EPS rolling sum; fallback to trailingEps field."""
    tk = yf.Ticker(ticker)
    qis = tk.quarterly_income_stmt  # rows=items, cols=period end dates
    if isinstance(qis, pd.DataFrame) and not qis.empty:
        eps_row = _first_match_row(qis, _EPS_ROWS)
        if eps_row is not None:
            eps = pd.to_numeric(eps_row.sort_index(), errors="coerce").dropna()
            if len(eps) >= 4:
                return float(eps.tail(4).sum())
    # fallback
    info = getattr(tk, "get_info", lambda: {})()
    if not isinstance(info, dict):
        info = getattr(tk, "info", {}) or {}
    ttm = info.get("trailingEps")
    return float(ttm) if ttm is not None else None


def get_equity_start_end(ticker: str, target_years: int = 5) -> Tuple[Optional[float], Optional[float], int]:
    """
    Return (equity_start, equity_end, years_between_points).
    Uses ANNUAL balance sheet equity. If fewer than 5 years,
    uses earliest → latest available and returns the actual span.
    """
    tk = yf.Ticker(ticker)
    abs_df = tk.balance_sheet  # annual
    if not isinstance(abs_df, pd.DataFrame) or abs_df.empty:
        return (None, None, target_years)

    equity_row = _first_match_row(abs_df, _EQUITY_ROWS)
    if equity_row is None:
        return (None, None, target_years)

    eq = pd.to_numeric(equity_row.sort_index(), errors="coerce").dropna()
    if eq.empty:
        return (None, None, target_years)

    dates = list(eq.index)
    latest_date = dates[-1]
    latest_val = float(eq.iloc[-1])

    def _years_between(d1, d2) -> int:
        return abs(pd.Timestamp(d2).year - pd.Timestamp(d1).year)

    # pick earliest value that is roughly target_years back; else first available
    pick = 0
    for i, d in enumerate(dates):
        if _years_between(d, latest_date) >= (target_years - 1):
            pick = i
            break
    start_val = float(eq.iloc[pick])
    years = max(1, _years_between(dates[pick], latest_date))
    return (start_val, latest_val, years)


def get_total_liabilities_plus_equity_latest(ticker: str) -> Optional[float]:
    """
    Latest Liabilities + Equity:
      1) Prefer 'Total Assets' (equal to L+E)
      2) Else sum Liabilities + Equity rows
      3) Else try a combined 'Liabilities and Stockholders' Equity' row
    We choose the latest column by actual date (handles tz-aware/naive & ordering).
    """
    tk = yf.Ticker(ticker)
    df = tk.quarterly_balance_sheet
    if not isinstance(df, pd.DataFrame) or df.empty:
        df = tk.balance_sheet
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    # Choose true latest column by date (not by position)
    cols = pd.Index(df.columns)
    parsed = pd.to_datetime(cols, errors="coerce")
    latest_col = cols[parsed.argmax()] if parsed.notna().any() else cols[0]

    # 1) Prefer Total Assets (identical to Liabilities + Equity)
    assets = _first_match_row(df, _TOTAL_ASSETS_ROWS)
    if assets is not None:
        val = pd.to_numeric(assets[latest_col], errors="coerce")
        if pd.notna(val):
            return float(val)

    # 2) Sum Liabilities + Equity
    liab = _first_match_row(df, _LIAB_ROWS)
    eqty = _first_match_row(df, _EQUITY_ROWS)
    if liab is not None and eqty is not None:
        liab_val = pd.to_numeric(liab[latest_col], errors="coerce")
        eqty_val = pd.to_numeric(eqty[latest_col], errors="coerce")
        if pd.notna(liab_val) and pd.notna(eqty_val):
            return float(liab_val + eqty_val)

    # 3) Combined row (rare on Yahoo)
    combined = _first_match_row(df, _LIAB_EQTY_COMBINED_ROWS)
    if combined is not None:
        val = pd.to_numeric(combined[latest_col], errors="coerce")
        if pd.notna(val):
            return float(val)

    return None




def get_pe_min_max_5y(ticker: str) -> tuple[Optional[float], Optional[float]]:
    """
    Compute monthly P/E over ~5y:
      - monthly close price (auto-adjusted)
      - TTM EPS time series built from quarterly EPS rolling sum
    Return (min_PE, max_PE) ignoring non-positive EPS values.
    """
    tk = yf.Ticker(ticker)

    def _drop_tz_index(s: pd.Series) -> pd.Series:
        idx = pd.DatetimeIndex(s.index)
        # If tz-aware, convert to UTC then drop tz; else just ensure DatetimeIndex
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        s.index = idx
        return s

    # 1) Quarterly EPS → rolling TTM → resample to month-end
    ttm_eps_monthly = None
    qis = tk.quarterly_income_stmt
    if isinstance(qis, pd.DataFrame) and not qis.empty:
        eps_row = _first_match_row(qis, _EPS_ROWS)
        if eps_row is not None:
            eps = pd.to_numeric(eps_row.sort_index(), errors="coerce").dropna()
            ttm_q = eps.rolling(4).sum().dropna()
            ttm_eps_monthly = (
                ttm_q.to_frame("ttm_eps")
                .set_index(pd.to_datetime(ttm_q.index))
                .resample("ME")  # month-end (M is deprecated)
                .ffill()["ttm_eps"]
            )
            ttm_eps_monthly = _drop_tz_index(ttm_eps_monthly)

    # 2) Prices: 5y monthly close
    px = tk.history(period="5y", interval="1mo", auto_adjust=True)
    if not isinstance(px, pd.DataFrame) or px.empty:
        return (None, None)

    close_m = px["Close"] if "Close" in px.columns else px.get("Adj Close")
    if close_m is None or close_m.empty:
        return (None, None)
    if isinstance(close_m, pd.DataFrame):
        close_m = close_m.squeeze("columns")
    close_m = pd.to_numeric(close_m, errors="coerce").dropna()
    close_m.index = pd.to_datetime(close_m.index)
    close_m = _drop_tz_index(close_m)
    # Normalize to exact month-end stamps to match EPS
    close_m = close_m.resample("ME").last()
    close_m.name = "close"

    # 3) Join & compute P/E
    if ttm_eps_monthly is None or ttm_eps_monthly.empty:
        # fallback: static P/E using current TTM across 5y prices
        ttm = get_ttm_eps(ticker)
        if ttm is None or ttm <= 0:
            return (None, None)
        pe = close_m / ttm
    else:
        joined = pd.concat([close_m, ttm_eps_monthly.rename("ttm_eps")], axis=1).dropna()
        joined = joined[joined["ttm_eps"] > 0]
        if joined.empty:
            return (None, None)
        pe = joined["close"] / joined["ttm_eps"]

    last_60 = pe.tail(60)
    if last_60.empty:
        return (None, None)
    return (float(last_60.min()), float(last_60.max()))



# ---------- facade ----------

def fetch_stock_basics(ticker: str) -> StockBasics:
    """Return a populated StockBasics for the given ticker."""
    ttm_eps = get_ttm_eps(ticker)
    eq_start, eq_end, years = get_equity_start_end(ticker, target_years=5)
    liab_plus_eq = get_total_liabilities_plus_equity_latest(ticker)
    pe_min, pe_max = get_pe_min_max_5y(ticker)

    return StockBasics(
        ticker=ticker,
        ttm_eps=ttm_eps,
        equity_start=eq_start,
        equity_end=eq_end,
        equity_cagr_years=years,
        total_liabilities_and_equity_latest=liab_plus_eq,
        analyst_5y_growth_estimate=None,   # intentionally None for now
        pe_5y_min=pe_min,
        pe_5y_max=pe_max,
    )
