# --- add imports at top if missing ---
import re
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Iterable, Tuple
# -------------------------------------

# ==== Robust label matching ====
_EQUITY_ROWS = [
    "Total Stockholder Equity", "Total Stockholders Equity", "Total Stockholders' Equity",
    "Total Shareholder Equity", "Total Shareholders' Equity",
    "Common Stock Equity", "Total Equity", "Total Equity Gross Minority Interest",
]
_LIAB_ROWS = [
    "Total Liab", "Total Liabilities", "Total Liabilities Net Minority Interest",
]
_LIAB_EQTY_COMBINED_ROWS = [
    "Total Liabilities & Stockholders' Equity", "Total Liabilities And Stockholders Equity",
    "Liabilities & Stockholders' Equity", "Liabilities And Stockholders Equity",
    "Total Liabilities and Stockholders' Equity",
]
_TOTAL_ASSETS_ROWS = ["Total Assets", "Total Assets Net Minority Interest"]

_EPS_ROWS = [
    "Diluted EPS", "EPS (Diluted)", "Basic EPS", "EPS (Basic)", "Earnings Per Share",
]

def _norm_key(s: str) -> str:
    """lowercase; replace & with and; strip non-alnum."""
    s = str(s).lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", s)

def _first_match_row(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    # Normalize index once
    idx_map = {_norm_key(i): i for i in df.index}
    # 1) exact normalized match
    for c in candidates:
        key = _norm_key(c)
        if key in idx_map:
            return df.loc[idx_map[key]]
    # 2) contains match
    for c in candidates:
        key = _norm_key(c)
        for k, orig in idx_map.items():
            if key in k or k in key:
                return df.loc[orig]
    return None

# ==== Equity start/end (annual) ====
def get_equity_start_end(ticker: str, target_years: int = 5) -> Tuple[Optional[float], Optional[float], int]:
    tk = yf.Ticker(ticker)
    abs_df = tk.balance_sheet  # annual BS (rows=items, cols=period ends)
    if not isinstance(abs_df, pd.DataFrame) or abs_df.empty:
        return (None, None, target_years)

    equity_row = _first_match_row(abs_df, _EQUITY_ROWS)
    if equity_row is None:
        # try combined row as a weak fallback (rare)
        equity_row = _first_match_row(abs_df, _LIAB_EQTY_COMBINED_ROWS)
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

    # pick earliest that is ~5y back; else first
    pick = 0
    for i, d in enumerate(dates):
        if _years_between(d, latest_date) >= (target_years - 1):
            pick = i
            break
    start_val = float(eq.iloc[pick])
    years = max(1, _years_between(dates[pick], latest_date))
    return (start_val, latest_val, years)

# ==== Liabilities + Equity (latest) ====
def get_total_liabilities_plus_equity_latest(ticker: str) -> Optional[float]:
    """
    Compute Liabilities + Equity for the latest available period.
    Try combined row; else sum liabilities + equity; final fallback = total assets.
    Picks the true latest column by date (handles tz-aware/naive and ordering).
    """
    tk = yf.Ticker(ticker)
    df = tk.quarterly_balance_sheet
    if not isinstance(df, pd.DataFrame) or df.empty:
        df = tk.balance_sheet
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    # pick latest column by date (not by position)
    cols = list(df.columns)
    parsed = pd.to_datetime(pd.Index(cols), errors="coerce")
    if parsed.notna().any():
        latest_idx = int(parsed.fillna(pd.Timestamp("1900-01-01")).argmax())
        latest_col = cols[latest_idx]
    else:
        latest_col = cols[0]  # fallback: assume first is most recent

    # Try explicit combined row first
    combined = _first_match_row(df, _LIAB_EQTY_COMBINED_ROWS)
    if combined is not None:
        val = pd.to_numeric(combined.get(latest_col), errors="coerce")
        return float(val) if pd.notna(val) else None

    # Sum liabilities + equity
    liab = _first_match_row(df, _LIAB_ROWS)
    eqty = _first_match_row(df, _EQUITY_ROWS)
    if liab is not None and eqty is not None:
        liab_val = pd.to_numeric(liab.get(latest_col), errors="coerce")
        eqty_val = pd.to_numeric(eqty.get(latest_col), errors="coerce")
        if pd.notna(liab_val) and pd.notna(eqty_val):
            return float(liab_val + eqty_val)

    # Final fallback: Total Assets ≈ Liabilities + Equity
    assets = _first_match_row(df, _TOTAL_ASSETS_ROWS)
    if assets is not None:
        val = pd.to_numeric(assets.get(latest_col), errors="coerce")
        return float(val) if pd.notna(val) else None

    return None


# ==== TTM EPS ====
def get_ttm_eps(ticker: str) -> Optional[float]:
    tk = yf.Ticker(ticker)
    qis = tk.quarterly_income_stmt
    if isinstance(qis, pd.DataFrame) and not qis.empty:
        eps_row = _first_match_row(qis, _EPS_ROWS)
        if eps_row is not None:
            eps = pd.to_numeric(eps_row.sort_index(), errors="coerce").dropna()
            if len(eps) >= 4:
                return float(eps.tail(4).sum())
    # fallback to trailingEps field
    info = getattr(tk, "get_info", lambda: {})()
    if not isinstance(info, dict):
        info = getattr(tk, "info", {}) or {}
    ttm = info.get("trailingEps")
    return float(ttm) if ttm is not None else None

# ==== 5y min/max P/E (robust) ====
def get_pe_min_max_5y(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    tk = yf.Ticker(ticker)

    # Build monthly TTM EPS series from quarterly EPS (preferred)
    ttm_eps_m = None
    qis = tk.quarterly_income_stmt
    if isinstance(qis, pd.DataFrame) and not qis.empty:
        eps_row = _first_match_row(qis, _EPS_ROWS)
        if eps_row is not None:
            eps = pd.to_numeric(eps_row.sort_index(), errors="coerce").dropna()
            ttm_q = eps.rolling(4).sum().dropna()
            ttm_eps_m = (
                ttm_q.to_frame("ttm_eps")
                .set_index(pd.to_datetime(ttm_q.index))
                .resample("ME")  # month-end
                .ffill()["ttm_eps"]
            )

    # Monthly prices
    px = tk.history(period="5y", interval="1mo", auto_adjust=True)
    if isinstance(px, pd.DataFrame) and not px.empty:
        close_m = px["Close"] if "Close" in px.columns else px.get("Adj Close")
        if isinstance(close_m, pd.DataFrame):
            close_m = close_m.squeeze("columns")
    else:
        return (None, None)

    if close_m is None or close_m.empty:
        return (None, None)
    close_m = pd.to_numeric(close_m, errors="coerce").dropna()
    close_m.index = pd.to_datetime(close_m.index)
    close_m.name = "close"

    if ttm_eps_m is None or ttm_eps_m.empty:
        # fallback: static P/E using current TTM across 5y prices
        ttm = get_ttm_eps(ticker)
        if ttm is None or ttm <= 0:
            return (None, None)
        pe = close_m / ttm
    else:
        joined = pd.conca
