# main.py
from yfinance_api import fetch_stock_basics

def fmt_pct(x): return f"{x*100:.2f}%" if x is not None else "—"
def fmt_money(x): return "—" if x is None else f"${x/1e9:.1f}B"
def fmt_range(r): return "—" if not r else f"({r[0]:.2f}, {r[1]:.2f})"


if __name__ == "__main__":
    for t in ["MSFT", "AAPL"]:
        s = fetch_stock_basics(t)
        print(f"\n{t}  —  TTM EPS: {s.ttm_eps}")
        print(f"Equity start/end ({s.equity_cagr_years}y): {s.equity_start} → {s.equity_end}  |  CAGR: {fmt_pct(s.equity_cagr)}")
        print(f"Liabilities + Equity (latest): {s.total_liabilities_and_equity_latest}")
        print(f"5y P/E range: {s.pe_range or '—'}")

    # --- optional: quick label debug (uncomment if needed) ---
    # import yfinance as yf
    # tk = yf.Ticker("MSFT")
    # print(list(map(str, tk.balance_sheet.index)))
    # print(list(map(str, tk.quarterly_balance_sheet.index)))
    # print(list(map(str, tk.quarterly_income_stmt.index)))
