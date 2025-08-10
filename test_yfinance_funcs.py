
import pandas as pd
import yfinance as yf

ticker = yf.Ticker('TSLA')
eps_ttm = ticker.info.get("trailingEps")
print("TTM EPS:", eps_ttm)

breakpoint()


