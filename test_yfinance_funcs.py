
import pandas as pd
import yfinance as yf

symbol = "AAPL"
ticker = yf.Ticker(symbol)

eps_ttm = ticker.info.get("trailingEps")
print("TTM EPS:", eps_ttm)

# 1. Get historical prices (daily)
hist = ticker.history(period="4y")

# 2. Get annual income statement (financials)
annual_income_stmt = ticker.financials

# 3. Extract Net Income for each year
net_income = annual_income_stmt.loc['Net Income'].sort_index()

# 4. Get shares outstanding (current)
# fixme yfinance can't grab the sharesOutstanding at the date of a given report, so it skews everything to current share quantities
# Can fix by pulling annual reports and parsing...
shares_outstanding = ticker.info.get('sharesOutstanding')

# 5. Calculate annual EPS = Net Income / Shares Outstanding
annual_eps = net_income / shares_outstanding

print("Annual EPS:")
print(annual_eps)

# 6. Get closing price on the last trading day of each year
prices = hist['Close']
year_end_prices = prices.resample('Y').last().sort_index()

print("\nYear-end Prices:")
print(year_end_prices)

# 7. Calculate P/E ratio for each year
pe_ratios = year_end_prices / annual_eps

print("\nAnnual P/E Ratios:")
print(pe_ratios)

print(f"\nMin Annual P/E over last 5 years: {pe_ratios.min()}")
print(f"Max Annual P/E over last 5 years: {pe_ratios.max()}")


breakpoint()
