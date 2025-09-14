# stock.py
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass(slots=True)
class StockBasics:
    """
    Minimal per-stock metrics for Stonky.
    Units:
      - EPS, equity, liabilities+equity in currency (e.g., USD)
      - Growth/CAGR as decimals (0.10 = 10%)
      - P/E is a ratio
    """
    ticker: str

    # Core fields you’ll populate
    ttm_eps: Optional[float] = None                               # 12-month trailing EPS
    equity_start: Optional[float] = None                           # equity value ~5 yrs ago
    equity_end: Optional[float] = None                             # latest equity value
    equity_cagr_years: int = 5                                     # actual span used
    total_liabilities_and_equity_latest: Optional[float] = None    # from Balance Sheet
    analyst_5y_growth_estimate: Optional[float] = None             # decimal (optional)
    pe_5y_min: Optional[float] = None                              # 5-year min P/E
    pe_5y_max: Optional[float] = None                              # 5-year max P/E

    # Convenience/computed properties
    @property
    def equity_cagr(self) -> Optional[float]:
        """CAGR from start → end over equity_cagr_years (as decimal)."""
        if (
            self.equity_start is None
            or self.equity_end is None
            or self.equity_start <= 0
            or self.equity_cagr_years <= 0
        ):
            return None
        return (self.equity_end / self.equity_start) ** (1 / self.equity_cagr_years) - 1

    @property
    def pe_range(self) -> Optional[Tuple[float, float]]:
        """(min_PE, max_PE) if both are available."""
        if self.pe_5y_min is None or self.pe_5y_max is None:
            return None
        return (self.pe_5y_min, self.pe_5y_max)

    def __post_init__(self):
        self.ticker = self.ticker.upper()
