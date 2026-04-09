"""Economic data release feeds for resolution sniping.

Monitors FRED (Federal Reserve Economic Data) for economic indicator
releases (CPI, unemployment, GDP) and generates signals when values
cross contract-relevant thresholds.
"""

from moneygone.data.economic.releases import (
    EconomicRelease,
    EconomicReleaseFeed,
    EconomicSignal,
    FREDConfig,
)

__all__ = [
    "EconomicRelease",
    "EconomicReleaseFeed",
    "EconomicSignal",
    "FREDConfig",
]
