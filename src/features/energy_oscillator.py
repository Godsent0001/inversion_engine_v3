import pandas as pd
import numpy as np

class MarketEnergyOscillator:
    def __init__(self, period=14, smooth=3):
        self.period = period
        self.smooth = smooth

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates 'Market Energy' based on True Range vs Historical Range.
        0 = Dead Market | 100 = Explosive/Exhaustion Phase.
        """
        # 1. Calculate True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # 2. Normalize Energy (0-100)
        # We compare current volatility to the max/min of the lookback period
        rolling_min = true_range.rolling(window=self.period * 5).min()
        rolling_max = true_range.rolling(window=self.period * 5).max()
        
        raw_energy = 100 * (true_range - rolling_min) / (rolling_max - rolling_min + 1e-8)

        # 3. Smoothing & Slope
        df['market_energy'] = raw_energy.ewm(span=self.smooth).mean()
        df['energy_slope'] = df['market_energy'].diff(3) # Velocity of energy build-up
        
        # 4. Energy Zones (Categorical for LightGBM)
        df['is_high_energy'] = (df['market_energy'] > 70).astype(int)
        df['is_low_energy'] = (df['market_energy'] < 30).astype(int)

        return df