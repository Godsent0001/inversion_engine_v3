import pandas as pd
import numpy as np

class MarketPressureOscillator:
    def __init__(self, structure_lookback=30, compress_period=14):
        self.lookback = structure_lookback
        self.compress_p = compress_period

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MPO = Structure Bias * (Compression + Expansion)
        Optimized for XAUUSD 5m Volatility.
        """
        # 1. Structure Bias (SB): -1 (Bearish) to 1 (Bullish)
        # Uses Midpoint of recent High/Low range
        hi = df['high'].rolling(self.lookback).max()
        lo = df['low'].rolling(self.lookback).min()
        mid = (hi + lo) / 2
        df['mpo_bias'] = np.where(df['close'] > mid, 1, -1)

        # 2. Compression Index (CI): 0 to 1
        # Measures ATR contraction (Price coiling before a move)
        # Using a Standard Deviation of range for better 'squeeze' detection
        range_std = (df['high'] - df['low']).rolling(self.compress_p).std()
        min_std = range_std.rolling(self.compress_p * 3).min()
        max_std = range_std.rolling(self.compress_p * 3).max()
        
        # CI is 1 when market is extremely tight (coiled)
        df['compression_index'] = 1 - ((range_std - min_std) / (max_std - min_std + 1e-8))

        # 3. Expansion Trigger (ET): 0 to 1
        # Measures if current candle is exploding out of the average range
        avg_range = (df['high'] - df['low']).rolling(self.compress_p).mean()
        df['expansion_trigger'] = np.clip((df['high'] - df['low']) / (avg_range * 1.5), 0, 1)

        # 4. Final MPO Formula (-100 to +100)
        # MPO = SB * (50% Compression potential + 50% Expansion active)
        raw_mpo = df['mpo_bias'] * (df['compression_index'] * 50 + df['expansion_trigger'] * 50)
        
        # Smooth with EMA for cleaner LightGBM decision trees
        df['mpo'] = raw_mpo.ewm(span=5).mean()
        
        return df