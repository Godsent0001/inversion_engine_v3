import pandas as pd
import numpy as np

class EVLabeler:
    @staticmethod
    def label_candidates(df: pd.DataFrame, candidates: pd.DataFrame, rrr=2.0, atr_mult=2.0):
        """
        For every candidate, check if it hits TP or SL first.
        Outcome is recorded in 'R-multiples'.
        -1.0 = Hit SL
        +2.0 = Hit TP
        """
        results = []
        prices = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        indices = df.index
        
        # Process each candidate (entry point)
        for idx, row in candidates.iterrows():
            entry_price = row['close']
            atr = row['atr']
            
            # SL and TP levels based on ATR
            sl_dist = atr * atr_mult
            tp_dist = sl_dist * rrr
            
            sl_price = entry_price - sl_dist # For a LONG dummy
            tp_price = entry_price + tp_dist
            
            # Look forward in time to find exit
            outcome = 0
            start_pos = df.index.get_loc(idx) + 1
            
            # Scan future candles (up to 200 bars)
            for i in range(start_pos, min(start_pos + 200, len(df))):
                if lows[i] <= sl_price:
                    outcome = -1.0 # Loss
                    break
                if highs[i] >= tp_price:
                    outcome = rrr # Win
                    break
            
            results.append(outcome)
            
        candidates['outcome_r'] = results
        return candidates