import pandas as pd
import numpy as np

class EVLabeler:
    @staticmethod
    def label_single(df: pd.DataFrame, idx: int, atr_mult: float, rrr: float, side: int):
        """
        Labels a single candle for a specific side (1=Buy, 0=Sell).
        Returns a Series containing features + the target (failure) + inverted outcome.
        """
        # Get entry details
        row = df.loc[idx].copy()
        entry_price = row['close']
        atr = row['atr']
        
        # Calculate SL and TP distances
        sl_dist = atr * atr_mult
        tp_dist = sl_dist * rrr
        
        # Set boundaries based on side
        if side == 1:  # Retail tries to BUY
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:          # Retail tries to SELL
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist
            
        # Scan future candles (up to 300 bars for Gold M5)
        outcome = 0 # 0 = Timed out (ignored by miner)
        start_pos = df.index.get_loc(idx) + 1
        future_data = df.iloc[start_pos : start_pos + 300]
        
        highs = future_data['high'].values
        lows = future_data['low'].values
        
        for i in range(len(future_data)):
            if side == 1: # BUY Scenario
                if lows[i] <= sl_price:
                    outcome = -1.0  # Retail hit SL
                    break
                if highs[i] >= tp_price:
                    outcome = rrr   # Retail hit TP
                    break
            else: # SELL Scenario
                if highs[i] >= sl_price:
                    outcome = -1.0  # Retail hit SL
                    break
                if lows[i] <= tp_price:
                    outcome = rrr   # Retail hit TP
                    break
        
        # --- THE INVERSION LOGIC ---
        # Target = 1 means "Retail Failed" (This is what our model predicts)
        row['target'] = 1 if outcome == -1.0 else 0
        
        # Inverted Outcome R:
        # If retail failed (-1.0), we hit our TP (+RRR)
        # If retail succeeded (+RRR), we hit our SL (-1.0)
        if outcome == -1.0:
            row['inverted_outcome_r'] = rrr
        elif outcome > 0:
            row['inverted_outcome_r'] = -1.0
        else:
            row['inverted_outcome_r'] = 0
            
        row['side'] = side
        return row

    @staticmethod
    def label_candidates(df: pd.DataFrame, candidates: pd.DataFrame, atr_mult: float, rrr: float):
        """
        Legacy support: Labels a batch of candidates using a fixed side.
        Used if you want to run traditional single-direction audits.
        """
        labeled_results = []
        for idx in candidates.index:
            labeled_results.append(EVLabeler.label_single(df, idx, atr_mult, rrr, side=1))
        
        return pd.DataFrame(labeled_results)