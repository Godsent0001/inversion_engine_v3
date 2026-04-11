import pandas as pd
import numpy as np

class EVLabeler:
    @staticmethod
    def label_single(df: pd.DataFrame, idx: int, atr_mult: float, rrr: float, side: int):
        """
        Labels a single candle for a specific side (1=Buy, 0=Sell).
        Returns a Series containing features + the target (failure) + inverted outcome.
        Optimized using NumPy vectorization.
        """
        # Get entry details
        row = df.loc[idx].copy()
        entry_price = row['close']
        atr = row['atr']
        
        # Calculate SL and TP distances
        sl_dist = atr * atr_mult
        tp_dist = sl_dist * rrr

        # Scan future candles (up to 300 bars for Gold M5)
        start_pos = df.index.get_loc(idx) + 1
        future_data = df.iloc[start_pos : start_pos + 300]

        if future_data.empty:
            row['target'] = 0
            row['inverted_outcome_r'] = 0
            row['side'] = side
            return row

        highs = future_data['high'].values
        lows = future_data['low'].values

        outcome = 0
        if side == 1:  # Retail attempted BUY
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist

            sl_hits = np.where(lows <= sl_price)[0]
            tp_hits = np.where(highs >= tp_price)[0]

            first_sl = sl_hits[0] if len(sl_hits) > 0 else 1000
            first_tp = tp_hits[0] if len(tp_hits) > 0 else 1000

            if first_sl < first_tp:
                outcome = -1.0
            elif first_tp < first_sl:
                outcome = rrr
        else:          # Retail attempted SELL
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

            sl_hits = np.where(highs >= sl_price)[0]
            tp_hits = np.where(lows <= tp_price)[0]

            first_sl = sl_hits[0] if len(sl_hits) > 0 else 1000
            first_tp = tp_hits[0] if len(tp_hits) > 0 else 1000

            if first_sl < first_tp:
                outcome = -1.0
            elif first_tp < first_sl:
                outcome = rrr

        # --- THE INVERSION LOGIC ---
        row['target'] = 1 if outcome == -1.0 else 0
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