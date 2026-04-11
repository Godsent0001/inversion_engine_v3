import numpy as np
import pandas as pd

class CandidateSampler:
    @staticmethod
    def get_random_candidates(df: pd.DataFrame, count=3000):
        """
        Selects random points in time to analyze for potential Liquidity Traps.
        In V4, we use pure randomness to let the AI learn the difference
        between 'Standard Price Action' and 'Failure Patterns'.
        """
        # 1. Look-ahead Protection:
        # We must ignore the most recent 300 candles (approx 25 hours on M5)
        # so the EVLabeler has enough future data to calculate outcomes.
        if len(df) < 400:
            return df.copy()

        valid_range = df.iloc[:-300]
        
        # 2. Reset the random seed
        # This ensures that every time 'main_mine.py' loops, it gets a
        # unique set of candles, creating a diversified strategy league.
        np.random.seed(None)

        # 3. Determine actual sample size
        # If the dataset is smaller than the requested count, take everything valid.
        actual_count = min(len(valid_range), count)

        # 4. Perform the random selection
        # We use 'replace=False' to ensure we don't train on the same candle twice.
        sample_indices = np.random.choice(
            valid_range.index,
            size=actual_count,
            replace=False
        )
        
        # 5. Extract and sort
        candidates = df.loc[sample_indices].copy()
        candidates.sort_index(inplace=True)

        print(f"🎲 Sampled {len(candidates)} unique points for AI Trap Analysis.")
        return candidates