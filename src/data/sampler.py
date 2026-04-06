import numpy as np
import pandas as pd

class CandidateSampler:
    @staticmethod
    def get_random_candidates(df: pd.DataFrame, target_monthly_count=3000):
        """
        Uses a probability mask to select potential 'Failure Points'.
        Each strategy run will get a DIFFERENT set of candidates.
        """
        # Calculate probability: Total candles in 6 months vs target
        total_candles = len(df)
        # 6 months x 3000 = 18,000 target points
        prob = (target_monthly_count * 6) / total_candles
        
        # Ensure prob doesn't exceed 1.0
        prob = min(prob, 0.8) 

        # Generate a random mask (Bernoulli trial per candle)
        np.random.seed(None) # Ensure different seed every time we mine
        df['is_candidate'] = np.random.choice(
            [0, 1], 
            size=len(df), 
            p=[1 - prob, prob]
        )
        
        candidates = df[df['is_candidate'] == 1].copy()
        print(f"🎲 Sampled {len(candidates)} candidates for this mining session.")
        return candidates