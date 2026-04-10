import pandas as pd
import numpy as np

class WalkForwardValidator:
    @staticmethod
    def validate_robustness(model, df, features, n_slices=3, multiplier=2.0, spread_tax=0.15):
        """
        Splits the data into 3 chronological windows.
        The strategy MUST be profitable (EV > 0) in at least 2 out of 3 slices,
        and the average EV across all slices must be positive.
        """
        # Ensure data is sorted by time before splitting
        df = df.sort_index()
        
        # Split data into chronological chunks
        # np.array_split is used to maintain the time-series order
        slices = np.array_split(df, n_slices)
        slice_results = []

        for i, test_slice in enumerate(slices):
            if test_slice.empty:
                continue
                
            # 1. Predict failures on this slice
            probs = model.predict(test_slice[features])
            
            # 2. Calculate Inverted Returns 
            # In V4, we use 'inverted_outcome_r' which respects the AI's Auto-RRR
            returns = np.where(probs >= 0.75, test_slice['inverted_outcome_r'], 0)
            
            # 3. Apply Spread Tax only to active trades
            active_returns = returns[returns != 0]
            
            if len(active_returns) > 0:
                # Subtract the spread tax from each trade
                net_active = active_returns - spread_tax
                slice_ev = np.mean(net_active)
            else:
                # If the expert found NO trades in this 2-month window, it's not robust
                slice_ev = -1.0 
            
            slice_results.append(slice_ev)

        # --- THE ROBUSTNESS SCORE ---
        # 1. Calculate how many slices were profitable
        profitable_slices = sum(1 for res in slice_results if res > 0)
        
        # 2. Average EV across the whole period
        avg_ev = np.mean(slice_results)

        # A strategy is ROBUST if:
        # - It made money in at least 2 out of 3 time periods
        # - The average EV is above your minimum alpha threshold
        is_robust = (profitable_slices >= 2) and (avg_ev > 0.05)
        
        results_clean = [round(float(r), 3) for r in slice_results]
        status = "✅ ROBUST" if is_robust else "❌ FRAGILE"
        
        print(f"🔄 WFA {status} | Slice EVs: {results_clean} | Avg: {avg_ev:.3f} | ATR: {multiplier}")
        
        return is_robust