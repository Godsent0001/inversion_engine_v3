import pandas as pd
import numpy as np

class WalkForwardValidator:
    @staticmethod
    def validate_robustness(model, df, features, n_slices=3, multiplier=2.0, spread_tax=0.05):
        """
        Splits the 6-month data into 3 rolling windows.
        A strategy MUST have a positive Inverted EV in all 3 slices to pass.
        """
        # Ensure we are working with a DataFrame if it's coming from np.array_split
        if isinstance(df, pd.DataFrame):
            slices = np.array_split(df, n_slices)
        else:
            slices = np.array_split(df, n_slices)

        slice_results = []

        for i in range(len(slices)):
            test_slice = slices[i]
            # Convert back to DataFrame if np.array_split turned it into an ndarray
            if isinstance(test_slice, np.ndarray):
                test_slice = pd.DataFrame(test_slice, columns=df.columns)
            
            # Predict failures on this slice
            probs = model.predict(test_slice[features])
            
            # Calculate Inverted Returns (Flip the -1.0Rs into +1.0Rs)
            # Simplified: If prob > 0.75, we take the negative of the outcome
            # Ensure outcome_r is numeric
            outcomes = test_slice['outcome_r'].values.astype(float)
            returns = np.where(probs >= 0.75, -outcomes, 0)
            
            # Apply Spread Tax
            active_returns = returns[returns != 0]
            slice_ev = np.mean(active_returns - spread_tax) if len(active_returns) > 0 else -1
            slice_results.append(slice_ev)

        # Robustness Check: Are all slices profitable?
        is_robust = all(res > 0.05 for res in slice_results)
        print(f"🔄 Walk-Forward EVs: {[round(r, 3) for r in slice_results]} | Robust: {is_robust} (ATR Mult: {multiplier})")
        return is_robust