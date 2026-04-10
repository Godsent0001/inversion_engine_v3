import pandas as pd
import numpy as np

class SignalGenerator:
    def __init__(self, failure_threshold=0.75):
        """
        V4 Signal Generator: Handles Dual-Side Prediction & Dynamic Math.
        """
        self.threshold = failure_threshold

    def generate_v4_signal(self, model, feature_row, strategy_params, current_price, current_atr):
        """
        Determines if a trade should be taken by checking both Buy-Trap and Sell-Trap.
        
        feature_row: The current indicators (MEO, MPO, etc.)
        strategy_params: Dictionary containing 'atr_multiplier' and 'rrr' for THIS model.
        """
        # 1. Prepare Dual-Side Inputs
        # We need to ask the model: "What if Retail Buys?" AND "What if Retail Sells?"
        buy_context = feature_row.copy()
        buy_context['side'] = 1  # Retail Buy Attempt
        
        sell_context = feature_row.copy()
        sell_context['side'] = 0 # Retail Sell Attempt

        # 2. Get Failure Probabilities
        # We need to convert the row to a DataFrame format for LightGBM
        prob_buy_fails = model.predict(pd.DataFrame([buy_context]))[0]
        prob_sell_fails = model.predict(pd.DataFrame([sell_context]))[0]

        # 3. Dynamic Math Settings
        atr_mult = strategy_params['atr_multiplier']
        rrr = strategy_params['rrr']
        sl_dist = current_atr * atr_mult
        tp_dist = sl_dist * rrr

        # 4. Conflict Resolution & Signal Generation
        # Logic: If Retail Buy is a trap (>= 0.75) AND it's a stronger trap than Sell
        if prob_buy_fails >= self.threshold and prob_buy_fails > prob_sell_fails:
            return {
                'type': 'SELL',  # Invert the Buy-Trap
                'entry': current_price,
                'sl': current_price + sl_dist,
                'tp': current_price - tp_dist,
                'confidence': round(prob_buy_fails, 3),
                'side_id': 'INVERT_BUY'
            }

        # Logic: If Retail Sell is a trap (>= 0.75) AND it's a stronger trap than Buy
        elif prob_sell_fails >= self.threshold and prob_sell_fails > prob_buy_fails:
            return {
                'type': 'BUY',   # Invert the Sell-Trap
                'entry': current_price,
                'sl': current_price - sl_dist,
                'tp': current_price + tp_dist,
                'confidence': round(prob_sell_fails, 3),
                'side_id': 'INVERT_SELL'
            }

        return None # No high-confidence trap detected