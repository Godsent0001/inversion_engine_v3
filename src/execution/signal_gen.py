import pandas as pd
import numpy as np

class SignalGenerator:
    def __init__(self, failure_threshold=0.75):
        """
        failure_threshold: If probability of failure > 75%, we flip.
        """
        self.threshold = failure_threshold

    def generate_inverted_signal(self, lgbm_model, feature_row, current_price, atr):
        """
        Takes the current market state and returns a Trade Dictionary.
        """
        # 1. Get Failure Probability (0.0 to 1.0)
        prob_failure = lgbm_model.predict(feature_row)[0]
        
        signal = None
        
        # 2. INVERSION LOGIC
        # If the 'Dummy Long' is highly likely to fail, we go SHORT.
        if prob_failure >= self.threshold:
            sl_dist = atr * 2.0  # Using our 2.0 ATR multiplier
            tp_dist = sl_dist * 2.0 # Using our 2.0 RRR
            
            signal = {
                'type': 'SELL',
                'entry': current_price,
                'sl': current_price + sl_dist,
                'tp': current_price - tp_dist,
                'confidence': prob_failure
            }
            
        # Conversely, if prob_failure is very LOW, the 'Dummy Long' is likely to WIN.
        elif prob_failure <= (1 - self.threshold):
            sl_dist = atr * 2.0
            tp_dist = sl_dist * 2.0
            
            signal = {
                'type': 'BUY',
                'entry': current_price,
                'sl': current_price - sl_dist,
                'tp': current_price + tp_dist,
                'confidence': (1 - prob_failure)
            }
            
        return signal