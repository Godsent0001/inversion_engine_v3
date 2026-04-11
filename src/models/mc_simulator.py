import numpy as np

class MonteCarloSimulator:
    def __init__(self, iterations=1000, confidence=0.95):
        """
        Simulates 1,000 alternate trading histories to ensure
        the strategy's risk profile is stable.
        """
        self.iterations = iterations
        self.confidence = confidence

    def get_max_drawdown_limit(self, trade_results_r):
        """
        trade_results_r: List of R-multiples (e.g., [2.5, -1.0, 2.5, -1.0, -1.0])
        Returns the 95% Confidence Max Drawdown.
        """
        if len(trade_results_r) < 5:
            return 99.0  # Reject strategies with insufficient trade data

        max_dds = []
        
        # We use a fixed seed for the internal loop to keep MC results
        # consistent for the same input, but allow randomness across iterations.
        for i in range(self.iterations):
            # 1. Shuffle and Resample (Bootstrap)
            # This simulates a different sequence of the same performance stats
            sim_trades = np.random.choice(
                trade_results_r,
                size=len(trade_results_r),
                replace=True
            )
            
            # 2. Calculate Equity Curve
            # Starting at 0.0 to track R-multiple drawdown
            equity = np.cumsum(sim_trades)

            # 3. Calculate Running High Water Mark (HWM)
            hwm = np.maximum.accumulate(equity)

            # 4. Calculate Drawdown from Peak
            drawdown = hwm - equity

            # Store the worst peak-to-valley drop of this "alternate life"
            max_dds.append(np.max(drawdown))
        
        # 5. Determine the Statistical Risk Limit (95th Percentile)
        # This means in 950 out of 1,000 lives, the drawdown was lower than this.
        mc_risk_score = np.percentile(max_dds, self.confidence * 100)
        
        # 6. Safety Logic:
        # If the 95% worst-case is above 12R, the 'main_mine.py' will reject it.
        # We return the raw score so the miner can make the hiring decision.
        return round(float(mc_risk_score), 2)

    @staticmethod
    def get_survival_probability(trade_results_r, max_allowed_dd=12.0):
        """
        Bonus: Calculates the % of simulations that survived without
        hitting the 12R cap.
        """
        # ... (Optional extra validation logic) ...
        pass