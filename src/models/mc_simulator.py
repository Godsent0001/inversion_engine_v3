import numpy as np

class MonteCarloSimulator:
    def __init__(self, iterations=1000, confidence=0.95):
        self.iterations = iterations
        self.confidence = confidence

    def get_max_drawdown_limit(self, trade_results_r):
        """
        Shuffles the trade history 1,000 times.
        Calculates the 95th percentile Max Drawdown.
        """
        max_dds = []
        
        for _ in range(self.iterations):
            # Resample with replacement to simulate alternate histories
            sim_trades = np.random.choice(trade_results_r, size=len(trade_results_r), replace=True)
            
            # Calculate Equity Curve
            equity = np.cumsum(sim_trades)
            hwm = np.maximum.accumulate(equity)
            drawdown = hwm - equity
            max_dds.append(np.max(drawdown))
        
        # Determine the statistical limit (95% confidence)
        mc_limit = np.percentile(max_dds, self.confidence * 100)
        
        # CAP AT 12R (Your specific rule)
        final_limit = min(mc_limit, 12.0)
        
        return final_limit