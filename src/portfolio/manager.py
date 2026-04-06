import pandas as pd
import numpy as np

class PortfolioManager:
    def __init__(self, max_strategies=30):
        self.max_strategies = max_strategies
        self.strategies = []

    def check_correlation(self, new_strategy_trades, threshold=0.3):
        """
        Ensures the new strategy doesn't trade exactly like the existing ones.
        """
        for existing in self.strategies:
            corr = np.corrcoef(new_strategy_trades, existing['trades'])[0,1]
            if corr > threshold:
                return False
        return True

    def add_strategy(self, strategy_dict):
        if len(self.strategies) < self.max_strategies:
            self.strategies.append(strategy_dict)
            return True
        return False