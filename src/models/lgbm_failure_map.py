import lightgbm as lgb
import pandas as pd
import numpy as np

class LGBMFailureMapper:
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'verbose': -1
        }
        self.model = None

    def train_failure_map(self, X, y):
        """
        X: Features (Energy, Pressure, etc.)
        y: 1 if trade was a LOSS (-1.0R), 0 if WIN (+2.0R)
        We are mapping the FAILURE.
        """
        # Create Dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Train the model to find the 'Losers'
        self.model = lgb.train(self.params, train_data, num_boost_round=100)
        return self.model

    def get_failure_probability(self, X):
        """
        Returns 0.0 to 1.0 (Probability of being a Loser)
        """
        if self.model:
            return self.model.predict(X)
        return np.zeros(len(X))