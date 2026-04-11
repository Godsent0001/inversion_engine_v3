import lightgbm as lgb
import pandas as pd
import numpy as np

class LGBMFailureMapper:
    def __init__(self, params=None):
        """
        LightGBM implementation for mapping the probability of Retail Failure.
        V4 Focus: Learning direction-specific liquidity traps using 'side'.
        """
        self.params = params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        self.model = None

    def train_failure_map(self, X, y):
        """
        Trains the model to identify 'Traps'.
        X: Combined Features (Primary, Retail, Session, and SIDE).
        y: 1 if Retail hit Stop Loss (Failure), 0 if they hit Take Profit.
        """
        # Define categorical features if they exist in the dataset
        # 'side' and 'is_ny_open' are better treated as categories
        cat_features = [col for col in ['side', 'is_ny_open', 'hour_of_day'] if col in X.columns]

        # Create the LightGBM Dataset
        train_data = lgb.Dataset(
            X,
            label=y,
            categorical_feature=cat_features,
            free_raw_data=False
        )
        
        # Train the model
        # 100 rounds is usually the 'sweet spot' for 5m Gold data to avoid overfitting
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100
        )
        return self.model

    def predict(self, X):
        """
        Returns the probability (0.0 to 1.0) of the retail trader failing.
        In V4, the executor calls this twice: once with side=1, once with side=0.
        """
        if self.model:
            return self.model.predict(X)
        return np.zeros(len(X))

    def save_model(self, path):
        """Saves the LightGBM model to a text file for fast MT5 execution."""
        if self.model:
            self.model.save_model(path)
        else:
            raise ValueError("❌ Cannot save model: No model has been trained yet.")

    def load_model(self, path):
        """Loads a model for prediction/execution."""
        self.model = lgb.Booster(model_file=path)