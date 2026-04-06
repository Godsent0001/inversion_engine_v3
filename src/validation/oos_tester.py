import pandas as pd

class OOSTester:
    @staticmethod
    def validate(model, test_df, features, threshold=0.75):
        """
        Runs the model on Month 6. If Net Profit < 0, REJECT strategy.
        """
        # 1. Predict on unseen data
        test_df['prob_fail'] = model.predict(test_df[features])
        
        # 2. Simulate Inverted Trades
        # If Prob Fail > Threshold, we go Short (-1 * Outcome)
        # If Prob Fail < (1-Threshold), we go Long (1 * Outcome)
        test_df['strat_ret'] = 0.0
        
        # Shorts (Inverting a high prob failure)
        test_df.loc[test_df['prob_fail'] >= threshold, 'strat_ret'] = -test_df['outcome_r']
        
        # Longs (Confirming a low prob failure)
        test_df.loc[test_df['prob_fail'] <= (1-threshold), 'strat_ret'] = test_df['outcome_r']
        
        net_profit = test_df['strat_ret'].sum()
        trade_count = len(test_df[test_df['strat_ret'] != 0])
        
        print(f"🧪 OOS Test: Net {net_profit:.2f}R over {trade_count} trades.")
        return net_profit > 0 and trade_count > 5 # Minimum 5 trades in test month