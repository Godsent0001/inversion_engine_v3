import pandas as pd
import numpy as np

class OOSTester:
    @staticmethod
    def validate(model, test_df, features, threshold=0.75, spread_tax=0.15):
        """
        Validates the expert on the Out-of-Sample (OOS) period.
        Ensures the strategy survives the 'Spread Tax' and maintains
        positive Expected Value on unseen data.
        """
        if test_df.empty:
            return False

        # 1. Predict Failure Probability on the OOS set
        # In V4, 'test_df' contains both side 0 and side 1 rows.
        test_df = test_df.copy()
        test_df['prob_fail'] = model.predict(test_df[features])
        
        # 2. Simulate our Inverted Trades
        # We only take trades where the AI is highly confident in a Retail Trap (prob >= 0.75)
        # 'inverted_outcome_r' is already calculated by the EVLabeler (RRR for win, -1 for loss)
        test_df['strat_ret'] = np.where(
            test_df['prob_fail'] >= threshold,
            test_df['inverted_outcome_r'],
            0.0
        )

        # 3. Apply the Spread Tax to active trades
        active_trades = test_df[test_df['strat_ret'] != 0].copy()
        active_trades['net_ret'] = active_trades['strat_ret'] - spread_tax
        
        # 4. Calculate Performance Metrics
        total_net_profit = active_trades['net_ret'].sum()
        trade_count = len(active_trades)
        win_rate = (active_trades['net_ret'] > 0).mean() if trade_count > 0 else 0
        
        # 5. Final Hiring Decision
        # Strategy must:
        # - Be profitable after the spread tax (Total Net > 0)
        # - Have a minimum trade frequency (to ensure it wasn't a lucky single trade)
        is_robust = total_net_profit > 0 and trade_count >= 5
        
        status_emoji = "✅" if is_robust else "❌"
        print(f"🧪 OOS Test: {status_emoji} Net {total_net_profit:.2f}R | Trades: {trade_count} | WR: {win_rate:.1%}")
        
        return is_robust