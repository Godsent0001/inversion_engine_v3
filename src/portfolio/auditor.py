import pandas as pd
import json

class PortfolioAuditor:
    def __init__(self, portfolio_path="storage/active_portfolio.json"):
        self.path = portfolio_path

    def audit_performance(self, trade_logs_df):
        """
        Reads the trade logs and compares every strategy 
        against its Monte Carlo 12R limit.
        """
        with open(self.path, 'r') as f:
            portfolio = json.load(f)

        updated_portfolio = []
        fired_any = False

        for strat in portfolio:
            # Calculate current drawdown for THIS specific strategy ID
            strat_trades = trade_logs_df[trade_logs_df['strat_id'] == strat['id']]
            
            if len(strat_trades) > 0:
                equity = strat_trades['r_result'].cumsum()
                max_dd = (equity.cummax() - equity).max()
                
                # THE 12R RULE
                if max_dd >= 12.0:
                    print(f"🔥 AUDITOR: Strategy {strat['id']} BREACHED 12R. Terminating.")
                    strat['status'] = 'FIRED'
                    fired_any = True
            
            updated_portfolio.append(strat)

        if fired_any:
            with open(self.path, 'w') as f:
                json.dump(updated_portfolio, f)
        
        return fired_any