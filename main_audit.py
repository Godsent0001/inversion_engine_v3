import pandas as pd
import yaml

def run_audit():
    with open("config/settings.yaml", 'r') as f:
        config = yaml.safe_load(f)

    portfolio_df = pd.read_pickle("storage/active_portfolio/full_steam_30.pkl")
    fired_count = 0

    for idx, strategy in portfolio_df.iterrows():
        # Check current live drawdown (from audit_logs)
        current_dd = strategy.get('live_drawdown', 0)
        
        if current_dd <= -config['max_strategy_drawdown']: # -12R
            print(f"🔥 FIRING STRATEGY {strategy['id']}: Hit 12R Limit.")
            strategy['status'] = 'TERMINATED'
            fired_count += 1
            
    if fired_count > 0:
        # Save updated portfolio and trigger main_mine.py to replace them
        portfolio_df.to_pickle("storage/active_portfolio/full_steam_30.pkl")
        print(f"♻️ {fired_count} strategies terminated. Run main_mine.py to refill.")

if __name__ == "__main__":
    run_audit()