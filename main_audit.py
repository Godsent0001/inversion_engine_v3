import pandas as pd
import yaml
import os
import shutil

def run_audit():
    """
    V4 Audit Engine: Surgical Performance Evaluation & Hot-Swapping.
    Checks if any active Expert has breached its unique MC-Limit.
    """
    # 1. LOAD CONFIGURATION & PORTFOLIO
    try:
        with open("config/settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Error: config/settings.yaml not found.")
        return

    portfolio_path = "storage/active_portfolio/full_steam_30.pkl"
    reserve_dir = "storage/reserve/"
    model_dir = "storage/models/"

    # Ensure directories exist
    os.makedirs(reserve_dir, exist_ok=True)

    if not os.path.exists(portfolio_path):
        print(f"❌ Error: Active portfolio file not found at {portfolio_path}")
        return

    # Load the active portfolio (The 'League of 30')
    portfolio_df = pd.read_pickle(portfolio_path)
    fired_count = 0

    print(f"🕵️  Inversion Engine v4: Auditing {len(portfolio_df)} active experts...")

    # 2. SURGICAL EVALUATION
    for idx, strategy in portfolio_df.iterrows():
        if strategy['status'] != 'ACTIVE':
            continue

        # THE KILL SWITCH LOGIC (Custom per Strategy):
        # We allow the specific MC Limit found during mining + 1.0R for real-world slippage.
        # Example: If MC Limit was 8.4R, the Kill Threshold is 9.4R.
        kill_threshold = strategy['mc_limit'] + 1.0
        
        # 'live_drawdown' must be updated by your execution/tracker script
        current_dd = strategy.get('live_drawdown', 0.0)

        if current_dd >= kill_threshold:
            print(f"🔥 FIRING {strategy['id']}: Drawdown {current_dd:.2f}R exceeded limit of {kill_threshold:.2f}R.")

            # 3. HOT-SWAP REPLACEMENT
            # Look for vetted models in the reserve folder (created by a separate mining run)
            reserve_files = sorted([f for f in os.listdir(reserve_dir) if f.endswith('.txt')])

            if len(reserve_files) > 0:
                # Grab the oldest reserve model (First-In, First-Out)
                new_model_name = reserve_files[0]
                source_path = os.path.join(reserve_dir, new_model_name)

                # Maintain the file naming convention for the execution engine (e.g., strat_5.txt)
                # We extract the index from the strategy ID
                strategy_num = strategy['id'].split('_')[-1]
                dest_filename = f"strat_{strategy_num}.txt"
                dest_path = os.path.join(model_dir, dest_filename)

                # Move the reserve model into production
                shutil.move(source_path, dest_path)

                # Reset Metadata for the new Expert in this slot
                portfolio_df.at[idx, 'status'] = 'ACTIVE'
                portfolio_df.at[idx, 'live_drawdown'] = 0.0
                portfolio_df.at[idx, 'expected_ev'] = "REPLACED_RESERVE"

                print(f"✅ SUCCESS: Slot {strategy['id']} replaced with fresh model: {new_model_name}")
            else:
                # No backup available; deactivate the slot to prevent further loss
                portfolio_df.at[idx, 'status'] = 'TERMINATED'
                print(f"⚠️ WARNING: No reserve models available. Slot {strategy['id']} is now OFFLINE.")

            fired_count += 1
            
    # 4. COMMIT CHANGES
    if fired_count > 0:
        portfolio_df.to_pickle(portfolio_path)
        print(f"♻️  Audit Complete: {fired_count} experts swapped or terminated.")
    else:
        print("✅ Audit Complete: All experts operating within acceptable risk boundaries.")

if __name__ == "__main__":
    run_audit()