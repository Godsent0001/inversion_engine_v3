import pandas as pd
import numpy as np
import optuna
import yaml
import json
import os
import lightgbm as lgb
from src.data.loader import DataLoader
from src.data.sampler import CandidateSampler
from src.features.ta_factory import TAFactory
from src.features.energy_oscillator import MarketEnergyOscillator
from src.features.pressure_oscillator import MarketPressureOscillator
from src.labeling.ev_labeler import EVLabeler
from src.models.lgbm_failure_map import LGBMFailureMapper
from src.models.mc_simulator import MonteCarloSimulator
from src.validation.walk_forward import WalkForwardValidator

# --- INITIALIZE CONFIG ---
with open("config/settings.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Flatten feature list from yaml - Includes 'side' for directional identity
FEATURES = (
    config['primary_features'] + 
    config['retail_features'] + 
    config['direction_logic'] + 
    config['session_features']
)

# 1. LOAD & ENRICH DATA
# Ensure xauusd_5m.csv is in data/raw/
df_raw = DataLoader.load_clean_data("data/raw/xauusd_5m.csv")
df_raw = TAFactory.add_indicators(df_raw)

meo = MarketEnergyOscillator()
mpo = MarketPressureOscillator()
df_enriched = meo.apply(df_raw)
df_enriched = mpo.apply(df_enriched)

# 2. DEFINE THE SEARCH (OPTUNA BAYESIAN)
def objective(trial, raw_candidates):
    # Optuna picks AI hyperparameters
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'objective': 'binary',
        'verbose': -1,
        'seed': 42
    }
    
    # Optuna picks the MATH (ATR Multiplier and the Auto-RRR)
    atr_mult = trial.suggest_float('atr_multiplier', config['min_atr_multiplier'], config['max_atr_multiplier'])
    rrr = trial.suggest_float('rrr', config['min_rrr'], config['max_rrr'])
    
    # --- DOUBLE LABELING LOGIC ---
    # We transform each candidate into TWO scenarios: a Retail Buy and a Retail Sell
    labeled_list = []
    for idx in raw_candidates.index:
        # Scenario 1: Retail attempted a BUY (Side 1) -> AI predicts if it fails
        labeled_list.append(EVLabeler.label_single(df_enriched, idx, atr_mult, rrr, side=1))
        # Scenario 2: Retail attempted a SELL (Side 0) -> AI predicts if it fails
        labeled_list.append(EVLabeler.label_single(df_enriched, idx, atr_mult, rrr, side=0))

    sampled_data = pd.DataFrame(labeled_list)

    # Split into Train (85%) and Test (15%)
    split = int(len(sampled_data) * config['train_test_split'])
    train_df = sampled_data.iloc[:split]
    test_df = sampled_data.iloc[split:]
    
    # Train model to predict FAILURE (target=1 means the trade hit SL)
    # The 'side' feature tells the model which direction the failure is for.
    y_train = train_df['target']
    mapper = LGBMFailureMapper(params)
    model = mapper.train_failure_map(train_df[FEATURES], y_train)
    
    # Calculate Inverted EV on Test Data
    probs = model.predict(test_df[FEATURES])
    
    # Logic: If prob of failure >= 75%, we take the INVERTED trade.
    # Returns test_df['inverted_outcome_r'] (which is the RRR if we win, -1 if we lose)
    returns = np.where(probs >= 0.75, test_df['inverted_outcome_r'], 0)
    
    # Filter for active signals and apply SPREAD TAX (0.15R)
    active_returns = returns[returns != 0]
    net_returns = active_returns - config['spread_tax_r']
    
    # Return mean net profit (Minimum 10 trades to avoid statistical noise)
    return np.mean(net_returns) if len(active_returns) >= config['min_trades_required'] else -1

# 3. THE MAIN MINING LOOP
def start_mining():
    portfolio = []
    strategy_id = 0
    os.makedirs("storage/models", exist_ok=True)
    os.makedirs("storage/active_portfolio", exist_ok=True)
    
    print(f"💎 --- STARTING V4 ENGINE MINING --- 💎")
    print(f"Settings: {config['min_atr_multiplier']}-{config['max_atr_multiplier']} ATR | {config['min_rrr']}-{config['max_rrr']} RRR")

    while len(portfolio) < config['portfolio_size']:
        print(f"\n🚀 [Expert {len(portfolio)+1}/{config['portfolio_size']}] Hunting for high-EV patterns...")
        
        # Pure Random Sampling for diverse "Expert" perspectives
        candidates = CandidateSampler.get_random_candidates(df_enriched, count=config['random_candidate_count'])
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, candidates), n_trials=40)
        
        best_ev = study.best_value
        best_params = study.best_params
        
        if best_ev > config['min_inverted_alpha_r']:
            atr_mult = best_params['atr_multiplier']
            rrr = best_params['rrr']
            
            # Re-generate full labeled data for final training on all candidates
            final_list = []
            for idx in candidates.index:
                final_list.append(EVLabeler.label_single(df_enriched, idx, atr_mult, rrr, side=1))
                final_list.append(EVLabeler.label_single(df_enriched, idx, atr_mult, rrr, side=0))
            
            final_df = pd.DataFrame(final_list)
            ai_params = {k: v for k, v in best_params.items() if k not in ['atr_multiplier', 'rrr']}
            
            # Final Model Build
            final_mapper = LGBMFailureMapper(ai_params)
            final_model = final_mapper.train_failure_map(final_df[FEATURES], final_df['target'])
            
            # Robustness: Monte Carlo 12R Test
            mc = MonteCarloSimulator()
            probs = final_model.predict(final_df[FEATURES])
            trades = np.where(probs >= 0.75, final_df['inverted_outcome_r'] - config['spread_tax_r'], 0)
            active_trades = trades[trades != 0]
            
            limit = mc.get_max_drawdown_limit(active_trades)
            
            if limit <= config['max_strategy_drawdown']:
                # HIRED!
                strategy_data = {
                    'id': f"XAU_V4_{strategy_id}",
                    'atr_multiplier': round(atr_mult, 2),
                    'rrr': round(rrr, 2),
                    'mc_limit': round(limit, 2),
                    'expected_ev': round(best_ev, 3),
                    'status': 'ACTIVE'
                }
                portfolio.append(strategy_data)
                
                # Save Model (.txt format for fast MT5 execution)
                final_model.save_model(f"storage/models/strat_{strategy_id}.txt")
                strategy_id += 1
                print(f"✅ HIRED! EV: {best_ev:.3f} | ATR: {atr_mult:.2f} | RRR: {rrr:.2f} | MC: {limit:.1f}R")
            else:
                print(f"❌ REJECTED: MC Limit {limit:.1f}R exceeds safety cap.")
        else:
            print(f"❌ REJECTED: EV {best_ev:.3f} below alpha threshold.")

    # Save the League Portfolio
    portfolio_df = pd.DataFrame(portfolio)
    portfolio_df.to_pickle("storage/active_portfolio/full_steam_30.pkl")
    
    # Save a human-readable JSON backup
    with open("storage/active_portfolio/league_summary.json", "w") as f:
        json.dump(portfolio, f, indent=4)

    print(f"\n🏁 MINING COMPLETE. 30 Expert Models are now locked in storage/models/")

if __name__ == "__main__":
    start_mining()