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

with open("config/features.yaml", 'r') as f:
    feat_config = yaml.safe_load(f)

# 1. LOAD & ENRICH DATA
df_raw = DataLoader.load_clean_data("data/raw/xauusd_5m.csv")
df_raw = TAFactory.add_indicators(df_raw)

meo = MarketEnergyOscillator()
mpo = MarketPressureOscillator()
df_enriched = meo.apply(df_raw)
df_enriched = mpo.apply(df_enriched)

# 2. DEFINE THE SEARCH (OPTUNA BAYESIAN)
def objective(trial, sampled_data):
    # Optuna picks the "Failure Logic"
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'objective': 'binary',
        'verbose': -1
    }
    
    # DYNAMIC ATR MULTIPLIER (The "Spread Shield")
    # We set a floor of 6.0 to ensure the Stop Loss is wide enough for Gold spreads.
    atr_mult = trial.suggest_float('atr_multiplier', 6.0, 15.0)
    
    # Split Sampled Data into Train (85%) and Test (15%)
    split = int(len(sampled_data) * config['train_test_split'])
    train_df = sampled_data.iloc[:split]
    test_df = sampled_data.iloc[split:]
    
    features = feat_config['primary_features'] + feat_config['retail_features']
    
    # Train model to find failures (outcome_r == -1.0)
    y_train = (train_df['outcome_r'] < 0).astype(int)
    
    mapper = LGBMFailureMapper(params)
    model = mapper.train_failure_map(train_df[features], y_train)
    
    # Calculate Inverted EV on Test Data
    probs = model.predict(test_df[features])
    
    # Logic: If prob of failure >= 75%, we do the opposite.
    # We use the dynamic multiplier from this specific trial.
    returns = np.where(probs >= 0.75, -test_df['outcome_r'], 0)
    
    # SPREAD TAX (0.15R): Account for Bid/Ask gap + Slippage on Gold
    net_returns = returns[returns != 0] - 0.15
    
    return np.mean(net_returns) if len(net_returns) > 5 else -1

# 3. THE MAIN MINING LOOP
def start_mining():
    portfolio = []
    strategy_id = 0
    os.makedirs("storage/models", exist_ok=True)
    
    while len(portfolio) < config['portfolio_size']:
        print(f"\n🚀 --- HUNTING FOR STRATEGY #{len(portfolio) + 1} ---")
        
        # Fresh Sampling for every strategy to ensure lack of correlation
        candidates = CandidateSampler.get_random_candidates(df_enriched)
        labeled_data = EVLabeler.label_candidates(df_enriched, candidates)
        
        # Bayesian Search for the best Failure Logic + ATR Multiplier
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, labeled_data), n_trials=40)
        
        best_params = study.best_params
        best_ev = study.best_value
        
        # Only proceed if the Inverted EV is high enough after the 0.15R tax
        if best_ev > config['min_inverted_alpha_r']:
            features = feat_config['primary_features'] + feat_config['retail_features']
            
            # Extract the AI params separate from the multiplier
            ai_params = {k: v for k, v in best_params.items() if k != 'atr_multiplier'}
            atr_mult = best_params['atr_multiplier']

            final_mapper = LGBMFailureMapper(ai_params)
            y = (labeled_data['outcome_r'] < 0).astype(int)
            final_model = final_mapper.train_failure_map(labeled_data[features], y)
            
            # B. Walk-Forward Validation (3-Stage Adaptivity Test)
            # Passing the dynamic multiplier to the validator
            wfa = WalkForwardValidator.validate_robustness(final_model, labeled_data, features, multiplier=atr_mult)
            
            if wfa:
                # C. Monte Carlo 12R Test
                mc = MonteCarloSimulator()
                probs = final_model.predict(labeled_data[features])
                
                # Apply the 0.15R Tax to the simulation for reality check
                trades = np.where(probs >= 0.75, -labeled_data['outcome_r'] - 0.15, 0)
                active_trades = trades[trades != 0]
                
                limit = mc.get_max_drawdown_limit(active_trades)
                
                if limit <= 12.0:
                    # HIRED!
                    strategy_data = {
                        'id': f"XAU_INV_{strategy_id}",
                        'params': ai_params,
                        'atr_multiplier': round(atr_mult, 2),
                        'mc_limit': round(limit, 2),
                        'expected_ev': round(best_ev, 3),
                        'status': 'ACTIVE'
                    }
                    portfolio.append(strategy_data)
                    
                    # Save Model Binary (LightGBM format)
                    final_model.save_model(f"storage/models/strat_{strategy_id}.txt")
                    strategy_id += 1
                    print(f"✅ HIRED! EV: {best_ev:.3f} | ATR Mult: {atr_mult:.1f} | MC Limit: {limit:.1f}R")
                else:
                    print(f"❌ REJECTED: MC Limit {limit:.1f}R exceeds 12R cap.")
            else:
                print("❌ REJECTED: Failed Walk-Forward Robustness.")
        else:
            print(f"❌ REJECTED: Low EV ({best_ev:.3f} after 0.15R tax)")

    # SAVE FINAL PORTFOLIO JSON
    with open("storage/active_portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=4)
    
    print("\n🏁 MINING COMPLETE. 30 Strategies Hired and Saved with High-ATR Protection.")

if __name__ == "__main__":
    start_mining()