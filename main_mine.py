import pandas as pd
import numpy as np
import optuna
import yaml
import json
import os
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
    # Flip the losers: If prob > 0.75, assume it fails and do opposite
    returns = np.where(probs >= 0.75, -test_df['outcome_r'], 0)
    
    # We subtract the SPREAD TAX (0.05R) from every active trade
    net_returns = returns[returns != 0] - config['spread_tax_r']
    
    return np.mean(net_returns) if len(net_returns) > 5 else -1

# 3. THE MAIN MINING LOOP
def start_mining():
    portfolio = []
    strategy_id = 0
    
    while len(portfolio) < config['portfolio_size']:
        print(f"\n🚀 --- HUNTING FOR STRATEGY #{len(portfolio) + 1} ---")
        
        # Fresh Sampling for every strategy to ensure lack of correlation
        candidates = CandidateSampler.get_random_candidates(df_enriched)
        labeled_data = EVLabeler.label_candidates(df_enriched, candidates)
        
        # Bayesian Search for the best Failure Logic
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, labeled_data), n_trials=30)
        
        best_params = study.best_params
        best_ev = study.best_value
        
        if best_ev > config['min_inverted_alpha_r']:
            # A. Train Final Model
            features = feat_config['primary_features'] + feat_config['retail_features']
            final_mapper = LGBMFailureMapper(best_params)
            y = (labeled_data['outcome_r'] < 0).astype(int)
            final_model = final_mapper.train_failure_map(labeled_data[features], y)
            
            # B. Walk-Forward Validation (3-Stage Adaptivity Test)
            wfa = WalkForwardValidator.validate_robustness(final_model, labeled_data, features)
            
            if wfa:
                # C. Monte Carlo 12R Test
                mc = MonteCarloSimulator()
                # Get the hypothetical trade returns for MC
                probs = final_model.predict(labeled_data[features])
                trades = np.where(probs >= 0.75, -labeled_data['outcome_r'] - 0.05, 0)
                active_trades = trades[trades != 0]
                
                limit = mc.get_max_drawdown_limit(active_trades)
                
                if limit <= 12.0:
                    # HIRED!
                    strategy_data = {
                        'id': f"XAU_INV_{strategy_id}",
                        'params': best_params,
                        'mc_limit': round(limit, 2),
                        'expected_ev': round(best_ev, 3),
                        'status': 'ACTIVE'
                    }
                    portfolio.append(strategy_data)
                    # Save Model Binary
                    final_model.save_model(f"storage/models/strat_{strategy_id}.txt")
                    strategy_id += 1
                    print(f"✅ HIRED! EV: {best_ev:.3f} | MC Limit: {limit:.1f}R")
                else:
                    print(f"❌ REJECTED: MC Limit {limit:.1f}R exceeds 12R cap.")
            else:
                print("❌ REJECTED: Failed Walk-Forward Robustness.")
        else:
            print(f"❌ REJECTED: Low EV ({best_ev:.3f})")

    # SAVE FINAL PORTFOLIO JSON
    with open("storage/active_portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=4)
    
    print("\n🏁 MINING COMPLETE. 30 Strategies Hired and Saved.")

if __name__ == "__main__":
    start_mining()