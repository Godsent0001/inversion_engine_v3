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
    labeled_list = []
    for idx in raw_candidates.index:
        labeled_list.append(EVLabeler.label_single(df_enriched, idx, atr_mult, rrr, side=1))
        labeled_list.append(EVLabeler.label_single(df_enriched, idx, atr_mult, rrr, side=0))

    sampled_data = pd.DataFrame(labeled_list)

    split = int(len(sampled_data) * config['train_test_split'])
    train_df = sampled_data.iloc[:split]
    test_df = sampled_data.iloc[split:]
    
    y_train = train_df['target']
    mapper = LGBMFailureMapper(params)
    model = mapper.train_failure_map(train_df[FEATURES], y_train)
    
    probs = model.predict(test_df[FEATURES])
    returns = np.where(probs >= 0.75, test_df['inverted_outcome_r'], 0)
    
    active_returns = returns[returns != 0]
    net_returns = active_returns - config['spread_tax_r']
    
    return np.mean(net_returns) if len(active_returns) >= config['min_trades_required'] else -1

# 3. THE MAIN MINING LOOP
def start_mining():
    os.makedirs("storage/models", exist_ok=True)
    os.makedirs("storage/active_portfolio", exist_ok=True)
    os.makedirs("storage/reserve", exist_ok=True)

    # RESUME LOGIC: Scan for existing experts and reserves
    existing_experts = [f for f in os.listdir("storage/models") if f.startswith("strat_") and f.endswith(".txt")]
    existing_reserves = [f for f in os.listdir("storage/reserve") if f.startswith("reserve_") and f.endswith(".txt")]

    strategy_id = len(existing_experts)
    reserve_count = len(existing_reserves)

    portfolio = []
    if os.path.exists("storage/active_portfolio/league_summary.json"):
        with open("storage/active_portfolio/league_summary.json", "r") as f:
            portfolio = json.load(f)
            # Keep what we have files for
            portfolio = portfolio[:strategy_id]
    
    print(f"📊 Resuming: {len(portfolio)} experts in portfolio, {strategy_id} total expert files, {reserve_count} reserve files.")

    print(f"💎 --- STARTING V4 ENGINE MINING --- 💎")

    # EXPERT MINING
    while len(portfolio) < config['portfolio_size']:
        print(f"\n🚀 [Expert {len(portfolio)+1}/{config['portfolio_size']}] Hunting...")
        candidates = CandidateSampler.get_random_candidates(df_enriched, count=config['random_candidate_count'])
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, candidates), n_trials=40, n_jobs=-1)
        
        best_ev = study.best_value
        best_params = study.best_params
        
        if best_ev > config['min_inverted_alpha_r']:
            atr_mult = best_params['atr_multiplier']
            rrr = best_params['rrr']

            final_list = []
            for idx in candidates.index:
                final_list.append(EVLabeler.label_single(df_enriched, idx, atr_mult, rrr, side=1))
                final_list.append(EVLabeler.label_single(df_enriched, idx, atr_mult, rrr, side=0))

            final_df = pd.DataFrame(final_list)
            ai_params = {k: v for k, v in best_params.items() if k not in ['atr_multiplier', 'rrr']}

            final_mapper = LGBMFailureMapper(ai_params)
            final_model = final_mapper.train_failure_map(final_df[FEATURES], final_df['target'])

            wfa = WalkForwardValidator.validate_robustness(
                final_model, final_df, FEATURES,
                multiplier=atr_mult, spread_tax=config['spread_tax_r']
            )

            if not wfa:
                print(f"❌ REJECTED: Failed Walk-Forward Robustness.")
                continue

            mc = MonteCarloSimulator()
            probs = final_model.predict(final_df[FEATURES])
            trades = np.where(probs >= 0.75, final_df['inverted_outcome_r'] - config['spread_tax_r'], 0)
            active_trades = trades[trades != 0]

            limit = mc.get_max_drawdown_limit(active_trades)

            if limit <= config['max_strategy_drawdown']:
                strategy_data = {
                    'id': f"XAU_V4_{strategy_id}",
                    'atr_multiplier': round(atr_mult, 2),
                    'rrr': round(rrr, 2),
                    'mc_limit': round(limit, 2),
                    'expected_ev': round(best_ev, 3),
                    'status': 'ACTIVE'
                }
                portfolio.append(strategy_data)
                final_model.save_model(f"storage/models/strat_{strategy_id}.txt")
                strategy_id += 1
                print(f"✅ HIRED! EV: {best_ev:.3f} | ATR: {atr_mult:.2f} | RRR: {rrr:.2f} | MC: {limit:.1f}R")
                # Incremental Save
                portfolio_df = pd.DataFrame(portfolio)
                portfolio_df.to_pickle("storage/active_portfolio/full_steam_30.pkl")
                with open("storage/active_portfolio/league_summary.json", "w") as f:
                    json.dump(portfolio, f, indent=4)
            else:
                print(f"❌ REJECTED: MC Limit {limit:.1f}R exceeds safety cap.")
        else:
            print(f"❌ REJECTED: EV {best_ev:.3f} below alpha threshold.")


    # RESERVE MINING
    print(f"\n🛡️ --- STARTING RESERVE MINING (Generating 10 Backups) --- 🛡️")
    # reserve_count is already set by resume logic
    while reserve_count < 10:
        print(f"🚀 [Reserve {reserve_count+1}/10] Hunting...")
        candidates = CandidateSampler.get_random_candidates(df_enriched, count=config['random_candidate_count'])
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, candidates), n_trials=40, n_jobs=-1)

        best_ev = study.best_value
        best_params = study.best_params

        if best_ev > config['min_inverted_alpha_r']:
            atr_mult = best_params['atr_multiplier']
            rrr = best_params['rrr']

            final_list = []
            for idx in candidates.index:
                final_list.append(EVLabeler.label_single(df_enriched, idx, atr_mult, rrr, side=1))
                final_list.append(EVLabeler.label_single(df_enriched, idx, atr_mult, rrr, side=0))

            final_df = pd.DataFrame(final_list)
            ai_params = {k: v for k, v in best_params.items() if k not in ['atr_multiplier', 'rrr']}

            final_mapper = LGBMFailureMapper(ai_params)
            final_model = final_mapper.train_failure_map(final_df[FEATURES], final_df['target'])

            wfa = WalkForwardValidator.validate_robustness(
                final_model, final_df, FEATURES,
                multiplier=atr_mult, spread_tax=config['spread_tax_r']
            )

            if not wfa:
                print(f"❌ REJECTED: Failed Walk-Forward Robustness.")
                continue

            mc = MonteCarloSimulator()
            probs = final_model.predict(final_df[FEATURES])
            trades = np.where(probs >= 0.75, final_df['inverted_outcome_r'] - config['spread_tax_r'], 0)
            active_trades = trades[trades != 0]
            limit = mc.get_max_drawdown_limit(active_trades)

            if limit <= config['max_strategy_drawdown']:
                final_model.save_model(f"storage/reserve/reserve_{reserve_count}.txt")
                reserve_count += 1
                print(f"✅ RESERVE HIRED! EV: {best_ev:.3f} | ATR: {atr_mult:.2f} | RRR: {rrr:.2f} | MC: {limit:.1f}R")
            else:
                print(f"❌ REJECTED: MC Limit {limit:.1f}R exceeds safety cap.")
        else:
            print(f"❌ REJECTED: EV {best_ev:.3f} below alpha threshold.")

    print(f"\n🏁 MINING COMPLETE.")

if __name__ == "__main__":
    start_mining()
