import MetaTrader5 as mt5
import pandas as pd
import yaml
import time
import os
import lightgbm as lgb
from datetime import datetime
from src.features.ta_factory import TAFactory
from src.execution.signal_gen import SignalGenerator
from src.execution.risk_engine import RiskEngine

# --- INITIALIZE ---
with open("config/settings.yaml", 'r') as f:
    config = yaml.safe_load(f)

PORTFOLIO_PATH = "storage/active_portfolio/full_steam_30.pkl"
MODEL_STORAGE = "storage/models/"

# Global dictionary to track the last candle traded by each Expert
# Format: { 'strat_1': '2026-04-10 09:30', 'strat_2': ... }
trade_locks = {}

def send_order_with_retry(request, retries=3):
    """Handles Requotes (10004) and Price Changes (10020) with a retry loop."""
    for i in range(retries):
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return result
        
        # If Requote or Price Change, update price and try again
        if result.retcode in [10004, 10020]:
            print(f"⚠️ Requote detected. Retrying {i+1}/{retries}...")
            sym = request['symbol']
            # Update price to current market
            request['price'] = mt5.symbol_info_tick(sym).ask if request['type'] == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(sym).bid
            time.sleep(0.1) # Tiny pause
            continue
        else:
            # Fatal error (No money, Invalid stops, etc.)
            return result
    return result

def execute_logic():
    if not mt5.initialize(): return

    portfolio_df = pd.read_pickle(PORTFOLIO_PATH)
    sig_gen = SignalGenerator(failure_threshold=0.75)

    print("🚀 Inversion Engine v4 Online. Protection: REQUOTE-RESISTANT & CANDLE-LOCKED.")

    while True:
        now = datetime.now()
        
        # Trigger check
        if now.minute % 5 == 0 and now.second < 2:
            # 1. Get Market State
            rates = mt5.copy_rates_from_pos(config['symbol'], mt5.TIMEFRAME_M5, 0, 50)
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = TAFactory.add_indicators(df)
            # (Apply Energy/Pressure oscillators here as well)

            last_row = df.iloc[-1].to_dict()
            candle_id = str(last_row['time']) # Unique ID for this 5m bar

            acc_info = mt5.account_info()
            sym_info = mt5.symbol_info(config['symbol'])
            risk_mgr = RiskEngine(acc_info.balance)

            # 2. Loop Experts
            for idx, strategy in portfolio_df.iterrows():
                strat_id = strategy['id']

                # --- GUARD 1: CANDLE LOCK ---
                # If this expert already traded on THIS specific 5m candle, skip it.
                if trade_locks.get(strat_id) == candle_id:
                    continue

                # Load and Predict
                model_file = f"{MODEL_STORAGE}strat_{strat_id.split('_')[-1]}.txt"
                if not os.path.exists(model_file): continue
                model = lgb.Booster(model_file=model_file)

                signal = sig_gen.generate_v4_signal(model, last_row, strategy, sym_info.bid, last_row['atr'])

                if signal:
                    lots = risk_mgr.calculate_lot_size(signal['entry'], signal['sl'])
                    if lots <= 0: continue

                    # Prepare Order with Deviation (Slippage Buffer)
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": config['symbol'],
                        "volume": lots,
                        "type": mt5.ORDER_TYPE_BUY if signal['type'] == 'BUY' else mt5.ORDER_TYPE_SELL,
                        "price": mt5.symbol_info_tick(config['symbol']).ask if signal['type'] == 'BUY' else mt5.symbol_info_tick(config['symbol']).bid,
                        "sl": signal['sl'],
                        "tp": signal['tp'],
                        "deviation": 20, # Allow 20 points of slippage to reduce requotes
                        "magic": 404040,
                        "comment": f"V4_{strat_id}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    # --- GUARD 2: REQUOTE RETRY ---
                    result = send_order_with_retry(request)

                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        # LOCK the strategy for the rest of this candle
                        trade_locks[strat_id] = candle_id
                        print(f"✅ {strat_id} Traded on {candle_id}. Expert Locked until next candle.")
                    else:
                        print(f"❌ {strat_id} Failed: {result.retcode}")

            time.sleep(10) # Wait for heartbeat to pass
        time.sleep(0.5)

if __name__ == "__main__":
    execute_logic()