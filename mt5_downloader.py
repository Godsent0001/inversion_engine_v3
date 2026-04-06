import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os

def download_from_mt5():
    # 1. Initialize connection to MT5
    if not mt5.initialize():
        print("❌ MT5 Initialization failed. Make sure MT5 is OPEN on your PC.")
        return

    symbol = "XAUUSD" # or "GOLD" depending on your broker
    print(f"🚀 Pulling 6 months of 5m data for {symbol}...")

    # 2. Set the timeframe (5 Minutes) and range
    timeframe = mt5.TIMEFRAME_M5
    # Approximately 6 months = 180 days
    # (30,000 to 50,000 bars usually)
    
    # Get bars from 'now' going back 6 months
    utc_to = datetime.now()
    # Pull 50,000 bars (roughly 6 months of 5m candles)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 50000)

    if rates is None or len(rates) == 0:
        print(f"❌ Could not find {symbol}. Check if it's in your 'Market Watch' window.")
        mt5.shutdown()
        return

    # 3. Convert to DataFrame
    df = pd.DataFrame(rates)
    
    # Convert time from seconds to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Rename columns to match our Engine's expectations
    df.rename(columns={
        'time': 'datetime',
        'tick_volume': 'volume'
    }, inplace=True)

    # 4. Save to CSV
    os.makedirs("data/raw", exist_ok=True)
    file_path = "data/raw/xauusd_5m.csv"
    df.to_csv(file_path, index=False)

    print(f"✅ SUCCESS! Saved {len(df)} bars to {file_path}")
    print(f"📅 Range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    mt5.shutdown()

if __name__ == "__main__":
    download_from_mt5()