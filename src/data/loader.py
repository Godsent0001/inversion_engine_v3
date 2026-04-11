import pandas as pd
import yaml
import os

class DataLoader:
    @staticmethod
    def load_clean_data(file_path):
        """
        Loads the XAUUSD data, standardizes MT5 naming conventions,
        and prepares it for the Inversion Engine's mining process.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Data file not found at {file_path}. Run your downloader first!")

        # Load Config for lookback window
        try:
            with open("config/settings.yaml", 'r') as f:
                config = yaml.safe_load(f)
        except Exception:
            config = {'lookback_months': 6} # Fallback default

        # 1. Load the CSV
        df = pd.read_csv(file_path)
        
        # 2. Standardize column names (lowercase and no spaces)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 3. Handle MT5 'time' vs 'datetime' column
        # MT5 often downloads as 'time'. We standardize to 'datetime' for the engine.
        if 'time' in df.columns:
            df.rename(columns={'time': 'datetime'}, inplace=True)
        elif 'datetime' not in df.columns:
            # If neither exist, assume the first column is our timestamp
            df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)

        # 4. Convert to datetime objects and set as index
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # 5. Filter for the Lookback Window
        # We only want the most recent 'X' months to keep models current
        max_date = df.index.max()
        months_back = config.get('lookback_months', 6)
        start_date = max_date - pd.DateOffset(months=months_back)
        df = df[df.index >= start_date]

        # 6. Standardize OHLCV
        # MT5 specific: 'tick_volume' is used for forex/metals instead of 'volume'
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)

        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise KeyError(f"❌ Missing required column: {col}. Check your CSV source!")

        # 7. Quality Control
        df = df[required].copy() # Keep only necessary columns
        df.dropna(inplace=True)   # Remove any broken candles

        # Ensure numeric types (important for TA-Lib and LGBM)
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"✅ Data Loaded: {len(df)} candles.")
        print(f"📅 Range: {df.index.min()} to {df.index.max()}")
        return df