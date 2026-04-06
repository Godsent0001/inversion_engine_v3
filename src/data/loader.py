import pandas as pd
import yaml
import os

class DataLoader:
    @staticmethod
    def load_clean_data(file_path):
        """
        Loads the XAUUSD data and prepares it for the Inversion Engine.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Data file not found at {file_path}. Run mt5_downloader.py first!")

        with open("config/settings.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # 1. Load the CSV
        df = pd.read_csv(file_path)
        
        # 2. Standardize column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 3. Handle 'datetime' column
        if 'datetime' not in df.columns:
            # If MT5 downloader named it 'time', rename it
            if 'time' in df.columns:
                df.rename(columns={'time': 'datetime'}, inplace=True)
            else:
                # If it's something else, find the first column with dates
                df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)

        # 4. Convert to datetime objects and set index
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # 5. Filter for 6-month window (or max available)
        max_date = df.index.max()
        start_date = max_date - pd.DateOffset(months=config.get('lookback_months', 6))
        df = df[df.index >= start_date]

        # 6. Ensure we have the required OHLCV columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                # MT5 sometimes calls volume 'tick_volume'
                if col == 'volume' and 'tick_volume' in df.columns:
                    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
                else:
                    raise KeyError(f"❌ Missing required column: {col}")

        print(f"✅ Data Loaded: {len(df)} rows. Range: {df.index.min()} to {df.index.max()}")
        return df