import pandas as pd
import pandas_ta as ta

class TAFactory:
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Wraps pandas_ta for high-speed indicator calculation.
        All features are SHIFTED by 1 to prevent look-ahead bias.
        """
        # 1. Volatility (ATR)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # 2. Momentum (RSI)
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        
        # 3. Structural Context (Bollinger Bands)
        bb = ta.bbands(df['close'], length=20, std=2)
        # Handle possible variation in pandas_ta column names
        upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
        lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
        mid_col = [c for c in bb.columns if c.startswith('BBM')][0]
        df['bb_width'] = (bb[upper_col] - bb[lower_col]) / (bb[mid_col] + 1e-8)
        
        # 4. Trend (EMA)
        df['ema_fast'] = ta.ema(df['close'], length=20)
        df['ema_slow'] = ta.ema(df['close'], length=50)
        
        # 5. Body Ratio (Candle Sentiment)
        df['body_size'] = (df['close'] - df['open']).abs()
        df['total_range'] = df['high'] - df['low']
        df['candle_body_ratio'] = df['body_size'] / (df['total_range'] + 1e-8)

        # 6. ATR Ratio
        df['atr_ratio'] = df['atr'] / (df['atr'].rolling(20).mean() + 1e-8)

        # 7. Session Features
        df['hour_of_day'] = df.index.hour
        df['is_ny_open'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)

        # IMPORTANT: SHIFT EVERYTHING BY 1
        # The AI must only see information from the COMPLETED candles.
        feature_cols = [
            'atr', 'rsi_14', 'bb_width', 'ema_fast', 'ema_slow',
            'candle_body_ratio', 'atr_ratio', 'hour_of_day', 'is_ny_open'
        ]
        df[feature_cols] = df[feature_cols].shift(1)
        
        return df.dropna()