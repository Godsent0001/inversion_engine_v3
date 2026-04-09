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
        # Handle inconsistent pandas_ta column naming
        u_col = [c for c in bb.columns if c.startswith('BBU')][0]
        l_col = [c for c in bb.columns if c.startswith('BBL')][0]
        m_col = [c for c in bb.columns if c.startswith('BBM')][0]
        df['bb_width'] = (bb[u_col] - bb[l_col]) / bb[m_col]
        
        # 4. Trend (EMA)
        df['ema_fast'] = ta.ema(df['close'], length=20)
        df['ema_slow'] = ta.ema(df['close'], length=50)
        
        # 5. Body Ratio (Candle Sentiment)
        df['body_size'] = (df['close'] - df['open']).abs()
        df['total_range'] = df['high'] - df['low']
        df['candle_body_ratio'] = df['body_size'] / (df['total_range'] + 1e-8)
        df['atr_ratio'] = df['atr'] / ta.sma(df['atr'], length=20)

        # IMPORTANT: SHIFT EVERYTHING BY 1
        # The AI must only see information from the COMPLETED candles.
        feature_cols = ['atr', 'rsi_14', 'bb_width', 'ema_fast', 'ema_slow', 'candle_body_ratio', 'atr_ratio']
        df[feature_cols] = df[feature_cols].shift(1)
        
        return df.dropna()