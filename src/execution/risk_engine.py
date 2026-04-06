import numpy as np

class RiskEngine:
    def __init__(self, account_balance, risk_pc=0.01):
        self.balance = account_balance
        self.risk_per_trade = account_balance * risk_pc

    def calculate_lot_size(self, entry_price, sl_price, instrument_factor=100):
        """
        Calculates the position size for XAUUSD.
        instrument_factor: 100 for standard Gold CFDs (1 lot = 100 oz).
        """
        # Distance to Stop Loss
        risk_distance = abs(entry_price - sl_price)
        
        if risk_distance == 0:
            return 0
            
        # Formula: Cash Risk / (SL Distance * Contract Size)
        # Example: $100 / ($5.00 move * 100 oz) = 0.20 Lots
        lots = self.risk_per_trade / (risk_distance * instrument_factor)
        
        # Round to 2 decimal places (Standard for MT5/Brokers)
        return round(lots, 2)

    def get_equity_protection(self, current_drawdown_r):
        """
        The 30R Buffer Check. 
        If account hits -25R, it returns False to halt all trading.
        """
        if current_drawdown_r <= -25.0:
            print("🚨 CIRCUIT BREAKER TRIGGERED: 25% Drawdown Reached.")
            return False
        return True