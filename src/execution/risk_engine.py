import numpy as np

class RiskEngine:
    def __init__(self, account_balance, risk_pc=0.01):
        """
        V4 Risk Engine: Pure Position Sizing & Account Protection.

        risk_pc: The fixed percentage of the account balance to risk per trade.
                 1% risk means if the SL is hit, the account drops exactly 1%.
        """
        self.balance = account_balance
        self.risk_per_trade_cash = account_balance * risk_pc

    def calculate_lot_size(self, entry_price, sl_price, instrument_factor=100):
        """
        Calculates the exact lot size for XAUUSD (Gold).

        entry_price: Market or Limit price for entry.
        sl_price: The Stop Loss price calculated by the Expert.
        instrument_factor: 100 for standard Gold CFDs (1 Lot = 100 Ounces).
        """
        # 1. Calculate the absolute distance to the Stop Loss
        risk_distance = abs(entry_price - sl_price)
        
        # 2. Safety Check: Avoid division by zero or unrealistic tight stops
        if risk_distance < 0.10: # Minimum 10 cent buffer for Gold M5
            print("⚠️ RISK REJECTED: Stop Loss distance is too small.")
            return 0.0
            
        # 3. Position Sizing Formula:
        # Lots = Cash Risk / (SL Distance * Contract Size)
        # Example: $100 Risk / ($5.00 SL move * 100 Contract Size) = 0.20 Lots
        lots = self.risk_per_trade_cash / (risk_distance * instrument_factor)

        # 4. Standard Broker Compliance
        # Round to 2 decimal places as required by MT5/MT4
        final_lots = round(lots, 2)
        
        # 5. Final validation for minimum trade volume
        if final_lots < 0.01:
            print(f"⚠️ RISK REJECTED: Calculated lot size {lots:.4f} is below broker minimum.")
            return 0.0

        return final_lots

    def check_circuit_breaker(self, current_drawdown_r):
        """
        The 25R Global Circuit Breaker.

        This protects your "War Chest." If the portfolio's cumulative drawdown
        reaches -25R (25%), all execution is halted to prevent further loss.
        """
        # 25R equals a 25% account drop at 1% risk per trade.
        if current_drawdown_r <= -25.0:
            print("🛑 CIRCUIT BREAKER TRIGGERED: Portfolio Drawdown exceeded -25R.")
            print("Action: All trading activity locked for audit.")
            return False

        return True

    def update_balance(self, new_balance):
        """
        Updates the internal balance for the next lot calculation.
        """
        self.balance = new_balance
        # Recalculate the cash risk based on the new equity
        self.risk_per_trade_cash = self.balance * (self.risk_per_trade_cash / self.balance)