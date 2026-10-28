#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np

# ==========================================
# 1. RISK MANAGEMENT CORE
# ==========================================
class RiskManager:
    def __init__(self, account_balance=10000.0, max_risk_pct=0.01):
        self.account_balance = account_balance
        self.max_risk_pct = max_risk_pct

    def calculate_position_size(self, entry, stop_loss):
        risk_amount = self.account_balance * self.max_risk_pct
        distance = abs(entry - stop_loss)
        if distance == 0: return 0
        return risk_amount / distance

    def validate_trade(self, entry, stop_loss, take_profit):
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0: 
            return False, "Zero risk distance", 0.0
            
        rr = reward / risk
        if rr < 2.0: 
            return False, f"R:R {rr:.2f} < 2.0", rr
            
        return True, "Approved", rr

# ==========================================
# 2. DATA ENGINE (Free OKX API - Paginated)
# ==========================================
def fetch_data(symbol, limit=1000):
    """Fetches up to 1000 historical 15m candles."""
    print(f"Fetching {limit} candles for {symbol}...")
    all_data = []
    after = ""
    
    while len(all_data) < limit:
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar=15m&limit=100"
        if after:
            url += f"&after={after}"
            
        try:
            r = requests.get(url, timeout=10)
            data = r.json()["data"]
            if not data: break
            all_data.extend(data)
            after = data[-1][0]
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=["ts", "o", "h", "l", "c", "v", "vc", "vq", "confirm"])
    df = df.iloc[::-1].reset_index(drop=True)
    for col in ["o", "h", "l", "c"]:
        df[col] = df[col].astype(float)
    df['ts'] = pd.to_datetime(df['ts'].astype("int64"), unit="ms")
    return df

# ==========================================
# 3. STRATEGY ENGINE (Historical Backtest)
# ==========================================
def backtest_smc(df):
    """Scans the ENTIRE dataset for Liquidity Sweeps."""
    setups_found = 0
    setups_approved = 0
    risk = RiskManager()
    
    # We start at candle 15 so we have history to look back at
    for i in range(15, len(df) - 1):
        candle = df.iloc[i]
        
        # 1. LIQUIDITY SWEEP LOGIC
        # Find lowest low of the 10 candles prior to this one
        prior_low = df['l'].iloc[i-10:i].min()
        
        # Did this candle sweep the low but close above it?
        if candle['l'] < prior_low and candle['c'] > prior_low:
            
            # 2. ORDER BLOCK LOGIC
            # Last bearish candle before the sweep
            ob_high, ob_low = candle['o'], candle['l']
            for j in range(i-1, max(0, i-6), -1):
                if df['c'].iloc[j] < df['o'].iloc[j]:
                    ob_high = df['h'].iloc[j]
                    ob_low = df['l'].iloc[j]
                    break
            
            entry = ob_high
            sl = candle['l'] * 0.999 # Stop below the sweep wick
            tp = df['h'].iloc[i-10:i].max() # Target previous swing high
            
            setups_found += 1
            
            # 3. RISK VALIDATION
            approved, reason, rr = risk.validate_trade(entry, sl, tp)
            
            if approved:
                size = risk.calculate_position_size(entry, sl)
                setups_approved += 1
                print(f"[{candle['ts']}] 🟢 SWEEP & OB FOUND | Entry: {entry:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | R:R {rr:.2f} | Size: {size:.2f}")
            else:
                # Only print rejects if they were actual sweeps but failed R:R
                print(f"[{candle['ts']}] 🔴 SWEEP FOUND BUT REJECTED: {reason}")

    print("\n=== BACKTEST COMPLETE ===")
    print(f"Total Sweeps Found: {setups_found}")
    print(f"Trades Approved by Risk Engine: {setups_approved}")
    print(f"Rejection Rate: {100 - (setups_approved/max(1, setups_found)*100):.1f}%")

if __name__ == "__main__":
    df = fetch_data("BTC-USDT-SWAP", limit=1000)
    backtest_smc(df)
