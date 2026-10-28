#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np
import sys

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

    def validate_trade(self, entry, stop_loss, take_profit, action):
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0: 
            return False, "Zero risk distance", 0.0
            
        rr = reward / risk
        if rr < 2.0: 
            return False, f"R:R ratio {rr:.2f} is below the 2.0 minimum.", rr
            
        return True, "Approved", rr

# ==========================================
# 2. DATA ENGINE (Free OKX API)
# ==========================================
def fetch_data(symbol):
    """Fetches latest 15m candles (max 100 per request)."""
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=15m&limit=100"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()["data"]
        df = pd.DataFrame(data, columns=["ts", "o", "h", "l", "c", "v", "vc", "vq", "confirm"])
        df = df.iloc[::-1].reset_index(drop=True) # Reverse to chronological
        for col in ["o", "h", "l", "c"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ==========================================
# 3. STRATEGY ENGINE (SMC Liquidity Sweep)
# ==========================================
def scan_smc(df):
    """Scans the last 15 candles for a Liquidity Sweep and Order Block."""
    recent = df.tail(20).copy()
    
    # Find the swing low of the 5 candles before the recent price action
    prior_swing_low = recent['l'].iloc[0:15].min()
    
    # We look at the last 3 candles to see if a sweep just happened
    for i in range(len(recent)-3, len(recent)):
        candle = recent.iloc[i]
        
        # SWEEP LOGIC: Wick breaks the prior low, but closes back above it.
        if candle['l'] < prior_swing_low and candle['c'] > prior_swing_low:
            
            # ORDER BLOCK LOGIC: Find the last bearish candle before the sweep
            ob_high, ob_low = candle['o'], candle['l']
            for j in range(i-1, max(0, i-6), -1):
                if recent['c'].iloc[j] < recent['o'].iloc[j]: # Bearish candle
                    ob_high = recent['h'].iloc[j]
                    ob_low = recent['l'].iloc[j]
                    break
            
            entry = ob_high
            sl = candle['l'] * 0.999 # 0.1% buffer below the sweep wick
            tp = recent['h'].iloc[0:15].max() # The previous swing high (liquidity target)
            
            return {
                "signal": "BUY",
                "sweep_time": pd.to_datetime(int(candle['ts']), unit="ms"),
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "ob_zone": f"{ob_low:.2f} - {ob_high:.2f}"
            }
    return None

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== STARTING SMC LIQUIDITY SCANNER ===")
    
    risk_engine = RiskManager(account_balance=10000.0)
    watchlist = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
    
    found_trades = 0
    
    for symbol in watchlist:
        df = fetch_data(symbol)
        if df is None or len(df) < 20:
            continue
            
        current_price = df['c'].iloc[-1]
        print(f"\n[{symbol}] Current Price: ${current_price:.2f}")
        
        setup = scan_smc(df)
        
        if setup:
            print(f"  🔥 LIQUIDITY SWEEP DETECTED at {setup['sweep_time']}")
            print(f"  Order Block Zone: {setup['ob_zone']}")
            print(f"  Entry: ${setup['entry']:.2f} | SL: ${setup['sl']:.2f} | TP: ${setup['tp']:.2f}")
            
            # Pass to Risk Engine
            approved, reason, rr = risk_engine.validate_trade(setup['entry'], setup['sl'], setup['tp'], "BUY")
            
            if approved:
                size = risk_engine.calculate_position_size(setup['entry'], setup['sl'])
                print(f"  ✅ TRADE APPROVED (R:R {rr:.2f})")
                print(f"  -> Execute BUY for {size:.4f} contracts at ${setup['entry']:.2f}")
                found_trades += 1
            else:
                print(f"  ❌ TRADE REJECTED: {reason}")
        else:
            print("  No recent sweep setups found.")
            
    print(f"\n=== SCAN COMPLETE. Found {found_trades} valid setups. ===")
