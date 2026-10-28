#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np

# ==========================================
# 1. DATA ENGINE (Free OKX API - 1000 Candles)
# ==========================================
def fetch_okx_data(inst, limit=1000):
    print(f"Fetching {limit} candles for {inst}...")
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={inst}&bar=15m&limit=100"
    all_data = []
    after = ""
    
    while len(all_data) < limit:
        req_url = url + (f"&after={after}" if after else "")
        r = requests.get(req_url, timeout=10).json()
        if r["code"] != "0" or not r["data"]: break
        all_data.extend(r["data"])
        after = r["data"][-1][0]
        
    df = pd.DataFrame(all_data, columns=["ts", "o", "h", "l", "c", "v", "vc", "vq", "confirm"])
    df = df.iloc[::-1].reset_index(drop=True)
    for col in ["o", "h", "l", "c"]:
        df[col] = df[col].astype(float)
    df['ts'] = pd.to_datetime(df['ts'].astype("int64"), unit="ms")
    
    # Filter future glitches
    today = pd.Timestamp.utcnow().tz_localize(None)
    df = df[df['ts'] <= today].reset_index(drop=True)
    return df

# ==========================================
# 2. SMC STRATEGY + MOCK EXECUTION
# ==========================================
def run_backtest(inst, initial_balance=10000.0, risk_pct=0.01):
    df = fetch_okx_data(inst)
    if len(df) < 100: return
    
    balance = initial_balance
    trades = []
    
    i = 15
    while i < len(df) - 2:
        sweep_candle = df.iloc[i]
        confirm_candle = df.iloc[i+1]
        
        # Find lowest low of prior 10 candles
        prior_low = df['l'].iloc[i-10:i].min()
        
        # 1. LIQUIDITY SWEEP (Wick below, close above)
        is_sweep = sweep_candle['l'] < prior_low and sweep_candle['c'] > prior_low
        
        # 2. DISPLACEMENT (Next candle breaks high, strong close)
        is_displacement = confirm_candle['c'] > sweep_candle['h'] and confirm_candle['c'] > confirm_candle['o']
        
        if is_sweep and is_displacement:
            # MARKET ORDER ENTRY: We enter at the close of the confirm candle
            entry_price = confirm_candle['c']
            
            # Stop Loss below the sweep
            sl = sweep_candle['l'] * 0.999
            
            # Take Profit (1:2.5 R:R)
            risk_distance = entry_price - sl
            tp = entry_price + (risk_distance * 2.5)
            
            # POSITION SIZING
            risk_amount = balance * risk_pct
            position_size = risk_amount / risk_distance
            
            # 3. FORWARD TESTING (Did it hit TP or SL?)
            outcome = "OPEN"
            exit_price = None
            
            # Look ahead up to 50 candles (12.5 hours) to see if it hits TP or SL
            for j in range(i+2, min(i+50, len(df))):
                future_candle = df.iloc[j]
                
                # Check Stop Loss First (Conservative - assumes worst case)
                if future_candle['l'] <= sl:
                    outcome = "LOSS"
                    exit_price = sl
                    break
                # Check Take Profit
                elif future_candle['h'] >= tp:
                    outcome = "WIN"
                    exit_price = tp
                    break
                    
            if outcome != "OPEN":
                pnl = (exit_price - entry_price) * position_size
                balance += pnl
                trades.append({
                    "time": sweep_candle['ts'],
                    "outcome": outcome,
                    "entry": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "pnl": pnl,
                    "balance": balance
                })
                # Jump i forward to the end of this trade to avoid overlapping
                i = j 
        i += 1

    # --- PRINT RESULTS ---
    if not trades:
        print(f"\n[{inst}] No valid SMC setups found in this period.")
        return

    wins = sum(1 for t in trades if t['outcome'] == 'WIN')
    losses = sum(1 for t in trades if t['outcome'] == 'LOSS')
    win_rate = (wins / len(trades)) * 100
    
    print(f"\n================ {inst} SMC BACKTEST ================")
    print(f"Period: {df['ts'].iloc[0].date()} to {df['ts'].iloc[-1].date()}")
    print(f"Total Trades: {len(trades)}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:    ${balance:,.2f}")
    print(f"Total Return:     {((balance/initial_balance)-1)*100:+.2f}%")
    print("-----------------------------------------------------")
    print("Last 3 Trades:")
    for t in trades[-3:]:
        print(f"  {t['time']} | {t['outcome']} | Entry: {t['entry']:.2f} | PnL: ${t['pnl']:,.2f}")
    print("=====================================================")

if __name__ == "__main__":
    run_backtest("BTC-USDT-SWAP")
    run_backtest("SOL-USDT-SWAP")
