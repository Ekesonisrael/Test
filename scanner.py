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
    today = pd.Timestamp.now('UTC').tz_localize(None)
    df = df[df['ts'] <= today].reset_index(drop=True)
    
    # Calculate 50 EMA Trend Filter (optimized for 15m responsiveness)
    df['ema_50'] = df['c'].ewm(span=50, adjust=False).mean()
    return df

# ==========================================
# 2. OPTIMIZED SMC STRATEGY + EXECUTION
# ==========================================
def run_backtest(inst, initial_balance=10000.0, risk_pct=0.01, leverage=3):
    df = fetch_okx_data(inst)
    if len(df) < 60: return
    
    balance = initial_balance
    trades = []
    
    i = 60 # Start after 50 EMA has warmed up
    while i < len(df) - 55:
        sweep_candle = df.iloc[i]
        confirm_candle = df.iloc[i+1]

        # --- OPTIMIZED FILTER: MACRO TREND FILTER (50 EMA) ---
        # Price must be above the 50 EMA for a Long setup
        if sweep_candle['c'] < sweep_candle['ema_50']:
            i += 1
            continue

        # Find lowest low of prior 10 candles (Liquidity pool)
        prior_low = df['l'].iloc[i-10:i].min()
        
        # 1. LIQUIDITY SWEEP
        is_sweep = sweep_candle['l'] < prior_low and sweep_candle['c'] > prior_low
        
        # 2. DISPLACEMENT
        is_displacement = confirm_candle['c'] > confirm_candle['o'] and confirm_candle['c'] > sweep_candle['o']
        
        if is_sweep and is_displacement:
            if i + 2 >= len(df):
                break
            entry_price = df.iloc[i+2]['o']
            
            # Stop Loss safely below the sweep wick
            sl = sweep_candle['l'] * 0.997
            
            # Take Profit (1:2.5 R:R)
            risk_distance = entry_price - sl
            if risk_distance <= 0:
                i += 1
                continue
                
            tp = entry_price + (risk_distance * 2.5)
            
            # ISOLATED POSITION SIZING
            risk_amount = balance * risk_pct
            position_size = risk_amount / risk_distance
            max_allowed_notional = balance * leverage
            
            if (position_size * entry_price) > max_allowed_notional:
                position_size = max_allowed_notional / entry_price
            
            # 3. FORWARD TESTING
            outcome = "TIMEOUT"
            exit_price = None
            
            end_lookahead = min(i + 52, len(df))
            for j in range(i+2, end_lookahead):
                future_candle = df.iloc[j]
                
                hit_sl = future_candle['l'] <= sl
                hit_tp = future_candle['h'] >= tp
                
                if hit_sl and hit_tp:
                    outcome = "LOSS"
                    exit_price = sl
                    break
                elif hit_sl:
                    outcome = "LOSS"
                    exit_price = sl
                    break
                elif hit_tp:
                    outcome = "WIN"
                    exit_price = tp
                    break
            
            if outcome == "TIMEOUT":
                exit_price = df.iloc[end_lookahead-1]['c']
                outcome = "WIN" if exit_price > entry_price else "LOSS"

            pnl = (exit_price - entry_price) * position_size
            balance += pnl
            
            trades.append({
                "time": df.iloc[i+2]['ts'],
                "outcome": outcome,
                "entry": entry_price,
                "exit": exit_price,
                "pnl": pnl,
                "balance": balance
            })
            
            i = j
        else:
            i += 1

    # --- PRINT RESULTS ---
    if not trades:
        print(f"\n[{inst}] No valid optimized SMC setups found.")
        return

    wins = sum(1 for t in trades if t['outcome'] == 'WIN')
    losses = sum(1 for t in trades if t['outcome'] == 'LOSS')
    win_rate = (wins / len(trades)) * 100
    
    print(f"\n========== {inst} OPTIMIZED SMC BACKTEST ==========")
    print(f"Period: {df['ts'].iloc[60].date()} to {df['ts'].iloc[-1].date()}")
    print(f"Total Trades: {len(trades)}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:    ${balance:,.2f}")
    print(f"Total Return:     {((balance/initial_balance)-1)*100:+.2f}%")
    print("-------------------------------------------------------------")
    print("Last 3 Trades:")
    for t in trades[-3:]:
        print(f"  {t['time']} | {t['outcome']} | Entry: {t['entry']:.2f} | Exit: {t['exit']:.2f} | PnL: ${t['pnl']:,.2f}")
    print("=============================================================")

if __name__ == "__main__":
    run_backtest("BTC-USDT-SWAP")
    run_backtest("SOL-USDT-SWAP")
    
