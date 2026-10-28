#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ==========================================
# 1. DATA ENGINE (Free OKX API)
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
    
    # FIX: Strict, timezone-aware cutoff to destroy the 2026 glitch
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    df['ts'] = df['ts'].astype("int64")
    df = df[df['ts'] <= now_ms].reset_index(drop=True)
    df['ts'] = pd.to_datetime(df['ts'], unit="ms")
    
    return df

# ==========================================
# 2. SMC STRATEGY + MOCK EXECUTION
# ==========================================
def run_backtest(inst, initial_balance=10000.0, risk_pct=0.01, leverage=3):
    df = fetch_okx_data(inst)
    if len(df) < 100: return
    
    balance = initial_balance
    trades = []
    
    i = 15
    while i < len(df) - 2:
        sweep_candle = df.iloc[i]
        confirm_candle = df.iloc[i+1]
        
        prior_low = df['l'].iloc[i-10:i].min()
        
        # 1. LIQUIDITY SWEEP
        is_sweep = sweep_candle['l'] < prior_low and sweep_candle['c'] > prior_low
        
        # 2. DISPLACEMENT (Loosened: Just needs to be a strong green candle)
        is_displacement = confirm_candle['c'] > confirm_candle['o'] and confirm_candle['c'] > sweep_candle['c']
        
        if is_sweep and is_displacement:
            # 3. FIXED EXECUTION (Entry at the open of the next candle)
            entry_price = df.iloc[i+2]['o']
            sl = sweep_candle['l'] * 0.999
            
            risk_distance = entry_price - sl
            if risk_distance <= 0:
                i += 1
                continue
                
            # 4. TIGHTER TP (1.5 R:R) so it actually hits
            tp = entry_price + (risk_distance * 1.5)
            
            # 5. MARGIN SIZING
            risk_amount = balance * risk_pct
            position_size = risk_amount / risk_distance
            notional_value = position_size * entry_price
            required_margin = notional_value / leverage
            
            if required_margin > (balance * 0.5): 
                i += 1
                continue
                
            # 6. FORWARD TESTING (Max 20 candles / 5 hours)
            outcome = "OPEN"
            exit_price = None
            max_forward = min(i + 20, len(df) - 1)
            
            for j in range(i+3, max_forward):
                future_candle = df.iloc[j]
                
                if future_candle['l'] <= sl:
                    outcome = "LOSS"
                    exit_price = sl
                    break
                elif future_candle['h'] >= tp:
                    outcome = "WIN"
                    exit_price = tp
                    break
                    
            # 7. HANDLE TIMEOUT
            if outcome == "OPEN":
                outcome = "TIMEOUT"
                exit_price = df.iloc[max_forward]['c']
                
            pnl = (exit_price - entry_price) * position_size
            balance += pnl
            trades.append({
                "time": df.iloc[i+2]['ts'],
                "outcome": outcome,
                "entry": entry_price,
                "sl": sl,
                "tp": tp,
                "pnl": pnl,
                "balance": balance
            })
            
            i = j 
        i += 1

    if not trades:
        print(f"\n[{inst}] No valid setups found.")
        return

    wins = sum(1 for t in trades if t['outcome'] == 'WIN')
    losses = sum(1 for t in trades if t['outcome'] == 'LOSS')
    timeouts = sum(1 for t in trades if t['outcome'] == 'TIMEOUT')
    total_trades = len(trades)
    
    win_rate = (wins / total_trades) * 100
    valid_win_rate = (wins / max(1, wins+losses)) * 100
    
    print(f"\n================ {inst} SMC BACKTEST ================")
    print(f"Total Trades: {total_trades} (Wins: {wins}, Losses: {losses}, Timeouts: {timeouts})")
    print(f"Win Rate (All): {win_rate:.1f}% | Win Rate (excl. Timeouts): {valid_win_rate:.1f}%")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:    ${balance:,.2f}")
    print(f"Total Return:     {((balance/initial_balance)-1)*100:+.2f}%")
    print("-----------------------------------------------------")
    print("Last 5 Trades:")
    for t in trades[-5:]:
        print(f"  {t['time']} | {t['outcome']} | Entry: {t['entry']:.2f} | PnL: ${t['pnl']:,.2f}")
    print("=====================================================")

if __name__ == "__main__":
    run_backtest("BTC-USDT-SWAP")
    run_backtest("SOL-USDT-SWAP")
