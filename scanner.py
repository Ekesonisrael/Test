#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np

# ==========================================
# 1. DATA ENGINE (10,000 Candles + Volume)
# ==========================================
def fetch_okx_data(inst, limit=10000):
    print(f"Fetching up to {limit} candles for {inst}...")
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={inst}&bar=15m&limit=100"
    all_data = []
    after = ""
    
    while len(all_data) < limit:
        req_url = url + (f"&after={after}" if after else "")
        r = requests.get(req_url, timeout=10).json()
        if r["code"] != "0" or not r["data"]: break
        all_data.extend(r["data"])
        after = r["data"][-1][0]
        if len(r["data"]) < 100: break
        
    df = pd.DataFrame(all_data, columns=["ts", "o", "h", "l", "c", "v", "vc", "vq", "confirm"])
    df = df.iloc[::-1].reset_index(drop=True)
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = df[col].astype(float)
    df['ts'] = pd.to_datetime(df['ts'].astype("int64"), unit="ms")
    
    today = pd.Timestamp.now('UTC').tz_localize(None)
    df = df[df['ts'] <= today].reset_index(drop=True)
    
    # 50 EMA Trend Filter & 20-period Volume Moving Average
    df['ema_50'] = df['c'].ewm(span=50, adjust=False).mean()
    df['vol_ma'] = df['v'].rolling(window=20).mean()
    return df

# ==========================================
# 2. REFINED SMC STRATEGY (VOLUME + BREAK-EVEN)
# ==========================================
def run_backtest(inst, initial_balance=10000.0, risk_pct=0.01, leverage=3):
    df = fetch_okx_data(inst)
    if len(df) < 60: return
    
    balance = initial_balance
    trades = []
    
    sl_buffer = 0.997  # 0.3% below sweep low for BTC
    rr_target = 2.0    # 1:2 R:R target
    
    i = 60
    while i < len(df) - 10:
        sweep_candle = df.iloc[i]
        confirm_candle = df.iloc[i+1]

        # Macro Trend Filter (50 EMA)
        if sweep_candle['c'] < sweep_candle['ema_50']:
            i += 1
            continue

        # Volume Filter: Displacement volume must exceed 20-period average volume
        if confirm_candle['v'] <= confirm_candle['vol_ma']:
            i += 1
            continue

        # 15-Candle Lookback
        prior_low = df['l'].iloc[i-15:i].min()
        
        # 1. LIQUIDITY SWEEP
        is_sweep = sweep_candle['l'] < prior_low and sweep_candle['c'] > prior_low
        
        # 2. DISPLACEMENT
        is_displacement = confirm_candle['c'] > confirm_candle['o'] and confirm_candle['c'] > sweep_candle['o']
        
        if is_sweep and is_displacement:
            if i + 2 >= len(df):
                break
            entry_price = df.iloc[i+2]['o']
            
            # Stop Loss & Take Profit
            sl = sweep_candle['l'] * sl_buffer
            risk_distance = entry_price - sl
            if risk_distance <= 0:
                i += 1
                continue
                
            tp = entry_price + (risk_distance * rr_target)
            
            # ISOLATED POSITION SIZING
            risk_amount = balance * risk_pct
            position_size = risk_amount / risk_distance
            max_allowed_notional = balance * leverage
            
            if (position_size * entry_price) > max_allowed_notional:
                position_size = max_allowed_notional / entry_price
            
            # 3. FORWARD TESTING with Break-Even Trailing Stop
            outcome = None
            exit_price = None
            be_triggered = False
            
            j = i + 2
            while j < len(df):
                future_candle = df.iloc[j]
                
                # Check for Break-Even activation (price reaches 1:1 reward distance)
                if not be_triggered and future_candle['h'] >= (entry_price + risk_distance):
                    sl = entry_price  # Move stop loss to entry
                    be_triggered = True
                
                hit_sl = future_candle['l'] <= sl
                hit_tp = future_candle['h'] >= tp
                
                if hit_sl and hit_tp:
                    outcome = "LOSS" if not be_triggered else "BREAK-EVEN"
                    exit_price = sl
                    break
                elif hit_sl:
                    outcome = "LOSS" if not be_triggered else "BREAK-EVEN"
                    exit_price = sl
                    break
                elif hit_tp:
                    outcome = "WIN"
                    exit_price = tp
                    break
                j += 1
            
            if outcome is None:
                break

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
        print(f"\n[{inst}] No valid setups found.")
        return

    wins = sum(1 for t in trades if t['outcome'] == 'WIN')
    losses = sum(1 for t in trades if t['outcome'] == 'LOSS')
    break_evens = sum(1 for t in trades if t['outcome'] == 'BREAK-EVEN')
    win_rate = (wins / len(trades)) * 100
    
    print(f"\n===== {inst} REFINED STRATEGY (VOLUME + BREAK-EVEN) =====")
    print(f"Period: {df['ts'].iloc[60].date()} to {df['ts'].iloc[-1].date()}")
    print(f"Total Trades: {len(trades)}")
    print(f"Wins: {wins} | Losses: {losses} | Break-Evens: {break_evens}")
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
