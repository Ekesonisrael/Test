#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np

# ==========================================
# 1. SOL-SPECIFIC DATA ENGINE (10,000 Candles)
# ==========================================
def fetch_okx_data(inst="SOL-USDT-SWAP", limit=10000):
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
    for col in ["o", "h", "l", "c"]:
        df[col] = df[col].astype(float)
    df['ts'] = pd.to_datetime(df['ts'].astype("int64"), unit="ms")
    
    today = pd.Timestamp.now('UTC').tz_localize(None)
    df = df[df['ts'] <= today].reset_index(drop=True)
    
    # 50 EMA Trend Filter
    df['ema_50'] = df['c'].ewm(span=50, adjust=False).mean()
    return df

# ==========================================
# 2. DUAL-SIDED SOL RE-ENGINEERED STRATEGY
# ==========================================
def run_sol_backtest(initial_balance=10000.0, risk_pct=0.01, leverage=3):
    df = fetch_okx_data("SOL-USDT-SWAP")
    if len(df) < 60: return
    
    balance = initial_balance
    trades = []
    
    # SOL-specific tunings: Wider stop buffer for volatility, higher R:R target
    sl_buffer = 0.012  # 1.2% stop buffer to dodge wicks
    rr_target = 3.5    # 1:3.5 target to reward wider stops
    lookback = 20      # 20-candle lookback for deeper liquidity pools
    
    i = lookback + 5
    while i < len(df) - 10:
        sweep_candle = df.iloc[i]
        confirm_candle = df.iloc[i+1]
        ema = sweep_candle['ema_50']
        
        is_bullish_trend = sweep_candle['c'] >= ema
        is_bearish_trend = sweep_candle['c'] < ema
        
        prior_low = df['l'].iloc[i-lookback:i].min()
        prior_high = df['h'].iloc[i-lookback:i].max()
        
        # --- LONG SETUP ---
        is_long_sweep = sweep_candle['l'] < prior_low and sweep_candle['c'] > prior_low
        is_long_disp = confirm_candle['c'] > confirm_candle['o'] and confirm_candle['c'] > sweep_candle['o']
        
        if is_bullish_trend and is_long_sweep and is_long_disp:
            if i + 2 >= len(df): break
            entry_price = df.iloc[i+2]['o']
            sl = sweep_candle['l'] * (1.0 - sl_buffer)
            risk_distance = entry_price - sl
            if risk_distance <= 0:
                i += 1
                continue
            tp = entry_price + (risk_distance * rr_target)
            side = "LONG"
            
        # --- SHORT SETUP ---
        elif is_bearish_trend and sweep_candle['h'] > prior_high and sweep_candle['c'] < prior_high:
            is_short_disp = confirm_candle['c'] < confirm_candle['o'] and confirm_candle['c'] < sweep_candle['o']
            if is_short_disp:
                if i + 2 >= len(df): break
                entry_price = df.iloc[i+2]['o']
                sl = sweep_candle['h'] * (1.0 + sl_buffer)
                risk_distance = sl - entry_price
                if risk_distance <= 0:
                    i += 1
                    continue
                tp = entry_price - (risk_distance * rr_target)
                side = "SHORT"
            else:
                i += 1
                continue
        else:
            i += 1
            continue

        # Position Sizing (3x Isolated Margin)
        risk_amount = balance * risk_pct
        position_size = risk_amount / risk_distance
        max_allowed_notional = balance * leverage
        if (position_size * entry_price) > max_allowed_notional:
            position_size = max_allowed_notional / entry_price

        # Forward Testing Loop
        outcome = None
        exit_price = None
        j = i + 2
        while j < len(df):
            future_candle = df.iloc[j]
            if side == "LONG":
                hit_sl = future_candle['l'] <= sl
                hit_tp = future_candle['h'] >= tp
                if hit_sl:
                    outcome = "LOSS"; exit_price = sl; break
                elif hit_tp:
                    outcome = "WIN"; exit_price = tp; break
            else:  # SHORT
                hit_sl = future_candle['h'] >= sl
                hit_tp = future_candle['l'] <= tp
                if hit_sl:
                    outcome = "LOSS"; exit_price = sl; break
                elif hit_tp:
                    outcome = "WIN"; exit_price = tp; break
            j += 1
            
        if outcome is None: break

        # PnL Calculation
        if side == "LONG":
            pnl = (exit_price - entry_price) * position_size
        else:
            pnl = (entry_price - exit_price) * position_size
            
        balance += pnl
        trades.append({
            "time": df.iloc[i+2]['ts'], "side": side, "outcome": outcome,
            "entry": entry_price, "exit": exit_price, "pnl": pnl, "balance": balance
        })
        i = j

    # --- RESULTS ---
    if not trades:
        print("\n[SOL-USDT] No valid setups found.")
        return

    wins = sum(1 for t in trades if t['outcome'] == 'WIN')
    losses = sum(1 for t in trades if t['outcome'] == 'LOSS')
    win_rate = (wins / len(trades)) * 100
    
    print(f"\n===== SOL-USDT RE-ENGINEERED SMC BACKTEST =====")
    print(f"Total Trades: {len(trades)} | Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:    ${balance:,.2f}")
    print(f"Total Return:     {((balance/initial_balance)-1)*100:+.2f}%")
    print("================================================")

if __name__ == "__main__":
    run_sol_backtest()
