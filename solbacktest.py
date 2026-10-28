#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np

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
    
    # 50 EMA Trend Filter & 14-period ATR for Volatility
    df['ema_50'] = df['c'].ewm(span=50, adjust=False).mean()
    
    high_low = df['h'] - df['l']
    high_close = np.abs(df['h'] - df['c'].shift())
    low_close = np.abs(df['l'] - df['c'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    return df

def run_sol_backtest(initial_balance=10000.0, risk_pct=0.01, leverage=3):
    df = fetch_okx_data("SOL-USDT-SWAP")
    if len(df) < 60: return
    
    balance = initial_balance
    trades = []
    
    rr_target = 2.5
    lookback = 15
    
    i = lookback + 15
    while i < len(df) - 100: # Ensure room for forward testing
        sweep_candle = df.iloc[i]
        confirm_candle = df.iloc[i+1]
        ema = sweep_candle['ema_50']
        atr = sweep_candle['atr']
        
        if pd.isna(atr) or atr == 0:
            i += 1
            continue
            
        is_bullish_trend = sweep_candle['c'] >= ema
        is_bearish_trend = sweep_candle['c'] < ema
        
        prior_low = df['l'].iloc[i-lookback:i].min()
        prior_high = df['h'].iloc[i-lookback:i].max()
        
        # --- LONG SETUP ---
        is_long_sweep = sweep_candle['l'] < prior_low and sweep_candle['c'] > prior_low
        is_long_disp = confirm_candle['c'] > confirm_candle['o'] and confirm_candle['c'] > sweep_candle['o']
        
        if is_bullish_trend and is_long_sweep and is_long_disp:
            entry_price = df.iloc[i+2]['o']
            # Dynamic ATR Stop Loss (1.5x ATR below swing low)
            sl = sweep_candle['l'] - (1.5 * atr)
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
                entry_price = df.iloc[i+2]['o']
                # Dynamic ATR Stop Loss (1.5x ATR above swing high)
                sl = sweep_candle['h'] + (1.5 * atr)
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

        # Position Sizing
        risk_amount = balance * risk_pct
        position_size = risk_amount / risk_distance
        max_allowed_notional = balance * leverage
        if (position_size * entry_price) > max_allowed_notional:
            position_size = max_allowed_notional / entry_price

        # Forward Testing Loop (With 50-candle Timeout)
        outcome = "OPEN"
        exit_price = None
        max_forward = min(i + 50, len(df) - 1)
        
        for j in range(i+3, max_forward):
            future_candle = df.iloc[j]
            if side == "LONG":
                hit_sl = future_candle['l'] <= sl
                hit_tp = future_candle['h'] >= tp
                if hit_sl:
                    outcome = "LOSS"; exit_price = sl; break
                elif hit_tp:
                    outcome = "WIN"; exit_price = tp; break
            else:
                hit_sl = future_candle['h'] >= sl
                hit_tp = future_candle['l'] <= tp
                if hit_sl:
                    outcome = "LOSS"; exit_price = sl; break
                elif hit_tp:
                    outcome = "WIN"; exit_price = tp; break
            
        if outcome == "OPEN":
            outcome = "TIMEOUT"
            exit_price = df.iloc[max_forward]['c']

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

    if not trades:
        print("\n[SOL-USDT] No valid setups found.")
        return

    wins = sum(1 for t in trades if t['outcome'] == 'WIN')
    losses = sum(1 for t in trades if t['outcome'] == 'LOSS')
    timeouts = sum(1 for t in trades if t['outcome'] == 'TIMEOUT')
    
    win_rate = (wins / max(1, wins+losses)) * 100 # Excluding timeouts
    
    print(f"\n===== SOL-USDT ATR-BASED SMC BACKTEST =====")
    print(f"Total Trades: {len(trades)} | Wins: {wins} | Losses: {losses} | Timeouts: {timeouts}")
    print(f"Win Rate (excl Timeouts): {win_rate:.1f}%")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:    ${balance:,.2f}")
    print(f"Total Return:     {((balance/initial_balance)-1)*100:+.2f}%")
    print("===========================================")

if __name__ == "__main__":
    run_sol_backtest()

