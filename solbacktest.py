#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np

def fetch_data(inst, limit=10000):
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
        if len(r["data"]) < 100: break
        
    df = pd.DataFrame(all_data, columns=["ts", "o", "h", "l", "c", "v", "vc", "vq", "confirm"])
    df = df.iloc[::-1].reset_index(drop=True)
    for col in ["o", "h", "l", "c"]:
        df[col] = df[col].astype(float)
    
    df['ts'] = pd.to_datetime(df['ts'].astype("int64"), unit="ms")
    today = pd.Timestamp.now('UTC').tz_localize(None)
    df = df[df['ts'] <= today].reset_index(drop=True)
    
    df['ema_50'] = df['c'].ewm(span=50, adjust=False).mean()
    tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    return df

def run_clean_sol_backtest(initial_balance=10000.0, risk_pct=0.01, leverage=3):
    sol_df = fetch_data("SOL-USDT-SWAP")
    btc_df = fetch_data("BTC-USDT-SWAP")
    
    if len(sol_df) < 60 or len(btc_df) < 60: return
    
    btc_trend = btc_df.set_index('ts')[['c', 'ema_50']].rename(columns={'c': 'btc_close', 'ema_50': 'btc_ema'})
    sol_df = sol_df.join(btc_trend, on='ts').dropna().reset_index(drop=True)
    
    balance = initial_balance
    trades = []
    macro_blocked = 0
    
    rr_target = 2.5       # FIXED: Back to our proven winning math
    lookback = 15
    slippage = 0.0005     # 0.05% slippage penalty on market fills
    
    i = lookback + 15
    while i < len(sol_df) - 100:
        sweep = sol_df.iloc[i]
        confirm = sol_df.iloc[i+1]
        entry_candle = sol_df.iloc[i+2]
        
        ema = sweep['ema_50']
        atr = sweep['atr']
        
        if pd.isna(atr) or atr == 0: 
            i += 1; continue
            
        is_bullish = sweep['c'] >= ema
        is_bearish = sweep['c'] < ema
        
        prior_low = sol_df['l'].iloc[i-lookback:i].min()
        prior_high = sol_df['h'].iloc[i-lookback:i].max()
        
        btc_is_bullish = sweep['btc_close'] >= sweep['btc_ema']
        btc_is_bearish = sweep['btc_close'] < sweep['btc_ema']
        
        side = None
        
        # --- LONG SETUP (Market Execution + Slippage) ---
        if is_bullish:
            if sweep['l'] < prior_low and sweep['c'] > prior_low:
                if confirm['c'] > confirm['o'] and confirm['c'] > sweep['o']:
                    if not btc_is_bullish:
                        macro_blocked += 1 # Macro filter saved us
                    else:
                        side = "LONG"
                        entry_price = entry_candle['o'] * (1.0 + slippage)
                        
        # --- SHORT SETUP (Market Execution + Slippage) ---
        elif is_bearish:
            if sweep['h'] > prior_high and sweep['c'] < prior_high:
                if confirm['c'] < confirm['o'] and confirm['c'] < sweep['o']:
                    if not btc_is_bearish:
                        macro_blocked += 1 # Macro filter saved us
                    else:
                        side = "SHORT"
                        entry_price = entry_candle['o'] * (1.0 - slippage)

        if side:
            if side == "LONG":
                sl = sweep['l'] - (1.5 * atr)
                risk_distance = entry_price - sl
                tp = entry_price + (risk_distance * rr_target)
            else:
                sl = sweep['h'] + (1.5 * atr)
                risk_distance = sl - entry_price
                tp = entry_price - (risk_distance * rr_target)

            if risk_distance > 0:
                risk_amount = balance * risk_pct
                position_size = risk_amount / risk_distance
                max_notional = balance * leverage
                if (position_size * entry_price) > max_notional:
                    position_size = max_notional / entry_price

                outcome = "OPEN"
                exit_price = None
                max_forward = min(i + 50, len(sol_df) - 1)
                
                for j in range(i+3, max_forward):
                    future = sol_df.iloc[j]
                    if side == "LONG":
                        if future['l'] <= sl: 
                            outcome = "LOSS"; exit_price = sl * (1.0 - slippage); break
                        elif future['h'] >= tp: 
                            outcome = "WIN"; exit_price = tp * (1.0 - slippage); break
                    else:
                        if future['h'] >= sl: 
                            outcome = "LOSS"; exit_price = sl * (1.0 + slippage); break
                        elif future['l'] <= tp: 
                            outcome = "WIN"; exit_price = tp * (1.0 + slippage); break
                        
                if outcome == "OPEN":
                    outcome = "TIMEOUT"
                    exit_price = sol_df.iloc[max_forward]['c']

                if side == "LONG":
                    pnl = (exit_price - entry_price) * position_size
                else:
                    pnl = (entry_price - exit_price) * position_size
                    
                balance += pnl
                trades.append({"time": entry_candle['ts'], "side": side, "outcome": outcome, "net_pnl": pnl})
                i = j 
        else:
            i += 1

    if not trades:
        print("\n[SOL-USDT] No valid setups found.")
        return

    wins = sum(1 for t in trades if t['outcome'] == 'WIN')
    losses = sum(1 for t in trades if t['outcome'] == 'LOSS')
    timeouts = sum(1 for t in trades if t['outcome'] == 'TIMEOUT')
    win_rate = (wins / max(1, wins+losses)) * 100
    
    print(f"\n===== FINAL SOL BACKTEST (Market Execution + BTC Filter) =====")
    print(f"Total Trades Executed: {len(trades)} | Wins: {wins} | Losses: {losses} | Timeouts: {timeouts}")
    print(f"Win Rate (excl Timeouts): {win_rate:.1f}%")
    print(f"Macro Filter Blocked Setups: {macro_blocked}")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:    ${balance:,.2f}")
    print(f"Total Return:     {((balance/initial_balance)-1)*100:+.2f}%")
    print("=============================================================")

if __name__ == "__main__":
    run_clean_sol_backtest()
