#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np

def fetch_data(inst, limit=10000):
    print(f"Fetching {limit} candles for {inst}...")
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={inst}&bar=1H&limit=100"
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
    
    df['ema_50'] = df['c'].ewm(span=50, adjust=False).mean()
    df['vol_ma'] = df['v'].rolling(window=20).mean()
    
    tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    return df

def run_proprietary_sol_backtest(initial_balance=10000.0, risk_pct=0.01, leverage=3):
    sol_df = fetch_data("SOL-USDT-SWAP")
    btc_df = fetch_data("BTC-USDT-SWAP")
    
    if len(sol_df) < 60 or len(btc_df) < 60: return
    
    btc_trend = btc_df.set_index('ts')[['c', 'ema_50']].rename(columns={'c': 'btc_close', 'ema_50': 'btc_ema'})
    sol_df = sol_df.join(btc_trend, on='ts').dropna().reset_index(drop=True)
    
    balance = initial_balance
    trades = []
    volume_blocked = 0
    macro_blocked = 0
    
    rr_target = 2.5       
    lookback = 20        
    slippage = 0.0005     
    
    i = lookback + 20
    while i < len(sol_df) - 100:
        sweep = sol_df.iloc[i]
        confirm = sol_df.iloc[i+1]
        entry_candle = sol_df.iloc[i+2]
        
        ema = sweep['ema_50']
        atr = sweep['atr']
        vol_ma = sweep['vol_ma']
        
        if pd.isna(atr) or atr == 0 or pd.isna(vol_ma) or vol_ma == 0: 
            i += 1; continue
            
        is_bullish = sweep['c'] >= ema
        is_bearish = sweep['c'] < ema
        
        prior_low = sol_df['l'].iloc[i-lookback:i].min()
        prior_high = sol_df['h'].iloc[i-lookback:i].max()
        
        btc_is_bullish = sweep['btc_close'] >= sweep['btc_ema']
        btc_is_bearish = sweep['btc_close'] < sweep['btc_ema']
        
        side = None
        
        # LOOSENED VOLUME FILTER: 1.2x instead of 1.4x
        has_volume = confirm['v'] > (vol_ma * 1.2)
        
        # --- LONG SETUP ---
        if is_bullish:
            if sweep['l'] < prior_low and sweep['c'] > prior_low:
                if confirm['c'] > confirm['o'] and confirm['c'] > sweep['o']:
                    if not btc_is_bullish:
                        macro_blocked += 1
                    elif not has_volume:
                        volume_blocked += 1 
                    else:
                        side = "LONG"
                        entry_price = entry_candle['o'] * (1.0 + slippage)
                        
        # --- SHORT SETUP ---
        elif is_bearish:
            if sweep['h'] > prior_high and sweep['c'] < prior_high:
                if confirm['c'] < confirm['o'] and confirm['c'] < sweep['o']:
                    if not btc_is_bearish:
                        macro_blocked += 1
                    elif not has_volume:
                        volume_blocked += 1 
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
                max_forward = min(i + 100, len(sol_df) - 1)
                
                # TRAILING STOP LOGIC
                current_sl = sl
                trailing_active = False
                
                for j in range(i+3, max_forward):
                    future = sol_df.iloc[j]
                    
                    if side == "LONG":
                        # Check if we hit initial TP
                        if future['h'] >= tp: 
                            outcome = "WIN"; exit_price = tp * (1.0 - slippage); break
                        # Check trailing stop
                        if trailing_active and future['l'] <= current_sl:
                            outcome = "TRAILED"; exit_price = current_sl * (1.0 - slippage); break
                        # Check hard SL
                        if not trailing_active and future['l'] <= sl:
                            outcome = "LOSS"; exit_price = sl * (1.0 - slippage); break
                            
                        # Update trailing stop (if price goes 1.5R in favor, trail to breakeven + 0.5R)
                        if future['h'] >= entry_price + (risk_distance * 1.5):
                            trailing_active = True
                            current_sl = max(current_sl, entry_price + (risk_distance * 0.5))
                            
                    else: # SHORT
                        if future['l'] <= tp: 
                            outcome = "WIN"; exit_price = tp * (1.0 + slippage); break
                        if trailing_active and future['h'] >= current_sl:
                            outcome = "TRAILED"; exit_price = current_sl * (1.0 + slippage); break
                        if not trailing_active and future['h'] >= sl:
                            outcome = "LOSS"; exit_price = sl * (1.0 + slippage); break
                            
                        if future['l'] <= entry_price - (risk_distance * 1.5):
                            trailing_active = True
                            current_sl = min(current_sl, entry_price - (risk_distance * 0.5))

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
    trailed = sum(1 for t in trades if t['outcome'] == 'TRAILED')
    timeouts = sum(1 for t in trades if t['outcome'] == 'TIMEOUT')
    
    win_rate = ((wins + trailed) / max(1, wins+losses+trailed)) * 100
    
    print(f"\n===== PROPRIETARY VOLUME-FILTERED 1H SOL BACKTEST =====")
    print(f"Total Trades Executed: {len(trades)} | Wins: {wins} | Trailed: {trailed} | Losses: {losses} | Timeouts: {timeouts}")
    print(f"Win Rate (excl Timeouts): {win_rate:.1f}%")
    print(f"Macro Filter Blocked: {macro_blocked} | Low-Volume Traps Blocked: {volume_blocked}")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:    ${balance:,.2f}")
    print(f"Total Return:     {((balance/initial_balance)-1)*100:+.2f}%")
    print(f"=======================================================")
    
    # --- ADDED TRADE LOG ---
    print(f"\n===== TRADE LOG =====")
    for t in trades:
        time_str = t['time'].strftime('%Y-%m-%d %H:%M')
        print(f"{time_str} | {t['side']:<5} | {t['outcome']:<7} | PnL: ${t['net_pnl']:>7.2f}")
    print(f"=====================")

if __name__ == "__main__":
    run_proprietary_sol_backtest()
