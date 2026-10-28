#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np

def fetch_data(inst, limit=10000):
    print(f"Fetching {limit} candles for {inst}...")
    # We switch to 1H timeframe to avoid 15m noise
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
    for col in ["o", "h", "l", "c"]:
        df[col] = df[col].astype(float)
    
    # HARDCODED DATE LOCK (No more 2026 glitches!)
    df['ts'] = pd.to_datetime(df['ts'].astype("int64"), unit="ms")
    safe_cutoff = pd.Timestamp('2024-05-20') # Hard stop
    df = df[df['ts'] <= safe_cutoff].reset_index(drop=True)
    
    # Indicators
    df['ema_9'] = df['c'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['c'].ewm(span=21, adjust=False).mean()
    df['ema_200'] = df['c'].ewm(span=200, adjust=False).mean()
    
    tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    return df

def run_trend_backtest(symbol, initial_balance=10000.0, risk_pct=0.01, leverage=3):
    print(f"\n=======================================================")
    print(f"STARTING TREND BACKTEST FOR: {symbol}")
    print(f"=======================================================")
    
    df = fetch_data(symbol)
    if df is None or len(df) < 200:
        print("Not enough data.")
        return
    
    balance = initial_balance
    trades = []
    
    slippage = 0.0005     
    
    i = 200 # Start after EMAs are calculated
    while i < len(df) - 100:
        current = df.iloc[i]
        prev = df.iloc[i-1]
        atr = current['atr']
        
        if pd.isna(atr) or atr == 0: 
            i += 1; continue
            
        # CROSSOVER LOGIC
        cross_up = (prev['ema_9'] <= prev['ema_21']) and (current['ema_9'] > current['ema_21'])
        cross_down = (prev['ema_9'] >= prev['ema_21']) and (current['ema_9'] < current['ema_21'])
        
        side = None
        entry_price = current['c'] * (1.0 + slippage) # Buy at close of crossover candle
        
        # LONG ENTRY
        if cross_up and current['c'] > current['ema_200']:
            side = "LONG"
            sl = entry_price - (2.0 * atr) # 2x ATR stop
            
        # SHORT ENTRY
        elif cross_down and current['c'] < current['ema_200']:
            side = "SHORT"
            entry_price = current['c'] * (1.0 - slippage)
            sl = entry_price + (2.0 * atr)
            
        if side:
            risk_distance = abs(entry_price - sl)
            risk_amount = balance * risk_pct
            position_size = risk_amount / risk_distance
            max_notional = balance * leverage
            
            if (position_size * entry_price) > max_notional:
                position_size = max_notional / entry_price

            outcome = "OPEN"
            exit_price = None
            max_forward = min(i + 200, len(df) - 1) # Give trend 200 hours to play out
            
            current_sl = sl
            
            for j in range(i+1, max_forward):
                future = df.iloc[j]
                
                if side == "LONG":
                    # Trailing Stop: Move SL up as price goes up
                    if future['c'] > entry_price:
                        current_sl = max(current_sl, future['c'] - (2.0 * atr))
                        
                    if future['l'] <= current_sl:
                        outcome = "TRAILED/STOP"; exit_price = current_sl * (1.0 - slippage); break
                        
                else: # SHORT
                    if future['c'] < entry_price:
                        current_sl = min(current_sl, future['c'] + (2.0 * atr))
                        
                    if future['h'] >= current_sl:
                        outcome = "TRAILED/STOP"; exit_price = current_sl * (1.0 + slippage); break

            if outcome == "OPEN":
                outcome = "TIMEOUT"
                exit_price = df.iloc[max_forward]['c']

            if side == "LONG":
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
                
            balance += pnl
            trades.append({"time": current['ts'], "side": side, "outcome": outcome, "net_pnl": pnl})
            i = j 
        else:
            i += 1

    if not trades:
        print(f"[{symbol}] No valid setups found.")
        return

    wins = sum(1 for t in trades if t['net_pnl'] > 0)
    losses = sum(1 for t in trades if t['net_pnl'] <= 0)
    win_rate = (wins / len(trades)) * 100
    
    print(f"===== RESULTS FOR {symbol} =====")
    print(f"Total Trades: {len(trades)} | Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Final Balance:    ${balance:,.2f}")
    print(f"Total Return:     {((balance/initial_balance)-1)*100:+.2f}%")
    
    print(f"\n--- LAST 5 TRADES ---")
    for t in trades[-5:]:
        time_str = t['time'].strftime('%Y-%m-%d %H:%M')
        print(f"{time_str} | {t['side']:<5} | {t['outcome']:<12} | PnL: ${t['net_pnl']:>7.2f}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    run_trend_backtest("XRP-USDT-SWAP")
    run_trend_backtest("HYPE-USDT-SWAP")
