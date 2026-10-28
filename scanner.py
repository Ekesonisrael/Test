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
    
    # Filter future glitches using modern pandas timestamp syntax
    today = pd.Timestamp.now('UTC').tz_localize(None)
    df = df[df['ts'] <= today].reset_index(drop=True)
    return df

# ==========================================
# 2. IMPROVED SMC STRATEGY + REALISTIC EXECUTION
# ==========================================
def run_backtest(inst, initial_balance=10000.0, risk_pct=0.01, leverage=3):
    df = fetch_okx_data(inst)
    if len(df) < 100: return
    
    balance = initial_balance
    trades = []
    
    i = 15
    while i < len(df) - 55: # Leave room for lookahead
        sweep_candle = df.iloc[i]
        confirm_candle = df.iloc[i+1]
        
        # Find lowest low of prior 10 candles (Liquidity pool)
        prior_low = df['l'].iloc[i-10:i].min()
        
        # 1. LIQUIDITY SWEEP: Wick below prior low, but closes back inside/above
        is_sweep = sweep_candle['l'] < prior_low and sweep_candle['c'] > prior_low
        
        # 2. IMPROVED DISPLACEMENT: Bullish close higher than previous candle's body/open (looser, more realistic)
        is_displacement = confirm_candle['c'] > confirm_candle['o'] and confirm_candle['c'] > sweep_candle['o']
        
        if is_sweep and is_displacement:
            # REALISTIC ENTRY: Fill at the OPEN of the candle following confirmation (i+2)
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
            
            # ISOLATED POSITION SIZING & LEVERAGE CHECK
            risk_amount = balance * risk_pct
            position_size = risk_amount / risk_distance
            notional_value = position_size * entry_price
            max_allowed_notional = balance * leverage
            
            # Cap position size if it exceeds isolated leverage limit
            if notional_value > max_allowed_notional:
                position_size = max_allowed_notional / entry_price
            
            # 3. FORWARD TESTING (Evaluating Outcome)
            outcome = "TIMEOUT"
            exit_price = None
            exit_time = None
            
            # Look ahead up to 50 candles
            end_lookahead = min(i + 52, len(df))
            for j in range(i+2, end_lookahead):
                future_candle = df.iloc[j]
                
                # Check if price hit SL or TP during this candle
                hit_sl = future_candle['l'] <= sl
                hit_tp = future_candle['h'] >= tp
                
                if hit_sl and hit_tp:
                    # Conservative tie-breaker: assume SL hit first if both breached in same 15m candle
                    outcome = "LOSS"
                    exit_price = sl
                    exit_time = future_candle['ts']
                    break
                elif hit_sl:
                    outcome = "LOSS"
                    exit_price = sl
                    exit_time = future_candle['ts']
                    break
                elif hit_tp:
                    outcome = "WIN"
                    exit_price = tp
                    exit_time = future_candle['ts']
                    break
            
            # If neither hit within 50 candles, close out at the final candle's close price
            if outcome == "TIMEOUT":
                exit_price = df.iloc[end_lookahead-1]['c']
                exit_time = df.iloc[end_lookahead-1]['ts']
                outcome = "WIN" if exit_price > entry_price else "LOSS"

            # Calculate PnL & update balance
            pnl = (exit_price - entry_price) * position_size
            balance += pnl
            
            trades.append({
                "time": df.iloc[i+2]['ts'],
                "outcome": outcome,
                "entry": entry_price,
                "exit": exit_price,
                "sl": sl,
                "tp": tp,
                "pnl": pnl,
                "balance": balance
            })
            
            # Jump index past this trade's duration to prevent overlapping executions
            i = j
        else:
            i += 1

    # --- PRINT RESULTS ---
    if not trades:
        print(f"\n[{inst}] No valid SMC setups found in this period.")
        return

    wins = sum(1 for t in trades if t['outcome'] == 'WIN')
    losses = sum(1 for t in trades if t['outcome'] == 'LOSS')
    win_rate = (wins / len(trades)) * 100
    
    print(f"\n================ {inst} SMC BACKTEST (FIXED) ================")
    print(f"Period: {df['ts'].iloc[0].date()} to {df['ts'].iloc[-1].date()}")
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
