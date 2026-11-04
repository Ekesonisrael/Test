import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# CONFIGURATION (Mirroring Claude's Live Code)
# ==========================================
SYMBOLS = {
    "SOL-USD": "Solana"
}
RISK_REWARD = 2.0
ATR_MULTIPLIER = 2.5         
VOLUME_MULTIPLIER = 1.5
LOOKBACK = 20
FEE_RATE = 0.0010            # 0.1% fee

# ✅ SIMULATING CLAUDE'S RISK CAPS
# 3% risk per trade. Max open risk is 9%. Therefore, max 3 concurrent trades.
MAX_CONCURRENT_TRADES = 3    

def run_backtest(ticker, name, split=False):
    print(f"\nFetching 60d of 15m data for {name} ({ticker})...")
    df = yf.download(ticker, period="60d", interval="15m", progress=False)
    
    if df.empty: return
    
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df.reset_index(inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df.set_index('Datetime', inplace=True)

    # 4H Trend
    df_4h = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    df_4h['ema20'] = df_4h['Close'].ewm(span=20, adjust=False).mean()
    df_4h['ema50'] = df_4h['Close'].ewm(span=50, adjust=False).mean()
    df_4h['trend'] = np.where(df_4h['ema20'] > df_4h['ema50'], 1, -1)
    
    df.reset_index(inplace=True)

    # ATR and Vol MA
    df['high_low'] = df['High'] - df['Low']
    df['high_close'] = (df['High'] - df['Close'].shift()).abs()
    df['low_close'] = (df['Low'] - df['Close'].shift()).abs()
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    df['Volume'] = df['Volume'].replace(0, 1).fillna(1)
    df['vol_ma'] = df['Volume'].rolling(window=20).mean()

    df['recent_low'] = df['Low'].rolling(window=LOOKBACK).min().shift(1)
    df['recent_high'] = df['High'].rolling(window=LOOKBACK).max().shift(1)

    df['4h_time'] = df['Datetime'].dt.floor('4h')
    df = pd.merge_asof(df, df_4h[['trend']], left_on='4h_time', right_index=True, direction='backward')

    if split:
        mid_point = len(df) // 2
        df = df.iloc[mid_point:].reset_index(drop=True) 
        print(f"--- {name} (OUT-OF-SAMPLE: Last 30 Days) ---")
    else:
        print(f"--- {name} (IN-SAMPLE: Full 60 Days) ---")

    trades = []
    active_trades = []
    attempted_sweeps = set()
    
    i = 50 

    while i < len(df) - 5:
        candle = df.iloc[i]
        
        # 1. CHECK ACTIVE TRADES FOR EXITS
        for trade in active_trades[:]: 
            exit_triggered = False
            result = None
            
            if trade['side'] == 'LONG':
                if candle['Low'] <= trade['sl']:
                    result = 'LOSS'
                    exit_triggered = True
                elif candle['High'] >= trade['tp']:
                    result = 'WIN'
                    exit_triggered = True
                    
            elif trade['side'] == 'SHORT':
                if candle['High'] >= trade['sl']:
                    result = 'LOSS'
                    exit_triggered = True
                elif candle['Low'] <= trade['tp']:
                    result = 'WIN'
                    exit_triggered = True
                    
            if exit_triggered:
                net_r = (RISK_REWARD if result == 'WIN' else -1.0) - trade['cost_r']
                trades.append({'result': result, 'r': net_r})
                active_trades.remove(trade)

        # 2. CHECK FOR NEW SIGNALS (Using closed candles: i-2 sweep, i-1 confirm, i entry)
        sweep_row = df.iloc[i-2]
        conf_row = df.iloc[i-1]
        
        sweep_key = f"{ticker}|{sweep_row['Datetime']}"
        
        if sweep_key not in attempted_sweeps:
            vol_ok = sweep_row['Volume'] > (sweep_row['vol_ma'] * VOLUME_MULTIPLIER)
            swept_low = (sweep_row['Low'] < sweep_row['recent_low']) and (sweep_row['Close'] > sweep_row['recent_low']) and vol_ok
            swept_high = (sweep_row['High'] > sweep_row['recent_high']) and (sweep_row['Close'] < sweep_row['recent_high']) and vol_ok

            conf_long = conf_row['Close'] > conf_row['Open']
            conf_short = conf_row['Close'] < conf_row['Open']

            side = None
            if sweep_row['trend'] == 1 and swept_low and conf_long:
                side = 'LONG'
            elif sweep_row['trend'] == -1 and swept_high and conf_short:
                side = 'SHORT'

            # Check if we hit our max concurrency limit (simulating the 9% risk cap)
            if side and len(active_trades) < MAX_CONCURRENT_TRADES:
                attempted_sweeps.add(sweep_key)
                entry = candle['Open']
                atr = sweep_row['atr']
                
                if side == 'LONG':
                    sl = entry - (atr * ATR_MULTIPLIER)
                    risk = entry - sl
                    tp = entry + (risk * RISK_REWARD)
                else:
                    sl = entry + (atr * ATR_MULTIPLIER)
                    risk = sl - entry
                    tp = entry - (risk * RISK_REWARD)
                
                cost_r = FEE_RATE / (risk / entry) 
                
                active_trades.append({
                    'side': side, 'entry': entry, 'sl': sl, 'tp': tp, 'cost_r': cost_r
                })
                
        i += 1

    # If we still have open trades at the end of the data, close them at the last price
    for trade in active_trades:
        trades.append({'result': 'OPEN', 'r': 0.0})

    if not trades:
        print("No trades found.")
        return

    total = len(trades)
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    total_r = sum(t['r'] for t in trades)
    
    cumulative_r = np.cumsum([t['r'] for t in trades])
    running_max = np.maximum.accumulate(cumulative_r)
    drawdown = cumulative_r - running_max
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0

    print(f"Total Trades: {total} | Wins: {wins} | Win Rate: {(wins/total)*100:.1f}%")
    print(f"Total R: {total_r:+.2f}R | Max Drawdown: {max_dd:.2f}R")

if __name__ == "__main__":
    for ticker, name in SYMBOLS.items():
        run_backtest(ticker, name, split=False)
        run_backtest(ticker, name, split=True)
