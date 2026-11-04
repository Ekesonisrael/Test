import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# CONFIGURATION (Mirroring Claude's Live Code)
# ==========================================
SYMBOLS = {
    "BTC-USD": "Bitcoin",
    "SOL-USD": "Solana"
}
RISK_REWARD = 2.0
ATR_MULTIPLIER = 2.5
VOLUME_MULTIPLIER = 1.5
LOOKBACK = 20
FEE_RATE = 0.0010 # 0.1% round-trip fee

# Claude's Stop Logic
USE_WICK_STOP = False 
WICK_BUFFER_ATR = 0.1

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
    
    # Yahoo volume fix: if volume is 0, set to 1 so filter doesn't permanently block
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
    attempted_sweeps = set()
    active_trade = False
    i = 50 

    while i < len(df) - 5:
        if active_trade:
            i += 1
            continue

        # ✅ CLAUDE'S FIX: Evaluate closed candles only.
        # i-2 is the sweep, i-1 is the confirmation. We enter at i.
        sweep_row = df.iloc[i-2]
        conf_row = df.iloc[i-1]
        entry_row = df.iloc[i]

        # Guard: Deduplication
        sweep_key = f"{ticker}|{sweep_row['Datetime']}"
        if sweep_key in attempted_sweeps:
            i += 1
            continue

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

        if side:
            attempted_sweeps.add(sweep_key)
            entry = entry_row['Open']
            atr = sweep_row['atr']
            
            # ✅ CLAUDE'S FIX: Stop loss goes beyond the sweep wick (optional)
            if side == 'LONG':
                sl = entry - (atr * ATR_MULTIPLIER)
                if USE_WICK_STOP:
                    sl = min(sl, float(sweep_row['Low']) - atr * WICK_BUFFER_ATR)
                risk = entry - sl
                tp = entry + (risk * RISK_REWARD)
            else:
                sl = entry + (atr * ATR_MULTIPLIER)
                if USE_WICK_STOP:
                    sl = max(sl, float(sweep_row['High']) + atr * WICK_BUFFER_ATR)
                risk = sl - entry
                tp = entry - (risk * RISK_REWARD)
            
            cost_r = FEE_RATE / (risk / entry) 

            # Look forward for exit
            exit_idx = None
            result = None
            for j in range(i+1, len(df)):
                candle = df.iloc[j]
                
                if side == 'LONG':
                    if candle['Low'] <= sl:
                        result = 'LOSS'
                        exit_idx = j; break
                    elif candle['High'] >= tp:
                        result = 'WIN'
                        exit_idx = j; break
                        
                elif side == 'SHORT':
                    if candle['High'] >= sl:
                        result = 'LOSS'
                        exit_idx = j; break
                    elif candle['Low'] <= tp:
                        result = 'WIN'
                        exit_idx = j; break

            if exit_idx:
                net_r = (RISK_REWARD if result == 'WIN' else -1.0) - cost_r
                trades.append({'result': result, 'r': net_r})
                active_trade = False
                # Move i to the exit candle to simulate MAX_TOTAL_OPEN = 1
                i = exit_idx + 1
            else:
                # Trade never hit TP or SL before data ended
                active_trade = True
                i += 1
        else:
            i += 1

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

    print(f"Trades: {total} | Wins: {wins} | Win Rate: {(wins/total)*100:.1f}%")
    print(f"Total R: {total_r:+.2f}R | Max Drawdown: {max_dd:.2f}R")

if __name__ == "__main__":
    for ticker, name in SYMBOLS.items():
        run_backtest(ticker, name, split=False)
        run_backtest(ticker, name, split=True)
