import ccxt
import pandas as pd
import numpy as np
import time

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = "SOL/USDT:USDT"
TIMEFRAME = "15m"
MONTHS_BACK = 6

RISK_REWARD = 2.0
ATR_MULTIPLIER = 2.5         
VOLUME_MULTIPLIER = 2.5      # ✅ STRICTER: 2.5x volume required
LOOKBACK = 20
FEE_RATE = 0.0010            # 0.1% fee
MAX_CONCURRENT_TRADES = 3    # Simulating the 9% max open risk cap (3% x 3)

# ✅ NEW STOP LOGIC
USE_WICK_STOP = True         # Put stop loss beyond the sweep wick
WICK_BUFFER_ATR = 0.1        # Add a small buffer beyond the wick

exchange = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_okx_data(symbol, timeframe, months):
    print(f"Fetching {months} months of {timeframe} data for {symbol} from OKX...")
    end_time = exchange.milliseconds()
    start_time = end_time - int(months * 30 * 24 * 60 * 60 * 1000)
    
    all_ohlcv = []
    since = start_time
    
    while since < end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=100)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Fetch error: {e}")
            time.sleep(2)
            
    df = pd.DataFrame(all_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    print(f"Fetched {len(df)} candles.")
    return df

def run_backtest(df, split_date, name):
    print(f"\n--- {name} ---")
    df.set_index('time', inplace=True)
    
    # 4H Trend (Dropping the unclosed 4H candle)
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    df_4h = df_4h.iloc[:-1] 
    df_4h['ema20'] = df_4h['close'].ewm(span=20, adjust=False).mean()
    df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
    df_4h['trend'] = np.where(df_4h['ema20'] > df_4h['ema50'], 1, -1)
    
    df.reset_index(inplace=True)
    
    # ATR and Vol MA
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = (df['high'] - df['close'].shift()).abs()
    df['low_close'] = (df['low'] - df['close'].shift()).abs()
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    
    df['recent_low'] = df['low'].rolling(window=LOOKBACK).min().shift(1)
    df['recent_high'] = df['high'].rolling(window=LOOKBACK).max().shift(1)

    df['4h_time'] = df['time'].dt.floor('4h')
    df = pd.merge_asof(df, df_4h[['trend']], left_on='4h_time', right_index=True, direction='backward')
    
    # True Out-of-Sample Split
    if name == "TRAIN (First 3 Months)":
        df = df[df['time'] < split_date].reset_index(drop=True)
    else:
        df = df[df['time'] >= split_date].reset_index(drop=True)

    trades = []
    active_trades = []
    attempted_sweeps = set()
    i = 50 

    while i < len(df) - 5:
        candle = df.iloc[i]
        
        # 1. CHECK EXITS (Conservative: If SL and TP hit in same candle, assume LOSS)
        for trade in active_trades[:]: 
            exit_triggered = False
            result = None
            
            if trade['side'] == 'LONG':
                if candle['low'] <= trade['sl']:
                    result, exit_triggered = 'LOSS', True
                elif candle['high'] >= trade['tp']:
                    result, exit_triggered = 'WIN', True
                    
            elif trade['side'] == 'SHORT':
                if candle['high'] >= trade['sl']:
                    result, exit_triggered = 'LOSS', True
                elif candle['low'] <= trade['tp']:
                    result, exit_triggered = 'WIN', True
                    
            if exit_triggered:
                net_r = (RISK_REWARD if result == 'WIN' else -1.0) - trade['cost_r']
                trades.append({'result': result, 'r': net_r})
                active_trades.remove(trade)

        # 2. CHECK SIGNALS (Using closed candles: i-2 sweep, i-1 confirm, i entry)
        sweep_row = df.iloc[i-2]
        conf_row = df.iloc[i-1]
        sweep_key = str(sweep_row['time'])
        
        if sweep_key not in attempted_sweeps:
            vol_ok = sweep_row['volume'] > (sweep_row['vol_ma'] * VOLUME_MULTIPLIER)
            swept_low = (sweep_row['low'] < sweep_row['recent_low']) and (sweep_row['close'] > sweep_row['recent_low']) and vol_ok
            swept_high = (sweep_row['high'] > sweep_row['recent_high']) and (sweep_row['close'] < sweep_row['recent_high']) and vol_ok

            conf_long = conf_row['close'] > conf_row['open']
            conf_short = conf_row['close'] < conf_row['open']

            side = None
            if sweep_row['trend'] == 1 and swept_low and conf_long:
                side = 'LONG'
            elif sweep_row['trend'] == -1 and swept_high and conf_short:
                side = 'SHORT'

            if side and len(active_trades) < MAX_CONCURRENT_TRADES:
                attempted_sweeps.add(sweep_key)
                entry = candle['open']
                atr = sweep_row['atr']
                
                # ✅ NEW STOP LOGIC (Wick Stop)
                if side == 'LONG':
                    sl = entry - (atr * ATR_MULTIPLIER)
                    if USE_WICK_STOP:
                        sl = min(sl, sweep_row['low'] - (atr * WICK_BUFFER_ATR))
                    risk = entry - sl
                    tp = entry + (risk * RISK_REWARD)
                else:
                    sl = entry + (atr * ATR_MULTIPLIER)
                    if USE_WICK_STOP:
                        sl = max(sl, sweep_row['high'] + (atr * WICK_BUFFER_ATR))
                    risk = sl - entry
                    tp = entry - (risk * RISK_REWARD)
                
                cost_r = FEE_RATE / (risk / entry) 
                active_trades.append({'side': side, 'entry': entry, 'sl': sl, 'tp': tp, 'cost_r': cost_r})
                
        i += 1

    total = len(trades)
    if total == 0:
        print("No trades found in this period.")
        return
        
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    total_r = sum(t['r'] for t in trades)
    cumulative_r = np.cumsum([t['r'] for t in trades])
    running_max = np.maximum.accumulate(cumulative_r)
    max_dd = np.min(cumulative_r - running_max) if len(cumulative_r) > 0 else 0

    print(f"Trades: {total} | Win Rate: {(wins/total)*100:.1f}%")
    print(f"Total R: {total_r:+.2f}R | Max Drawdown: {max_dd:.2f}R")

if __name__ == "__main__":
    df = fetch_okx_data(SYMBOL, TIMEFRAME, MONTHS_BACK)
    if not df.empty:
        # Split exactly in the middle (3 months train, 3 months test)
        split_date = df['time'].iloc[len(df) // 2]
        run_backtest(df, split_date, "TRAIN (First 3 Months)")
        run_backtest(df, split_date, "OUT-OF-SAMPLE (Last 3 Months)")
