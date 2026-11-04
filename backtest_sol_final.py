import ccxt
import pandas as pd
import numpy as np
import time

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOLS = ["BTC/USDT:USDT", "SOL/USDT:USDT"]
TIMEFRAME = "15m"
MONTHS_BACK = 3              
RISK_REWARD = 2.0
ATR_MULTIPLIER = 2.5         
VOLUME_MULTIPLIER = 1.5
LOOKBACK = 20
FEE_RATE = 0.0010            
USE_WICK_STOP = True         
WICK_BUFFER_ATR = 0.1        

# ✅ TESTING THE BEST CONFIGURATION
MAX_TRADES_PER_DAY = 2
MAX_TRADES_PER_WEEK = None

exchange = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_okx_data(symbol, timeframe, months):
    print(f"Fetching {months}m of {timeframe} for {symbol}...", flush=True)
    end_time = exchange.milliseconds()
    since = end_time - int(months * 30 * 24 * 60 * 60 * 1000)
    rows, fails = [], 0

    while since < end_time:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=100)
            if not batch: break
            rows.extend(batch)
            since = batch[-1][0] + 1
            fails = 0
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            fails += 1
            if fails > 5: break
            time.sleep(2)

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

def run_backtest(df, btc_df, symbol, label):
    df = df.copy()
    btc_df = btc_df.copy()
    
    # Calculate 4H Trend (Lookahead-free)
    df.set_index('time', inplace=True)
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    df_4h['ema20'] = df_4h['close'].ewm(span=20, adjust=False).mean()
    df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
    df_4h['trend'] = np.where(df_4h['ema20'] > df_4h['ema50'], 1, -1)
    df_4h['trend'] = df_4h['trend'].shift(1) 
    df.reset_index(inplace=True)
    
    # ✅ BTC DOMINANCE PROXY (15m EMA 50)
    btc_df['btc_ema50'] = btc_df['close'].ewm(span=50, adjust=False).mean()
    btc_df['btc_trend'] = np.where(btc_df['close'] > btc_df['btc_ema50'], 1, -1)
    
    # Merge BTC trend into the main dataframe
    df = pd.merge_asof(df, btc_df[['time', 'btc_trend']], on='time', direction='backward')
    
    # ATR & Volume MA
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
    
    trades = []
    active_trades = []
    attempted_sweeps = set()
    daily_trade_counts = {}
    
    i = 50 

    while i < len(df) - 5:
        candle = df.iloc[i]
        
        # MANAGE EXITS
        for trade in active_trades[:]: 
            exit_triggered = False
            result = None
            
            if trade['side'] == 'LONG':
                if candle['low'] <= trade['sl']: result, exit_triggered = 'LOSS', True
                elif candle['high'] >= trade['tp']: result, exit_triggered = 'WIN', True
            elif trade['side'] == 'SHORT':
                if candle['high'] >= trade['sl']: result, exit_triggered = 'LOSS', True
                elif candle['low'] <= trade['tp']: result, exit_triggered = 'WIN', True
                    
            if exit_triggered:
                net_r = (RISK_REWARD if result == 'WIN' else -1.0) - trade['cost_r']
                trades.append({'result': result, 'r': net_r})
                active_trades.remove(trade)

        # CHECK SIGNALS
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
            
            # ✅ APPLY BTC DOMINANCE FILTER
            btc_is_bullish = sweep_row['btc_trend'] == 1
            btc_is_bearish = sweep_row['btc_trend'] == -1
            
            if sweep_row['trend'] == 1 and swept_low and conf_long and btc_is_bullish: 
                side = 'LONG'
            elif sweep_row['trend'] == -1 and swept_high and conf_short and btc_is_bearish: 
                side = 'SHORT'

            if side:
                day_str = str(sweep_row['time'].date())
                d_count = daily_trade_counts.get(day_str, 0)
                
                can_trade = True
                if MAX_TRADES_PER_DAY and d_count >= MAX_TRADES_PER_DAY: can_trade = False
                
                if can_trade:
                    attempted_sweeps.add(sweep_key)
                    daily_trade_counts[day_str] = d_count + 1
                    
                    entry = candle['open']
                    atr = sweep_row['atr']
                    
                    if side == 'LONG':
                        sl = entry - (atr * ATR_MULTIPLIER)
                        if USE_WICK_STOP: sl = min(sl, sweep_row['low'] - (atr * WICK_BUFFER_ATR))
                        risk = entry - sl
                        tp = entry + (risk * RISK_REWARD)
                    else:
                        sl = entry + (atr * ATR_MULTIPLIER)
                        if USE_WICK_STOP: sl = max(sl, sweep_row['high'] + (atr * WICK_BUFFER_ATR))
                        risk = sl - entry
                        tp = entry - (risk * RISK_REWARD)
                    
                    cost_r = FEE_RATE / (risk / entry) 
                    active_trades.append({'side': side, 'entry': entry, 'sl': sl, 'tp': tp, 'cost_r': cost_r})
                
        i += 1

    total = len(trades)
    if total == 0: return f"{symbol} ({label}): No trades found."
        
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    total_r = sum(t['r'] for t in trades)
    win_rate = (wins/total)*100 if total > 0 else 0
    expectancy = total_r / total if total > 0 else 0
    
    return f"{symbol} ({label}): {total} trades | Win Rate: {win_rate:.1f}% | Total R: {total_r:+.2f}R | Expectancy: {expectancy:+.3f}R"

if __name__ == "__main__":
    print("Starting BTC Dominance Filter Backtest...", flush=True)
    
    # Fetch BTC data first for the filter
    btc_raw = fetch_okx_data("BTC/USDT:USDT", TIMEFRAME, MONTHS_BACK)
    
    for sym in SYMBOLS:
        raw = fetch_okx_data(sym, TIMEFRAME, MONTHS_BACK)
        if raw.empty: continue
        
        res = run_backtest(raw, btc_raw, sym, "Max 2/Day + BTC Filter")
        print("\n" + "="*60, flush=True)
        print(res, flush=True)
        print("="*60, flush=True)
