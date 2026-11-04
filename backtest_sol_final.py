import ccxt
import pandas as pd
import numpy as np
import time

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOLS = ["BTC/USDT:USDT", "SOL/USDT:USDT"]
MONTHS_BACK = 12             # 1 Year of data to get enough trades

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

# ==========================================
# BTC STRATEGY (15m SMC SWEEP)
# ==========================================
def backtest_btc_smc(df_15m, df_4h):
    df_15m = df_15m.copy()
    df_4h = df_4h.copy()
    
    # 4H Trend (Lookahead-free)
    df_4h['ema20'] = df_4h['close'].ewm(span=20, adjust=False).mean()
    df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
    df_4h['trend'] = np.where(df_4h['ema20'] > df_4h['ema50'], 1, -1)
    
    # ✅ CRITICAL FIX: Shift trend to use only CLOSED 4H candles
    df_4h['trend'] = df_4h['trend'].shift(1)
    
    # ✅ FIX: Set time as index to avoid column collision
    df_4h.set_index('time', inplace=True)
    
    df_15m['tr'] = pd.concat([df_15m['high'] - df_15m['low'], (df_15m['high'] - df_15m['close'].shift()).abs(), (df_15m['low'] - df_15m['close'].shift()).abs()], axis=1).max(axis=1)
    df_15m['atr'] = df_15m['tr'].rolling(window=14).mean()
    df_15m['vol_ma'] = df_15m['volume'].rolling(window=20).mean()
    df_15m['recent_low'] = df_15m['low'].rolling(window=20).min().shift(1)
    df_15m['recent_high'] = df_15m['high'].rolling(window=20).max().shift(1)
    
    df_15m['4h_time'] = df_15m['time'].dt.floor('4h')
    
    # ✅ FIX: Use right_index=True
    df_15m = pd.merge_asof(df_15m, df_4h[['trend']], left_on='4h_time', right_index=True, direction='backward')
    
    trades = []
    active_trades = []
    daily_trade_counts = {}
    attempted_sweeps = set()
    
    i = 50
    while i < len(df_15m) - 5:
        candle = df_15m.iloc[i]
        
        # Manage exits
        for trade in active_trades[:]:
            if trade['side'] == 'LONG':
                if candle['low'] <= trade['sl']: 
                    trades.append(-1.0); active_trades.remove(trade)
                elif candle['high'] >= trade['tp']: 
                    trades.append(2.0); active_trades.remove(trade)
            else:
                if candle['high'] >= trade['sl']: 
                    trades.append(-1.0); active_trades.remove(trade)
                elif candle['low'] <= trade['tp']: 
                    trades.append(2.0); active_trades.remove(trade)

        # Check signals
        sweep = df_15m.iloc[i-2]
        conf = df_15m.iloc[i-1]
        
        sweep_key = str(sweep['time'])
        if sweep_key not in attempted_sweeps:
            vol_ok = sweep['volume'] > (sweep['vol_ma'] * 1.5)
            swept_low = (sweep['low'] < sweep['recent_low']) and (sweep['close'] > sweep['recent_low']) and vol_ok
            swept_high = (sweep['high'] > sweep['recent_high']) and (sweep['close'] < sweep['recent_high']) and vol_ok
            
            conf_long = conf['close'] > conf['open']
            conf_short = conf['close'] < conf['open']
            
            side = None
            if sweep['trend'] == 1 and swept_low and conf_long: side = 'LONG'
            elif sweep['trend'] == -1 and swept_high and conf_short: side = 'SHORT'
            
            if side:
                day_str = str(sweep['time'].date())
                if daily_trade_counts.get(day_str, 0) < 2:
                    attempted_sweeps.add(sweep_key)
                    daily_trade_counts[day_str] = daily_trade_counts.get(day_str, 0) + 1
                    
                    entry = candle['open']
                    atr = sweep['atr']
                    
                    if side == 'LONG':
                        sl = min(entry - (atr * 2.5), sweep['low'] - (atr * 0.1))
                        tp = entry + ((entry - sl) * 2.0)
                    else:
                        sl = max(entry + (atr * 2.5), sweep['high'] + (atr * 0.1))
                        tp = entry - ((sl - entry) * 2.0)
                        
                    active_trades.append({'side': side, 'entry': entry, 'sl': sl, 'tp': tp})
        i += 1

    total = len(trades)
    if total == 0: return "No trades."
    wins = sum(1 for r in trades if r > 0)
    total_r = sum(trades)
    return f"Trades: {total} | Win Rate: {(wins/total)*100:.1f}% | Total R: {total_r:+.2f}R | Expectancy: {total_r/total:+.3f}R"

# ==========================================
# SOL STRATEGY (4H DONCHIAN TREND)
# ==========================================
def backtest_sol_trend(df_4h):
    df = df_4h.copy()
    
    df['tr'] = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['don_high'] = df['high'].rolling(20).max().shift(1)
    df['don_low'] = df['low'].rolling(20).min().shift(1)
    
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_s = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_s
    mdi = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()

    trades = []
    tr = None
    
    for i in range(300, len(df)):
        candle = df.iloc[i]
        
        if tr:
            if tr['side'] == 'LONG':
                if candle['low'] <= tr['sl']:
                    pnl = (tr['sl'] - tr['entry']) / (tr['entry'] - tr['initial_sl'])
                    trades.append(pnl)
                    tr = None
                else:
                    tr['sl'] = max(tr['sl'], candle['high'] - (3.0 * candle['atr']))
            else:
                if candle['high'] >= tr['sl']:
                    pnl = (tr['entry'] - tr['sl']) / (tr['initial_sl'] - tr['entry'])
                    trades.append(pnl)
                    tr = None
                else:
                    tr['sl'] = min(tr['sl'], candle['low'] + (3.0 * candle['atr']))
                    
        if not tr:
            s = df.iloc[i-1]
            is_trending = s['adx'] > 20
            vol_ok = s['volume'] > s['vol_ma'] * 1.5
            
            if is_trending and vol_ok and s['close'] > s['ema200'] and s['close'] > s['don_high']:
                entry = candle['open']
                sl = entry - (2.0 * s['atr'])
                tr = {'side': 'LONG', 'entry': entry, 'sl': sl, 'initial_sl': sl}
            elif is_trending and vol_ok and s['close'] < s['ema200'] and s['close'] < s['don_low']:
                entry = candle['open']
                sl = entry + (2.0 * s['atr'])
                tr = {'side': 'SHORT', 'entry': entry, 'sl': sl, 'initial_sl': sl}

    total = len(trades)
    if total == 0: return "No trades."
    wins = sum(1 for r in trades if r > 0)
    total_r = sum(trades)
    return f"Trades: {total} | Win Rate: {(wins/total)*100:.1f}% | Total R: {total_r:+.2f}R | Expectancy: {total_r/total:+.3f}R"

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("Starting Master Bot Backtest (1 Year, Lookahead-Fixed)...", flush=True)
    
    for sym in SYMBOLS:
        print(f"\n{'='*60}", flush=True)
        print(f"  TESTING {sym}", flush=True)
        print(f"{'='*60}", flush=True)
        
        df_4h = fetch_okx_data(sym, "4h", MONTHS_BACK)
        if df_4h.empty: continue
        
        if sym == "BTC/USDT:USDT":
            df_15m = fetch_okx_data(sym, "15m", MONTHS_BACK)
            if df_15m.empty: continue
            res = backtest_btc_smc(df_15m, df_4h)
            print(f"BTC 15m SMC (Max 2/Day): {res}", flush=True)
        else:
            res = backtest_sol_trend(df_4h)
            print(f"SOL 4H Trend: {res}", flush=True)
