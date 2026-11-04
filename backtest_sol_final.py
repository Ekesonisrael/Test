import time
import ccxt
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION (BTC ONLY)
# ==========================================
SYMBOL = "BTC/USDT:USDT"     # ✅ CHANGED TO BTC
TIMEFRAME = "4h"
MONTHS_BACK = 48             # 4 Years 

# Exits (R-based)
ATR_MULTIPLIER = 2.0         
TRAIL_ATR_MULTIPLIER = 3.0   

# Costs
FEE_PER_SIDE = 0.0005        
SLIPPAGE_PER_SIDE = 0.0004   
FUNDING_PER_8H = 0.0001      

# Indicators
ADX_PERIOD = 14
ADX_THRESHOLD = 20
ATR_PERIOD = 14
EMA_PERIOD = 200
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_MULT = 1.5
WARMUP_BARS = 300            

exchange = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})

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
            print(f"Fetch error: {e}", flush=True)
            fails += 1
            if fails > 5: break
            time.sleep(2)

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

def add_indicators(df):
    df = df.copy()
    prev_close = df["close"].shift()
    df["tr"] = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    a = 1 / ADX_PERIOD
    tr_s = df["tr"].ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=a, adjust=False).mean() / tr_s
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=a, adjust=False).mean() / tr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=a, adjust=False).mean()

    df["atr"] = df["tr"].ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()  
    df["ema200"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()
    df["don_high"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["don_low"] = df["low"].rolling(DONCHIAN_PERIOD).min().shift(1)
    return df

def backtest(df):
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    atr, adx, ema = df["atr"].values, df["adx"].values, df["ema200"].values
    vol, vma = df["volume"].values, df["vol_ma"].values
    dhi, dlo = df["don_high"].values, df["don_low"].values

    trades = []
    tr = None

    def close_trade(t, i, raw_exit, reason):
        if t["side"] == "LONG":
            px = raw_exit * (1 - SLIPPAGE_PER_SIDE)
            pnl = (px - t["entry"]) / t["entry"]
        else:
            px = raw_exit * (1 + SLIPPAGE_PER_SIDE)
            pnl = (t["entry"] - px) / t["entry"]
        
        held_hours = (i - t["entry_idx"] + 1) * 4 
        funding = FUNDING_PER_8H * held_hours / 8
        net_r = (pnl - 2 * FEE_PER_SIDE - funding) / t["risk_pct"]
        trades.append({"side": t["side"], "r": net_r, "reason": reason})

    def manage(t, i):
        if t["side"] == "LONG":
            if l[i] <= t["sl"]:
                close_trade(t, i, min(o[i], t["sl"]), "STOP") 
                return True
            t["peak"] = max(t["peak"], h[i])
            t["sl"] = max(t["sl"], t["peak"] - TRAIL_ATR_MULTIPLIER * atr[i])
        else:
            if h[i] >= t["sl"]:
                close_trade(t, i, max(o[i], t["sl"]), "STOP")
                return True
            t["peak"] = min(t["peak"], l[i])
            t["sl"] = min(t["sl"], t["peak"] + TRAIL_ATR_MULTIPLIER * atr[i])
        return False

    for i in range(max(WARMUP_BARS, 1), len(df)):
        had_trade = tr is not None
        if had_trade and manage(tr, i):
            tr = None

        if had_trade:
            continue

        s = i - 1  
        vals = (atr[s], adx[s], ema[s], vma[s], dhi[s], dlo[s])
        if any(np.isnan(v) for v in vals):
            continue

        trending = adx[s] > ADX_THRESHOLD
        vol_ok = vol[s] > vma[s] * VOL_MULT

        if trending and vol_ok and c[s] > ema[s] and c[s] > dhi[s]:
            entry = o[i] * (1 + SLIPPAGE_PER_SIDE)
            sl = entry - ATR_MULTIPLIER * atr[s]
            tr = {"side": "LONG", "entry": entry, "sl": sl, "peak": entry,
                  "risk_pct": (entry - sl) / entry, "entry_idx": i}
        elif trending and vol_ok and c[s] < ema[s] and c[s] < dlo[s]:
            entry = o[i] * (1 - SLIPPAGE_PER_SIDE)
            sl = entry + ATR_MULTIPLIER * atr[s]
            tr = {"side": "SHORT", "entry": entry, "sl": sl, "peak": entry,
                  "risk_pct": (sl - entry) / entry, "entry_idx": i}

        if tr is not None and manage(tr, i):
            tr = None

    if tr is not None:
        close_trade(tr, len(df) - 1, c[-1], "EOD")

    return trades

if __name__ == "__main__":
    print("Starting BTC 4H Backtest...", flush=True)
    raw = fetch_okx_data(SYMBOL, TIMEFRAME, MONTHS_BACK)
    if raw.empty:
        print("No data.")
        exit()
        
    print(f"Data Loaded. Range: {raw['time'].iloc[0]} to {raw['time'].iloc[-1]}")
    print(f"Total Bars: {len(raw)}")
    
    df = add_indicators(raw)
    trades = backtest(df)

    n = len(trades)
    if n == 0:
        print("No trades found.")
    else:
        r = np.array([t["r"] for t in trades])
        wins, losses = r[r > 0], r[r <= 0]
        cum = np.cumsum(r)
        max_dd = float(np.min(cum - np.maximum.accumulate(cum)))

        streak = longest = 0
        for x in r:
            streak = streak + 1 if x <= 0 else 0
            longest = max(longest, streak)

        gross_loss = abs(losses.sum())
        pf = wins.sum() / gross_loss if gross_loss > 0 else float("inf")

        print("\n" + "="*50)
        print("  BTC 4H MONSTER BACKTEST RESULTS (4 YEARS)")
        print("="*50)
        print(f"  Trades: {n} | Win rate: {len(wins) / n * 100:.1f}%")
        print(f"  Total R: {r.sum():+.2f} | Expectancy: {r.mean():+.3f}R/trade | PF: {pf:.2f}")
        print(f"  Avg win: {wins.mean() if len(wins) else 0:.2f}R | Avg loss: {losses.mean() if len(losses) else 0:.2f}R")
        print(f"  Max DD: {max_dd:.2f}R | Longest losing streak: {longest}")
        print("="*50)
