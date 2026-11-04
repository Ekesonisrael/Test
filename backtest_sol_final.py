import time

import ccxt
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
TIMEFRAMES = ["1h", "4h"]
MONTHS_BACK = 12

# Exits (R-based, so account risk % does not change the R results)
ATR_MULTIPLIER = 2.0         # initial stop distance
TRAIL_ATR_MULTIPLIER = 3.0   # chandelier trail distance

# Costs
FEE_PER_SIDE = 0.0005        # 0.05% taker per side -> 0.10% round trip
SLIPPAGE_PER_SIDE = 0.0004   # 0.04% adverse slippage per side (assumption)
FUNDING_PER_8H = 0.0001      # 0.01% per 8h, charged as a cost both ways (conservative)

# Indicators
ADX_PERIOD = 14
ADX_THRESHOLD = 20
ATR_PERIOD = 14
EMA_PERIOD = 200
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_MULT = 1.5

WARMUP_BARS = 300            # EMA200 must be settled before the first trade
MIN_TRADES = 30              # below this, results are noise

TF_HOURS = {"15m": 0.25, "30m": 0.5, "1h": 1, "2h": 2, "4h": 4, "1d": 24}

exchange = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})


# ==========================================
# DATA
# ==========================================
def fetch_okx_data(symbol, timeframe, months):
    print(f"Fetching {months}m of {timeframe} for {symbol}...", flush=True)
    end_time = exchange.milliseconds()
    since = end_time - int(months * 30 * 24 * 60 * 60 * 1000)
    rows, fails = [], 0

    while since < end_time:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=100)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + 1
            fails = 0
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Fetch error: {e}", flush=True)
            fails += 1
            if fails > 5:
                break
            time.sleep(2)

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def check_integrity(df, timeframe, months):
    """If this fails, every number downstream is garbage."""
    step = pd.Timedelta(hours=TF_HOURS[timeframe])
    gaps = int((df["time"].diff() > step).sum())
    expected = int(months * 30 * 24 / TF_HOURS[timeframe])
    coverage = len(df) / expected
    print(f"Bars: {len(df)} (expected ~{expected}, coverage {coverage:.0%}) | gaps: {gaps}", flush=True)
    if gaps > 0 or coverage < 0.95:
        print("WARNING: data has gaps or is incomplete. Do not trust results.", flush=True)
        return False
    return True


# ==========================================
# INDICATORS (all causal, computed once on the full series)
# ==========================================
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

    df["atr"] = df["tr"].ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()  # Wilder ATR
    df["ema200"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()
    df["don_high"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["don_low"] = df["low"].rolling(DONCHIAN_PERIOD).min().shift(1)
    return df


# ==========================================
# BACKTEST
# ==========================================
def backtest(df, start, end, tf_hours):
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
        held_hours = (i - t["entry_idx"] + 1) * tf_hours
        funding = FUNDING_PER_8H * held_hours / 8
        net_r = (pnl - 2 * FEE_PER_SIDE - funding) / t["risk_pct"]
        trades.append({"side": t["side"], "r": net_r, "reason": reason})

    def manage(t, i):
        """Stop is checked against the level known BEFORE this candle.
        The trail is updated AFTER the candle closes, so it only applies from i+1."""
        if t["side"] == "LONG":
            if l[i] <= t["sl"]:
                close_trade(t, i, min(o[i], t["sl"]), "STOP")  # gap-aware
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

    for i in range(max(start, 1), end):
        had_trade = tr is not None
        if had_trade and manage(tr, i):
            tr = None

        # No same-candle re-entry after an exit: entry is at the open, the exit came later.
        if had_trade:
            continue

        s = i - 1  # signal candle, fully closed
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

        # The entry candle gets its own stop check immediately.
        if tr is not None and manage(tr, i):
            tr = None

    if tr is not None:  # don't silently drop open trades
        close_trade(tr, end - 1, c[end - 1], "EOD")

    return trades


# ==========================================
# REPORTING
# ==========================================
def report(trades, label):
    print(f"\n  [{label}]", flush=True)
    n = len(trades)
    if n == 0:
        print("  No trades.", flush=True)
        return
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

    print(f"  Trades: {n} | Win rate: {len(wins) / n * 100:.1f}%", flush=True)
    print(f"  Total R: {r.sum():+.2f} | Expectancy: {r.mean():+.3f}R/trade | PF: {pf:.2f}", flush=True)
    print(f"  Avg win: {wins.mean() if len(wins) else 0:.2f}R | Avg loss: {losses.mean() if len(losses) else 0:.2f}R", flush=True)
    print(f"  Max DD: {max_dd:.2f}R | Longest losing streak: {longest}", flush=True)
    if n < MIN_TRADES:
        print(f"  !! Only {n} trades (<{MIN_TRADES}). Statistically meaningless.", flush=True)


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("Starting Monster Backtest v2...", flush=True)
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"\n{'=' * 50}\n{symbol} | {tf}\n{'=' * 50}", flush=True)
            try:
                raw = fetch_okx_data(symbol, tf, MONTHS_BACK)
                if raw.empty or not check_integrity(raw, tf, MONTHS_BACK):
                    print("Skipping: bad data.", flush=True)
                    continue
                df = add_indicators(raw)
                mid = len(df) // 2
                tfh = TF_HOURS[tf]
                report(backtest(df, WARMUP_BARS, mid, tfh), "TRAIN (first half, after warmup)")
                report(backtest(df, mid, len(df), tfh), "OUT-OF-SAMPLE (second half)")
            except Exception as e:
                print(f"Failed on {symbol} {tf}: {e}", flush=True)
