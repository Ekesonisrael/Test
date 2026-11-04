import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime

# ==========================================
# PRE-REGISTERED PASS CRITERIA (Don't change these after seeing results)
# ==========================================
MIN_SHARPE = 1.0
MIN_PROFIT_FACTOR = 1.2
MIN_EXPECTANCY = 0.0  # Positive return

# ==========================================
# CONFIGURATION
# ==========================================
# A wider basket, acknowledging survivorship bias (we can only test what OKX still lists)
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOGE/USDT:USDT",
    "DOT/USDT:USDT", "LINK/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT", 
    "UNI/USDT:USDT", "ATOM/USDT:USDT", "ETC/USDT:USDT", "FIL/USDT:USDT",
    "APT/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT"
]

TIMEFRAME = "4h"
MONTHS_BACK = 60             # 5 years (Covers 2022 Bear Market)
FEE_PER_SIDE = 0.0005        # 0.05%
SLIPPAGE_PER_SIDE = 0.0004   # 0.04%
FUNDING_PER_8H = 0.0001      # 0.01% per 8 hours
INITIAL_CAPITAL = 10000      

exchange = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def fetch_okx_data(symbol, timeframe, months):
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

    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

def run_simulation(df_open, df_close, lookback_bars, top_n):
    # Calculate returns bar-by-bar
    df_ret = df_open.pct_change()
    
    # Calculate historical volatility (for inverse weighting)
    df_vol = df_ret.rolling(window=lookback_bars).std()
    
    # Rebalance every 7 days (42 bars)
    rebalance_bars = 7 * 6 
    
    equity = INITIAL_CAPITAL
    equity_curve = []
    
    # We start after the lookback period
    for i in range(lookback_bars, len(df_close) - rebalance_bars, rebalance_bars):
        
        # Get current momentum scores and vol
        # Drop NaNs per-row to avoid dropping the whole dataset
        mom_scores = df_close.pct_change(lookback_bars).iloc[i].dropna()
        vols = df_vol.iloc[i].dropna()
        
        # Only trade coins that have enough data
        valid_coins = mom_scores.index.intersection(vols.index)
        if len(valid_coins) < top_n * 2:
            continue
            
        mom_scores = mom_scores[valid_coins]
        vols = vols[valid_coins]
        
        longs = mom_scores.nlargest(top_n).index.tolist()
        shorts = mom_scores.nsmallest(top_n).index.tolist()
        
        # Inverse Volatility Weighting
        # Weight = (1/Vol) / sum(1/Vol)
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
        
        long_weights = weights[longs]
        short_weights = weights[shorts]
        
        # Normalize weights so total gross exposure is 100% (50% long, 50% short)
        long_weights = long_weights / long_weights.sum() * 0.5
        short_weights = short_weights / short_weights.sum() * 0.5
        
        # Track daily equity within the holding period
        for j in range(i + 1, i + rebalance_bars + 1):
            if j >= len(df_close): break
            
            # Calculate bar return (using next open to next open to avoid look-ahead)
            bar_ret = df_ret.iloc[j]
            
            # Long PnL
            long_pnl = sum(bar_ret[sym] * long_weights[sym] for sym in longs)
            
            # Short PnL
            short_pnl = sum(-bar_ret[sym] * short_weights[sym] for sym in shorts)
            
            # Funding costs (4h bar = 0.5 * 8h funding)
            # Longs pay funding if positive, shorts receive. We assume positive funding environment for alts.
            funding_cost = FUNDING_PER_8H * 0.5 * 0.5 # 50% exposure, 0.5 of 8h
            net_pnl = long_pnl + short_pnl - funding_cost
            
            # Apply entry costs (spread over the 42 bars)
            trade_cost = (FEE_PER_SIDE * 2 + SLIPPAGE_PER_SIDE * 2) / rebalance_bars
            
            equity = equity * (1 + net_pnl - trade_cost)
            equity_curve.append({'time': df_close.index[j], 'equity': equity})

    return pd.DataFrame(equity_curve).set_index('time')

def calculate_metrics(eq_df):
    if eq_df.empty: return None
    
    # Daily resampling for Sharpe
    daily_eq = eq_df['equity'].resample('D').last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    
    # Sharpe (Annualized)
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() > 0 else 0
    
    # Max Drawdown
    running_max = np.maximum.accumulate(daily_eq)
    drawdowns = (daily_eq - running_max) / running_max
    max_dd = drawdowns.min() * 100
    
    # Profit Factor & Expectancy (based on weekly returns)
    weekly_eq = eq_df['equity'].resample('W').last().dropna()
    weekly_ret = weekly_eq.pct_change().dropna()
    
    wins = weekly_ret[weekly_ret > 0]
    losses = weekly_ret[weekly_ret <= 0]
    
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy = weekly_ret.mean()
    
    total_return = (daily_eq.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    return {
        'total_ret': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'pf': pf,
        'expectancy': expectancy,
        'trades': len(weekly_ret)
    }

def run_backtest():
    print("Fetching data (this will take a few minutes)...", flush=True)
    data_open = {}
    data_close = {}
    
    for sym in SYMBOLS:
        df = fetch_okx_data(sym, TIMEFRAME, MONTHS_BACK)
        if not df.empty:
            df.set_index('time', inplace=True)
            data_open[sym] = df['open']
            data_close[sym] = df['close']
    
    if not data_open:
        print("No data fetched.")
        return

    df_open = pd.DataFrame(data_open)
    df_close = pd.DataFrame(data_close)
    
    # Check date range
    print(f"\nData Loaded. Range: {df_close.index[0]} to {df_close.index[-1]}", flush=True)
    print(f"Total 4H Bars: {len(df_close)}", flush=True)
    print(f"Total Coins: {len(df_close.columns)}", flush=True)

    # SPLIT DATA: Train (First 50%) / Test (Last 50%)
    split_idx = len(df_close) // 2
    split_date = df_close.index[split_idx]
    
    train_open, train_close = df_open.iloc[:split_idx], df_close.iloc[:split_idx]
    test_open, test_close = df_open.iloc[split_idx:], df_close.iloc[split_idx:]

    print(f"\nTrain Period: {train_close.index[0]} to {train_close.index[-1]}")
    print(f"Test Period:  {test_close.index[0]} to {test_close.index[-1]}")

    # SENSITIVITY TABLE
    lookbacks = [180, 360, 540] # 30 days, 60 days, 90 days
    top_ns = [3, 5]

    print("\n" + "="*80)
    print("  PRE-REGISTERED CRITERIA:")
    print(f"  Sharpe > {MIN_SHARPE} | PF > {MIN_PROFIT_FACTOR} | Expectancy > {MIN_EXPECTANCY*100}% (Weekly)")
    print("="*80)

    for lb in lookbacks:
        for top_n in top_ns:
            print(f"\n--- PARAMETERS: Lookback={lb} bars ({lb//6} days), Top N={top_n} ---")
            
            # Train
            train_eq = run_simulation(train_open, train_close, lb, top_n)
            train_metrics = calculate_metrics(train_eq)
            
            if not train_metrics:
                print("  Train: No trades.")
                continue
                
            print(f"  [TRAIN] Ret: {train_metrics['total_ret']:+.1f}% | Sharpe: {train_metrics['sharpe']:.2f} | DD: {train_metrics['max_dd']:.1f}% | PF: {train_metrics['pf']:.2f} | Exp: {train_metrics['expectancy']*100:+.2f}%")

            # Test
            test_eq = run_simulation(test_open, test_close, lb, top_n)
            test_metrics = calculate_metrics(test_eq)
            
            if not test_metrics:
                print("  Test: No trades.")
                continue

            # Check criteria
            passed = (test_metrics['sharpe'] > MIN_SHARPE and 
                      test_metrics['pf'] > MIN_PROFIT_FACTOR and 
                      test_metrics['expectancy'] > MIN_EXPECTANCY)
            
            status = "✅ PASS" if passed else "❌ FAIL"
            
            print(f"  [TEST]  Ret: {test_metrics['total_ret']:+.1f}% | Sharpe: {test_metrics['sharpe']:.2f} | DD: {test_metrics['max_dd']:.1f}% | PF: {test_metrics['pf']:.2f} | Exp: {test_metrics['expectancy']*100:+.2f}%  -> {status}")

if __name__ == "__main__":
    print("Starting V2 Momentum Backtest...", flush=True)
    run_backtest()
