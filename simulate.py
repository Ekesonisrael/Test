#!/usr/bin/env python3
import pandas as pd
import requests
import numpy as np

def fetch_okx(inst):
    url = "https://www.okx.com/api/v5/market/history-candles"
    rows, after = [], None
    while True:
        p = {"instId": inst, "bar": "1Dutc", "limit": "100"}
        if after: p["after"] = after
        r = requests.get(url, params=p, timeout=20)
        j = r.json()
        if not j["data"]: break
        rows += j["data"]
        after = j["data"][-1][0]
    
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v", "vc", "vq", "confirm"])
    df = df[df.confirm == "1"]
    df.index = pd.to_datetime(df.ts.astype("int64"), unit="ms").dt.normalize()
    
    # FIX: Drop future dates
    today_naive = pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()
    df = df[df.index <= today_naive]
    
    df = df.sort_index()[["c"]].astype(float)
    df.columns = ["close"]
    return df[~df.index.duplicated()]

def analyze(inst):
    df = fetch_okx(inst)
    df['sma'] = df['close'].rolling(200).mean()
    df = df.dropna() # Drop the first 200 days where SMA isn't calculated
    
    if len(df) == 0:
        print(f"Not enough data for {inst}")
        return

    # Calculate daily returns for Buy & Hold vs Strategy
    df['ret_bh'] = df['close'].pct_change()
    
    # Strategy: If yesterday's close > SMA, we hold today. Else, we sit in cash (0% return)
    df['signal'] = (df['close'].shift(1) > df['sma'].shift(1)).astype(int)
    df['ret_strat'] = df['ret_bh'] * df['signal']

    # Group by year
    df['year'] = df.index.year
    yearly = df.groupby('year').agg(
        bh_return=('ret_bh', lambda x: (1 + x).prod() - 1),
        strat_return=('ret_strat', lambda x: (1 + x).prod() - 1),
        days_in_cash=('signal', lambda x: (x == 0).sum())
    )
    
    print(f"\n================ {inst} (200-SMA) ================")
    print(f"{'Year':<6} | {'Buy & Hold':<12} | {'Strategy':<12} | {'Days in Cash'}")
    print("-" * 52)
    
    for year, row in yearly.iterrows():
        bh = f"{row['bh_return']*100:+.1f}%"
        st = f"{row['strat_return']*100:+.1f}%"
        cash = int(row['days_in_cash'])
        print(f"{year:<6} | {bh:<12} | {st:<12} | {cash}")

if __name__ == "__main__":
    analyze("BTC-USDT-SWAP")
    analyze("SOL-USDT-SWAP") 
