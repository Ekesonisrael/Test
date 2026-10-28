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
    
    # FIX: Hardcode end date to avoid OKX API glitch
    df = df[df.index <= pd.Timestamp('2024-12-31')]
    
    df = df.sort_index()[["c"]].astype(float)
    df.columns = ["close"]
    return df[~df.index.duplicated()]

def analyze_futures(inst):
    print(f"\nFetching data for {inst}...")
    df = fetch_okx(inst)
    df['sma'] = df['close'].rolling(200).mean()
    df = df.dropna() 
    
    if len(df) == 0:
        print(f"Not enough data for {inst}")
        return

    df['ret_bh'] = df['close'].pct_change()
    
    # FUTURES LOGIC:
    # If Close > SMA, Signal = 1 (Long)
    # If Close < SMA, Signal = -1 (Short)
    df['signal'] = np.where(df['close'].shift(1) > df['sma'].shift(1), 1, -1)
    
    # Apply 3x Leverage
    LEVERAGE = 3.0
    
    # Calculate raw strategy return (Long or Short)
    df['ret_strat'] = df['ret_bh'] * df['signal'] * LEVERAGE
    
    # Apply a simplified funding cost (approx 0.01% per day deducted)
    df['ret_strat'] = df['ret_strat'] - 0.0001 

    # Group by year
    df['year'] = df.index.year
    yearly = df.groupby('year').agg(
        bh_return=('ret_bh', lambda x: (1 + x).prod() - 1),
        strat_return=('ret_strat', lambda x: (1 + x).prod() - 1),
        long_days=('signal', lambda x: (x == 1).sum()),
        short_days=('signal', lambda x: (x == -1).sum())
    )
    
    print(f"\n================ {inst} FUTURES (3x Leverage) ================")
    print(f"{'Year':<6} | {'Buy & Hold':<12} | {'Futures Strategy':<18} | {'Long/Short Days'}")
    print("-" * 65)
    
    for year, row in yearly.iterrows():
        bh = f"{row['bh_return']*100:+.1f}%"
        st = f"{row['strat_return']*100:+.1f}%"
        ls = f"{int(row['long_days'])}L / {int(row['short_days'])}S"
        print(f"{year:<6} | {bh:<12} | {st:<18} | {ls}")
    print("==============================================================")

if __name__ == "__main__":
    analyze_futures("BTC-USDT-SWAP")
    analyze_futures("SOL-USDT-SWAP")
