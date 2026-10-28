#!/usr/bin/env python3
"""
Simulates the 200-SMA bot with real money math on historical data.
Requires ZERO API keys. Runs entirely on public data.
"""
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
    df = df.sort_index()[["c"]].astype(float)
    df.columns = ["close"]
    return df[~df.index.duplicated()]

def simulate(inst, initial_cash=5000.0):
    df = fetch_okx(inst)
    df['sma'] = df['close'].rolling(200).mean()
    df = df.dropna()
    
    cash = initial_cash
    position = 0.0
    trades = []
    
    for i in range(1, len(df)):
        price = df['close'].iloc[i-1] # Buy at yesterday's close
        signal = df['close'].iloc[i-1] > df['sma'].iloc[i-1]
        
        # Buy
        if signal and position == 0:
            position = cash / price
            cash = 0
            trades.append(f"{df.index[i].date()}: BOUGHT at ${price:.2f}")
            
        # Sell
        elif not signal and position > 0:
            cash = position * price
            position = 0
            trades.append(f"{df.index[i].date()}: SOLD at ${price:.2f}")
            
    # Final value
    final_price = df['close'].iloc[-1]
    final_val = cash + (position * final_price)
    
    print(f"\n--- {inst} SIMULATION ---")
    print(f"Start Date: {df.index[0].date()} | End Date: {df.index[-1].date()}")
    print(f"Start Cash: ${initial_cash:.2f}")
    print(f"Final Value: ${final_val:.2f}")
    print(f"Total Return: {((final_val/initial_cash)-1)*100:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    print("Last 5 trades:")
    for t in trades[-5:]:
        print(f"  {t}")

if __name__ == "__main__":
    simulate("BTC-USDT-SWAP")
    simulate("SOL-USDT-SWAP")
