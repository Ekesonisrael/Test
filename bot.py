#!/usr/bin/env python3
"""
Daily 200-SMA Trend Gate Bot (OKX).
- If Close > 200 SMA: Long.
- If Close < 200 SMA: Flat (Cash).
- Catastrophic server-side stop placed on every entry.
"""
import os
import sys
import time
import requests
import pandas as pd
import ccxt

# ---------------- CONFIGURATION ----------------
# Loaded from GitHub Actions Environment Variables
API_KEY = os.environ.get("OKX_API_KEY")
API_SECRET = os.environ.get("OKX_API_SECRET")
API_PASS = os.environ.get("OKX_API_PASS")
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")
IS_DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"

# Strategy Parameters
SYMBOLS = ["BTC/USDT:USDT", "SOL/USDT:USDT"]
SMA_LEN = 200
NOTIONAL_SIZE_USD = 1000  # How much USD value to allocate per coin
STOP_PCT = 0.25           # 25% catastrophic stop

# ---------------- TELEGRAM ----------------
def send_telegram(msg):
    if not TG_TOKEN or not TG_CHAT:
        print(f"TG NOT CONFIGURED. MSG: {msg}")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        print(f"Failed to send TG: {e}")

# ---------------- EXCHANGE SETUP ----------------
def get_exchange():
    ex = ccxt.okx({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'password': API_PASS,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })
    if IS_DRY_RUN:
        ex.set_sandbox_mode(True)
    return ex

# ---------------- STRATEGY LOGIC ----------------
def get_signal(ex, symbol):
    """Returns True if Close > 200 SMA, False otherwise."""
    # Fetch enough candles to calculate SMA + current
    ohlcv = ex.fetch_ohlcv(symbol, timeframe='1d', limit=SMA_LEN + 10)
    df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    
    # Drop the current, still-forming candle
    df = df.iloc[:-1]
    
    sma = df['c'].rolling(window=SMA_LEN).mean().iloc[-1]
    current_close = df['c'].iloc[-1]
    
    signal = current_close > sma
    print(f"{symbol}: Close {current_close:.2f} | SMA{SMA_LEN} {sma:.2f} -> {'LONG' if signal else 'FLAT'}")
    return signal

# ---------------- EXECUTION ----------------
def reconcile_position(ex, symbol, target_long):
    """Checks exchange position and enters/exits if it doesn't match target."""
    positions = ex.fetch_positions([symbol])
    pos = next((p for p in positions if p['symbol'] == symbol), None)
    
    current_amt = float(pos['contracts']) if pos and pos['contracts'] else 0.0
    
    # Determine target size in contracts
    ticker = ex.fetch_ticker(symbol)
    price = ticker['last']
    contract_size = ex.market(symbol)['contractSize']
    
    if target_long:
        # Calculate target contracts (rounded to lot size)
        target_amt = (NOTIONAL_SIZE_USD / price) / contract_size
        target_amt = ex.amount_to_precision(symbol, target_amt)
        target_amt = float(target_amt)
    else:
        target_amt = 0.0

    # If already at target (within 1 contract tolerance), do nothing
    if abs(current_amt - target_amt) < 1e-6:
        return f"✅ {symbol}: Holding target ({current_amt} contracts). No action needed."

    msg = f"⚙️ {symbol}: Adjusting position from {current_amt} -> {target_amt} contracts.\n"
    
    try:
        if current_amt > 0 and target_amt == 0:
            # Close existing
            if not IS_DRY_RUN:
                ex.create_market_sell_order(symbol, current_amt, params={'reduceOnly': True})
            msg += "Action: CLOSED position."
            
        elif current_amt == 0 and target_amt > 0:
            # Open new
            if not IS_DRY_RUN:
                order = ex.create_market_buy_order(symbol, target_amt)
                
                # Place Catastrophic Exchange-Side Stop
                stop_price = price * (1 - STOP_PCT)
                stop_price = ex.price_to_precision(symbol, stop_price)
                
                stop_params = {
                    'stopPrice': stop_price,
                    'reduceOnly': True,
                    'triggerPrice': stop_price
                }
                ex.create_order(symbol, 'stop', 'sell', target_amt, None, stop_params)
                msg += f"Action: OPENED LONG. Catastrophic stop at {stop_price}."
                
        elif current_amt > 0 and target_amt > 0:
            # Resize (rare in binary system, but handles drift)
            diff = target_amt - current_amt
            side = 'buy' if diff > 0 else 'sell'
            if not IS_DRY_RUN:
                ex.create_order(symbol, 'market', side, abs(diff))
            msg += f"Action: RESIZED position by {diff}."

    except Exception as e:
        msg += f"❌ ERROR: {e}"
        print(msg)
        return msg
        
    return msg

# ---------------- MAIN ----------------
def main():
    print(f"=== STARTING BOT (Dry Run: {IS_DRY_RUN}) ===")
    ex = get_exchange()
    
    report = [f"🤖 *Daily Bot Run* (Dry Run: {IS_DRY_RUN})"]
    
    try:
        balance = ex.fetch_balance()
        usdt_bal = balance['USDT']['free']
        report.append(f"Balance: {usdt_bal:.2f} USDT\n")
        
        for sym in SYMBOLS:
            is_long = get_signal(ex, sym)
            result = reconcile_position(ex, sym, is_long)
            report.append(result)
            
        final_msg = "\n".join(report)
        print(final_msg)
        send_telegram(final_msg)
        
    except Exception as e:
        err_msg = f"🚨 BOT CRASHED:\n{str(e)}"
        print(err_msg)
        send_telegram(err_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
