#!/usr/bin/env python3
"""
Daily 200-SMA Trend Gate Bot (OKX).
- If Close > 200 SMA: Long.
- If Close < 200 SMA: Flat (Cash).
- PUBLIC MODE: If no API keys are provided, it runs as a read-only simulation.
"""
import os
import sys
import time
import requests
import pandas as pd
import ccxt

# ---------------- CONFIGURATION ----------------
API_KEY = os.environ.get("OKX_API_KEY", "")
API_SECRET = os.environ.get("OKX_API_SECRET", "")
API_PASS = os.environ.get("OKX_API_PASS", "")
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")
IS_DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"

# If no key is found, we run in public mode (no login required)
PUBLIC_MODE = not bool(API_KEY)

# Strategy Parameters
SYMBOLS = ["BTC/USDT:USDT", "SOL/USDT:USDT"]
SMA_LEN = 200
NOTIONAL_SIZE_USD = 1000  # How much USD value to allocate per coin
STOP_PCT = 0.25           # 25% catastrophic stop

# ---------------- TELEGRAM ----------------
def send_telegram(msg):
    if not TG_TOKEN or not TG_CHAT:
        return # Silently skip if no telegram
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    except Exception:
        pass

# ---------------- EXCHANGE SETUP ----------------
def get_exchange():
    config = {'enableRateLimit': True, 'options': {'defaultType': 'swap'}}
    
    if not PUBLIC_MODE:
        config.update({'apiKey': API_KEY, 'secret': API_SECRET, 'password': API_PASS})
        if IS_DRY_RUN:
            ex = ccxt.okx(config)
            ex.set_sandbox_mode(True)
            return ex
    return ccxt.okx(config)

# ---------------- STRATEGY LOGIC ----------------
def get_signal(ex, symbol):
    """Returns True if Close > 200 SMA, False otherwise."""
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
    if PUBLIC_MODE:
        action = "BUY / LONG" if target_long else "SELL / FLAT"
        return f"🧪 [SIMULATION] {symbol}: Signal is {action}. (No API keys provided, skipping order)."

    positions = ex.fetch_positions([symbol])
    pos = next((p for p in positions if p['symbol'] == symbol), None)
    current_amt = float(pos['contracts']) if pos and pos['contracts'] else 0.0
    
    ticker = ex.fetch_ticker(symbol)
    price = ticker['last']
    contract_size = ex.market(symbol)['contractSize']
    
    if target_long:
        target_amt = (NOTIONAL_SIZE_USD / price) / contract_size
        target_amt = float(ex.amount_to_precision(symbol, target_amt))
    else:
        target_amt = 0.0

    if abs(current_amt - target_amt) < 1e-6:
        return f"✅ {symbol}: Holding target ({current_amt} contracts). No action needed."

    msg = f"⚙️ {symbol}: Adjusting position from {current_amt} -> {target_amt} contracts.\n"
    
    try:
        if current_amt > 0 and target_amt == 0:
            if not IS_DRY_RUN:
                ex.create_market_sell_order(symbol, current_amt, params={'reduceOnly': True})
            msg += "Action: CLOSED position."
            
        elif current_amt == 0 and target_amt > 0:
            if not IS_DRY_RUN:
                order = ex.create_market_buy_order(symbol, target_amt)
                stop_price = ex.price_to_precision(symbol, price * (1 - STOP_PCT))
                stop_params = {'stopPrice': stop_price, 'reduceOnly': True, 'triggerPrice': stop_price}
                ex.create_order(symbol, 'stop', 'sell', target_amt, None, stop_params)
                msg += f"Action: OPENED LONG. Catastrophic stop at {stop_price}."
                
        elif current_amt > 0 and target_amt > 0:
            diff = target_amt - current_amt
            side = 'buy' if diff > 0 else 'sell'
            if not IS_DRY_RUN:
                ex.create_order(symbol, 'market', side, abs(diff))
            msg += f"Action: RESIZED position by {diff}."

    except Exception as e:
        msg += f"❌ ERROR: {e}"
        
    return msg

# ---------------- MAIN ----------------
def main():
    mode_str = "PUBLIC SIMULATION" if PUBLIC_MODE else f"LIVE (Dry Run: {IS_DRY_RUN})"
    print(f"=== STARTING BOT [{mode_str}] ===")
    ex = get_exchange()
    
    report = [f"🤖 *Daily Bot Run* [{mode_str}]"]
    
    try:
        if not PUBLIC_MODE:
            balance = ex.fetch_balance()
            report.append(f"Balance: {balance['USDT']['free']:.2f} USDT\n")
        else:
            report.append("⚠️ *No API keys found. Running read-only data test.*\n")
        
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
