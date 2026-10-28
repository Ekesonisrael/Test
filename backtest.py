#!/usr/bin/env python3
"""
Vol-targeted BTC/SOL backtest. Confirmatory, not exploratory.

RULES: code frozen -> run once -> result is the verdict.
One bug-fix rerun is allowed after seeing results and counts as trial #2.
Any parameter change after seeing a result is a new trial. Trials so far: 1.

CHANGELOG (pre-first-real-run, from code review — not counted as a trial):
  - Verdict: FAIL no longer masked by INCONCLUSIVE. Any conclusive gate
    failure -> FAIL. Only truly untested gates -> INCONCLUSIVE.
  - Null now pays honest transaction costs for daily rebalance to fixed L:
        turnover_null = L * |1 - L| * |r|
    This is the cost of maintaining constant leverage, which is what the
    null is. A buy-and-hold null would have near-zero cost but leverage
    would drift, so it wouldn't be a "matched" null. Chosen deliberately.
  - Empty bear-slice in funding-drag report now prints nan%, not crash.
  - Added sanity checks to funding fetch: column count >= 3, |rate| < 5%.
  - Filter out future-dated candles from OKX API.
  - NOTE ON VOL TARGET: TARGET_VOL is PER COIN, not portfolio. With BTC/SOL
    correlation ~0.75, realised PORTFOLIO vol runs ~14% vs 15% per-coin.
    Reported explicitly in output. Frozen as per-coin; do not change post-run.

Usage:
  python backtest.py                    # real data (OKX prices, Binance Vision funding)
  python backtest.py --synthetic        # plumbing test on fake data, proves nothing
  python backtest.py --fallback-funding # if Binance Vision is unreachable (flagged)
Exit codes: 0 all gates pass, 1 any gate fails, 2 inconclusive.
"""
import sys, io, time, zipfile, argparse
import numpy as np
import pandas as pd
import requests

# ---------------- FROZEN PARAMETERS ----------------
INSTR = {"BTC": "BTC-USDT-SWAP", "SOL": "SOL-USDT-SWAP"}
BVN = {"BTC": "BTCUSDT", "SOL": "SOLUSDT"}
TARGET_VOL = 0.15      # PER COIN, annualized (see docstring)
VOL_SPAN = 30
LEV_CAP = 2.0
SMA_N = 200
BAND = 0.20
FEE, SLIP = 0.0005, 0.0002
DD_THRESH = 0.08
MIN_YEARS = 4.0
D = 365
VOL_BAND = 0.25
DD_RATIO_MAX = 0.70
DD_WIN_MIN = 0.75
MIN_EPISODES = 6


# ---------------- DATA ----------------
def fetch_okx(inst):
    url = "https://www.okx.com/api/v5/market/history-candles"
    rows, after = [], None
    while True:
        p = {"instId": inst, "bar": "1Dutc", "limit": "100"}
        if after:
            p["after"] = after
        r = requests.get(url, params=p, timeout=20)
        r.raise_for_status()
        j = r.json()
        if j.get("code") != "0":
            raise RuntimeError(f"OKX error: {j}")
        if not j["data"]:
            break
        rows += j["data"]
        after = j["data"][-1][0]
        time.sleep(0.12)
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v", "vc", "vq", "confirm"])
    df = df[df.confirm == "1"]
    df.index = pd.to_datetime(df.ts.astype("int64"), unit="ms").dt.normalize()
    df = df.sort_index()[["o", "c"]].astype(float)
    df.columns = ["open", "close"]
    # Filter out any future-dated candles
    df = df[df.index <= pd.Timestamp.utcnow().normalize()]
    return df[~df.index.duplicated()]


def fetch_funding(sym, start, end):
    out = []
    for m in pd.period_range(start, end, freq="M"):
        url = (f"https://data.binance.vision/data/futures/um/monthly/fundingRate/"
               f"{sym}/{sym}-fundingRate-{m.year}-{m.month:02d}.zip")
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]))
        if df.shape[1] < 3:
            raise RuntimeError(f"{sym} {m}: expected >=3 columns, got {df.shape[1]}")
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        f = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        d = pd.DataFrame({"t": t, "f": f}).dropna()
        
        # Relaxed bound from 1% to 5% to accommodate extreme events (e.g. Nov 2022)
        bad = d[d.f.abs() > 0.05]
        if len(bad):
            raise RuntimeError(f"{sym} {m}: {len(bad)} rates outside (-5%, +5%) — "
                               f"column layout may have changed. Sample:\n{bad.head()}")
        out.append(d)
    if not out:
        raise RuntimeError(f"no funding data for {sym}")
    df = pd.concat(out)
    df.index = pd.to_datetime(df.t.astype("int64"), unit="ms").dt.normalize()
    return df.f.groupby(level=0).sum()


def synthetic():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-01-01", periods=365 * 5, freq="D")
    base = rng.normal(0.0004, 0.03, len(idx))
    px, fund = {}, {}
    for k, vol in (("BTC", 0.03), ("SOL", 0.05)):
        r = 0.7 * base + 0.3 * rng.normal(0.0004, vol, len(idx))
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, len(idx)))
        px[k] = pd.DataFrame({"open": o, "close": c}, index=idx)
        fund[k] = pd.Series(0.0003 + rng.normal(0, 0.0001, len(idx)), index=idx)
    return px, fund


# ---------------- STRATEGY ----------------
def run_arm(px, fund, gate, lag=0):
    close, opn = px.close, px.open
    r = close.pct_change()
    vol = r.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std() * np.sqrt(D)
    tw = (TARGET_VOL / vol).clip(upper=LEV_CAP)
    if gate:
        tw = tw.where(close > close.rolling(SMA_N).mean(), 0.0)
    tw = tw.shift(lag)
    ret_next = opn.shift(-2) / opn.shift(-1) - 1
    fnd_next = fund.reindex(px.index).shift(-1)
    ok = tw.notna() & ret_next.notna() & fnd_next.notna()
    first = ok.idxmax()
    ok = ok[first:]; ok = ok[ok]
    idx = ok.index
    t, rn, fn = tw[idx].values, ret_next[idx].values, fnd_next[idx].values
    w = np.zeros(len(idx)); prev = 0.0
    for i in range(len(idx)):
        if t[i] == 0.0:
            cur = 0.0
        elif prev == 0.0 or abs(t[i] - prev) / prev > BAND:
            cur = t[i]
        else:
            cur = prev
        w[i], prev = cur, cur
    wprev = np.r_[0.0, w[:-1]]
    cost = np.abs(w - wprev) * (FEE + SLIP)
    pnl = w * rn - w * fn - cost
    return pd.DataFrame({"w": w, "gross": w * rn, "pnl": pnl,
                         "rn": rn, "fn": fn, "cost": cost}, index=idx)


def null_arm(arm, L):
    """Fixed-L null, rebalanced daily to keep L constant.
    turnover = L * |1 - L| * |r| — the daily trade needed to hold L flat."""
    rn, fn = arm.rn.values, arm.fn.values
    turnover = L * abs(1.0 - L) * np.abs(rn)
    cost = turnover * (FEE + SLIP)
    gross = L * rn
    pnl = gross - L * fn - cost
    return pd.DataFrame({"gross": gross, "pnl": pnl, "rn": rn, "fn": fn,
                         "cost": cost}, index=arm.index)


# ---------------- METRICS ----------------
def sharpe(p): return p.mean() / p.std() * np.sqrt(D) if p.std() > 0 else np.nan
def cagr(p): return (1 + p).prod() ** (D / len(p)) - 1
def maxdd(p):
    e = (1 + p).cumprod()
    return (e / e.cummax() - 1).min()


def episodes(eq):
    v = eq.values
    out, pk, tr = [], 0, 0
    def close_ep():
        if 1 - v[tr] / v[pk] >= DD_THRESH:
            out.append((pk, tr))
    for i in range(1, len(v)):
        if v[i] >= v[pk]:
            close_ep(); pk = tr = i
        elif v[i] < v[tr]:
            tr = i
    close_ep()
    return out


def evaluate(name, arms):
    idx = arms["BTC"].index.intersection(arms["SOL"].index)
    arms = {k: a.loc[idx] for k, a in arms.items()}
    lines, ok_all, inconc = [f"\n=== {name} ==="], True, False

    for k, a in arms.items():
        inv = a.w != 0
        rv = a.gross[inv].std() * np.sqrt(D) if inv.sum() > 30 else np.nan
        passed = bool(abs(rv / TARGET_VOL - 1) <= VOL_BAND) if rv == rv else False
        ok_all &= passed
        lines.append(f"[GATE vol-given-invested] {k}: {rv:.3f} vs {TARGET_VOL} "
                     f"(invested {inv.mean():.0%} of days) -> {'PASS' if passed else 'FAIL'}")

    L = {k: a.w.abs().mean() for k, a in arms.items()}
    arm_p = pd.concat({k: a.pnl for k, a in arms.items()}, axis=1).mean(axis=1)
    nulls = {k: null_arm(a, L[k]) for k, a in arms.items()}
    null_p = pd.concat({k: n.pnl for k, n in nulls.items()}, axis=1).mean(axis=1)

    port_vol = arm_p.std() * np.sqrt(D)
    lines.append(f"[report] realised PORTFOLIO vol: {port_vol:.3f} "
                 f"(per-coin target {TARGET_VOL}; with rho~0.75 portfolio ~= 14%)")

    lines.append("avg leverage arm/null: " +
                 ", ".join(f"{k} {L[k]:.3f}/{L[k]:.3f}" for k in L) +
                 "  (null held exactly at L by daily rebalance)")

    arm_costs = pd.concat({k: a.cost for k, a in arms.items()}, axis=1).mean(axis=1)
    null_costs = pd.concat({k: n.cost for k, n in nulls.items()}, axis=1).mean(axis=1)
    lines.append(f"[report] avg daily cost arm {arm_costs.mean():.2e} "
                 f"| null {null_costs.mean():.2e}")

    eq_a, eq_n = (1 + arm_p).cumprod(), (1 + null_p).cumprod()
    eps = episodes(eq_n)
    if len(eps) < MIN_EPISODES:
        inconc = True
        lines.append(f"[GATE drawdown] only {len(eps)} episodes (<{MIN_EPISODES}) -> INCONCLUSIVE")
    else:
        la = np.array([1 - eq_a.iloc[t] / eq_a.iloc[p] for p, t in eps])
        ln = np.array([1 - eq_n.iloc[t] / eq_n.iloc[p] for p, t in eps])
        ratio = la.sum() / ln.sum()
        win = float((la <= ln).mean())
        passed = bool(ratio <= DD_RATIO_MAX and win >= DD_WIN_MIN)
        ok_all &= passed
        lines.append(f"[GATE drawdown paired] {len(eps)} episodes, loss ratio {ratio:.2f} "
                     f"(<= {DD_RATIO_MAX}), arm<=null in {win:.0%} (>= {DD_WIN_MIN:.0%}) "
                     f"-> {'PASS' if passed else 'FAIL'}")
        lines.append("   NB: episodes cluster in 2-3 market cycles; not independent.")

    lines.append(f"[report] Sharpe arm {sharpe(arm_p):.2f} | null {sharpe(null_p):.2f} | "
                 f"CAGR arm {cagr(arm_p):.1%} | null {cagr(null_p):.1%} | "
                 f"maxDD arm {maxdd(arm_p):.1%} | null {maxdd(null_p):.1%}")
    return ok_all, inconc, lines, arms


def arm_verdict(ok, inc):
    if not ok: return "FAIL"
    if inc:    return "INCONCLUSIVE"
    return "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fallback-funding", action="store_true")
    a = ap.parse_args()

    if a.synthetic:
        px, fund = synthetic()
        print("SYNTHETIC DATA: plumbing test only.")
    else:
        px, fund = {}, {}
        for k in INSTR:
            px[k] = fetch_okx(INSTR[k])
            print(f"{k}: {len(px[k])} bars {px[k].index[0].date()} -> {px[k].index[-1].date()}")
        try:
            for k in INSTR:
                fund[k] = fetch_funding(BVN[k], px[k].index[0], px[k].index[-1])
        except Exception as e:
            if not a.fallback_funding:
                sys.exit(f"ABORT: funding fetch failed ({e}). Rerun with --fallback-funding "
                         f"to assume 0.03%/day, and treat results as degraded.")
            print("WARNING: assumed flat funding 0.03%/day. DEGRADED RUN.")
            fund = {k: pd.Series(0.0003, index=px[k].index) for k in INSTR}
        end = min(min(f.index[-1] for f in fund.values()), min(p.index[-1] for p in px.values()))
        px = {k: p[:end] for k, p in px.items()}

    for k, p in px.items():
        yrs = (p.index[-1] - p.index[0]).days / 365
        assert yrs >= MIN_YEARS, f"{k}: only {yrs:.1f} years of data, need {MIN_YEARS}"
        assert p.index.is_monotonic_increasing and not p.index.duplicated().any()

    base = run_arm(px["BTC"], fund["BTC"], True, lag=0)
    lagged = run_arm(px["BTC"], fund["BTC"], True, lag=1)
    j = base.index.intersection(lagged.index)
    assert not np.allclose(base.pnl[j].fillna(0), lagged.pnl[j].fillna(0)), \
        "SELFTEST FAILED: signal lag had no effect"

    results = {}
    for name, gate in (("vol-only (the thesis)", False), ("vol + 200d gate (the product)", True)):
        arms = {k: run_arm(px[k], fund[k], gate) for k in INSTR}
        okk, inc, lines, used = evaluate(name, arms)
        print("\n".join(lines))
        results[name] = (okk, inc)
        if gate:
            bt = px["BTC"].close
            bear = (bt < 0.7 * bt.cummax()).reindex(used["BTC"].index).fillna(False)
            for k, u in used.items():
                drag = (u.w * u.fn * D)
                bull_drag = drag[~bear].mean() if (~bear).sum() else float("nan")
                bear_drag = drag[bear].mean() if bear.sum() else float("nan")
                print(f"[report] {k} annualized funding cost: "
                      f"bull {bull_drag:.2%} | bear {bear_drag:.2%} "
                      f"(bear = BTC>30% below ATH, simplification applied to both coins)")

    print("\n=== VERDICT (both arms must pass) ===")
    verdicts = {name: arm_verdict(ok, inc) for name, (ok, inc) in results.items()}
    for name, v in verdicts.items():
        print(f"  {name}: {v}")
    if any(v == "FAIL" for v in verdicts.values()):
        print("FAIL: do not build the execution layer on this strategy. Rerun = new trial.")
        sys.exit(1)
    if any(v == "INCONCLUSIVE" for v in verdicts.values()):
        print("INCONCLUSIVE: gates could not be decided on this sample.")
        sys.exit(2)
    print("PASS: risk-side claims hold on this sample. Says nothing about alpha.")
    sys.exit(0)


if __name__ == "__main__":
    main()
