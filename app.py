"""
EMA Crossover Web Platform
Flask backend — serves UI, runs backtest/optimizer API, manages live bot thread.
"""

import os, sys, json, time, threading, hashlib, logging, csv, io, secrets
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from flask import Flask, jsonify, request, send_from_directory, Response, session, redirect, url_for
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder="static", template_folder=".")
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
from datetime import timedelta
app.permanent_session_lifetime = timedelta(days=30)  # stay logged in 30 days

# ─────────────────────────────────────────────────────────────────
# PATHS / GLOBALS
# ─────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
CACHE_DIR      = BASE_DIR / "candle_cache"
RESULT_DIR     = BASE_DIR / "backtest_results"
BOT_STATE_FILE = BASE_DIR / "live_bot_state.json"
BOT_CONFIG_FILE= BASE_DIR / "live_bot_config.json"
SIGNALS_FILE   = BASE_DIR / "signals.csv"

CACHE_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

AUTH_FILE = BASE_DIR / "auth.json"

# ─────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────
def load_auth():
    """Load credentials. Creates default on first run."""
    if AUTH_FILE.exists():
        with open(AUTH_FILE) as f:
            return json.load(f)
    default = {"username": "admin", "password_hash": _hash_pw("changeme123")}
    with open(AUTH_FILE, "w") as f:
        json.dump(default, f, indent=2)
    return default

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.strip().encode()).hexdigest()

def check_credentials(username: str, password: str) -> bool:
    auth = load_auth()
    return (username.strip() == auth["username"] and
            _hash_pw(password) == auth["password_hash"])

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated

@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        if check_credentials(u, p):
            session["logged_in"] = True
            session["username"]  = u
            session.permanent    = True
            return redirect("/")
        error = "Invalid username or password."
    return _login_page(error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/api/auth/change-password", methods=["POST"])
@login_required
def api_change_password():
    body    = request.json or {}
    current = body.get("current", "")
    new_pw  = body.get("new", "")
    if not check_credentials(session.get("username",""), current):
        return jsonify({"error": "Current password incorrect"}), 400
    if len(new_pw) < 8:
        return jsonify({"error": "New password must be at least 8 characters"}), 400
    auth = load_auth()
    auth["password_hash"] = _hash_pw(new_pw)
    with open(AUTH_FILE, "w") as f:
        json.dump(auth, f, indent=2)
    return jsonify({"ok": True})

@app.route("/api/auth/change-username", methods=["POST"])
@login_required
def api_change_username():
    body         = request.json or {}
    new_username = body.get("username","").strip()
    password     = body.get("password","")
    if not check_credentials(session.get("username",""), password):
        return jsonify({"error": "Password incorrect"}), 400
    if len(new_username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    auth = load_auth()
    auth["username"] = new_username
    with open(AUTH_FILE, "w") as f:
        json.dump(auth, f, indent=2)
    session["username"] = new_username
    return jsonify({"ok": True})

def _login_page(error=""):
    err_html = f"<div class=\'error\'>{error}</div>" if error else ""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>EMAX — Login</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#080a0d;color:#dce3ee;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;justify-content:center;min-height:100vh}}
.box{{width:340px;background:#0e1117;border:1px solid #1a2030;padding:36px}}
.logo{{font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:3px;color:#f0b90b;margin-bottom:4px}}
.logo span{{color:#00d4ff}}
.sub{{font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3d4d66;margin-bottom:32px}}
label{{display:block;font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#3d4d66;margin-bottom:4px;margin-top:16px}}
input{{width:100%;background:#131820;border:1px solid #222d40;color:#dce3ee;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;padding:9px 10px;outline:none;border-radius:2px;transition:border .15s}}
input:focus{{border-color:#00d4ff}}
.error{{background:rgba(246,70,93,.08);border:1px solid rgba(246,70,93,.25);color:#f6465d;font-size:0.65rem;padding:8px 10px;margin-top:12px;border-radius:2px}}
button{{width:100%;margin-top:24px;background:#f0b90b;border:none;color:#000;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;letter-spacing:1px;text-transform:uppercase;font-weight:600;padding:11px;cursor:pointer;border-radius:2px;transition:background .12s}}
button:hover{{background:#d4a30a}}
.hint{{font-size:0.58rem;color:#3d4d66;margin-top:16px;text-align:center}}
</style>
</head>
<body>
<div class="box">
  <div class="logo">EMA<span>X</span></div>
  <div class="sub">Trading Platform</div>
  <form method="POST" action="/login">
    <label>Username</label>
    <input type="text" name="username" autocomplete="username" autofocus>
    <label>Password</label>
    <input type="password" name="password" autocomplete="current-password">
    {err_html}
    <button type="submit">Sign In</button>
  </form>
  <div class="hint">Default: admin / changeme123<br>Change after first login in Settings</div>
</div>
</body>
</html>"""

# Live bot thread state
_bot_thread: threading.Thread = None
_bot_stop_event = threading.Event()
_bot_lock = threading.Lock()

log = logging.getLogger("webapp")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(BASE_DIR / "app.log"), logging.StreamHandler()])

# ─────────────────────────────────────────────────────────────────
# DEFAULT CONFIGS
# ─────────────────────────────────────────────────────────────────
DEFAULT_BACKTEST_CONFIG = {
    "exchange":     "bybit",
    "symbol":       "XMR/USDT:USDT",
    "timeframe":    "30m",
    "intrabar_tf":  "1m",
    "use_intrabar": True,
    "capital":      1000.0,
    "fee_pct":      0.055,
    "slippage_pct": 0.0143,
    "compounding":  True,
    "tsl_mode":     "intrabar",
    "params": {
        "fast_length":    9,
        "slow_length":    21,
        "use_trailing":   True,
        "use_all_exits":  False,
        "tp_perc":        2.0,
        "sl_perc":        1.0,
        "trail_perc":     1.0,
        "use_atr_tsl":    True,
        "atr_length":     9,
        "atr_multiplier": 10.0,
        "use_tiered_tsl": False,
        "tier1_profit":   5.0,
        "tier1_tsl":      3.0,
        "tier2_profit":   10.0,
        "tier2_tsl":      2.0,
        "tier3_tsl":      1.0,
        "use_vol_filter": False,
        "vol_multiplier": 1.5,
        "use_adx_filter": False,
        "adx_length":     14,
        "adx_threshold":  25,
    }
}

DEFAULT_BOT_CONFIG = {
    "exchange":       "bybit",
    "symbol":         "XMR/USDT:USDT",
    "timeframe":      "30m",
    "fast_length":    9,
    "slow_length":    21,
    "atr_length":     9,
    "atr_multiplier": 10.0,
    "tp_perc":        2.0,
    "trail_perc":     1.0,
    "use_trailing":   True,
    "use_atr_tsl":    True,
    "tick_size":      0.01,
    "tsl_mode":       "barclose",
    "webhook_url":    "",
    "signal_type":    "",
    "pionex_symbol":  "XMR_USDT",
    "contracts":      "0.1",
    "enabled":        False,
}

# ─────────────────────────────────────────────────────────────────
# INDICATOR HELPERS  (copied from backtest_app.py)
# ─────────────────────────────────────────────────────────────────
def compute_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def compute_atr(df, length):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def compute_adx(df, length):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    dm_plus  = np.where((h-h.shift(1))>(l.shift(1)-l), (h-h.shift(1)).clip(lower=0), 0)
    dm_minus = np.where((l.shift(1)-l)>(h-h.shift(1)), (l.shift(1)-l).clip(lower=0), 0)
    tr_s     = pd.Series(tr).ewm(alpha=1/length, adjust=False).mean()
    dip = 100 * pd.Series(dm_plus).ewm(alpha=1/length, adjust=False).mean()  / tr_s
    dim = 100 * pd.Series(dm_minus).ewm(alpha=1/length, adjust=False).mean() / tr_s
    dx  = 100 * (dip-dim).abs() / (dip+dim)
    return dx.ewm(alpha=1/length, adjust=False).mean()

def get_trail_dist(direction, ref_price, entry_price, atr_val, p):
    tick = p.get("tick_size", 0.01)
    profit_pct = ((ref_price-entry_price)/entry_price*100) if direction=="long" \
              else ((entry_price-ref_price)/entry_price*100)
    if p.get("use_atr_tsl") and atr_val > 0:
        if p.get("use_tiered_tsl"):
            if profit_pct < p["tier1_profit"]:   mult = p["tier1_tsl"]
            elif profit_pct < p["tier2_profit"]:  mult = p["tier2_tsl"]
            else:                                  mult = p["tier3_tsl"]
            return atr_val * mult * tick
        return atr_val * p["atr_multiplier"] * tick
    if p.get("use_tiered_tsl"):
        if profit_pct < p["tier1_profit"]:   return p["tier1_tsl"]/100*entry_price
        elif profit_pct < p["tier2_profit"]:  return p["tier2_tsl"]/100*entry_price
        return p["tier3_tsl"]/100*entry_price
    return p["trail_perc"]/100*entry_price

def build_intra_lookup(df_base, df_intra):
    lookup = {}
    base_ts  = df_base["timestamp"].values
    intra_ts = df_intra["timestamp"].values
    ih       = df_intra["high"].values
    il       = df_intra["low"].values
    ic       = df_intra["close"].values
    n = len(df_base)
    for i in range(n):
        t_start = base_ts[i]
        t_end   = base_ts[i+1] if i+1<n else base_ts[i]+np.timedelta64(1,'ns')*int(1e18)
        mask = (intra_ts>=t_start)&(intra_ts<t_end)
        idx  = np.where(mask)[0]
        if len(idx)>0:
            lookup[i] = list(zip(ih[idx], il[idx], ic[idx]))
        else:
            r = df_base.iloc[i]
            lookup[i] = [(r["high"], r["low"], r["close"])]
    return lookup

def _compute_dd(equity_curve, capital):
    arr  = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    dd   = (arr-peak)/peak*100
    return float(dd.min()) if len(dd) else 0.0

# ─────────────────────────────────────────────────────────────────
# MAIN BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────
def run_ema_crossover(df_base, df_intra, p, fee_pct, capital):
    initial_capital = capital
    df = df_base.copy().reset_index(drop=True)
    df["fast_ema"] = compute_ema(df["close"], p["fast_length"])
    df["slow_ema"] = compute_ema(df["close"], p["slow_length"])
    df["atr"]      = compute_atr(df, p["atr_length"])
    if p.get("use_vol_filter"):
        df["avg_vol"] = df["volume"].rolling(20).mean()
    if p.get("use_adx_filter"):
        df["adx"] = compute_adx(df, p["adx_length"])

    fe = df["fast_ema"].values
    se = df["slow_ema"].values
    n  = len(df)
    cross_up   = np.zeros(n, dtype=bool)
    cross_down = np.zeros(n, dtype=bool)
    cross_up[2:]   = (fe[1:-1]>se[1:-1]) & (fe[:-2]<=se[:-2])
    cross_down[2:] = (fe[1:-1]<se[1:-1]) & (fe[:-2]>=se[:-2])

    vol_ok = np.ones(n, dtype=bool)
    adx_ok = np.ones(n, dtype=bool)
    if p.get("use_vol_filter"):
        avg_v = df["avg_vol"].values
        vol_v = df["volume"].values
        with np.errstate(invalid="ignore"):
            vol_ok = vol_v > avg_v * p["vol_multiplier"]
        vol_ok[np.isnan(avg_v)] = False
    if p.get("use_adx_filter"):
        adx_ok = df["adx"].values > p["adx_threshold"]

    long_sig  = cross_up   & vol_ok & adx_ok
    short_sig = cross_down & vol_ok & adx_ok

    intra_lookup = None
    if df_intra is not None and len(df_intra) > 0:
        intra_lookup = build_intra_lookup(df, df_intra)

    trades       = []
    position     = None
    equity_curve = []

    for i in range(2, n):
        row     = df.iloc[i]
        atr_val = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
        close   = float(row["close"])

        if position is not None:
            ep  = position["entry_price"]
            d   = position["direction"]
            trail_active = position["trail_active"]

            if intra_lookup is not None:
                bars = intra_lookup.get(i, [(row["high"], row["low"], close)])
            else:
                bars = [(float(row["high"]), float(row["low"]), close)]

            closed = False; exit_price = None; exit_reason = None

            for (ih, il, ic) in bars:
                if closed: break
                if d == "long":
                    if ih >= position["tp"]:
                        exit_price, exit_reason, closed = position["tp"], "TP", True
                    elif not trail_active and il <= position["sl"]:
                        exit_price, exit_reason, closed = position["sl"], "SL", True
                else:
                    if il <= position["tp"]:
                        exit_price, exit_reason, closed = position["tp"], "TP", True
                    elif not trail_active and ih >= position["sl"]:
                        exit_price, exit_reason, closed = position["sl"], "SL", True

            if not closed and trail_active:
                dist = get_trail_dist(d, close, ep, atr_val, p)
                tsl_mode = p.get("tsl_mode", "intrabar")
                for (ih2, il2, ic2) in bars:
                    if d == "long":
                        if not position["trail_activated"]:
                            if ih2 >= ep + dist:
                                position["trail_activated"] = True
                                position["trail_stop"] = ih2 - dist
                        elif ih2 - dist > position["trail_stop"]:
                            position["trail_stop"] = ih2 - dist
                    else:
                        if not position["trail_activated"]:
                            if il2 <= ep - dist:
                                position["trail_activated"] = True
                                position["trail_stop"] = il2 + dist
                        elif position["trail_stop"] is None or il2+dist < position["trail_stop"]:
                            position["trail_stop"] = il2 + dist

                if position["trail_activated"] and position["trail_stop"] is not None:
                    if tsl_mode == "barclose":
                        if d=="long" and close<=position["trail_stop"]:
                            exit_price,exit_reason,closed = position["trail_stop"],"TSL",True
                        elif d=="short" and close>=position["trail_stop"]:
                            exit_price,exit_reason,closed = position["trail_stop"],"TSL",True
                    else:
                        for (ih2,il2,ic2) in bars:
                            if closed: break
                            if d=="long" and il2<=position["trail_stop"]:
                                exit_price,exit_reason,closed = position["trail_stop"],"TSL",True
                            elif d=="short" and ih2>=position["trail_stop"]:
                                exit_price,exit_reason,closed = position["trail_stop"],"TSL",True

            if not closed:
                if d=="long" and short_sig[i]:
                    exit_price, exit_reason, closed = close, "Signal", True
                elif d=="short" and long_sig[i]:
                    exit_price, exit_reason, closed = close, "Signal", True

            if closed:
                slip = p.get("slippage_pct", 0.0143)/100
                exit_price = exit_price*(1-slip) if d=="long" else exit_price*(1+slip)
                fee_open  = ep*position["qty"]*fee_pct/100
                fee_close = exit_price*position["qty"]*fee_pct/100
                gross = (exit_price-ep)*position["qty"] if d=="long" else (ep-exit_price)*position["qty"]
                net_pnl = gross - fee_open - fee_close
                pnl_pct = net_pnl/position["capital_at_entry"]*100
                trades.append({
                    "entry_time":  position["entry_time"].isoformat() if hasattr(position["entry_time"],"isoformat") else str(position["entry_time"]),
                    "exit_time":   row["timestamp"].isoformat() if hasattr(row["timestamp"],"isoformat") else str(row["timestamp"]),
                    "direction":   d,
                    "entry_price": round(ep, 6),
                    "exit_price":  round(exit_price, 6),
                    "exit_reason": exit_reason,
                    "pnl_usdt":    round(net_pnl, 4),
                    "pnl_pct":     round(pnl_pct, 4),
                    "fee_usdt":    round(fee_open+fee_close, 4),
                })
                capital += net_pnl
                position = None

        if position is not None:
            ep_ = position["entry_price"]; qty_ = position["qty"]
            unr = (close-ep_)*qty_ if position["direction"]=="long" else (ep_-close)*qty_
            equity_curve.append(capital+unr)
        else:
            equity_curve.append(capital)

        if position is None:
            dir_ = None
            if i>0 and long_sig[i-1]:  dir_ = "long"
            elif i>0 and short_sig[i-1]: dir_ = "short"
            if dir_ is not None:
                slip   = p.get("slippage_pct", 0.0143)/100
                raw_ep = float(row["open"])
                ep     = raw_ep*(1+slip) if dir_=="long" else raw_ep*(1-slip)
                trade_cap = capital if p.get("compounding", True) else initial_capital
                qty    = trade_cap/ep
                tp     = ep*(1+p["tp_perc"]/100) if dir_=="long" else ep*(1-p["tp_perc"]/100)
                sl     = ep*(1-p["sl_perc"]/100) if dir_=="long" else ep*(1+p["sl_perc"]/100)
                position = {
                    "direction":        dir_,
                    "entry_time":       row["timestamp"],
                    "entry_price":      ep,
                    "qty":              qty,
                    "capital_at_entry": trade_cap,
                    "tp": tp, "sl": sl,
                    "trail_active":     p.get("use_trailing") or p.get("use_all_exits"),
                    "trail_activated":  False,
                    "trail_stop":       None,
                }

    return trades, capital, equity_curve

# ─────────────────────────────────────────────────────────────────
# OPTIMIZATION WORKER  (top-level for multiprocessing)
# ─────────────────────────────────────────────────────────────────
def _opt_worker(args):
    combo, keys, base_params, base_rec, intra_rec, fee_pct, capital = args
    df_base = pd.DataFrame(base_rec)
    df_base["timestamp"] = pd.to_datetime(df_base["timestamp"], utc=True)
    df_intra = None
    if intra_rec is not None:
        df_intra = pd.DataFrame(intra_rec)
        df_intra["timestamp"] = pd.to_datetime(df_intra["timestamp"], utc=True)
    p = base_params.copy()
    for k, v in zip(keys, combo):
        p[k] = v
    try:
        trades, final_cap, eq_curve = run_ema_crossover(df_base, df_intra, p, fee_pct, capital)
    except Exception:
        return None
    if not trades or len(trades) < 5:
        return None
    df = pd.DataFrame(trades)
    wins = (df["pnl_pct"] > 0).sum()
    ret  = (final_cap - capital) / capital * 100
    wr   = wins / len(trades) * 100
    max_dd = _compute_dd(eq_curve, capital) if eq_curve else 0.0
    worst_trade = float(df["pnl_pct"].min())
    df["duration"] = (pd.to_datetime(df["exit_time"]) - pd.to_datetime(df["entry_time"])).dt.total_seconds() / 1800
    avg_dur = float(df["duration"].mean())
    max_safe_lev = 1
    for lev in [2,3,5,7,10]:
        if max_dd*lev >= -60: max_safe_lev = lev
    lev_ret = ret*max_safe_lev
    lev_dd  = max(max_dd*max_safe_lev, -100.0)
    calmar  = lev_ret/max(abs(lev_dd), 1.0)
    gw = df[df["pnl_usdt"]>0]["pnl_usdt"].sum()
    gl = df[df["pnl_usdt"]<=0]["pnl_usdt"].abs().sum()
    pf = round(gw/gl, 2) if gl > 0 else 99.0
    score = calmar*(1+wr/100)*min(pf/2.0, 3.0)
    return {
        **{k: v for k, v in zip(keys, combo)},
        "trades": len(trades), "avg_dur": round(avg_dur,1),
        "win_rate": round(wr,2), "return": round(ret,2),
        "max_dd": round(max_dd,2), "worst_trade": round(worst_trade,2),
        "pf": pf, "max_lev": max_safe_lev, "score": round(score,2),
    }

# ─────────────────────────────────────────────────────────────────
# DATA FETCHING / CACHE
# ─────────────────────────────────────────────────────────────────
def cache_path(exchange, symbol, timeframe):
    sym = symbol.replace("/","_").replace(":","_")
    return CACHE_DIR / f"{exchange}_{sym}_{timeframe}.csv"

def load_cache(exchange, symbol, timeframe):
    p = cache_path(exchange, symbol, timeframe)
    if p.exists():
        df = pd.read_csv(p, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    return None

def save_cache(df, exchange, symbol, timeframe):
    cache_path(exchange, symbol, timeframe).parent.mkdir(exist_ok=True)
    df.to_csv(cache_path(exchange, symbol, timeframe), index=False)

def get_exchange(name):
    import ccxt
    name = name.lower()
    if name == "bybit":   return ccxt.bybit({"enableRateLimit": True})
    if name == "pionex":  return ccxt.pionex({"enableRateLimit": True})
    raise ValueError(f"Unknown exchange: {name}")

def timeframe_to_ms(tf):
    units = {"m":60000,"h":3600000,"d":86400000}
    return int(tf[:-1]) * units.get(tf[-1], 60000)

def fetch_ohlcv_update(exchange_name, symbol, timeframe, max_candles=10000):
    """Fetch+update cache. Returns (df, new_count, total_count)."""
    ex = get_exchange(exchange_name)
    ex.load_markets()
    cached = load_cache(exchange_name, symbol, timeframe)
    per_req = 1000

    if cached is not None and len(cached) > 0:
        last_ts = int(cached["timestamp"].iloc[-1].timestamp()*1000)
        new_batches = []; since = last_ts+1
        while True:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=per_req)
            if not batch: break
            new_batches.extend(batch)
            if len(batch) < per_req: break
            since = batch[-1][0]+1
            time.sleep(0.2)
        if new_batches:
            new_df = pd.DataFrame(new_batches, columns=["timestamp","open","high","low","close","volume"])
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
            df = pd.concat([cached, new_df], ignore_index=True)
            df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
            save_cache(df, exchange_name, symbol, timeframe)
            return df, len(new_batches), len(df)
        return cached, 0, len(cached)

    # Full fetch
    all_candles = []
    since = None
    batch = ex.fetch_ohlcv(symbol, timeframe, since=None, limit=per_req)
    if batch:
        all_candles = batch[:]
        while len(all_candles) < max_candles:
            earliest_ts = all_candles[0][0]
            tf_ms = timeframe_to_ms(timeframe)
            fetch_since = earliest_ts - (per_req * tf_ms)
            batch2 = ex.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=per_req)
            if not batch2 or batch2[0][0] >= earliest_ts: break
            new_c = [c for c in batch2 if c[0] < earliest_ts]
            if not new_c: break
            all_candles = new_c + all_candles
            time.sleep(0.2)
    df = pd.DataFrame(all_candles, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    save_cache(df, exchange_name, symbol, timeframe)
    return df, len(df), len(df)

# ─────────────────────────────────────────────────────────────────
# RESULT CACHING  (hash settings → JSON file)
# ─────────────────────────────────────────────────────────────────
def settings_hash(cfg):
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def result_path(h):
    return RESULT_DIR / f"{h}.json"

def load_result(h):
    p = result_path(h)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def save_result(h, data):
    with open(result_path(h), "w") as f:
        json.dump(data, f)

# ─────────────────────────────────────────────────────────────────
# BOT CONFIG I/O
# ─────────────────────────────────────────────────────────────────
def load_bot_config():
    if BOT_CONFIG_FILE.exists():
        with open(BOT_CONFIG_FILE) as f:
            cfg = DEFAULT_BOT_CONFIG.copy()
            cfg.update(json.load(f))
            return cfg
    return DEFAULT_BOT_CONFIG.copy()

def save_bot_config(cfg):
    with open(BOT_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def load_bot_state():
    if BOT_STATE_FILE.exists():
        with open(BOT_STATE_FILE) as f:
            return json.load(f)
    return {"position": None, "entry_price": None, "entry_time": None,
            "trail_stop": None, "trail_activated": False}

def save_bot_state(state):
    with open(BOT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ─────────────────────────────────────────────────────────────────
# LIVE BOT  (runs in background thread)
# ─────────────────────────────────────────────────────────────────
def _bot_get_trail_dist(entry_price, atr_val, cfg):
    if cfg.get("use_atr_tsl") and atr_val > 0:
        return atr_val * cfg["atr_multiplier"] * cfg.get("tick_size", 0.01)
    return cfg["trail_perc"]/100*entry_price

def _bot_send_webhook(action, price, note, cfg):
    import requests as req_lib
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sig_file = SIGNALS_FILE
    has_header = not sig_file.exists()
    if action=="buy" and "open long" in note:
        direction="long"; pos_size=cfg["contracts"]
    elif action=="sell" and "open short" in note:
        direction="short"; pos_size=f"-{cfg['contracts']}"
    elif action=="sell":
        direction="long"; pos_size="0"
    else:
        direction="short"; pos_size="0"
    detail = f"{direction} {cfg['pionex_symbol']} {note} price:{price:.4f}"
    with open(sig_file, "a", newline="", encoding="utf-8") as f:
        if has_header:
            f.write("Time (UTC),Symbol,Action,Direction,Price,Contracts,Note\n")
        f.write(f"{now_str},{cfg['pionex_symbol']},{action},{direction},{price:.4f},{cfg['contracts']},{note}\n")
    if cfg.get("webhook_url"):
        try:
            payload = {
                "data": {"action": action, "contracts": cfg["contracts"], "position_size": pos_size},
                "price": str(round(price, 4)),
                "signal_param": "{}",
                "signal_type": cfg.get("signal_type",""),
                "symbol": cfg["pionex_symbol"],
                "time": now_str,
            }
            headers = {"Content-Type":"application/json","User-Agent":"Mozilla/5.0",
                       "Origin":"https://www.tradingview.com","Referer":"https://www.tradingview.com/"}
            r = req_lib.post(cfg["webhook_url"], json=payload, headers=headers, timeout=10)
            log.info(f"Webhook {action}: {r.status_code}")
        except Exception as e:
            log.warning(f"Webhook error: {e}")

def _bot_get_signals(df, cfg):
    df = df.copy()
    df["fast_ema"] = compute_ema(df["close"], cfg["fast_length"])
    df["slow_ema"] = compute_ema(df["close"], cfg["slow_length"])
    df["atr"]      = compute_atr(df, cfg["atr_length"])
    fe = df["fast_ema"].values; se = df["slow_ema"].values
    i  = len(df)-2
    long_sig  = bool(fe[i]>se[i] and fe[i-1]<=se[i-1])
    short_sig = bool(fe[i]<se[i] and fe[i-1]>=se[i-1])
    last_atr  = float(df["atr"].iloc[i])
    last_open = float(df["open"].iloc[-1])
    return long_sig, short_sig, last_atr, last_open

def _bot_loop(stop_event):
    import ccxt as _ccxt
    log.info("Bot thread started")
    cfg = load_bot_config()
    try:
        exchange = _ccxt.bybit({"enableRateLimit": True})
    except Exception as e:
        log.error(f"Bot: exchange init failed: {e}")
        return

    state = load_bot_state()
    last_bar_processed = None

    while not stop_event.is_set():
        try:
            cfg = load_bot_config()  # reload each iteration so UI changes take effect
            if not cfg.get("enabled"):
                time.sleep(30)
                continue

            now = datetime.now(timezone.utc)
            tf_min = int(cfg["timeframe"].replace("m","").replace("h","")) * (60 if "h" in cfg["timeframe"] else 1)
            floored = (now.minute // tf_min) * tf_min
            current_bar_open = now.replace(minute=floored, second=0, microsecond=0)
            last_completed = current_bar_open - pd.Timedelta(minutes=tf_min)

            bar_just_closed = (last_completed != last_bar_processed and
                               (now - current_bar_open).total_seconds() >= 5)

            if bar_just_closed:
                last_bar_processed = last_completed
                log.info(f"Bot: bar close {last_completed.strftime('%H:%M')}")
                try:
                    ohlcv = exchange.fetch_ohlcv(cfg["symbol"], cfg["timeframe"], limit=100)
                    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    long_sig, short_sig, atr_val, entry_open = _bot_get_signals(df, cfg)

                    last_close = float(df["close"].iloc[-2])
                    last_high  = float(df["high"].iloc[-2])
                    last_low   = float(df["low"].iloc[-2])
                    pos        = state["position"]
                    ep         = state["entry_price"]
                    trail_stp  = state["trail_stop"]
                    trail_act  = state["trail_activated"]

                    log.info(f"  close={last_close:.4f} pos={pos or 'flat'} "
                             f"{'LONG' if long_sig else 'SHORT' if short_sig else 'no signal'}")

                    if pos is not None and ep is not None:
                        dist = _bot_get_trail_dist(ep, atr_val, cfg)
                        bars = [(last_high, last_low, last_close)]
                        closed=False; exit_p=None; exit_why=None
                        for (ih,il,ic) in bars:
                            if pos=="long":
                                if not trail_act:
                                    if ih>=ep+dist: trail_act=True; trail_stp=ih-dist
                                elif ih-dist>(trail_stp or 0): trail_stp=ih-dist
                            else:
                                if not trail_act:
                                    if il<=ep-dist: trail_act=True; trail_stp=il+dist
                                elif trail_stp is None or il+dist<trail_stp: trail_stp=il+dist
                        if trail_act and trail_stp is not None:
                            if pos=="long" and last_low<=trail_stp:  exit_p,exit_why,closed=trail_stp,"TSL",True
                            elif pos=="short" and last_high>=trail_stp: exit_p,exit_why,closed=trail_stp,"TSL",True
                        if not closed and ep:
                            tp = ep*(1+cfg["tp_perc"]/100) if pos=="long" else ep*(1-cfg["tp_perc"]/100)
                            if pos=="long" and last_high>=tp: exit_p,exit_why,closed=tp,"TP",True
                            elif pos=="short" and last_low<=tp: exit_p,exit_why,closed=tp,"TP",True
                        if not closed:
                            if pos=="long" and short_sig: exit_p,exit_why,closed=last_close,"Signal",True
                            elif pos=="short" and long_sig: exit_p,exit_why,closed=last_close,"Signal",True
                        if closed:
                            pnl = (exit_p-ep)/ep*100 if pos=="long" else (ep-exit_p)/ep*100
                            log.info(f"  EXIT {pos.upper()} @ {exit_p:.4f} {exit_why} P&L:{pnl:+.2f}%")
                            _bot_send_webhook("sell" if pos=="long" else "buy", exit_p,
                                              f"close {pos} | {exit_why} | P&L {pnl:+.2f}%", cfg)
                            state.update({"position":None,"entry_price":None,"entry_time":None,
                                          "trail_stop":None,"trail_activated":False})
                            trail_stp=None; trail_act=False; pos=None

                    if pos is None:
                        if long_sig:
                            log.info(f"  ENTRY LONG @ {entry_open:.4f}")
                            _bot_send_webhook("buy", entry_open, "open long", cfg)
                            state.update({"position":"long","entry_price":entry_open,
                                          "entry_time":now.isoformat(),"trail_stop":None,"trail_activated":False})
                        elif short_sig:
                            log.info(f"  ENTRY SHORT @ {entry_open:.4f}")
                            _bot_send_webhook("sell", entry_open, "open short", cfg)
                            state.update({"position":"short","entry_price":entry_open,
                                          "entry_time":now.isoformat(),"trail_stop":None,"trail_activated":False})

                    state["trail_stop"] = trail_stp
                    state["trail_activated"] = trail_act
                    state["last_atr"] = atr_val
                    save_bot_state(state)
                except Exception as e:
                    log.error(f"Bot bar-close error: {e}")

            time.sleep(20)
        except Exception as e:
            log.error(f"Bot outer loop error: {e}")
            time.sleep(60)

    log.info("Bot thread stopped")

def start_bot_thread():
    global _bot_thread, _bot_stop_event
    with _bot_lock:
        if _bot_thread and _bot_thread.is_alive():
            return
        _bot_stop_event = threading.Event()
        _bot_thread = threading.Thread(target=_bot_loop, args=(_bot_stop_event,), daemon=True)
        _bot_thread.start()

def stop_bot_thread():
    global _bot_thread, _bot_stop_event
    with _bot_lock:
        if _bot_stop_event:
            _bot_stop_event.set()

# ─────────────────────────────────────────────────────────────────
# FLASK ROUTES — static
# ─────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return send_from_directory(".", "index.html")

# ─────────────────────────────────────────────────────────────────
# API — DATA
# ─────────────────────────────────────────────────────────────────
@app.route("/api/cache/status")
@login_required
def api_cache_status():
    exchange  = request.args.get("exchange", "bybit")
    symbol    = request.args.get("symbol", "XMR/USDT:USDT")
    timeframe = request.args.get("timeframe", "30m")
    df = load_cache(exchange, symbol, timeframe)
    if df is None:
        return jsonify({"exists": False, "count": 0, "first": None, "last": None})
    return jsonify({
        "exists": True, "count": len(df),
        "first":  df["timestamp"].iloc[0].isoformat(),
        "last":   df["timestamp"].iloc[-1].isoformat(),
    })

_fetch_progress = {}

@app.route("/api/data/fetch", methods=["POST"])
@login_required
def api_data_fetch():
    body      = request.json or {}
    exchange  = body.get("exchange", "bybit")
    symbol    = body.get("symbol", "XMR/USDT:USDT")
    timeframe = body.get("timeframe", "30m")
    force_full= body.get("force_full", False)

    def do_fetch():
        key = f"{exchange}_{symbol}_{timeframe}"
        _fetch_progress[key] = {"status": "running", "message": "Connecting..."}
        try:
            if force_full:
                p = cache_path(exchange, symbol, timeframe)
                if p.exists(): p.unlink()
            _fetch_progress[key]["message"] = "Fetching candles..."
            df, new_count, total = fetch_ohlcv_update(exchange, symbol, timeframe)
            _fetch_progress[key] = {
                "status": "done", "new": new_count, "total": total,
                "first": df["timestamp"].iloc[0].isoformat(),
                "last":  df["timestamp"].iloc[-1].isoformat(),
            }
        except Exception as e:
            _fetch_progress[key] = {"status": "error", "message": str(e)}

    t = threading.Thread(target=do_fetch, daemon=True)
    t.start()
    return jsonify({"ok": True})

@app.route("/api/data/fetch/status")
@login_required
def api_fetch_status():
    exchange  = request.args.get("exchange", "bybit")
    symbol    = request.args.get("symbol", "XMR/USDT:USDT")
    timeframe = request.args.get("timeframe", "30m")
    key = f"{exchange}_{symbol}_{timeframe}"
    return jsonify(_fetch_progress.get(key, {"status": "idle"}))

# ─────────────────────────────────────────────────────────────────
# API — BACKTEST
# ─────────────────────────────────────────────────────────────────
_backtest_jobs = {}

@app.route("/api/backtest/run", methods=["POST"])
@login_required
def api_backtest_run():
    cfg = request.json or {}
    h   = settings_hash(cfg)

    cached = load_result(h)
    if cached and not cfg.get("force_rerun"):
        return jsonify({"ok": True, "hash": h, "cached": True})

    def do_backtest(h, cfg):
        _backtest_jobs[h] = {"status": "running"}
        try:
            exchange  = cfg.get("exchange", "bybit")
            symbol    = cfg.get("symbol", "XMR/USDT:USDT")
            timeframe = cfg.get("timeframe", "30m")
            params    = cfg.get("params", DEFAULT_BACKTEST_CONFIG["params"])
            capital   = float(cfg.get("capital", 1000))
            fee_pct   = float(cfg.get("fee_pct", 0.055))

            df_base = load_cache(exchange, symbol, timeframe)
            if df_base is None or len(df_base) == 0:
                _backtest_jobs[h] = {"status": "error", "message": "No data cached. Fetch data first."}
                return

            df_intra = None
            if cfg.get("use_intrabar"):
                df_intra = load_cache(exchange, symbol, cfg.get("intrabar_tf", "1m"))

            run_params = {**params, "tick_size": cfg.get("tick_size",0.01),
                          "slippage_pct": cfg.get("slippage_pct",0.0143),
                          "compounding": cfg.get("compounding",True),
                          "tsl_mode": cfg.get("tsl_mode","intrabar")}

            trades, final_cap, equity_curve = run_ema_crossover(df_base, df_intra, run_params, fee_pct, capital)

            if not trades:
                _backtest_jobs[h] = {"status": "error", "message": "No trades generated."}
                return

            df_t = pd.DataFrame(trades)
            wins = int((df_t["pnl_pct"]>0).sum())
            total_ret = (final_cap-capital)/capital*100
            max_dd = _compute_dd(equity_curve, capital)
            gw = float(df_t[df_t["pnl_usdt"]>0]["pnl_usdt"].sum())
            gl = float(df_t[df_t["pnl_usdt"]<=0]["pnl_usdt"].abs().sum())
            pf = round(gw/gl, 3) if gl>0 else 99.0

            # Buy and hold
            bh_start = float(df_base["close"].iloc[0])
            bh_end   = float(df_base["close"].iloc[-1])
            bh_dates = [str(ts)[:10] for ts in df_base["timestamp"].tolist()]
            bh_equity = [round(capital*(float(p_/bh_start)),2) for p_ in df_base["close"].tolist()]

            # Equity curve aligned to trade exit times
            equity_dates = [str(t["exit_time"])[:16] for t in trades]
            equity_vals  = [round(capital+float(pd.DataFrame(trades[:i+1])["pnl_usdt"].sum()),2)
                            for i in range(len(trades))]

            # Drawdown series
            cumul = capital + df_t["pnl_usdt"].cumsum()
            peak  = cumul.expanding().max()
            dd_series = [round(float(v),2) for v in ((cumul-peak)/peak*100).tolist()]

            result = {
                "hash": h,
                "summary": {
                    "total_trades": len(trades),
                    "wins": wins,
                    "losses": len(trades)-wins,
                    "win_rate": round(wins/len(trades)*100,2),
                    "total_return": round(total_ret,2),
                    "final_capital": round(final_cap,2),
                    "initial_capital": capital,
                    "max_dd": round(max_dd,2),
                    "profit_factor": pf,
                    "total_fees": round(float(df_t["fee_usdt"].sum()),2),
                    "avg_win": round(float(df_t[df_t["pnl_pct"]>0]["pnl_pct"].mean()) if wins>0 else 0,3),
                    "avg_loss": round(float(df_t[df_t["pnl_pct"]<=0]["pnl_pct"].mean()) if len(trades)-wins>0 else 0,3),
                    "exit_tp": int(df_t["exit_reason"].eq("TP").sum()),
                    "exit_tsl": int(df_t["exit_reason"].eq("TSL").sum()),
                    "exit_sl": int(df_t["exit_reason"].eq("SL").sum()),
                    "exit_signal": int(df_t["exit_reason"].eq("Signal").sum()),
                    "data_from": df_base["timestamp"].iloc[0].isoformat(),
                    "data_to":   df_base["timestamp"].iloc[-1].isoformat(),
                },
                "equity_curve":   {"dates": equity_dates, "values": equity_vals},
                "drawdown_series": {"dates": equity_dates, "values": dd_series},
                "bh_curve":       {"dates": bh_dates, "values": bh_equity},
                "trades": trades[-200:],  # last 200 trades for UI
                "all_trades_count": len(trades),
                "params": run_params,
            }
            save_result(h, result)
            _backtest_jobs[h] = {"status": "done", "hash": h}
        except Exception as e:
            import traceback
            _backtest_jobs[h] = {"status": "error", "message": str(e), "trace": traceback.format_exc()}

    t = threading.Thread(target=do_backtest, args=(h, cfg), daemon=True)
    t.start()
    return jsonify({"ok": True, "hash": h, "cached": False})

@app.route("/api/backtest/status/<h>")
@login_required
def api_backtest_status(h):
    return jsonify(_backtest_jobs.get(h, {"status": "idle"}))

@app.route("/api/backtest/result/<h>")
@login_required
def api_backtest_result(h):
    r = load_result(h)
    if r is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(r)

@app.route("/api/backtest/config")
@login_required
def api_backtest_config():
    return jsonify(DEFAULT_BACKTEST_CONFIG)

# ─────────────────────────────────────────────────────────────────
# API — OPTIMIZER
# ─────────────────────────────────────────────────────────────────
_opt_jobs = {}

@app.route("/api/optimizer/run", methods=["POST"])
@login_required
def api_optimizer_run():
    body = request.json or {}
    job_id = settings_hash(body)

    def do_opt(job_id, body):
        import multiprocessing as mp
        from itertools import product as iproduct
        _opt_jobs[job_id] = {"status": "running", "progress": 0, "total": 0, "results": []}
        try:
            cfg        = body.get("config", {})
            opt_ranges = body.get("ranges", {})
            dd_limit   = float(body.get("dd_limit", -60))
            wf_split   = float(body.get("wf_split", 0.7))
            use_cache  = body.get("use_cache", True)

            exchange  = cfg.get("exchange","bybit")
            symbol    = cfg.get("symbol","XMR/USDT:USDT")
            timeframe = cfg.get("timeframe","30m")
            capital   = float(cfg.get("capital",1000))
            fee_pct   = float(cfg.get("fee_pct",0.055))
            params    = cfg.get("params", DEFAULT_BACKTEST_CONFIG["params"])

            df_base = load_cache(exchange, symbol, timeframe)
            if df_base is None:
                _opt_jobs[job_id] = {"status":"error","message":"No data. Fetch first."}
                return

            df_intra = None
            if cfg.get("use_intrabar"):
                df_intra = load_cache(exchange, symbol, cfg.get("intrabar_tf","1m"))

            base_params = {**params, "tick_size":cfg.get("tick_size",0.01),
                           "slippage_pct":cfg.get("slippage_pct",0.0143),
                           "compounding":cfg.get("compounding",True),
                           "tsl_mode":cfg.get("tsl_mode","intrabar")}

            wf_idx  = int(len(df_base)*wf_split)
            wf_time = df_base.iloc[wf_idx]["timestamp"]
            df_opt  = df_base.iloc[:wf_idx].reset_index(drop=True)
            df_hold = df_base.iloc[wf_idx:].reset_index(drop=True)
            intra_opt = intra_hold = None
            if df_intra is not None:
                intra_opt  = df_intra[df_intra["timestamp"]<wf_time].reset_index(drop=True)
                intra_hold = df_intra[df_intra["timestamp"]>=wf_time].reset_index(drop=True)

            keys   = list(opt_ranges.keys())
            ranges = [opt_ranges[k] for k in keys]
            combos = list(iproduct(*ranges))
            total  = len(combos)
            _opt_jobs[job_id]["total"] = total

            cols = ["timestamp","open","high","low","close","volume"]
            base_rec  = df_opt[cols].assign(timestamp=df_opt["timestamp"].astype(str)).to_dict("records")
            intra_rec = None
            if intra_opt is not None and len(intra_opt)>0:
                intra_rec = intra_opt[cols].assign(timestamp=intra_opt["timestamp"].astype(str)).to_dict("records")

            # Check cache for already-done combos
            opt_cache_key = settings_hash({"exchange":exchange,"symbol":symbol,"timeframe":timeframe,
                                           "keys":keys,"wf_split":wf_split,"base_params":base_params})
            opt_cache_file = RESULT_DIR / f"opt_{opt_cache_key}.json"
            cached_results = {}
            if use_cache and opt_cache_file.exists():
                with open(opt_cache_file) as f:
                    cached_results = json.load(f)

            args_list = [(c, keys, base_params, base_rec, intra_rec, fee_pct, capital) for c in combos]
            n_cores = max(1, mp.cpu_count()-1)
            results = []
            done = 0

            with mp.Pool(processes=n_cores) as pool:
                for result in pool.imap_unordered(_opt_worker, args_list, chunksize=max(1,total//(n_cores*8))):
                    done += 1
                    _opt_jobs[job_id]["progress"] = done
                    if result is not None:
                        results.append(result)

            results.sort(key=lambda x: x["score"], reverse=True)

            # Run OOS on top 20
            oos_results = []
            if results and intra_hold is not None:
                hold_rec = df_hold[cols].assign(timestamp=df_hold["timestamp"].astype(str)).to_dict("records")
                hold_intra_rec = intra_hold[cols].assign(timestamp=intra_hold["timestamp"].astype(str)).to_dict("records") if len(intra_hold)>0 else None
            else:
                hold_rec = df_hold[cols].assign(timestamp=df_hold["timestamp"].astype(str)).to_dict("records")
                hold_intra_rec = None
            top20 = results[:20]
            for r in top20:
                p_oos = {**base_params, **{k: r[k] for k in keys}}
                df_h2 = pd.DataFrame(hold_rec)
                df_h2["timestamp"] = pd.to_datetime(df_h2["timestamp"], utc=True)
                df_hi2 = None
                if hold_intra_rec:
                    df_hi2 = pd.DataFrame(hold_intra_rec)
                    df_hi2["timestamp"] = pd.to_datetime(df_hi2["timestamp"], utc=True)
                t_o, c_o, eq_o = run_ema_crossover(df_h2, df_hi2, p_oos, fee_pct, capital)
                if t_o:
                    df_o = pd.DataFrame(t_o)
                    oos_ret = (c_o-capital)/capital*100
                    oos_wr  = (df_o["pnl_pct"]>0).sum()/len(t_o)*100
                    oos_dd  = _compute_dd(eq_o, capital)
                    holds   = oos_ret>0 and oos_dd>-60
                    oos_results.append({**r,"oos_ret":round(oos_ret,2),"oos_wr":round(oos_wr,2),
                                        "oos_dd":round(oos_dd,2),"oos_n":len(t_o),"holds":holds})
                else:
                    oos_results.append({**r,"oos_ret":0,"oos_wr":0,"oos_dd":0,"oos_n":0,"holds":False})

            _opt_jobs[job_id] = {
                "status": "done",
                "total": total, "progress": total,
                "results": results[:50],
                "oos_results": oos_results,
                "keys": keys,
                "opt_period": {"from": df_opt["timestamp"].iloc[0].isoformat(),
                               "to":   df_opt["timestamp"].iloc[-1].isoformat(),
                               "bars": len(df_opt)},
                "hold_period": {"from": df_hold["timestamp"].iloc[0].isoformat(),
                                "to":   df_hold["timestamp"].iloc[-1].isoformat(),
                                "bars": len(df_hold)},
            }
            # Save cache
            with open(opt_cache_file, "w") as f:
                json.dump(results[:200], f)

        except Exception as e:
            import traceback
            _opt_jobs[job_id] = {"status":"error","message":str(e),"trace":traceback.format_exc()}

    t = threading.Thread(target=do_opt, args=(job_id, body), daemon=True)
    t.start()
    return jsonify({"ok": True, "job_id": job_id})

@app.route("/api/optimizer/status/<job_id>")
@login_required
def api_optimizer_status(job_id):
    j = _opt_jobs.get(job_id, {"status":"idle"})
    # Don't send results in poll, only progress
    return jsonify({k: v for k, v in j.items() if k != "results" and k != "oos_results"})

@app.route("/api/optimizer/result/<job_id>")
@login_required
def api_optimizer_result(job_id):
    j = _opt_jobs.get(job_id)
    if not j:
        return jsonify({"error":"Not found"}), 404
    return jsonify(j)

# ─────────────────────────────────────────────────────────────────
# API — LIVE BOT
# ─────────────────────────────────────────────────────────────────
@app.route("/api/bot/config", methods=["GET", "POST"])
@login_required
def api_bot_config():
    if request.method == "POST":
        new_cfg = request.json or {}
        cfg = load_bot_config()
        cfg.update(new_cfg)
        save_bot_config(cfg)
        # If enabling, ensure thread is running
        if cfg.get("enabled"):
            start_bot_thread()
        return jsonify({"ok": True, "config": cfg})
    return jsonify(load_bot_config())

@app.route("/api/bot/status")
@login_required
def api_bot_status():
    state = load_bot_state()
    cfg   = load_bot_config()
    global _bot_thread
    running = _bot_thread is not None and _bot_thread.is_alive()
    return jsonify({
        "running": running,
        "enabled": cfg.get("enabled", False),
        "state":   state,
        "symbol":  cfg.get("symbol"),
        "timeframe": cfg.get("timeframe"),
    })

@app.route("/api/bot/signals")
@login_required
def api_bot_signals():
    limit = int(request.args.get("limit", 50))
    if not SIGNALS_FILE.exists():
        return jsonify({"signals": []})
    rows = []
    with open(SIGNALS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return jsonify({"signals": rows[-limit:][::-1]})

@app.route("/api/bot/start", methods=["POST"])
@login_required
def api_bot_start():
    cfg = load_bot_config()
    cfg["enabled"] = True
    save_bot_config(cfg)
    start_bot_thread()
    return jsonify({"ok": True})

@app.route("/api/bot/stop", methods=["POST"])
@login_required
def api_bot_stop():
    cfg = load_bot_config()
    cfg["enabled"] = False
    save_bot_config(cfg)
    return jsonify({"ok": True})

# ─────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Auto-start bot thread if it was enabled
    cfg = load_bot_config()
    if cfg.get("enabled"):
        start_bot_thread()
    app.run(host="0.0.0.0", port=5000, debug=False)
