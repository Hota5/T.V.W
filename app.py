"""
EMA Crossover Web Platform v2
Flask backend — all features, unified cache, persistent jobs, fixed bugs.
"""

import os, sys, json, time, threading, hashlib, logging, csv, io, secrets, glob
from pathlib import Path
from datetime import datetime, timezone, timedelta
from functools import wraps
from itertools import product as iproduct
from flask import (Flask, jsonify, request, send_from_directory,
                   Response, session, redirect)
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder=".")
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.permanent_session_lifetime = timedelta(days=30)

BASE_DIR        = Path(__file__).parent
CACHE_DIR       = BASE_DIR / "candle_cache"
RESULT_DIR      = BASE_DIR / "backtest_results"
OPT_DIR         = BASE_DIR / "opt_jobs"
PRESET_FILE     = BASE_DIR / "presets.json"
BOT_STATE_FILE  = BASE_DIR / "live_bot_state.json"
BOT_CONFIG_FILE = BASE_DIR / "live_bot_config.json"
SIGNALS_FILE    = BASE_DIR / "signals.csv"
AUTH_FILE       = BASE_DIR / "auth.json"
BYBIT_DATA_DIR  = BASE_DIR / "bybit_data"
TICK_DIR        = BASE_DIR / "tick_cache"

for d in [CACHE_DIR, RESULT_DIR, OPT_DIR, BYBIT_DATA_DIR, TICK_DIR]:
    d.mkdir(exist_ok=True)

log = logging.getLogger("webapp")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(BASE_DIR/"app.log"), logging.StreamHandler()])

# AUTH
def _hash_pw(pw): return hashlib.sha256(pw.strip().encode()).hexdigest()
def load_auth():
    if AUTH_FILE.exists():
        with open(AUTH_FILE) as f: return json.load(f)
    d = {"username":"admin","password_hash":_hash_pw("changeme123")}
    AUTH_FILE.write_text(json.dumps(d, indent=2)); return d
def check_creds(u, p):
    a = load_auth()
    return u.strip()==a["username"] and _hash_pw(p)==a["password_hash"]
def login_required(f):
    @wraps(f)
    def dec(*a,**kw):
        if not session.get("logged_in"):
            return (jsonify({"error":"Unauthorized"}),401) if request.path.startswith("/api/") else redirect("/login")
        return f(*a,**kw)
    return dec

@app.route("/login", methods=["GET","POST"])
def login():
    err=""
    if request.method=="POST":
        if check_creds(request.form.get("username",""), request.form.get("password","")):
            session["logged_in"]=True; session["username"]=request.form["username"]
            session.permanent=True; return redirect("/")
        err="Invalid username or password."
    return _login_html(err)

@app.route("/logout")
def logout():
    session.clear(); return redirect("/login")

@app.route("/api/auth/change-password", methods=["POST"])
@login_required
def api_change_password():
    b=request.json or {}
    if not check_creds(session.get("username",""), b.get("current","")): return jsonify({"error":"Current password wrong"}),400
    if len(b.get("new",""))<8: return jsonify({"error":"Min 8 characters"}),400
    a=load_auth(); a["password_hash"]=_hash_pw(b["new"])
    AUTH_FILE.write_text(json.dumps(a,indent=2)); return jsonify({"ok":True})

@app.route("/api/auth/change-username", methods=["POST"])
@login_required
def api_change_username():
    b=request.json or {}
    if not check_creds(session.get("username",""), b.get("password","")): return jsonify({"error":"Password wrong"}),400
    u=b.get("username","").strip()
    if len(u)<3: return jsonify({"error":"Min 3 chars"}),400
    a=load_auth(); a["username"]=u; AUTH_FILE.write_text(json.dumps(a,indent=2))
    session["username"]=u; return jsonify({"ok":True})

def _login_html(err=""):
    e=f"<div class='error'>{err}</div>" if err else ""
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>EMAX</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#080a0d;color:#dce3ee;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;justify-content:center;min-height:100vh}}
.box{{width:360px;background:#0e1117;border:1px solid #1a2030;padding:44px}}
.logo{{font-family:'Bebas Neue',sans-serif;font-size:2.4rem;letter-spacing:4px;color:#f0b90b;margin-bottom:6px}}
.logo span{{color:#00d4ff}}.sub{{font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#3d4d66;margin-bottom:36px}}
label{{display:block;font-size:0.65rem;letter-spacing:1.5px;text-transform:uppercase;color:#3d4d66;margin-bottom:5px;margin-top:18px}}
input{{width:100%;background:#131820;border:1px solid #222d40;color:#dce3ee;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;padding:11px 12px;outline:none;border-radius:2px;transition:border .15s}}
input:focus{{border-color:#00d4ff}}.error{{background:rgba(246,70,93,.08);border:1px solid rgba(246,70,93,.25);color:#f6465d;font-size:0.75rem;padding:10px 12px;margin-top:14px;border-radius:2px}}
button{{width:100%;margin-top:28px;background:#f0b90b;border:none;color:#000;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;letter-spacing:1.5px;text-transform:uppercase;font-weight:600;padding:13px;cursor:pointer;border-radius:2px}}
button:hover{{background:#d4a30a}}.hint{{font-size:0.65rem;color:#3d4d66;margin-top:18px;text-align:center}}</style></head>
<body><div class="box"><div class="logo">EMA<span>X</span></div><div class="sub">Trading Platform</div>
<form method="POST" action="/login">
<label>Username</label><input type="text" name="username" autofocus autocomplete="username">
<label>Password</label><input type="password" name="password" autocomplete="current-password">
{e}<button type="submit">Sign In</button></form>
<div class="hint">Default: admin / changeme123</div></div></body></html>"""

# DEFAULTS
DEFAULT_PARAMS = dict(fast_length=9,slow_length=21,use_trailing=True,use_all_exits=False,
    tp_perc=2.0,sl_perc=1.0,trail_perc=1.0,use_atr_tsl=True,atr_length=9,atr_multiplier=10.0,
    use_tiered_tsl=False,tier1_profit=5.0,tier1_tsl=3.0,tier2_profit=10.0,tier2_tsl=2.0,tier3_tsl=1.0,
    use_vol_filter=False,vol_multiplier=1.5,use_adx_filter=False,adx_length=14,adx_threshold=25,
    tick_size=0.01,slippage_pct=0.0143,tsl_mode="intrabar")

DEFAULT_BOT_CONFIG = {**DEFAULT_PARAMS,"exchange":"bybit","symbol":"XMR/USDT:USDT","timeframe":"30m",
    "tsl_mode":"barclose","webhook_url":"","signal_type":"","pionex_symbol":"XMR_USDT","contracts":"0.1","enabled":False}

# INDICATORS
def compute_ema(s, n): return s.ewm(span=n, adjust=False).mean()
def compute_atr(df, n):
    h,l,c=df["high"],df["low"],df["close"]
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def compute_adx(df, n):
    h,l,c=df["high"],df["low"],df["close"]
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    dmp=np.where((h-h.shift(1))>(l.shift(1)-l),(h-h.shift(1)).clip(lower=0),0)
    dmm=np.where((l.shift(1)-l)>(h-h.shift(1)),(l.shift(1)-l).clip(lower=0),0)
    trs=pd.Series(tr).ewm(alpha=1/n,adjust=False).mean()
    dip=100*pd.Series(dmp).ewm(alpha=1/n,adjust=False).mean()/trs
    dim=100*pd.Series(dmm).ewm(alpha=1/n,adjust=False).mean()/trs
    dx=100*(dip-dim).abs()/(dip+dim)
    return dx.ewm(alpha=1/n,adjust=False).mean()
def get_trail_dist(direction, ref, ep, atr, p):
    tick=p.get("tick_size",0.01)
    pct=(ref-ep)/ep*100 if direction=="long" else (ep-ref)/ep*100
    if p.get("use_atr_tsl") and atr>0:
        if p.get("use_tiered_tsl"):
            m=p["tier1_tsl"] if pct<p["tier1_profit"] else p["tier2_tsl"] if pct<p["tier2_profit"] else p["tier3_tsl"]
            return atr*m*tick
        return atr*p["atr_multiplier"]*tick
    if p.get("use_tiered_tsl"):
        t=p["tier1_tsl"] if pct<p["tier1_profit"] else p["tier2_tsl"] if pct<p["tier2_profit"] else p["tier3_tsl"]
        return t/100*ep
    return p["trail_perc"]/100*ep

# ─────────────────────────────────────────────────────────────────
# TICK CACHE  (parquet, graceful fallback if pyarrow not installed)
# ─────────────────────────────────────────────────────────────────
try:
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    _HAVE_PARQUET = True
except ImportError:
    _HAVE_PARQUET = False
    log.warning("pyarrow not installed — tick mode disabled. Run: pip install pyarrow")

def _tick_path(ex, sym):
    s = sym.replace("/","_").replace(":","_")
    return TICK_DIR / f"{ex}_{s}_ticks.parquet"

def tick_cache_info(ex, sym):
    if not _HAVE_PARQUET:
        return {"exists": False, "error": "pyarrow not installed"}
    p = _tick_path(ex, sym)
    if not p.exists():
        return {"exists": False}
    try:
        meta = _pq.read_metadata(str(p))
        df_ts = _pq.read_table(str(p), columns=["timestamp"]).to_pandas()
        size_mb = round(p.stat().st_size / 1048576, 1)
        return {"exists": True, "rows": meta.num_rows, "size_mb": size_mb,
                "first": str(df_ts["timestamp"].min())[:16],
                "last":  str(df_ts["timestamp"].max())[:16]}
    except Exception as e:
        return {"exists": True, "error": str(e)}

def save_ticks(df, ex, sym):
    if not _HAVE_PARQUET:
        raise RuntimeError("pyarrow not installed")
    p = _tick_path(ex, sym)
    if p.exists():
        existing = _pq.read_table(str(p)).to_pandas()
        df = pd.concat([existing, df], ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    table = _pa.Table.from_pandas(df, preserve_index=False)
    _pq.write_table(table, str(p), compression="snappy")
    return df

def build_tick_lookup(ex, sym, bar_timestamps):
    if not _HAVE_PARQUET:
        return None
    p = _tick_path(ex, sym)
    if not p.exists():
        return None
    try:
        table = _pq.read_table(str(p), columns=["timestamp","price"])
        df = table.to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        ts_arr = df["timestamp"].values
        px_arr = df["price"].values
        lookup = {}
        bar_ts = sorted(bar_timestamps)
        for i, bar_start in enumerate(bar_ts):
            bar_end = bar_ts[i+1] if i+1 < len(bar_ts) else bar_start + pd.Timedelta("30min")
            mask = (ts_arr >= bar_start) & (ts_arr < bar_end)
            prices = px_arr[mask]
            if len(prices) > 0:
                lookup[bar_start] = prices
        return lookup if lookup else None
    except Exception as e:
        log.warning(f"Tick lookup build error: {e}")
        return None

def build_intra_lookup(df_base, df_intra):
    # Returns dict: bar_index -> list of (high, low, close, timestamp)
    lookup={}; bt=df_base["timestamp"].values; it=df_intra["timestamp"].values
    ih=df_intra["high"].values; il=df_intra["low"].values; ic=df_intra["close"].values
    n=len(df_base)
    fallback_ts=None  # no intra timestamp for fallback — use bar close ts
    for i in range(n):
        ts=bt[i]; te=bt[i+1] if i+1<n else bt[i]+np.timedelta64(1,"ns")*int(1e18)
        idx=np.where((it>=ts)&(it<te))[0]
        if len(idx)>0:
            lookup[i]=list(zip(ih[idx],il[idx],ic[idx],it[idx]))
        else:
            r=df_base.iloc[i]
            lookup[i]=[(r["high"],r["low"],r["close"],bt[i])]
    return lookup
def _compute_dd(eq, cap):
    a=np.array(eq); pk=np.maximum.accumulate(a); dd=(a-pk)/pk*100
    return float(dd.min()) if len(dd) else 0.0

# BACKTEST ENGINE
def run_ema_crossover(df_base, df_intra, p, fee_pct, capital, tick_lookup=None):
    init_cap=capital
    df=df_base.copy().reset_index(drop=True)
    df["fast_ema"]=compute_ema(df["close"],p["fast_length"])
    df["slow_ema"]=compute_ema(df["close"],p["slow_length"])
    df["atr"]=compute_atr(df,p["atr_length"])
    if p.get("use_vol_filter"): df["avg_vol"]=df["volume"].rolling(20).mean()
    if p.get("use_adx_filter"): df["adx"]=compute_adx(df,p["adx_length"])
    fe=df["fast_ema"].values; se=df["slow_ema"].values; n=len(df)
    cu=np.zeros(n,bool); cd=np.zeros(n,bool)
    cu[2:]=(fe[1:-1]>se[1:-1])&(fe[:-2]<=se[:-2])
    cd[2:]=(fe[1:-1]<se[1:-1])&(fe[:-2]>=se[:-2])
    vol_ok=np.ones(n,bool); adx_ok=np.ones(n,bool)
    if p.get("use_vol_filter"):
        av=df["avg_vol"].values; vv=df["volume"].values
        with np.errstate(invalid="ignore"): vol_ok=vv>av*p["vol_multiplier"]
        vol_ok[np.isnan(av)]=False
    if p.get("use_adx_filter"):
        adx_ok=df["adx"].values>p["adx_threshold"]
    ls=cu&vol_ok&adx_ok; ss=cd&vol_ok&adx_ok
    intra=build_intra_lookup(df,df_intra) if df_intra is not None and len(df_intra)>0 else None
    use_ticks = tick_lookup is not None
    trades=[]; pos=None; eq=[]; slip=p.get("slippage_pct",0.0143)/100
    for i in range(2,n):
        row=df.iloc[i]; atr=float(row["atr"]) if not np.isnan(row["atr"]) else 0.0; close=float(row["close"])
        if pos:
            ep=pos["entry_price"]; d=pos["direction"]
            bar_ts = df["timestamp"].iloc[i] if "timestamp" in df.columns else None
            bar_ts_val = df["timestamp"].iloc[i]
            if use_ticks and bar_ts is not None and bar_ts in tick_lookup:
                tick_prices = tick_lookup[bar_ts]
                # (high, low, close, timestamp) — each tick is its own point
                bars = [(float(px), float(px), float(px), bar_ts) for px in tick_prices]
                if not bars: bars = [(float(row["high"]),float(row["low"]),close,bar_ts_val)]
            elif intra:
                # lookup already returns (h, l, c, ts) tuples
                bars=intra.get(i,[(row["high"],row["low"],close,bar_ts_val)])
            else:
                bars=[(float(row["high"]),float(row["low"]),close,bar_ts_val)]
            closed=False; xp=None; xr=None; exit_ts=bar_ts_val
            for (ih,il,ic,its) in bars:
                if closed: break
                if d=="long":
                    if ih>=pos["tp"]: xp,xr,closed,exit_ts=pos["tp"],"TP",True,its
                    elif not pos["trail_active"] and il<=pos["sl"]: xp,xr,closed,exit_ts=pos["sl"],"SL",True,its
                else:
                    if il<=pos["tp"]: xp,xr,closed,exit_ts=pos["tp"],"TP",True,its
                    elif not pos["trail_active"] and ih>=pos["sl"]: xp,xr,closed,exit_ts=pos["sl"],"SL",True,its
            if not closed and pos["trail_active"]:
                dist=get_trail_dist(d,close,ep,atr,p)
                for (ih2,il2,ic2,its2) in bars:
                    if d=="long":
                        if not pos["trail_on"]:
                            if ih2>=ep+dist: pos["trail_on"]=True; pos["tsl"]=ih2-dist
                        elif ih2-dist>pos["tsl"]: pos["tsl"]=ih2-dist
                    else:
                        if not pos["trail_on"]:
                            if il2<=ep-dist: pos["trail_on"]=True; pos["tsl"]=il2+dist
                        elif pos["tsl"] is None or il2+dist<pos["tsl"]: pos["tsl"]=il2+dist
                if pos["trail_on"] and pos["tsl"] is not None:
                    if p.get("tsl_mode","intrabar")=="barclose":
                        if d=="long" and close<=pos["tsl"]: xp,xr,closed,exit_ts=pos["tsl"],"TSL",True,bar_ts_val
                        elif d=="short" and close>=pos["tsl"]: xp,xr,closed,exit_ts=pos["tsl"],"TSL",True,bar_ts_val
                    else:
                        for (ih2,il2,ic2,its2) in bars:
                            if closed: break
                            if d=="long" and il2<=pos["tsl"]: xp,xr,closed,exit_ts=pos["tsl"],"TSL",True,its2
                            elif d=="short" and ih2>=pos["tsl"]: xp,xr,closed,exit_ts=pos["tsl"],"TSL",True,its2
            if not closed:
                if d=="long" and ss[i]: xp,xr,closed,exit_ts=close,"Signal",True,bar_ts_val
                elif d=="short" and ls[i]: xp,xr,closed,exit_ts=close,"Signal",True,bar_ts_val
            if closed:
                xp=xp*(1-slip) if d=="long" else xp*(1+slip)
                fo=ep*pos["qty"]*fee_pct/100; fc=xp*pos["qty"]*fee_pct/100
                gross=(xp-ep)*pos["qty"] if d=="long" else (ep-xp)*pos["qty"]
                net=gross-fo-fc
                _exit_ts = exit_ts if exit_ts is not None else row["timestamp"]
                if hasattr(_exit_ts,"item"): _exit_ts=pd.Timestamp(_exit_ts)  # numpy -> pandas
                trades.append({"entry_time":pos["et"].isoformat() if hasattr(pos["et"],"isoformat") else str(pos["et"]),
                    "exit_time":_exit_ts.isoformat() if hasattr(_exit_ts,"isoformat") else str(_exit_ts),
                    "direction":d,"entry_price":round(ep,6),"exit_price":round(xp,6),"exit_reason":xr,
                    "pnl_usdt":round(net,4),"pnl_pct":round(net/pos["cap"]*100,4),"fee_usdt":round(fo+fc,4),"qty":round(pos["qty"],6)})
                capital+=net; pos=None
        if pos:
            unr=(close-pos["entry_price"])*pos["qty"] if pos["direction"]=="long" else (pos["entry_price"]-close)*pos["qty"]
            eq.append(capital+unr)
        else: eq.append(capital)
        if pos is None:
            d_=None
            if ls[i-1]: d_="long"
            elif ss[i-1]: d_="short"
            if d_:
                raw=float(row["open"]); ep=raw*(1+slip) if d_=="long" else raw*(1-slip)
                cap=capital if p.get("compounding",True) else init_cap
                qty=cap/ep
                tp=ep*(1+p["tp_perc"]/100) if d_=="long" else ep*(1-p["tp_perc"]/100)
                sl=ep*(1-p["sl_perc"]/100) if d_=="long" else ep*(1+p["sl_perc"]/100)
                pos={"direction":d_,"et":row["timestamp"],"entry_price":ep,"qty":qty,"cap":cap,
                     "tp":tp,"sl":sl,"trail_active":bool(p.get("use_trailing") or p.get("use_all_exits")),"trail_on":False,"tsl":None}
    return trades, capital, eq

# COMBO CACHE (unified backtest + optimizer)
def _combo_key(symbol, timeframe, date_from, date_to, params_dict):
    obj={"sym":symbol,"tf":timeframe,"from":str(date_from)[:10],"to":str(date_to)[:10],"p":params_dict}
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()[:20]
def _combo_path(key): return RESULT_DIR/f"c_{key}.json"
def load_combo(key):
    p=_combo_path(key)
    if p.exists():
        with open(p) as f: return json.load(f)
    return None
def save_combo(key, data): _combo_path(key).write_text(json.dumps(data))
def _score_combo(trades, final_cap, capital, eq):
    if not trades or len(trades)<3: return None
    df=pd.DataFrame(trades); wins=int((df["pnl_pct"]>0).sum())
    ret=(final_cap-capital)/capital*100; wr=wins/len(trades)*100
    max_dd=_compute_dd(eq,capital); worst=float(df["pnl_pct"].min())
    df["dur"]=(pd.to_datetime(df["exit_time"])-pd.to_datetime(df["entry_time"])).dt.total_seconds()/1800
    avg_dur=float(df["dur"].mean()); max_lev=1
    for lv in [2,3,5,7,10]:
        if max_dd*lv>=-60: max_lev=lv
    gw=df[df["pnl_usdt"]>0]["pnl_usdt"].sum(); gl=df[df["pnl_usdt"]<=0]["pnl_usdt"].abs().sum()
    pf=round(float(gw/gl),2) if gl>0 else 99.0
    lev_ret=ret*max_lev; lev_dd=max(max_dd*max_lev,-100.0); calmar=lev_ret/max(abs(lev_dd),1.0)
    score=calmar*(1+wr/100)*min(pf/2.0,3.0)
    return {"trades":len(trades),"wins":wins,"avg_dur":round(avg_dur,1),"win_rate":round(wr,2),
            "return":round(ret,2),"max_dd":round(max_dd,2),"worst":round(worst,2),"pf":pf,
            "max_lev":max_lev,"lev_ret":round(lev_ret,1),"lev_dd":round(lev_dd,1),"score":round(score,2)}

# DATA CACHE
def _cache_path(ex, sym, tf):
    s=sym.replace("/","_").replace(":","_"); return CACHE_DIR/f"{ex}_{s}_{tf}.csv"
def load_cache(ex, sym, tf):
    p=_cache_path(ex,sym,tf)
    if not p.exists(): return None
    df=pd.read_csv(p,parse_dates=["timestamp"])
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)
def save_cache(df, ex, sym, tf): df.to_csv(_cache_path(ex,sym,tf),index=False)
def _tf_ms(tf):
    u={"m":60000,"h":3600000,"d":86400000}; return int(tf[:-1])*u.get(tf[-1],60000)
def get_exchange(name):
    import ccxt
    if name=="bybit": return ccxt.bybit({"enableRateLimit":True})
    if name=="pionex": return ccxt.pionex({"enableRateLimit":True})
    raise ValueError(f"Unknown: {name}")
def fetch_and_update_cache(ex_name, sym, tf, force_full=False):
    ex=get_exchange(ex_name); ex.load_markets()
    cached=None if force_full else load_cache(ex_name,sym,tf); per=1000
    if cached is not None and len(cached)>0:
        last_ts=int(cached["timestamp"].iloc[-1].timestamp()*1000)
        batches=[]; since=last_ts+1
        while True:
            b=ex.fetch_ohlcv(sym,tf,since=since,limit=per)
            if not b: break
            batches.extend(b)
            if len(b)<per: break
            since=b[-1][0]+1; time.sleep(0.2)
        if batches:
            nd=pd.DataFrame(batches,columns=["timestamp","open","high","low","close","volume"])
            nd["timestamp"]=pd.to_datetime(nd["timestamp"],unit="ms",utc=True)
            df=pd.concat([cached,nd],ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
            save_cache(df,ex_name,sym,tf); return df,len(batches),len(df)
        return cached,0,len(cached)
    all_c=[]; b=ex.fetch_ohlcv(sym,tf,limit=per)
    if b:
        all_c=b[:]
        for _ in range(50):
            fs=all_c[0][0]-per*_tf_ms(tf)
            b2=ex.fetch_ohlcv(sym,tf,since=fs,limit=per)
            if not b2 or b2[0][0]>=all_c[0][0]: break
            new=[c for c in b2 if c[0]<all_c[0][0]]
            if not new: break
            all_c=new+all_c; time.sleep(0.2)
    df=pd.DataFrame(all_c,columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms",utc=True)
    df=df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    save_cache(df,ex_name,sym,tf); return df,len(df),len(df)

def load_bybit_csv_files():
    files=list(BYBIT_DATA_DIR.glob("*.csv"))
    if not files: return None
    dfs=[]
    for f in files:
        try:
            df=pd.read_csv(f); col_map={}
            for c in df.columns:
                cl=c.lower().strip()
                if cl in ["timestamp","time","open time","date"]: col_map[c]="timestamp"
                elif cl=="open": col_map[c]="open"
                elif cl=="high": col_map[c]="high"
                elif cl=="low": col_map[c]="low"
                elif cl in ["close","close price"]: col_map[c]="close"
                elif cl in ["volume","vol"]: col_map[c]="volume"
            df=df.rename(columns=col_map)
            needed={"timestamp","open","high","low","close","volume"}
            if not needed.issubset(df.columns): continue
            df=df[list(needed)].copy()
            df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True,errors="coerce")
            df=df.dropna(subset=["timestamp"])
            for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
            dfs.append(df.dropna())
        except Exception as e: log.warning(f"CSV load {f}: {e}")
    if not dfs: return None
    merged=pd.concat(dfs,ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return merged

def check_cache_integrity(ex, sym, tf):
    df=load_cache(ex,sym,tf)
    if df is None or len(df)==0: return {"error":"No cache found"}
    tf_mins={"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"1d":1440}
    interval=pd.Timedelta(minutes=tf_mins.get(tf,30)); gaps=[]; ts=df["timestamp"].values
    for i in range(1,len(ts)):
        delta=pd.Timestamp(ts[i])-pd.Timestamp(ts[i-1])
        if delta>interval*3:
            gaps.append({"from":str(pd.Timestamp(ts[i-1]))[:16],"to":str(pd.Timestamp(ts[i]))[:16],
                         "hours":round(delta.total_seconds()/3600,1),"missing_bars":int(delta/interval)-1})
    return {"total":len(df),"first":df["timestamp"].iloc[0].isoformat(),"last":df["timestamp"].iloc[-1].isoformat(),
            "gaps":gaps,"duplicates":int(df.duplicated("timestamp").sum())}

# PRESETS
def load_presets():
    if PRESET_FILE.exists():
        with open(PRESET_FILE) as f: return json.load(f)
    return {}
def save_presets(p): PRESET_FILE.write_text(json.dumps(p,indent=2))

# JOB STORE (persistent)
_jobs={}; _jobs_lock=threading.Lock()
def _job_path(jid): return OPT_DIR/f"job_{jid}.json"
def get_job(jid):
    with _jobs_lock:
        if jid in _jobs: return _jobs[jid]
    p=_job_path(jid)
    if p.exists():
        with open(p) as f: return json.load(f)
    return None
def set_job(jid, data):
    with _jobs_lock: _jobs[jid]=data
    if data.get("status") in ("done","error"):
        _job_path(jid).write_text(json.dumps(data))

_fetch_jobs={}

# BOT
_bot_thread=None; _bot_stop=threading.Event(); _bot_lock=threading.Lock()
_bot_enabled=False  # in-memory flag, no disk read needed for status
def load_bot_config():
    if BOT_CONFIG_FILE.exists():
        cfg=DEFAULT_BOT_CONFIG.copy(); cfg.update(json.load(open(BOT_CONFIG_FILE))); return cfg
    return DEFAULT_BOT_CONFIG.copy()
def save_bot_config(cfg): BOT_CONFIG_FILE.write_text(json.dumps(cfg,indent=2))
def load_bot_state():
    if BOT_STATE_FILE.exists(): return json.load(open(BOT_STATE_FILE))
    return {"position":None,"entry_price":None,"entry_time":None,"trail_stop":None,"trail_activated":False,"last_atr":0}
def save_bot_state(s): BOT_STATE_FILE.write_text(json.dumps(s,indent=2))

def _bot_send_webhook(action, price, note, cfg):
    import requests as rq
    now=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    has_hdr=not SIGNALS_FILE.exists()
    direction="long" if (action=="buy" and "open" in note) else "short" if (action=="sell" and "open" in note) else ("long" if action=="sell" else "short")
    with open(SIGNALS_FILE,"a",newline="",encoding="utf-8") as f:
        if has_hdr: f.write("Time (UTC),Symbol,Action,Direction,Price,Contracts,Note\n")
        f.write(f"{now},{cfg['pionex_symbol']},{action},{direction},{price:.4f},{cfg['contracts']},{note}\n")
    if cfg.get("webhook_url"):
        try:
            ps="0" if "close" in note else cfg["contracts"]
            payload={"data":{"action":action,"contracts":cfg["contracts"],"position_size":ps},
                     "price":str(round(price,4)),"signal_param":"{}","signal_type":cfg.get("signal_type",""),
                     "symbol":cfg["pionex_symbol"],"time":now}
            rq.post(cfg["webhook_url"],json=payload,timeout=10,
                    headers={"Content-Type":"application/json","User-Agent":"Mozilla/5.0","Origin":"https://www.tradingview.com"})
        except Exception as e: log.warning(f"Webhook: {e}")

def _bot_loop(stop_ev):
    log.info("Bot thread started"); state=load_bot_state(); last_bar=None
    while not stop_ev.is_set():
        try:
            cfg=load_bot_config()
            if not _bot_enabled: time.sleep(15); continue
            try:
                import ccxt as _cx; ex=_cx.bybit({"enableRateLimit":True})
            except: time.sleep(30); continue
            now=datetime.now(timezone.utc)
            tf_min=int(cfg["timeframe"].replace("m","").replace("h",""))*(60 if "h" in cfg["timeframe"] else 1)
            fl=(now.minute//tf_min)*tf_min; bar_open=now.replace(minute=fl,second=0,microsecond=0)
            last_comp=bar_open-pd.Timedelta(minutes=tf_min)
            if last_comp!=last_bar and (now-bar_open).total_seconds()>=5:
                last_bar=last_comp; log.info(f"Bot bar {last_comp.strftime('%H:%M')}")
                try:
                    raw=ex.fetch_ohlcv(cfg["symbol"],cfg["timeframe"],limit=150)
                    df=pd.DataFrame(raw,columns=["timestamp","open","high","low","close","volume"])
                    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms",utc=True)
                    df["fast_ema"]=compute_ema(df["close"],cfg["fast_length"])
                    df["slow_ema"]=compute_ema(df["close"],cfg["slow_length"])
                    df["atr"]=compute_atr(df,cfg["atr_length"])
                    fe=df["fast_ema"].values; se=df["slow_ema"].values; i=len(df)-2
                    long_sig=bool(fe[i]>se[i] and fe[i-1]<=se[i-1])
                    short_sig=bool(fe[i]<se[i] and fe[i-1]>=se[i-1])
                    if cfg.get("use_vol_filter"):
                        av=df["volume"].rolling(20).mean().iloc[i]
                        if not np.isnan(av) and df["volume"].iloc[i]<=av*cfg.get("vol_multiplier",1.5): long_sig=False; short_sig=False
                    if cfg.get("use_adx_filter"):
                        adx_v=compute_adx(df,cfg.get("adx_length",14)).iloc[i]
                        if adx_v<=cfg.get("adx_threshold",25): long_sig=False; short_sig=False
                    atr_val=float(df["atr"].iloc[i]); last_close=float(df["close"].iloc[i])
                    last_high=float(df["high"].iloc[i]); last_low=float(df["low"].iloc[i]); entry_open=float(df["open"].iloc[-1])
                    pos=state["position"]; ep=state["entry_price"]; tsl=state["trail_stop"]; ta=state["trail_activated"]
                    log.info(f"  close={last_close:.4f} pos={pos or 'flat'} {'LONG' if long_sig else 'SHORT' if short_sig else '-'}")
                    if pos and ep:
                        dist=get_trail_dist(pos,last_close,ep,atr_val,cfg); closed=False; xp=None; xr=None
                        if pos=="long":
                            if not ta:
                                if last_high>=ep+dist: ta=True; tsl=last_high-dist
                            elif last_high-dist>tsl: tsl=last_high-dist
                        else:
                            if not ta:
                                if last_low<=ep-dist: ta=True; tsl=last_low+dist
                            elif tsl is None or last_low+dist<tsl: tsl=last_low+dist
                        if ta and tsl:
                            if pos=="long" and last_low<=tsl: xp,xr,closed=tsl,"TSL",True
                            elif pos=="short" and last_high>=tsl: xp,xr,closed=tsl,"TSL",True
                        if not closed and ep:
                            tp=ep*(1+cfg["tp_perc"]/100) if pos=="long" else ep*(1-cfg["tp_perc"]/100)
                            if pos=="long" and last_high>=tp: xp,xr,closed=tp,"TP",True
                            elif pos=="short" and last_low<=tp: xp,xr,closed=tp,"TP",True
                        if not closed:
                            if pos=="long" and short_sig: xp,xr,closed=last_close,"Signal",True
                            elif pos=="short" and long_sig: xp,xr,closed=last_close,"Signal",True
                        if closed:
                            pnl=(xp-ep)/ep*100 if pos=="long" else (ep-xp)/ep*100
                            log.info(f"  EXIT {pos.upper()} @ {xp:.4f} {xr} P&L:{pnl:+.2f}%")
                            _bot_send_webhook("sell" if pos=="long" else "buy",xp,f"close {pos}|{xr}|P&L {pnl:+.2f}%",cfg)
                            state.update({"position":None,"entry_price":None,"entry_time":None,"trail_stop":None,"trail_activated":False})
                            tsl=None; ta=False; pos=None
                    if pos is None:
                        if long_sig:
                            log.info(f"  ENTRY LONG @ {entry_open:.4f}")
                            _bot_send_webhook("buy",entry_open,"open long",cfg)
                            state.update({"position":"long","entry_price":entry_open,"entry_time":now.isoformat(),"trail_stop":None,"trail_activated":False})
                        elif short_sig:
                            log.info(f"  ENTRY SHORT @ {entry_open:.4f}")
                            _bot_send_webhook("sell",entry_open,"open short",cfg)
                            state.update({"position":"short","entry_price":entry_open,"entry_time":now.isoformat(),"trail_stop":None,"trail_activated":False})
                    state["trail_stop"]=tsl; state["trail_activated"]=ta; state["last_atr"]=atr_val
                    save_bot_state(state)
                except Exception as e: log.error(f"Bot bar error: {e}")
            time.sleep(15)
        except Exception as e: log.error(f"Bot outer: {e}"); time.sleep(60)
    log.info("Bot stopped")

def start_bot():
    global _bot_thread, _bot_stop, _bot_enabled
    with _bot_lock:
        _bot_enabled=True
        if _bot_thread and _bot_thread.is_alive(): return
        _bot_stop=threading.Event()
        _bot_thread=threading.Thread(target=_bot_loop,args=(_bot_stop,),daemon=True)
        _bot_thread.start()
def stop_bot():
    global _bot_enabled
    with _bot_lock:
        _bot_enabled=False
        if _bot_stop: _bot_stop.set()

# OPTIMIZER WORKER
def _opt_worker(args):
    combo, keys, base_params, base_rec, intra_rec, fee_pct, capital, sym, tf, d_from, d_to = args
    try:
        p={**base_params,**dict(zip(keys,combo))}
        key_params={k:p[k] for k in list(base_params.keys())[:10]}
        key_params.update(dict(zip(keys,combo)))
        ck=_combo_key(sym,tf,d_from,d_to,key_params)
        cached=load_combo(ck)
        if cached and "score" in cached: return {**cached,"combo_key":ck,"was_cached":True,**dict(zip(keys,combo))}
        df_b=pd.DataFrame(base_rec); df_b["timestamp"]=pd.to_datetime(df_b["timestamp"],utc=True)
        df_i=None
        if intra_rec:
            df_i=pd.DataFrame(intra_rec); df_i["timestamp"]=pd.to_datetime(df_i["timestamp"],utc=True)
        trades,final_cap,eq=run_ema_crossover(df_b,df_i,p,fee_pct,capital)
        sc=_score_combo(trades,final_cap,capital,eq)
        if sc is None: return None
        result={**sc,**dict(zip(keys,combo)),"combo_key":ck,"was_cached":False,"saved_at":datetime.now(timezone.utc).isoformat()}
        save_combo(ck,result); return result
    except: return None

# ROUTES
@app.route("/")
@login_required
def index(): return send_from_directory(".","index.html")

@app.route("/api/cache/status")
@login_required
def api_cache_status():
    ex=request.args.get("exchange","bybit"); sym=request.args.get("symbol","XMR/USDT:USDT"); tf=request.args.get("timeframe","30m")
    df=load_cache(ex,sym,tf)
    # Also return 1m status so frontend can warn about missing intrabar data
    df1m=load_cache(ex,sym,"1m") if tf!="1m" else df
    intra_info={"exists":False,"count":0}
    if df1m is not None:
        intra_info={"exists":True,"count":len(df1m),
                    "first":df1m["timestamp"].iloc[0].isoformat()[:16],
                    "last":df1m["timestamp"].iloc[-1].isoformat()[:16]}
    if df is None: return jsonify({"exists":False,"count":0,"intra_1m":intra_info})
    return jsonify({"exists":True,"count":len(df),
                    "first":df["timestamp"].iloc[0].isoformat(),
                    "last":df["timestamp"].iloc[-1].isoformat(),
                    "intra_1m":intra_info})

@app.route("/api/data/fetch", methods=["POST"])
@login_required
def api_data_fetch():
    b=request.json or {}; ex=b.get("exchange","bybit"); sym=b.get("symbol","XMR/USDT:USDT"); tf=b.get("timeframe","30m"); force=b.get("force_full",False)
    key=f"{ex}_{sym}_{tf}"
    def do():
        _fetch_jobs[key]={"status":"running","message":"Connecting..."}
        try:
            csv_df=load_bybit_csv_files()
            if csv_df is not None and len(csv_df)>0:
                existing=load_cache(ex,sym,tf)
                if existing is not None:
                    merged=pd.concat([csv_df,existing],ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
                    save_cache(merged,ex,sym,tf)
                else: save_cache(csv_df,ex,sym,tf)
                _fetch_jobs[key]["message"]="CSV merged, fetching API updates..."
            _fetch_jobs[key]["message"]="Fetching from API..."
            df,new_c,total=fetch_and_update_cache(ex,sym,tf,force_full=force)
            _fetch_jobs[key]={"status":"done","new":new_c,"total":total,"first":df["timestamp"].iloc[0].isoformat(),"last":df["timestamp"].iloc[-1].isoformat()}
        except Exception as e: _fetch_jobs[key]={"status":"error","message":str(e)}
    threading.Thread(target=do,daemon=True).start(); return jsonify({"ok":True})

@app.route("/api/data/fetch/status")
@login_required
def api_fetch_status():
    ex=request.args.get("exchange","bybit"); sym=request.args.get("symbol","XMR/USDT:USDT"); tf=request.args.get("timeframe","30m")
    return jsonify(_fetch_jobs.get(f"{ex}_{sym}_{tf}",{"status":"idle"}))

@app.route("/api/cache/integrity")
@login_required
def api_cache_integrity():
    return jsonify(check_cache_integrity(request.args.get("exchange","bybit"),request.args.get("symbol","XMR/USDT:USDT"),request.args.get("timeframe","30m")))

@app.route("/api/bybit-files")
@login_required
def api_bybit_files():
    files=[f.name for f in BYBIT_DATA_DIR.glob("*.csv")]
    return jsonify({"files":files,"count":len(files)})

_bt_jobs={}

@app.route("/api/backtest/run", methods=["POST"])
@login_required
def api_backtest_run():
    cfg=request.json or {}
    p={**DEFAULT_PARAMS,**cfg.get("params",{})}
    p.update({"slippage_pct":cfg.get("slippage_pct",0.0143),"compounding":cfg.get("compounding",True),"tsl_mode":cfg.get("tsl_mode","intrabar"),"tick_size":cfg.get("tick_size",0.01)})
    sym=cfg.get("symbol","XMR/USDT:USDT"); tf=cfg.get("timeframe","30m")
    df_test=load_cache(cfg.get("exchange","bybit"),sym,tf)
    if df_test is None: return jsonify({"error":"No data cached"}),400
    d_from=cfg.get("date_from") or str(df_test["timestamp"].iloc[0])[:10]
    d_to=cfg.get("date_to") or str(df_test["timestamp"].iloc[-1])[:10]
    ck=_combo_key(sym,tf,d_from,d_to,p)
    cached=load_combo(ck)
    if cached and cached.get("summary") and not cfg.get("force_rerun"):
        jid=f"bt_{ck}"; _bt_jobs[jid]={"status":"done","combo_key":ck,"cached":True,"cached_at":cached.get("saved_at","")}
        return jsonify({"ok":True,"jid":jid,"cached":True})
    jid=f"bt_{ck}"
    def do():
        _bt_jobs[jid]={"status":"running"}
        try:
            ex_name=cfg.get("exchange","bybit"); df_base=load_cache(ex_name,sym,tf)
            if df_base is None: _bt_jobs[jid]={"status":"error","message":"No data"}; return
            if cfg.get("date_from"): df_base=df_base[df_base["timestamp"]>=pd.Timestamp(cfg["date_from"],tz="UTC")]
            if cfg.get("date_to"): df_base=df_base[df_base["timestamp"]<=pd.Timestamp(cfg["date_to"],tz="UTC")]
            df_base=df_base.reset_index(drop=True)
            df_intra=None; tick_lookup=None
            if cfg.get("use_tick_mode"):
                # Build tick lookup for bars that matter (more accurate)
                bar_ts_list = list(df_base["timestamp"].values)
                ex_name2 = cfg.get("exchange","bybit")
                tl = build_tick_lookup(ex_name2, sym, bar_ts_list)
                if tl:
                    tick_lookup = tl
                    log.info(f"Tick mode: {len(tl)} bars have tick data")
                else:
                    log.info("Tick mode requested but no tick cache found, falling back to 1m")
            intra_warning=None
            if tick_lookup is None and cfg.get("use_intrabar"):
                df_intra=load_cache(ex_name,sym,cfg.get("intrabar_tf","1m"))
                if df_intra is not None:
                    if cfg.get("date_from"): df_intra=df_intra[df_intra["timestamp"]>=pd.Timestamp(cfg["date_from"],tz="UTC")]
                    if cfg.get("date_to"): df_intra=df_intra[df_intra["timestamp"]<=pd.Timestamp(cfg["date_to"],tz="UTC")]
                    df_intra=df_intra.reset_index(drop=True)
                    if len(df_intra)==0:
                        intra_warning="1m cache exists but no data in selected date range — intrabar simulation disabled"
                        df_intra=None
                else:
                    intra_warning="No 1m cache — intrabar simulation disabled, using bar close only. Upload tick files or run Update Cache to fix."
            fee=float(cfg.get("fee_pct",0.055)); capital=float(cfg.get("capital",1000))
            trades,final_cap,eq=run_ema_crossover(df_base,df_intra,p,fee,capital,tick_lookup=tick_lookup)
            if not trades: _bt_jobs[jid]={"status":"error","message":"No trades generated"}; return
            df_t=pd.DataFrame(trades); wins=int((df_t["pnl_pct"]>0).sum()); ret=(final_cap-capital)/capital*100
            max_dd=_compute_dd(eq,capital)
            gw=float(df_t[df_t["pnl_usdt"]>0]["pnl_usdt"].sum()); gl=float(df_t[df_t["pnl_usdt"]<=0]["pnl_usdt"].abs().sum())
            pf=round(gw/gl,3) if gl>0 else 99.0
            # Buy & Hold: $capital invested at first bar close, held to end
            bh_start=float(df_base["close"].iloc[0])
            bh_qty=capital/bh_start  # how many units you could buy
            bh_eq=[round(bh_qty*float(p_),2) for p_ in df_base["close"]]
            bh_dates=[str(t)[:16] for t in df_base["timestamp"]]
            eq_vals=[round(capital+float(df_t["pnl_usdt"].iloc[:i+1].sum()),2) for i in range(len(df_t))]
            eq_dates=[str(t["exit_time"])[:16] for t in trades]
            arr=np.array(eq_vals); pk=np.maximum.accumulate(arr); dd_vals=[round(float(v),2) for v in ((arr-pk)/pk*100)]
            result={"combo_key":ck,"saved_at":datetime.now(timezone.utc).isoformat(),
                "summary":{"total_trades":len(trades),"wins":wins,"losses":len(trades)-wins,
                    "win_rate":round(wins/len(trades)*100,2),"total_return":round(ret,2),"final_capital":round(final_cap,2),
                    "initial_capital":capital,"max_dd":round(max_dd,2),"profit_factor":pf,
                    "total_fees":round(float(df_t["fee_usdt"].sum()),2),
                    "avg_win":round(float(df_t[df_t["pnl_pct"]>0]["pnl_pct"].mean()) if wins>0 else 0,3),
                    "avg_loss":round(float(df_t[df_t["pnl_pct"]<=0]["pnl_pct"].mean()) if len(trades)-wins>0 else 0,3),
                    "exit_tp":int(df_t["exit_reason"].eq("TP").sum()),"exit_tsl":int(df_t["exit_reason"].eq("TSL").sum()),
                    "exit_sl":int(df_t["exit_reason"].eq("SL").sum()),"exit_signal":int(df_t["exit_reason"].eq("Signal").sum()),
                    "data_from":df_base["timestamp"].iloc[0].isoformat(),"data_to":df_base["timestamp"].iloc[-1].isoformat()},
                "equity_curve":{"dates":eq_dates,"values":eq_vals},
                "drawdown_series":{"dates":eq_dates,"values":dd_vals},
                "bh_curve":{"dates":bh_dates,"values":bh_eq},
                "trades":trades[-300:],"all_trades_count":len(trades),"params":p}
            save_combo(ck,result); _bt_jobs[jid]={"status":"done","combo_key":ck}
        except Exception as e:
            import traceback; _bt_jobs[jid]={"status":"error","message":str(e),"trace":traceback.format_exc()}
    threading.Thread(target=do,daemon=True).start(); return jsonify({"ok":True,"jid":jid,"cached":False})

@app.route("/api/backtest/status/<jid>")
@login_required
def api_bt_status(jid): return jsonify(_bt_jobs.get(jid,{"status":"idle"}))

@app.route("/api/backtest/result/<ck>")
@login_required
def api_bt_result(ck):
    r=load_combo(ck)
    if not r: return jsonify({"error":"Not found"}),404
    return jsonify(r)

@app.route("/api/backtest/export-csv/<ck>")
@login_required
def api_bt_export_csv(ck):
    r=load_combo(ck)
    if not r or "trades" not in r: return jsonify({"error":"Not found"}),404
    trades=r["trades"]; out=io.StringIO(); w=csv.writer(out)
    w.writerow(["Source","Trade #","Trade Date (UTC)","PnL (USDT)","Action","Date & Time (UTC)","Price (USDT)","Fee (USDT)","Total (USDT)"])
    capital=r["summary"]["initial_capital"]
    for i,t in enumerate(trades,1):
        pnl=t["pnl_usdt"]; qty=t.get("qty",0); fee_each=t["fee_usdt"]/2
        entry_act="Buy" if t["direction"]=="long" else "Sell"; exit_act="Sell" if t["direction"]=="long" else "Buy"
        w.writerow(["EMAX",i,t["exit_time"][:16],round(pnl,4),exit_act,t["exit_time"][:16],t["exit_price"],round(fee_each,8),round(t["exit_price"]*qty,5)])
        w.writerow(["EMAX",i,t["entry_time"][:16],"",entry_act,t["entry_time"][:16],t["entry_price"],round(fee_each,8),round(t["entry_price"]*qty,5)])
    out.seek(0)
    return Response(out.getvalue(),mimetype="text/csv",headers={"Content-Disposition":f"attachment;filename=trades_{ck[:8]}.csv"})

@app.route("/api/presets", methods=["GET"])
@login_required
def api_presets_get(): return jsonify(load_presets())

@app.route("/api/presets", methods=["POST"])
@login_required
def api_presets_save():
    b=request.json or {}; name=b.get("name","").strip()
    if not name: return jsonify({"error":"Name required"}),400
    p=load_presets(); p[name]={"config":b.get("config",{}),"stats":b.get("stats",{}),"saved_at":datetime.now(timezone.utc).isoformat()}
    save_presets(p); return jsonify({"ok":True})

@app.route("/api/presets/<name>", methods=["DELETE"])
@login_required
def api_presets_delete(name):
    p=load_presets(); p.pop(name,None); save_presets(p); return jsonify({"ok":True})

@app.route("/api/optimizer/run", methods=["POST"])
@login_required
def api_optimizer_run():
    import multiprocessing as mp  # cpu_count only
    body=request.json or {}
    jid=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()[:16]
    existing=get_job(jid)
    if existing and existing.get("status")=="done":
        return jsonify({"ok":True,"jid":jid,"cached":True})
    def do():
        set_job(jid,{"status":"running","progress":0,"total":0,"cached_count":0})
        try:
            cfg=body.get("config",{}); ranges=body.get("ranges",{})
            dd_limit=float(body.get("dd_limit",-60)); wf_split=body.get("wf_split",None)
            ex=cfg.get("exchange","bybit"); sym=cfg.get("symbol","XMR/USDT:USDT"); tf=cfg.get("timeframe","30m")
            capital=float(cfg.get("capital",1000)); fee=float(cfg.get("fee_pct",0.055))
            params={**DEFAULT_PARAMS,**cfg.get("params",{})}
            params.update({"slippage_pct":cfg.get("slippage_pct",0.0143),"compounding":cfg.get("compounding",True),
                           "tsl_mode":cfg.get("tsl_mode","intrabar"),"tick_size":cfg.get("tick_size",0.01)})
            df_full=load_cache(ex,sym,tf)
            if df_full is None: set_job(jid,{"status":"error","message":"No data cached"}); return
            if cfg.get("date_from"): df_full=df_full[df_full["timestamp"]>=pd.Timestamp(cfg["date_from"],tz="UTC")]
            if cfg.get("date_to"): df_full=df_full[df_full["timestamp"]<=pd.Timestamp(cfg["date_to"],tz="UTC")]
            df_full=df_full.reset_index(drop=True)
            df_intra_full=load_cache(ex,sym,cfg.get("intrabar_tf","1m")) if cfg.get("use_intrabar") else None
            if wf_split and float(wf_split)<1.0:
                wf=float(wf_split); idx=int(len(df_full)*wf); wf_time=df_full.iloc[idx]["timestamp"]
                df_opt=df_full.iloc[:idx].reset_index(drop=True); df_hold=df_full.iloc[idx:].reset_index(drop=True)
                df_intra_opt=df_intra_hold=None
                if df_intra_full is not None:
                    df_intra_opt=df_intra_full[df_intra_full["timestamp"]<wf_time].reset_index(drop=True)
                    df_intra_hold=df_intra_full[df_intra_full["timestamp"]>=wf_time].reset_index(drop=True)
                opt_from=df_opt["timestamp"].iloc[0]; opt_to=df_opt["timestamp"].iloc[-1]
                hold_from=df_hold["timestamp"].iloc[0]; hold_to=df_hold["timestamp"].iloc[-1]
            else:
                df_opt=df_full; df_intra_opt=df_intra_full; df_hold=None; df_intra_hold=None
                opt_from=df_full["timestamp"].iloc[0]; opt_to=df_full["timestamp"].iloc[-1]; hold_from=hold_to=None
            keys=list(ranges.keys()); combos=list(iproduct(*[ranges[k] for k in keys])); total=len(combos)
            set_job(jid,{"status":"running","progress":0,"total":total,"cached_count":0})
            cols=["timestamp","open","high","low","close","volume"]
            base_rec=df_opt[cols].assign(timestamp=df_opt["timestamp"].astype(str)).to_dict("records")
            intra_rec=None
            if df_intra_opt is not None and len(df_intra_opt)>0:
                intra_rec=df_intra_opt[cols].assign(timestamp=df_intra_opt["timestamp"].astype(str)).to_dict("records")
            d_from=str(opt_from)[:10]; d_to=str(opt_to)[:10]
            args_list=[(c,keys,params,base_rec,intra_rec,fee,capital,sym,tf,d_from,d_to) for c in combos]
            from concurrent.futures import ThreadPoolExecutor, as_completed
            n_cores=max(1,min(mp.cpu_count()-1, 8)); results=[]; done=0; cached_count=0
            update_every=max(1,total//50)
            with ThreadPoolExecutor(max_workers=n_cores) as pool:
                futs={pool.submit(_opt_worker,a):a for a in args_list}
                for fut in as_completed(futs):
                    done+=1
                    try:
                        r=fut.result()
                    except Exception:
                        r=None
                    if r:
                        if r.get("was_cached"): cached_count+=1
                        results.append({k:v for k,v in r.items() if k!="was_cached"})
                    if done%update_every==0 or done==total:
                        set_job(jid,{"status":"running","progress":done,"total":total,"cached_count":cached_count})
            results.sort(key=lambda x:x.get("score",0),reverse=True)
            oos_results=[]
            if df_hold is not None and len(df_hold)>0:
                hold_rec=df_hold[cols].assign(timestamp=df_hold["timestamp"].astype(str)).to_dict("records")
                hold_intra_rec=None
                if df_intra_hold is not None and len(df_intra_hold)>0:
                    hold_intra_rec=df_intra_hold[cols].assign(timestamp=df_intra_hold["timestamp"].astype(str)).to_dict("records")
                for r in results[:20]:
                    p_oos={**params,**{k:r[k] for k in keys if k in r}}
                    dh=pd.DataFrame(hold_rec); dh["timestamp"]=pd.to_datetime(dh["timestamp"],utc=True)
                    dhi=None
                    if hold_intra_rec:
                        dhi=pd.DataFrame(hold_intra_rec); dhi["timestamp"]=pd.to_datetime(dhi["timestamp"],utc=True)
                    to,co,eo=run_ema_crossover(dh,dhi,p_oos,fee,capital)
                    if to:
                        dfo=pd.DataFrame(to)
                        oos_results.append({**r,"oos_ret":round((co-capital)/capital*100,2),
                            "oos_wr":round((dfo["pnl_pct"]>0).sum()/len(to)*100,2),"oos_dd":round(_compute_dd(eo,capital),2),
                            "oos_n":len(to),"holds":bool((co-capital)/capital*100>0 and _compute_dd(eo,capital)>-60)})
                    else:
                        oos_results.append({**r,"oos_ret":0,"oos_wr":0,"oos_dd":0,"oos_n":0,"holds":False})
            job_data={"status":"done","total":total,"progress":total,"cached_count":cached_count,
                      "results":results[:50],"oos_results":oos_results,"keys":keys,
                      "opt_period":{"from":str(opt_from)[:16],"to":str(opt_to)[:16],"bars":len(df_opt)},
                      "hold_period":{"from":str(hold_from)[:16] if hold_from else None,"to":str(hold_to)[:16] if hold_to else None,"bars":len(df_hold) if df_hold is not None else 0},
                      "finished_at":datetime.now(timezone.utc).isoformat()}
            set_job(jid,job_data)
        except Exception as e:
            import traceback
            tb=traceback.format_exc()
            log.error(f"Optimizer error: {e}\n{tb}")
            set_job(jid,{"status":"error","message":str(e),"trace":tb})
    threading.Thread(target=do,daemon=True).start(); return jsonify({"ok":True,"jid":jid,"cached":False})

@app.route("/api/optimizer/status/<jid>")
@login_required
def api_opt_status(jid):
    j=get_job(jid)
    if not j: return jsonify({"status":"idle"})
    return jsonify({k:v for k,v in j.items() if k not in ("results","oos_results")})

@app.route("/api/optimizer/result/<jid>")
@login_required
def api_opt_result(jid):
    j=get_job(jid)
    if not j: return jsonify({"error":"Not found"}),404
    return jsonify(j)

@app.route("/api/optimizer/export-combos", methods=["POST"])
@login_required
def api_opt_export_combos():
    body=request.json or {}; ranges=body.get("ranges",{}); cfg=body.get("config",{})
    if not ranges: return jsonify({"error":"No ranges"}),400
    keys=list(ranges.keys()); combos=list(iproduct(*[ranges[k] for k in keys]))
    sym=cfg.get("symbol",""); tf=cfg.get("timeframe","")
    df_base=load_cache(cfg.get("exchange","bybit"),sym,tf)
    d_from=str(df_base["timestamp"].iloc[0])[:10] if df_base is not None else ""; d_to=str(df_base["timestamp"].iloc[-1])[:10] if df_base is not None else ""
    out=io.StringIO(); w=csv.writer(out)
    w.writerow(["combo_id","symbol","timeframe","date_from","date_to"]+keys)
    for i,c in enumerate(combos): w.writerow([i,sym,tf,d_from,d_to]+list(c))
    out.seek(0)
    return Response(out.getvalue(),mimetype="text/csv",headers={"Content-Disposition":"attachment;filename=combinations.csv"})

@app.route("/api/optimizer/upload-results", methods=["POST"])
@login_required
def api_opt_upload():
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    try:
        content=request.files["file"].read().decode("utf-8"); reader=csv.DictReader(io.StringIO(content)); rows=list(reader); imported=0
        for row in rows:
            try:
                sym=row.get("symbol",""); tf=row.get("timeframe",""); d_from=row.get("date_from",""); d_to=row.get("date_to","")
                skip={"symbol","timeframe","date_from","date_to","combo_key","saved_at"}
                p_vals={k:(float(v) if "." in str(v) else int(v)) for k,v in row.items() if k not in skip and v}
                ck=row.get("combo_key") or _combo_key(sym,tf,d_from,d_to,p_vals)
                data=dict(row); data["combo_key"]=ck; data["saved_at"]=datetime.now(timezone.utc).isoformat()
                save_combo(ck,data); imported+=1
            except: pass
        return jsonify({"ok":True,"imported":imported,"total":len(rows)})
    except Exception as e: return jsonify({"error":str(e)}),400

@app.route("/api/bot/config", methods=["GET","POST"])
@login_required
def api_bot_config():
    if request.method=="POST":
        cfg=load_bot_config(); cfg.update(request.json or {}); save_bot_config(cfg)
        if cfg.get("enabled"): start_bot()
        return jsonify({"ok":True,"config":cfg})
    return jsonify(load_bot_config())

@app.route("/api/bot/status")
@login_required
def api_bot_status():
    cfg=load_bot_config(); state=load_bot_state()
    running=_bot_thread is not None and _bot_thread.is_alive()
    # Use in-memory flag — no disk read, no flicker
    return jsonify({"running":running,"enabled":_bot_enabled,"state":state,"symbol":cfg.get("symbol"),"timeframe":cfg.get("timeframe")})

@app.route("/api/bot/signals")
@login_required
def api_bot_signals():
    limit=int(request.args.get("limit",50))
    if not SIGNALS_FILE.exists(): return jsonify({"signals":[]})
    rows=[]
    with open(SIGNALS_FILE,newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for row in reader: rows.append(dict(row))
    return jsonify({"signals":rows[-limit:][::-1]})

@app.route("/api/bot/candles")
@login_required
def api_bot_candles():
    cfg=load_bot_config(); sym=cfg.get("symbol","XMR/USDT:USDT"); tf=cfg.get("timeframe","30m"); limit=int(request.args.get("limit",120))
    try:
        import ccxt as _cx; ex=_cx.bybit({"enableRateLimit":True}); raw=ex.fetch_ohlcv(sym,tf,limit=limit)
        candles=[{"t":c[0],"o":c[1],"h":c[2],"l":c[3],"c":c[4],"v":c[5]} for c in raw]
        closes=[c[4] for c in raw]
        def ema_js(vals,n):
            k=2/(n+1); r=[vals[0]]
            for v in vals[1:]: r.append(v*k+r[-1]*(1-k))
            return r
        fast=ema_js(closes,cfg["fast_length"]); slow=ema_js(closes,cfg["slow_length"])
        state=load_bot_state()
        return jsonify({"candles":candles,"fast_ema":fast,"slow_ema":slow,"state":state,"symbol":sym,"timeframe":tf})
    except Exception as e: return jsonify({"error":str(e),"candles":[]})

@app.route("/api/bot/start", methods=["POST"])
@login_required
def api_bot_start():
    cfg=load_bot_config(); cfg["enabled"]=True; save_bot_config(cfg); start_bot(); return jsonify({"ok":True,"enabled":True})

@app.route("/api/bot/stop", methods=["POST"])
@login_required
def api_bot_stop():
    cfg=load_bot_config(); cfg["enabled"]=False; save_bot_config(cfg); stop_bot(); return jsonify({"ok":True,"enabled":False})


@app.route("/upload")
@login_required
def upload_page():
    return send_from_directory(".", "upload.html")


def _resample_to_tf(df_1m, rule="30min"):
    """Resample 1m OHLCV dataframe to a higher timeframe."""
    df = df_1m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    resampled = df.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["open"])
    return resampled.reset_index()

@app.route("/api/upload/candles", methods=["POST"])
@login_required
def api_upload_candles():
    """
    Accept Bybit CSV or CSV.GZ files (tick data OR OHLCV).
    Ported directly from original import_bybit_csv logic.
    - Tick data (timestamp,symbol,side,size,price,...) -> resample to 1m OHLCV
    - OHLCV data -> use directly as 1m
    Always saves 1m cache + auto-aggregates to 30m cache.
    """
    import gzip as gz_mod
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f   = request.files["file"]
    fname = f.filename or ""
    ex  = request.form.get("exchange", "bybit")
    sym = request.form.get("symbol",   "XMR/USDT:USDT")
    try:
        raw_bytes = f.read()
        if fname.endswith(".gz"):
            raw_bytes = gz_mod.decompress(raw_bytes)
        text = raw_bytes.decode("utf-8", errors="replace")
        from io import StringIO as _SIO
        raw = pd.read_csv(_SIO(text))

        cols_lower = [c.lower() for c in raw.columns]
        is_tick = "side" in cols_lower or ("price" in cols_lower and "open" not in cols_lower)

        if is_tick:
            # ── Tick data from public.bybit.com/trading/ ──
            raw.columns = [c.lower() for c in raw.columns]
            # timestamp is unix seconds (with decimals)
            raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="s", utc=True)
            raw = raw.sort_values("timestamp")
            ohlcv = raw.set_index("timestamp")["price"].resample("1min").agg(
                open="first", high="max", low="min", close="last"
            ).dropna()
            ohlcv["volume"] = raw.set_index("timestamp")["size"].resample("1min").sum()
            df_1m = ohlcv.reset_index()
            detected = "tick"
        else:
            # ── Standard OHLCV ──
            if "open" in cols_lower:
                raw.columns = [c.lower() for c in raw.columns]
                raw = raw.rename(columns={"open_time":"timestamp","time":"timestamp","opentime":"timestamp"})
            else:
                raw.columns = (["timestamp","open","high","low","close","volume"] +
                               [f"x{i}" for i in range(len(raw.columns)-6)])
            df_1m = raw[["timestamp","open","high","low","close","volume"]].copy()
            # Parse timestamp
            first = df_1m["timestamp"].iloc[0]
            if isinstance(first, str):
                df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], utc=True, errors="coerce")
            else:
                v = float(first)
                unit = "ms" if v > 1e12 else "s"
                df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], unit=unit, utc=True)
            for col in ["open","high","low","close","volume"]:
                df_1m[col] = pd.to_numeric(df_1m[col], errors="coerce")
            df_1m = df_1m.dropna()
            detected = "ohlcv"

        df_1m = df_1m.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        if len(df_1m) == 0:
            return jsonify({"error": "No valid rows after parsing"}), 400

        # ── Save raw ticks to parquet (if tick data and pyarrow available) ──
        tick_saved = False
        if is_tick and _HAVE_PARQUET:
            try:
                tick_df = raw[["timestamp","price","size"]].copy() if "size" in raw.columns else raw[["timestamp","price"]].copy()
                tick_df = tick_df.rename(columns={"size":"volume"})
                save_ticks(tick_df, ex, sym)
                tick_saved = True
                log.info(f"Saved {len(tick_df)} ticks to parquet")
            except Exception as te:
                log.warning(f"Tick parquet save failed: {te}")

        # ── Merge into 1m cache ──
        existing_1m = load_cache(ex, sym, "1m")
        if existing_1m is not None and len(existing_1m) > 0:
            merged_1m = pd.concat([existing_1m, df_1m], ignore_index=True)
        else:
            merged_1m = df_1m.copy()
        merged_1m = merged_1m.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        save_cache(merged_1m, ex, sym, "1m")

        # ── Auto-aggregate 1m -> 30m ──
        merged_30m = _resample_to_tf(merged_1m, "30min")
        existing_30m = load_cache(ex, sym, "30m")
        if existing_30m is not None and len(existing_30m) > 0:
            merged_30m = pd.concat([existing_30m, merged_30m], ignore_index=True)
        merged_30m = merged_30m.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        save_cache(merged_30m, ex, sym, "30m")

        return jsonify({
            "ok": True,
            "filename": fname,
            "detected": detected,
            "rows_in_file": len(df_1m),
            "total_1m": len(merged_1m),
            "total_30m": len(merged_30m),
            "first": str(merged_1m["timestamp"].iloc[0])[:16],
            "last":  str(merged_1m["timestamp"].iloc[-1])[:16],
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/api/upload/status")
@login_required
def api_upload_status():
    ex  = request.args.get("exchange",  "bybit")
    sym = request.args.get("symbol",    "XMR/USDT:USDT")
    tf  = request.args.get("timeframe", "30m")
    df  = load_cache(ex, sym, tf)
    ticks = tick_cache_info(ex, sym)
    if df is None:
        return jsonify({"exists": False, "count": 0, "ticks": ticks})
    return jsonify({"exists": True, "count": len(df),
                    "first": df["timestamp"].iloc[0].isoformat(),
                    "last":  df["timestamp"].iloc[-1].isoformat(),
                    "ticks": ticks})

@app.route("/api/tick/status")
@login_required
def api_tick_status():
    ex  = request.args.get("exchange", "bybit")
    sym = request.args.get("symbol",   "XMR/USDT:USDT")
    return jsonify(tick_cache_info(ex, sym))

if __name__=="__main__":
    cfg=load_bot_config()
    if cfg.get("enabled"):
        _bot_enabled=True
        start_bot()
    app.run(host="0.0.0.0",port=5000,debug=False)
