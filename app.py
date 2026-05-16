import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = Exception  # older yfinance versions don't export this

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantAI · Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
dark_mode = st.sidebar.toggle("Dark Mode", value=True, key="dark_mode")

_DARK = dict(
    bg="#0a0e1a", surface="#0f1629", border="#1e2d4a",
    text="#e2e8f0", muted="#64748b", dim="#475569",
    up="#10b981", down="#f43f5e", accent="#6366f1",
    accent2="#a78bfa", warn="#f59e0b", blue="#3b82f6",
)
_LIGHT = dict(
    bg="#f8fafc", surface="#ffffff", border="#e2e8f0",
    text="#0f172a", muted="#64748b", dim="#94a3b8",
    up="#059669", down="#dc2626", accent="#4f46e5",
    accent2="#7c3aed", warn="#d97706", blue="#2563eb",
)
C = _DARK if dark_mode else _LIGHT


def _get_css(C: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {C["bg"]};
    color: {C["text"]};
}}
.main {{ background-color: {C["bg"]}; }}
h1, h2, h3 {{ font-family: 'Space Mono', monospace; letter-spacing: -0.5px; }}
.stApp {{ background: {C["bg"]}; }}

[data-testid="stSidebar"] {{
    background: {C["surface"]} !important;
    border-right: 1px solid {C["border"]};
}}
[data-testid="metric-container"] {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 16px;
}}
.stButton > button {{
    background: {C["accent"]};
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    padding: 10px 24px;
    width: 100%;
    transition: opacity 0.2s;
}}
.stButton > button:hover {{ opacity: 0.85; }}
.stTextInput > div > div > input,
.stSelectbox > div > div > select {{
    background: {C["surface"]} !important;
    border: 1px solid {C["border"]} !important;
    color: {C["text"]} !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif;
}}

.signal-buy {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-left: 3px solid {C["up"]};
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
}}
.signal-sell {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-left: 3px solid {C["down"]};
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
}}
.signal-hold {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
}}

.ai-report {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-left: 3px solid {C["accent2"]};
    border-radius: 12px;
    padding: 24px;
    font-size: 15px;
    line-height: 1.7;
    white-space: pre-wrap;
}}
.po-ticker-card {{
    background: {C["surface"]};
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    border-top: 3px solid {C["accent"]};
    border-left: 1px solid {C["border"]};
    border-right: 1px solid {C["border"]};
    border-bottom: 1px solid {C["border"]};
}}
.mc-card {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}}
.mc-label {{ font-size: 11px; color: {C["muted"]}; margin-bottom: 6px; }}
.mc-value {{ font-family: 'Space Mono', monospace; font-size: 18px; font-weight: 700; }}
.mc-sub {{ font-size: 10px; color: {C["dim"]}; margin-top: 4px; }}

.risk-card {{
    background: {C["surface"]};
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    border: 1px solid {C["border"]};
}}
.risk-label {{ font-size: 11px; color: {C["muted"]}; margin-bottom: 6px; }}
.risk-value {{ font-family: 'Space Mono', monospace; font-size: 20px; font-weight: 700; }}
.risk-badge {{
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-top: 6px;
}}

.fund-card {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}}
.fund-label {{ font-size: 11px; color: {C["muted"]}; margin-bottom: 6px; letter-spacing: 0.3px; }}
.fund-value {{ font-family: 'Space Mono', monospace; font-size: 16px; font-weight: 700; color: {C["text"]}; }}
.fund-na {{ font-family: 'Space Mono', monospace; font-size: 16px; color: {C["dim"]}; }}

.news-item {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: border-color 0.2s;
}}
.news-item:hover {{ border-color: {C["blue"]}; }}
.news-item a {{ color: {C["text"]}; text-decoration: none; font-size: 14px; line-height: 1.5; flex: 1; }}
.news-item a:hover {{ color: {C["blue"]}; }}
.news-meta {{ font-size: 11px; color: {C["dim"]}; margin-top: 4px; white-space: nowrap; }}
.news-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: {C["border"]}; margin-top: 6px; flex-shrink: 0;
}}

.sentiment-wrap {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    margin-bottom: 16px;
}}
.sentiment-score-box {{ text-align: center; min-width: 90px; }}
.sentiment-score-num {{ font-family: 'Space Mono', monospace; font-size: 36px; font-weight: 700; line-height: 1; }}
.sentiment-score-label {{ font-size: 10px; color: {C["muted"]}; margin-top: 4px; letter-spacing: 0.5px; }}
.sentiment-right {{ flex: 1; }}
.sentiment-bar-track {{
    background: {C["border"]};
    border-radius: 4px; height: 8px; width: 100%;
    position: relative; margin-bottom: 6px; overflow: hidden;
}}
.sentiment-reason {{ font-size: 13px; color: {C["muted"]}; margin-top: 8px; }}

.bt-card {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}}
.bt-card .bt-label {{ font-size: 11px; color: {C["muted"]}; font-family: 'DM Sans', sans-serif; margin-bottom: 6px; }}
.bt-card .bt-value {{ font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; }}
.bt-card .bt-sub {{ font-size: 11px; color: {C["muted"]}; margin-top: 4px; }}

.header-banner {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
}}
.ticker-tag {{
    display: inline-block;
    background: {C["bg"]};
    border: 1px solid {C["border"]};
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: {C["muted"]};
    margin-right: 8px;
}}
</style>
"""

st.markdown(_get_css(C), unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def compute_bollinger(series: pd.Series, window: int = 20):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ma + 2 * std, ma, ma - 2 * std


def compute_regime(df: pd.DataFrame) -> dict:
    """
    Classify market state per bar using MA50 10-day normalised slope.

    slope[i] = (MA50[i] - MA50[i-10]) / MA50[i-10] / 10   (fractional per-day)
    epsilon   = daily_return_std * 0.1

    slope >  epsilon  → UPTREND
    slope < -epsilon  → DOWNTREND
    otherwise         → RANGE
    """
    daily_std = float(df["Close"].pct_change().std())
    epsilon   = daily_std * 0.1

    ma50  = df["MA50"]
    slope = ma50.pct_change(periods=10) / 10          # fractional per-day rate

    regimes = pd.Series("RANGE", index=df.index, dtype=object)
    valid = slope.notna()
    regimes[valid & (slope >  epsilon)] = "UPTREND"
    regimes[valid & (slope < -epsilon)] = "DOWNTREND"

    cur_slope  = float(slope.iloc[-1]) if pd.notna(slope.iloc[-1]) else 0.0
    cur_regime = str(regimes.iloc[-1])
    strength   = abs(cur_slope) / epsilon if epsilon > 0 else 0.0

    # ── Volatility filter ────────────────────────────────────────────────────
    # True Range per bar, normalised by close to make it comparable across prices
    prev_close = df["Close"].shift(1).fillna(df["Close"])
    tr = pd.concat([
        (df["High"] - df["Low"]),
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14      = tr.rolling(14, min_periods=1).mean()
    atr20_mean = atr14.rolling(20, min_periods=1).mean()
    vol_filter = atr14 > 2.0 * atr20_mean   # True = spike, block BUY

    cur_atr      = float(atr14.iloc[-1])
    cur_atr_mean = float(atr20_mean.iloc[-1])

    return {
        "regimes":       regimes,
        "slopes":        slope,
        "current":       cur_regime,
        "current_slope": cur_slope,
        "epsilon":       epsilon,
        "strength":      strength,
        "vol_filter":    vol_filter,
        "cur_atr":       cur_atr,
        "cur_atr_mean":  cur_atr_mean,
    }


def compute_rs_spread(stock_close, ref_close, idx: int = -1, window: int = 20) -> float | None:
    """Relative Strength as a percentage-point spread:
        rs = stock_window_return_pct - ref_window_return_pct
    rs = 0   → in line with reference (SPY or sector ETF)
    rs = +9  → stock outperformed reference by 9 percentage points
    rs = -5  → stock underperformed reference by 5 percentage points
    Returns None if there isn't enough history."""
    if stock_close is None or ref_close is None:
        return None
    sc = stock_close.values if hasattr(stock_close, "values") else stock_close
    rc = ref_close.values if hasattr(ref_close, "values") else ref_close
    n = len(sc)
    if n != len(rc) or n <= window:
        return None
    pos = idx if idx >= 0 else n + idx
    if pos < window or pos >= n:
        return None
    s_now, s_then = float(sc[pos]), float(sc[pos - window])
    r_now, r_then = float(rc[pos]), float(rc[pos - window])
    if s_then <= 0 or r_then <= 0:
        return None
    s_ret = (s_now / s_then - 1.0) * 100.0
    r_ret = (r_now / r_then - 1.0) * 100.0
    return s_ret - r_ret


def _spy_penalty_for_beta(beta: float | None) -> float:
    """Beta-tiered SPY-bear penalty applied to weighted_score:
       beta > 1.2          →  0.0  (high-beta: ignore SPY filter; stock has own alpha)
       0.9 ≤ beta ≤ 1.2    → -0.5  (neutral: moderate market-tracking penalty)
       beta < 0.9          → -1.0  (defensive: full penalty, follows broad market)
    Unknown beta defaults to neutral tier."""
    if beta is None:
        return -0.5
    if beta > 1.2:
        return 0.0
    if beta >= 0.9:
        return -0.5
    return -1.0


def _beta_tier_label(beta: float | None) -> tuple[str, str]:
    """Returns (label_zh, tier_key). tier_key ∈ {'high','neutral','defensive','unknown'}."""
    if beta is None:        return ("未知 Beta",        "unknown")
    if beta > 1.2:          return ("高Beta (>1.2)",   "high")
    if beta >= 0.9:         return ("中性 (0.9-1.2)",  "neutral")
    return ("防御 (<0.9)",  "defensive")


def compute_score(
    df: pd.DataFrame,
    idx: int,
    regime_info: dict,
    rsi_weight: int = 2,
) -> tuple[int, list[str]]:
    """
    Core scoring rule for a single bar — the single source of truth.

    All signal rules live here exactly once.  Both the live signal display
    (get_signal) and the historical backtest (compute_signal_scores) call this
    function, so rules can never drift out of sync between the two paths.

    Parameters
    ----------
    df          : full OHLCV + indicator DataFrame
    idx         : integer position of the bar to score (negative indexing supported)
    regime_info : output of compute_regime(df)
    rsi_weight  : RSI score magnitude (default 2; pass 1 for low-beta "稳健模式")

    Returns
    -------
    score   : int  — positive = bullish pressure, negative = bearish
    reasons : list[str] — human-readable explanations (ignored by backtest)
    """
    # Length guard: need at least 2 bars (current + previous) for crossover checks.
    # Normalise negative idx so checks are uniform.
    n = len(df)
    pos_idx = idx if idx >= 0 else n + idx
    if n < 2 or pos_idx < 1 or pos_idx >= n:
        return 0, []

    last   = df.iloc[idx]
    prev   = df.iloc[idx - 1]
    regime = str(regime_info["regimes"].iloc[idx])

    if pd.isna(last["RSI"]) or pd.isna(last["MACD"]) or pd.isna(last["BB_Upper"]):
        return 0, []

    score   = 0
    reasons: list[str] = []
    rsi     = float(last["RSI"])

    # ── RSI ──────────────────────────────────────────────────────────────────
    # Oversold BUY signal is suppressed in DOWNTREND (regime filter).
    # rsi_weight controls magnitude (1 for low-beta stocks, 2 standard).
    if rsi < 35:
        if regime == "DOWNTREND":
            reasons.append(f"RSI={rsi:.1f} → 超卖，但趋势向下，买入信号被抑制")
        else:
            score += rsi_weight
            reasons.append(f"RSI={rsi:.1f} → 超卖区间，有反弹预期")
    elif rsi > 70:
        score -= rsi_weight
        reasons.append(f"RSI={rsi:.1f} → 超买区间，存在回调风险")
    else:
        reasons.append(f"RSI={rsi:.1f} → 中性区间")

    # ── MACD crossover ───────────────────────────────────────────────────────
    if last["MACD"] > last["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        score += 2
        reasons.append("MACD 金叉 → 短期看涨信号")
    elif last["MACD"] < last["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        score -= 2
        reasons.append("MACD 死叉 → 短期看跌信号")

    # ── MA20 vs MA50 crossover ───────────────────────────────────────────────
    if (pd.notna(last.get("MA20")) and pd.notna(last.get("MA50")) and
            pd.notna(prev.get("MA20")) and pd.notna(prev.get("MA50"))):
        if last["MA20"] > last["MA50"] and prev["MA20"] <= prev["MA50"]:
            score += 1
            reasons.append("均线金叉 (MA20>MA50) → 中期趋势转强")
        elif last["MA20"] < last["MA50"] and prev["MA20"] >= prev["MA50"]:
            score -= 1
            reasons.append("均线死叉 (MA20<MA50) → 中期趋势转弱")

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    if last["Close"] < last["BB_Lower"]:
        score += 1
        reasons.append("价格跌破布林带下轨 → 短期超卖")
    elif last["Close"] > last["BB_Upper"]:
        score -= 1
        reasons.append("价格突破布林带上轨 → 短期超买")

    return score, reasons


def _sentiment_bonus(sentiment_score: float | None) -> float:
    """Map sentiment score to weighted_score adjustment.
    >+30: +0.3 (boost BUY), <-30: -0.3 (suppress BUY), otherwise: 0.
    Returns 0 when sentiment_score is None (no API Key configured)."""
    if sentiment_score is None:
        return 0.0
    if sentiment_score > 30:
        return 0.3
    if sentiment_score < -30:
        return -0.3
    return 0.0


def compute_signal_scores(
    df: pd.DataFrame,
    regime_info: dict | None = None,
    rsi_weight: int = 2,
) -> pd.Series:
    """
    Time-weighted score series for ALL bars (used by backtest).

    Pass 1 — raw score for every bar via compute_score (single source of truth).
    Pass 2 — 3-day exponential-style weighting:
        weighted[i] = 0.5 * raw[i] + 0.3 * raw[i-1] + 0.2 * raw[i-2]

    Note: news sentiment is intentionally NOT applied here — Yahoo Finance
    news has no historical record, so using today's sentiment for past bars
    would be look-ahead bias. Sentiment factor lives only in get_signal()
    for live decisions.
    """
    if regime_info is None:
        regime_info = compute_regime(df)
    n = len(df)

    # Pass 1: raw integer score for each bar (bar 0 stays 0 — no prev bar)
    raw = np.zeros(n, dtype=float)
    for i in range(1, n):
        s, _ = compute_score(df, i, regime_info, rsi_weight=rsi_weight)
        raw[i] = float(s)

    # Pass 2: 3-day time weighting with renormalised fallback
    _W = (0.5, 0.3, 0.2)
    weighted = np.zeros(n, dtype=float)
    for i in range(1, n):
        if i >= 3:
            weighted[i] = _W[0]*raw[i] + _W[1]*raw[i-1] + _W[2]*raw[i-2]
        elif i == 2:
            weighted[i] = (_W[0]*raw[i] + _W[1]*raw[i-1]) / (_W[0] + _W[1])
        else:                       # i == 1: only one scored bar
            weighted[i] = raw[i]

    return pd.Series(weighted, index=df.index)


def run_backtest(
    df: pd.DataFrame,
    holding_days: int = 5,
    stop_loss: float = 0.03,
    take_profit: float = 0.05,
    slippage: float = 0.001,
    commission: float = 0.001,
    time_stop_enabled: bool = True,
    time_stop_days: int = 5,
    time_stop_min_pnl: float = 0.005,
    uptrend_thr: float = 1.5,
    range_thr: float = 2.0,
    rsi_weight: int = 2,
    spy_full: pd.DataFrame | None = None,
    beta: float | None = None,
) -> dict:
    """
    Simulate the signal strategy on historical data.

    Score adjustment per bar (added to weighted_score before threshold check):
      - SPY bear (MA20 < MA50), beta-tiered penalty:
          beta > 1.2   →  0.0   (high-beta: skip SPY filter)
          0.9-1.2      → -0.5
          beta < 0.9   → -1.0
    Hard blocks:
      - DOWNTREND regime, volatility spike, 2-bar cooldown after DOWNTREND
      - Volume confirmation: current bar volume must exceed 20-day average
    NOT applied (look-ahead bias):
      - Insider trades, earnings proximity, news sentiment

    Exit priority: stop-loss → take-profit → 时间止损 → sell signal → max holding days.
    """
    regime_info = compute_regime(df)
    scores  = compute_signal_scores(df, regime_info=regime_info, rsi_weight=rsi_weight)
    regimes = regime_info["regimes"]
    closes  = df["Close"].values
    opens   = df["Open"].values
    highs   = df["High"].values
    lows    = df["Low"].values
    volumes = df["Volume"].values
    dates   = df.index
    n       = len(df)

    # Per-bar volume confirmation: today's volume must exceed 20-day average
    _vol_avg20 = pd.Series(volumes).rolling(20, min_periods=5).mean().values

    # Per-bar SPY bear flag (sector RS removed — see attribution analysis: it hurt Sharpe)
    if spy_full is not None and not spy_full.empty and {"MA20", "MA50"}.issubset(spy_full.columns):
        _spy_aligned = spy_full.reindex(df.index, method="ffill")
        _spy_bear_series = (_spy_aligned["MA20"] < _spy_aligned["MA50"]).fillna(False).values
    else:
        _spy_bear_series = np.zeros(n, dtype=bool)

    # Beta-tiered SPY penalty (precomputed once)
    _spy_penalty = _spy_penalty_for_beta(beta)

    # ATR-based position sizing: 14-day ATR vs 20-day rolling mean ATR
    _prev_c = np.roll(closes, 1); _prev_c[0] = closes[0]
    _tr = np.maximum(highs - lows,
          np.maximum(np.abs(highs - _prev_c), np.abs(lows - _prev_c)))
    _atr14 = np.full(n, np.nan)
    for _j in range(13, n):
        _atr14[_j] = _tr[_j - 13: _j + 1].mean()
    _atr_mean20 = pd.Series(_atr14).rolling(20, min_periods=10).mean().values

    def _position_size(idx: int) -> float:
        """Return position multiplier: 0.5 / 1.0 / 1.5 based on ATR ratio."""
        atr = _atr14[idx]
        avg = _atr_mean20[idx]
        if np.isnan(atr) or np.isnan(avg) or avg <= 0:
            return 1.0
        ratio = atr / avg
        if ratio > 1.5:
            return 0.5
        if ratio < 0.5:
            return 1.5
        return 1.0

    trades    = []
    in_trade  = False
    entry_idx = entry_price = entry_position = None

    # Round-trip cost: buy slippage + sell slippage + 2 × commission
    cost      = 2 * slippage + 2 * commission
    cooldown  = 0           # bars remaining after DOWNTREND exit
    prev_reg  = str(regimes.iloc[0])

    for i in range(1, n - 1):
        cur_reg = str(regimes.iloc[i])

        # Start 2-bar cooldown the moment we leave DOWNTREND
        if prev_reg == "DOWNTREND" and cur_reg != "DOWNTREND":
            cooldown = 2
        prev_reg = cur_reg

        if not in_trade:
            if cooldown > 0:
                cooldown -= 1
                continue
            # DOWNTREND: no entries at all
            if cur_reg == "DOWNTREND":
                continue
            # Volatility spike: ATR > 2× 20-day mean ATR blocks entry
            if regime_info["vol_filter"].iloc[i]:
                continue
            # Beta-tiered SPY bear penalty (sector/SPY RS removed — see attribution)
            _adj = _spy_penalty if _spy_bear_series[i] else 0.0
            entry_threshold = uptrend_thr if cur_reg == "UPTREND" else range_thr
            if (scores.iloc[i] + _adj) >= entry_threshold:
                # Volume confirmation: today's volume must exceed 20-day avg
                _va = _vol_avg20[i]
                if not np.isnan(_va) and _va > 0 and volumes[i] <= _va:
                    continue
                entry_idx = i + 1
                # Execute at next-day open (more realistic than close-to-close)
                entry_price    = float(opens[entry_idx]) * (1 + slippage + commission)
                entry_position = _position_size(i)   # ATR ratio at signal bar
                in_trade = True
        else:
            pnl = (float(closes[i]) - entry_price) / entry_price
            hold = i - entry_idx + 1

            if pnl <= -stop_loss:
                reason = f"止损 -{stop_loss:.0%}"
            elif pnl >= take_profit:
                reason = f"止盈 +{take_profit:.0%}"
            elif time_stop_enabled and hold >= time_stop_days and pnl < time_stop_min_pnl:
                reason = "时间止损"
            elif scores.iloc[i] <= -2:
                reason = "信号卖出"
            elif hold >= holding_days:
                reason = f"持有{holding_days}日"
            else:
                continue

            # SL/TP/时间止损: trigger on close, exit at that same close (intraday stop)
            # Signal/hold exits: execute at next-day open
            if reason.startswith("止") or reason == "时间止损":
                exit_idx = i
                exit_price = float(closes[i]) * (1 - slippage - commission)
                _exit_display = float(closes[i])
            else:
                exit_idx = min(i + 1, n - 1)
                exit_price = float(opens[exit_idx]) * (1 - slippage - commission)
                _exit_display = float(opens[exit_idx])

            raw_ret = (exit_price - entry_price) / entry_price
            pos_ret = raw_ret * entry_position   # position-weighted return
            trades.append({
                "entry_date":  dates[entry_idx],
                "exit_date":   dates[exit_idx],
                "entry_price": round(float(opens[entry_idx]), 2),
                "exit_price":  round(_exit_display, 2),
                "return":      pos_ret,
                "raw_return":  raw_ret,
                "position":    entry_position,
                "days":        exit_idx - entry_idx,
                "exit_reason": reason,
            })
            in_trade = False

    # Close any open position at end — use last bar's open as execution price
    if in_trade and entry_idx is not None:
        exit_price = float(opens[-1]) * (1 - slippage - commission)
        raw_ret = (exit_price - entry_price) / entry_price
        pos_ret = raw_ret * entry_position
        trades.append({
            "entry_date":  dates[entry_idx],
            "exit_date":   dates[-1],
            "entry_price": round(float(opens[entry_idx]), 2),
            "exit_price":  round(float(opens[-1]), 2),
            "return":      pos_ret,
            "raw_return":  raw_ret,
            "position":    entry_position,
            "days":        n - 1 - entry_idx,
            "exit_reason": "期末清仓",
        })

    if not trades:
        return {"trades": pd.DataFrame(), "metrics": None, "equity": None}

    trades_df = pd.DataFrame(trades)
    rets = trades_df["return"].values

    # Compound equity curve (trade exit points)
    equity_vals = np.cumprod(1 + rets)
    equity_series = pd.Series(
        np.concatenate([[1.0], equity_vals]),
        index=pd.Index([df.index[0]] + list(trades_df["exit_date"]))
    )

    # Metrics
    total_return = float(equity_vals[-1] - 1)
    win_rate = float((rets > 0).sum() / len(rets))
    bh_return = float((closes[-1] - closes[0]) / closes[0])

    # Max drawdown on equity curve
    eq = np.concatenate([[1.0], equity_vals])
    peak = np.maximum.accumulate(eq)
    max_drawdown = float(((eq - peak) / peak).min())

    # Annualized Sharpe (per-trade, annualized by avg holding period)
    avg_hold = float(trades_df["days"].mean())
    periods_per_year = 252 / max(avg_hold, 1)
    sharpe = float((rets.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else 0.0)

    # Downside deviation: annualised std of negative per-trade returns only
    neg_rets = rets[rets < 0]
    downside_dev = (
        float(neg_rets.std() * np.sqrt(periods_per_year))
        if len(neg_rets) > 1 else 0.0
    )

    metrics = {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "total_return": total_return,
        "bh_return": bh_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "downside_dev": downside_dev,
        "avg_return": float(rets.mean()),
        "avg_hold_days": avg_hold,
    }
    return {"trades": trades_df, "metrics": metrics, "equity": equity_series}


@st.cache_data
def run_robustness_test(
    rets_tuple: tuple,
    avg_hold: float,
    n_sims: int = 1000,
    noise_std: float = 0.005,
) -> dict:
    """
    Monte Carlo noise-injection robustness test.

    For each simulation, adds i.i.d. N(0, noise_std) to every trade return,
    then recomputes the annualised Sharpe. Returns the percentile distribution
    of 1000 Sharpe ratios to reveal whether the strategy's edge is real or
    an artefact of a lucky few trades.
    """
    rets = np.array(rets_tuple, dtype=float)
    n = len(rets)
    if n < 3:
        return {"error": "insufficient_trades"}

    rng = np.random.default_rng(42)
    # Shape: (n_sims, n_trades) — vectorised, no Python loop
    noise    = rng.normal(0.0, noise_std, size=(n_sims, n))
    sim_rets = rets[None, :] + noise          # broadcast original returns

    periods_per_year = 252.0 / max(float(avg_hold), 1.0)
    means   = sim_rets.mean(axis=1)           # (n_sims,)
    stds    = sim_rets.std(axis=1)
    sharpes = np.where(stds > 1e-9,
                       means / stds * np.sqrt(periods_per_year),
                       0.0)

    p5   = float(np.percentile(sharpes, 5))
    p50  = float(np.percentile(sharpes, 50))
    p95  = float(np.percentile(sharpes, 95))
    pct_pos = float((sharpes > 0).mean())     # fraction of sims with Sharpe > 0

    return {
        "p5": p5, "p50": p50, "p95": p95,
        "pct_positive": pct_pos,
        "sharpes": sharpes.tolist(),
        "n_sims": n_sims,
        "noise_std": noise_std,
        "error": None,
    }


def build_robustness_chart(rb: dict, actual_sharpe: float, C: dict) -> go.Figure:
    """Histogram of simulated Sharpe ratios with actual-Sharpe marker."""
    sharpes = np.array(rb["sharpes"])
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=sharpes,
        nbinsx=60,
        marker_color=C.get("accent2", "#6366f1"),
        opacity=0.75,
        name="模拟夏普分布",
        hovertemplate="夏普: %{x:.2f}<br>频次: %{y}<extra></extra>",
    ))

    # Percentile bands
    for pct, val, label in [
        (5,  rb["p5"],  "P5"),
        (50, rb["p50"], "中位"),
        (95, rb["p95"], "P95"),
    ]:
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color=C.get("muted", "#94a3b8"),
            line_width=1.2,
            annotation_text=f"{label} {val:.2f}",
            annotation_font_color=C.get("muted", "#94a3b8"),
            annotation_font_size=10,
        )

    # Actual Sharpe marker
    fig.add_vline(
        x=actual_sharpe,
        line_color=C.get("up", "#10b981"),
        line_width=2,
        annotation_text=f"实际 {actual_sharpe:.2f}",
        annotation_font_color=C.get("up", "#10b981"),
        annotation_font_size=11,
    )

    fig.update_layout(
        height=200,
        margin=dict(l=8, r=8, t=8, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=C.get("text", "#e2e8f0"),
        xaxis=dict(
            title="夏普比率", title_font_size=10,
            gridcolor=C.get("grid", "#1e2d4a"), tickfont_size=9,
        ),
        yaxis=dict(
            title="频次", title_font_size=10,
            gridcolor=C.get("grid", "#1e2d4a"), tickfont_size=9,
        ),
        showlegend=False,
        bargap=0.05,
    )
    return fig


def build_backtest_chart(equity: pd.Series, df: pd.DataFrame, trades_df: pd.DataFrame, C: dict) -> go.Figure:
    bh = df["Close"] / float(df["Close"].iloc[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bh.index, y=bh.values, name="买入持有",
        line=dict(color=C["dim"], width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, name="策略净值",
        line=dict(color=C["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
        mode="lines+markers",
        marker=dict(size=5, color=C["accent2"]),
    ))

    if not trades_df.empty:
        wins = trades_df[trades_df["return"] > 0]
        losses = trades_df[trades_df["return"] <= 0]
        win_equity = equity.reindex(wins["exit_date"])
        loss_equity = equity.reindex(losses["exit_date"])
        if not win_equity.empty:
            fig.add_trace(go.Scatter(
                x=wins["exit_date"], y=win_equity.values, mode="markers", name="盈利交易",
                marker=dict(color=C["up"], size=9, symbol="triangle-up"),
            ))
        if not loss_equity.empty:
            fig.add_trace(go.Scatter(
                x=losses["exit_date"], y=loss_equity.values, mode="markers", name="亏损交易",
                marker=dict(color=C["down"], size=9, symbol="triangle-down"),
            ))

    fig.update_layout(
        height=320,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], title="净值 (起始=1)"),
        legend=dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
                    orientation="h", y=1.08, font=dict(size=11)),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )
    return fig


def get_signal(
    df: pd.DataFrame,
    uptrend_thr: float = 1.5,
    range_thr: float = 2.0,
    rsi_weight: int = 2,
    spy_bear: bool = False,
    insider_adj: float = 0.0,
    near_earnings: bool = False,
    sentiment_score: float | None = None,
    rs_spy: float | None = None,
    beta: float | None = None,
) -> tuple[str, list[str], dict]:
    """Regime-filtered rule-based signal generator.

    Score adjustments (all stack on weighted_score):
      • Insider trades:        ±0.5 (CEO/CFO 大额)
      • News sentiment:        ±0.3 (>+30 / <-30)
      • SPY bear, beta-tiered:
          beta > 1.2   →  0.0  (high-beta: independent alpha, no penalty)
          0.9-1.2      → -0.5
          beta < 0.9   → -1.0
    Hard blocks (no score):
      • DOWNTREND regime:      buy_threshold=None (BUY impossible)
      • Volatility spike:      buy_threshold=None
      • Volume confirmation:   today's volume must exceed 20-day average
      • Earnings proximity:    BUY blocked within 3 days of earnings
    Display-only (no score effect):
      • RS_SPY spread (rs_spy): shown in reasons for context
    """
    regime_info = compute_regime(df)
    regime      = regime_info["current"]

    # Today's raw score + human-readable reasons
    raw_score, reasons = compute_score(df, -1, regime_info, rsi_weight=rsi_weight)
    _n = len(df)

    # 3-day time-weighted score — mirrors compute_signal_scores weighting exactly
    # compute_score(df, -k, ...) uses df.iloc[-k] and df.iloc[-k-1],
    # so -2 requires n>=3, -3 requires n>=4.
    _s0 = float(raw_score)
    _s1 = float(compute_score(df, -2, regime_info, rsi_weight=rsi_weight)[0]) if _n >= 3 else 0.0
    _s2 = float(compute_score(df, -3, regime_info, rsi_weight=rsi_weight)[0]) if _n >= 4 else 0.0

    if _n >= 4:
        weighted_score = 0.5*_s0 + 0.3*_s1 + 0.2*_s2
    elif _n >= 3:
        weighted_score = (0.5*_s0 + 0.3*_s1) / 0.8
    else:
        weighted_score = _s0

    # Regime-adjusted thresholds (caller-supplied, so Beta mode is already baked in)
    if regime == "UPTREND":
        buy_threshold = uptrend_thr
    elif regime == "RANGE":
        buy_threshold = range_thr
    else:                       # DOWNTREND — BUY completely forbidden
        buy_threshold = None

    # ── Filter 3: insider adj modifies weighted_score before threshold ──────────
    if insider_adj > 0:
        reasons.append(f"内部人(CEO/CFO)大额买入 → 加强信号 (+{insider_adj:.1f}分)")
    elif insider_adj < 0:
        reasons.append(f"内部人(CEO/CFO)大额卖出 → 压制信号 ({insider_adj:+.1f}分)")
    weighted_score = weighted_score + insider_adj

    # ── Sentiment factor: shift weighted_score by ±0.3 based on news sentiment ──
    _senti_bonus = _sentiment_bonus(sentiment_score)
    if _senti_bonus > 0:
        reasons.append(f"新闻情绪 {sentiment_score:+.0f} 偏正面 → 加强信号 (+{_senti_bonus:.1f}分)")
    elif _senti_bonus < 0:
        reasons.append(f"新闻情绪 {sentiment_score:+.0f} 偏负面 → 压制信号 ({_senti_bonus:+.1f}分)")
    weighted_score = weighted_score + _senti_bonus

    # ── SPY bear regime, beta-tiered penalty ────────────────────────────────────
    _beta_label, _beta_tier = _beta_tier_label(beta)
    _spy_pen = _spy_penalty_for_beta(beta) if spy_bear else 0.0
    if spy_bear and _spy_pen < 0:
        reasons.append(
            f"大盘趋势向下（SPY MA20<MA50）+ {_beta_label} → 压制信号 ({_spy_pen:+.1f}分)"
        )
    elif spy_bear and _spy_pen == 0:
        reasons.append(
            f"大盘趋势向下，但 {_beta_label} 跳过 SPY 过滤（独立 alpha）"
        )
    weighted_score = weighted_score + _spy_pen

    # ── RS_SPY: display only, no score effect ──────────────────────────────────
    if rs_spy is not None:
        if rs_spy > 4:
            reasons.append(f"相对强度 RS vs SPY = {rs_spy:+.1f}pp（跑赢大盘，仅参考）")
        elif rs_spy < -4:
            reasons.append(f"相对强度 RS vs SPY = {rs_spy:+.1f}pp（跑输大盘，仅参考）")
        else:
            reasons.append(f"相对强度 RS vs SPY = {rs_spy:+.1f}pp（与大盘同步）")

    # Volatility filter — hard block on buy_threshold regardless of regime
    if regime_info["vol_filter"].iloc[-1]:
        _atr_ratio = regime_info["cur_atr"] / max(regime_info["cur_atr_mean"], 1e-9)
        reasons.append(f"波动率过高（ATR {_atr_ratio:.1f}×均值），暂停买入")
        buy_threshold = None

    regime_info["raw_score"]      = raw_score
    regime_info["weighted_score"] = weighted_score
    regime_info["buy_threshold"]  = buy_threshold
    regime_info["rs_spy"]         = rs_spy
    regime_info["beta"]           = beta
    regime_info["beta_tier"]      = _beta_tier
    regime_info["beta_label"]     = _beta_label
    regime_info["spy_penalty"]    = _spy_pen

    if buy_threshold is not None and weighted_score >= buy_threshold:
        # ── Volume confirmation (hard block) ──────────────────────────────────
        _vol_series = df["Volume"]
        _vol_avg20  = _vol_series.rolling(20, min_periods=5).mean()
        _vol_now    = float(_vol_series.iloc[-1])
        _vol_avg    = float(_vol_avg20.iloc[-1]) if not pd.isna(_vol_avg20.iloc[-1]) else 0.0
        if _vol_avg > 0 and _vol_now <= _vol_avg:
            reasons.append(f"成交量({_vol_now/1e6:.1f}M)低于20日均量({_vol_avg/1e6:.1f}M)，BUY未确认")
            return "HOLD", reasons, regime_info
        # ── Earnings proximity (hard block) ───────────────────────────────────
        if near_earnings:
            reasons.append("距财报日≤3天，规避财报风险，暂停买入")
            return "HOLD", reasons, regime_info
        return "BUY", reasons, regime_info
    elif raw_score <= -2:           # SELL uses today's raw score — no history dampening
        return "SELL", reasons, regime_info
    else:
        return "HOLD", reasons, regime_info


@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    # auto_adjust=True: yfinance returns split- and dividend-adjusted prices in the
    # "Close" (and Open/High/Low) columns.  There is no separate "Adj Close" column —
    # df["Close"] IS the adjusted close, so all feature engineering is split-safe.
    #
    # Retry once on empty result: yfinance occasionally returns nothing for a
    # specific (ticker, period) pair due to transient rate-limits or network
    # blips.  Without a retry, @st.cache_data caches the empty DataFrame and
    # every subsequent page-render returns the cached failure for 5 minutes.
    df = pd.DataFrame()
    for _attempt in range(2):
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            break
        if _attempt == 0:
            time.sleep(1)   # brief pause before retry

    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(df["Close"])
    return df


@st.cache_data(ttl=300)
def scan_ticker(tkr: str, spy_bear: bool = False) -> dict:
    """Multi-ticker scan helper. Mirrors main-page get_signal context:
    same beta-mode threshold tier, same SPY bear filter (computed once, passed in).
    Skips per-ticker insider / earnings / sentiment (those would require
    per-ticker API calls — too slow for a scan)."""
    try:
        df = yf.download(tkr, period="3mo", progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return {"ticker": tkr, "error": "no_data"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"]  = df["Close"].rolling(50).mean()
        df["RSI"]   = compute_rsi(df["Close"])
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])
        df["BB_Upper"], df["BB_Mid"], df["BB_Lower"]   = compute_bollinger(df["Close"])
        df = df.dropna(subset=["RSI", "MACD", "BB_Upper"])
        if len(df) < 2:
            return {"ticker": tkr, "error": "no_data"}

        # Beta-mode tier (matches main page logic)
        try:
            _info = fetch_ticker_info(tkr)
            _b_raw = _info.get("beta")
            _b = float(_b_raw) if _b_raw is not None else None
        except Exception:
            _b = None
        if _b is not None and _b < 0.8:
            _upt, _rng, _rsiw = 1.0, 1.5, 1
        else:
            _upt, _rng, _rsiw = 1.5, 2.0, 2

        sig, _, _ri = get_signal(df,
                                 uptrend_thr=_upt, range_thr=_rng, rsi_weight=_rsiw,
                                 spy_bear=spy_bear)
        price = float(df["Close"].iloc[-1])
        chg   = (price - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2]) * 100
        rsi   = float(df["RSI"].iloc[-1])
        return {
            "ticker": tkr, "price": price, "signal": sig,
            "rsi": rsi, "chg_pct": chg,
            "sort_key": {"BUY": 0, "HOLD": 1, "SELL": 2}[sig],
            "error": None,
        }
    except Exception as e:
        return {"ticker": tkr, "error": str(e)}


@st.cache_data(ttl=300)
def fetch_ticker_info(ticker: str) -> dict:
    return yf.Ticker(ticker).info or {}


@st.cache_data(ttl=600)
def fetch_earnings_and_insider(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)

        # ── Earnings ──────────────────────────────────────────────────────
        cal = t.calendar or {}
        raw_dates = cal.get("Earnings Date", [])
        next_date = raw_dates[0] if raw_dates else None

        def _fmt_money(v):
            if v is None:
                return "N/A"
            v = float(v)
            return f"${v/1e9:.2f}B" if abs(v) >= 1e9 else f"${v/1e6:.1f}M"

        earnings = {
            "next_date":  str(next_date) if next_date else None,
            "eps_avg":    cal.get("Earnings Average"),
            "eps_high":   cal.get("Earnings High"),
            "eps_low":    cal.get("Earnings Low"),
            "rev_avg":    _fmt_money(cal.get("Revenue Average")),
            "rev_high":   _fmt_money(cal.get("Revenue High")),
            "rev_low":    _fmt_money(cal.get("Revenue Low")),
        }

        # Days until next earnings
        if next_date:
            try:
                delta = (pd.Timestamp(next_date) - pd.Timestamp("today")).days
                earnings["days_until"] = int(delta)
            except Exception:
                earnings["days_until"] = None
        else:
            earnings["days_until"] = None

        # ── Insider transactions ──────────────────────────────────────────
        def _tx_type(text: str) -> str:
            if not isinstance(text, str) or not text.strip():
                return "其他"
            t_low = text.lower()
            if "sale" in t_low:
                return "卖出"
            if "purchase" in t_low or "buy" in t_low or "acquisition" in t_low:
                return "买入"
            if "gift" in t_low:
                return "赠予"
            return "其他"

        insider_rows = []
        try:
            it = t.insider_transactions
            if it is not None and not it.empty:
                for _, row in it.head(10).iterrows():
                    val = row.get("Value")
                    val_str = (f"${float(val)/1e6:.2f}M"
                               if pd.notna(val) and float(val) > 0 else "—")
                    shares = int(row.get("Shares", 0))
                    insider_rows.append({
                        "日期":       str(row.get("Start Date", ""))[:10],
                        "内部人":     str(row.get("Insider", "")).title(),
                        "职位":       str(row.get("Position", "")),
                        "类型":       _tx_type(str(row.get("Text", ""))),
                        "股数":       f"{shares:,}",
                        "交易额":     val_str,
                        "shares_raw": shares,  # kept as int for signal computation
                    })
        except Exception:
            pass

        # Historical earnings dates for backtest earnings-proximity filter
        all_earnings_dates: list[str] = []
        try:
            _ед = t.get_earnings_dates(limit=20)
            if _ед is not None and not _ед.empty:
                all_earnings_dates = [str(d)[:10] for d in _ед.index.tolist()]
        except Exception:
            pass

        return {"earnings": earnings, "insider": insider_rows,
                "all_earnings_dates": all_earnings_dates, "error": None}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def fetch_news(ticker: str) -> list[dict]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    items = []
    for n in raw[:10]:
        content = n.get("content", {})
        if content:
            title = content.get("title", "")
            publisher = (content.get("provider") or {}).get("displayName", "")
            url = (content.get("canonicalUrl") or {}).get("url", "") or (content.get("clickThroughUrl") or {}).get("url", "")
            pub_raw = content.get("pubDate", "")
            try:
                from datetime import timezone
                dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                pub_time = dt.astimezone(timezone.utc).strftime("%m/%d %H:%M UTC")
            except Exception:
                pub_time = pub_raw[:10] if pub_raw else ""
        else:
            title = n.get("title", "")
            publisher = n.get("publisher", "")
            url = n.get("link", "")
            ts = n.get("providerPublishTime", 0)
            pub_time = datetime.fromtimestamp(ts).strftime("%m/%d %H:%M") if ts else ""
        if title:
            items.append({"title": title, "publisher": publisher, "pub_time": pub_time, "url": url})
    return items


_GEMINI_RATE_LIMIT_MSG = "API请求频率超限，请稍后再试"


def _handle_429(resp, attempt: int) -> bool:
    """Returns True if caller should retry (after waiting), False if caller should raise."""
    import time as _time
    if attempt == 0:
        _slot = st.empty()
        for _s in range(60, 0, -1):
            _slot.warning(f"⏸ 分析已暂停，{_s} 秒后自动重试…")
            _time.sleep(1)
        _slot.empty()
        return True
    return False


def _gemini_generate(prompt: str, api_key: str) -> str:
    """Call Gemini REST API directly; retries once after 60 s on 429; raises on other errors."""
    import requests as _requests
    import json as _json
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    body = _json.dumps(
        {"contents": [{"parts": [{"text": prompt}]}]},
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}

    for attempt in range(2):
        resp = _requests.post(url, data=body, headers=headers, timeout=30)
        if resp.status_code == 429:
            if _handle_429(resp, attempt):
                continue
            raise _requests.exceptions.HTTPError(_GEMINI_RATE_LIMIT_MSG, response=resp)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _deepseek_generate(prompt: str, api_key: str) -> str:
    """Call DeepSeek (OpenAI-compatible) chat completions endpoint."""
    import requests as _requests
    import json as _json
    url = "https://api.deepseek.com/v1/chat/completions"
    body = _json.dumps(
        {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(2):
        resp = _requests.post(url, data=body, headers=headers, timeout=60)
        if resp.status_code == 429:
            if _handle_429(resp, attempt):
                continue
            raise _requests.exceptions.HTTPError(_GEMINI_RATE_LIMIT_MSG, response=resp)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _ai_generate(prompt: str, api_key: str) -> str:
    """Dispatch to Gemini or DeepSeek based on API Key prefix."""
    k = (api_key or "").strip()
    # DeepSeek keys start with "sk-" but not "sk-ant-" (Anthropic) or "sk-proj-" (OpenAI project)
    if k.startswith("sk-") and not k.startswith("sk-ant-") and not k.startswith("sk-proj-"):
        return _deepseek_generate(prompt, k)
    return _gemini_generate(prompt, k)


def get_news_sentiment(ticker: str, headlines: list[str], api_key: str) -> dict:
    """
    One API call returns one integer per headline (one per line, -100..+100).
    Aggregates to a single score via average.
    Returns {'score': int, 'reason': str, 'individual': list[int]}.
    Never raises.
    """
    _key = str(api_key).strip()
    if not _key:
        return {"score": 0, "reason": "未提供 API Key", "error": "no_key"}
    if not headlines:
        return {"score": 0, "reason": "无新闻数据", "individual": []}
    try:
        numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
        n = len(headlines)
        prompt = (
            f"你是一位专业金融分析师。请对以下每条 {ticker} 新闻标题打分，"
            f"评估其对股价的情绪倾向，从 -100（极度悲观）到 +100（极度乐观），0 表示中性。\n\n"
            f"{numbered}\n\n"
            f"只输出 {n} 行数字，每行对应一条新闻的得分，不要有任何其他内容。"
        )
        text = _ai_generate(prompt, _key).strip()

        import re as _re
        scores = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Tolerate "1. 45", "50 (积极)", "Score: 30", "+25" etc.
            match = _re.search(r"-?\d+", line)
            if match:
                scores.append(max(-100, min(100, int(match.group()))))

        if not scores:
            return {"score": 0, "reason": "解析失败", "individual": [], "error": "no_scores"}

        avg = int(round(sum(scores) / len(scores)))
        parsed = len(scores)
        reason = f"基于 {parsed} 条新闻，平均情绪分 {avg:+d}"
        return {"score": avg, "reason": reason, "individual": scores}
    except Exception as exc:
        msg = str(exc)
        is_rl = msg.startswith(_GEMINI_RATE_LIMIT_MSG)
        return {"score": 0, "reason": msg if is_rl else "分析失败",
                "individual": [], "error": msg}


def generate_rule_report(
    ticker: str,
    df: pd.DataFrame,
    signal: str,
    info: dict,
) -> str:
    if len(df) < 2:
        return "⚠️ 数据不足，无法生成规则驱动分析（需要至少 2 个交易日）。"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    rsi = float(last["RSI"])
    macd = float(last["MACD"])
    macd_sig = float(last["MACD_Signal"])
    close = float(last["Close"])
    bb_upper = float(last["BB_Upper"])
    bb_lower = float(last["BB_Lower"])
    bb_mid = float(last["BB_Mid"])
    # iloc[-20] needs ≥20 bars; fall back to the earliest available bar otherwise
    _ref_idx = -20 if len(df) >= 20 else 0
    _ref_close = float(df["Close"].iloc[_ref_idx])
    price_20d_pct = (close - _ref_close) / _ref_close * 100 if _ref_close > 0 else 0.0

    parts = []

    # ── Technical summary ────────────────────────────────────────────────────
    if rsi < 35:
        parts.append(f"RSI 处于 {rsi:.0f} 的超卖区间，短期存在技术性反弹预期")
    elif rsi > 70:
        parts.append(f"RSI 高达 {rsi:.0f}，市场情绪偏热，短线存在获利回吐压力")
    else:
        parts.append(f"RSI 为 {rsi:.0f}，动能中性")

    macd_cross = ""
    if macd > macd_sig and float(prev["MACD"]) <= float(prev["MACD_Signal"]):
        macd_cross = "MACD 刚形成金叉，短期趋势转多"
    elif macd < macd_sig and float(prev["MACD"]) >= float(prev["MACD_Signal"]):
        macd_cross = "MACD 刚形成死叉，短期趋势转空"
    elif macd > macd_sig:
        macd_cross = "MACD 维持多头排列"
    else:
        macd_cross = "MACD 处于空头排列"
    parts.append(macd_cross)

    if close > bb_upper:
        parts.append("价格突破布林带上轨，超买特征明显")
    elif close < bb_lower:
        parts.append("价格跌破布林带下轨，短期超卖")
    else:
        pct_in_band = (close - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
        parts.append(f"价格位于布林带 {pct_in_band:.0f}% 位置，运行平稳")

    # ── Fundamental color ────────────────────────────────────────────────────
    pe = info.get("trailingPE")
    beta = info.get("beta")
    fund_notes = []
    if pe and not np.isnan(float(pe)):
        pe = float(pe)
        if pe > 40:
            fund_notes.append(f"市盈率 {pe:.0f}x 偏高，估值溢价明显")
        elif pe < 15:
            fund_notes.append(f"市盈率 {pe:.0f}x 处于低估区间")
        else:
            fund_notes.append(f"市盈率 {pe:.0f}x，估值合理")
    if beta and not np.isnan(float(beta)):
        beta = float(beta)
        if beta > 1.5:
            fund_notes.append(f"Beta {beta:.1f} 显示高弹性，波动显著大于大盘")
        elif beta < 0.7:
            fund_notes.append(f"Beta {beta:.1f} 低波动，防御属性较强")

    if fund_notes:
        parts.append("；".join(fund_notes))

    # ── Signal conclusion ────────────────────────────────────────────────────
    conclusion = {
        "BUY":  f"综合来看，多项指标共振向上，近20日涨幅 {price_20d_pct:+.1f}%，技术面偏多，可关注逢低布局机会。",
        "SELL": f"综合来看，多项指标发出预警，近20日涨幅 {price_20d_pct:+.1f}%，建议控制仓位，注意下行风险。",
        "HOLD": f"综合来看，技术面信号中性，近20日涨幅 {price_20d_pct:+.1f}%，建议观望等待更明确方向。",
    }[signal]

    body = "；".join(parts) + "。" + conclusion
    return f"⚠️ 以下为规则驱动分析，非 AI 生成，仅供参考。\n\n{body}"


def get_ai_analysis(ticker: str, df: pd.DataFrame, signal: str, reasons: list[str], api_key: str) -> str:
    """Returns analysis text. Never raises — returns error notice string on any failure."""
    import time as _time
    _key = str(api_key).strip()
    if not _key:
        return "未提供 API Key，无法生成 AI 分析。"
    _time.sleep(1)  # 1 s gap after sentiment call to avoid back-to-back 429
    if len(df) < 2:
        return "数据不足，无法生成 AI 分析（需要至少 2 个交易日）。"
    try:
        last = df.iloc[-1]
        _ref_idx = -20 if len(df) >= 20 else 0
        _ref_close = float(df["Close"].iloc[_ref_idx])
        price_change = ((float(df["Close"].iloc[-1]) - _ref_close) / _ref_close * 100) if _ref_close > 0 else 0.0

        prompt = f"""你是一位专业的美股量化分析师。请根据以下技术指标数据，对 {ticker} 股票给出简明专业的分析报告。

## 当前指标数据
- 最新收盘价: ${float(last['Close']):.2f}
- 近20日涨跌幅: {float(price_change):.2f}%
- RSI (14): {float(last['RSI']):.2f}
- MACD: {float(last['MACD']):.4f} | Signal: {float(last['MACD_Signal']):.4f}
- 布林带上轨: ${float(last['BB_Upper']):.2f} | 中轨: ${float(last['BB_Mid']):.2f} | 下轨: ${float(last['BB_Lower']):.2f}
- MA20: ${float(last['MA20']):.2f} | MA50: ${float(last['MA50']):.2f}

## 系统信号
综合信号：{signal}
信号依据：{'; '.join(reasons)}

## 请输出以下内容（用中文，简洁专业，控制在250字以内）：
1. 当前技术面总结（2-3句）
2. 主要风险提示（1-2点）
3. 操作建议（结合信号，给出具体思路）

注意：这仅为技术分析，不构成投资建议。"""
        return _ai_generate(prompt, _key)
    except Exception as exc:
        msg = str(exc)
        return msg if msg.startswith(_GEMINI_RATE_LIMIT_MSG) else f"AI 分析暂时不可用：{msg}"


@st.cache_data(ttl=600)
def run_garch_forecast(returns_tuple: tuple, dates_tuple: tuple, n_forecast: int = 30) -> dict:
    """
    Rolling-window GARCH(1,1) with strict out-of-sample conditional volatility.

    Each point in the historical vol series is a 1-step-ahead forecast produced
    by a model trained ONLY on past data — no future information used.

    Implementation details
    ──────────────────────
    • ROLL_WIN : rolling training window length (up to 252 days)
    • MIN_WIN  : minimum observations before first fit (63 days = ~3 months)
    • STRIDE   : refit every STRIDE days; between refits the vol is linearly
                 interpolated.  STRIDE is adaptive: targets ≤ 80 total refits
                 so runtime stays < ~8 s even on multi-year series.
    • Warm-start: each fit is initialised from the previous fit's parameters,
                 which dramatically reduces iterations and speeds up the loop.
    • Final fit : uses the most recent ROLL_WIN days; its horizon=30 forecast
                 drives the forward vol prediction cards.
    """
    try:
        from arch import arch_model
    except ImportError:
        return {"error": "arch_missing"}
    try:
        rets_raw    = np.array(returns_tuple, dtype=float)
        rets_scaled = rets_raw * 100          # scale to % for numerical stability
        n           = len(rets_scaled)

        MIN_WIN  = 63                         # ~3 months minimum
        ROLL_WIN = min(252, n)                # up to 1 year rolling window
        # Adaptive stride: target ≤ 80 refits so the loop stays fast
        STRIDE   = max(5, (n - MIN_WIN) // 80) if n > MIN_WIN else 5

        if n < MIN_WIN + 5:
            return {"error": "insufficient_data"}

        # ── Rolling OOS conditional volatility ────────────────────────────
        # For each fit point t, train on rets_scaled[start:t] and produce a
        # 1-step-ahead forecast representing vol for the NEXT trading day.
        # We assign that forecast to index t (the day being predicted).
        oos_vol      = np.full(n, np.nan)
        last_params  = None                   # warm-start cache

        fit_points = list(range(MIN_WIN, n, STRIDE))
        if (n - 1) not in fit_points:
            fit_points.append(n - 1)

        for t in fit_points:
            start  = max(0, t - ROLL_WIN)
            window = rets_scaled[start:t]
            if len(window) < 30:
                continue
            try:
                m   = arch_model(window, vol="Garch", p=1, q=1,
                                 dist="Normal", rescale=False)
                kw  = {"disp": "off", "show_warning": False}
                if last_params is not None:
                    kw["starting_values"] = last_params
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")   # mute StartingValueWarning etc.
                    fit = m.fit(**kw)
                last_params = fit.params.values.copy()   # save for warm-start

                # 1-step-ahead forecast → annualised %
                fc_var = float(
                    fit.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
                )
                oos_vol[t] = np.sqrt(fc_var) / 100 * np.sqrt(252) * 100
            except Exception:
                last_params = None            # reset on convergence failure

        # Fill sparse stride gaps by linear interpolation, then forward/back-fill
        oos_series = (
            pd.Series(oos_vol)
            .interpolate(method="linear")
            .bfill()
            .ffill()
        )

        # ── Final fit on most recent ROLL_WIN days → 30-day forward forecast
        final_window = rets_scaled[max(0, n - ROLL_WIN):]
        m_final  = arch_model(final_window, vol="Garch", p=1, q=1,
                              dist="Normal", rescale=False)
        kw_final = {"disp": "off", "show_warning": False}
        if last_params is not None:
            kw_final["starting_values"] = last_params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_final = m_final.fit(**kw_final)

        fc_final        = fitted_final.forecast(horizon=n_forecast, reindex=False)
        fc_var_arr      = fc_final.variance.iloc[-1].values
        fc_vol_daily    = np.sqrt(fc_var_arr) / 100 * 100   # daily vol %
        fc_vol_annual   = fc_vol_daily * np.sqrt(252)

        avg_fc = float(fc_vol_annual.mean())
        if avg_fc < 15:
            risk_level, risk_color = "低风险",   "#10b981"
        elif avg_fc < 30:
            risk_level, risk_color = "中等风险", "#f59e0b"
        elif avg_fc < 50:
            risk_level, risk_color = "高风险",   "#f97316"
        else:
            risk_level, risk_color = "极高风险", "#f43f5e"

        p     = fitted_final.params
        alpha = float(p.get("alpha[1]", 0))
        beta  = float(p.get("beta[1]",  0))

        # 20-day rolling realised volatility for comparison chart
        realized_pct = (
            pd.Series(rets_raw).rolling(20).std() * np.sqrt(252) * 100
        )

        current_vol = float(oos_series.iloc[-1])

        return {
            "dates":          list(dates_tuple),
            "cond_vol_pct":   oos_series.tolist(),      # ← strict OOS, no lookahead
            "realized_pct":   realized_pct.fillna(0).tolist(),
            "fc_vol_annual":  fc_vol_annual.tolist(),
            "fc_vol_daily":   fc_vol_daily.tolist(),
            "current_vol":    current_vol,
            "avg_fc_annual":  avg_fc,
            "risk_level":     risk_level,
            "risk_color":     risk_color,
            "alpha":          alpha,
            "beta":           beta,
            "persistence":    alpha + beta,
            "roll_window":    ROLL_WIN,
            "stride":         STRIDE,
            "n_refits":       len(fit_points),
            "error":          None,
        }
    except Exception as e:
        return {"error": str(e)}


def build_garch_hist_chart(g: dict, C: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["dates"], y=g["realized_pct"],
        name="已实现波动率（20日）",
        line=dict(color=C["muted"], width=1.5),
        opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        x=g["dates"], y=g["cond_vol_pct"],
        name="GARCH 样本外预测波动率",
        line=dict(color=C["accent2"], width=2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.07)",
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], title="年化波动率 (%)"),
        legend=dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
                    orientation="h", y=1.12, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    return fig


def build_garch_forecast_chart(g: dict, C: dict) -> go.Figure:
    days = list(range(1, len(g["fc_vol_annual"]) + 1))
    color = g["risk_color"]
    fig = go.Figure()
    fig.add_hline(
        y=g["current_vol"],
        line_dash="dot", line_color=C["dim"], opacity=0.8,
        annotation_text=f"当前 {g['current_vol']:.1f}%",
        annotation_font_color=C["muted"], annotation_font_size=10,
    )
    fig.add_trace(go.Scatter(
        x=days, y=g["fc_vol_annual"],
        name="预测年化波动率",
        line=dict(color=color, width=2.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
        mode="lines+markers",
        marker=dict(size=4, color=color),
        hovertemplate="第 %{x} 天: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis=dict(title="未来天数", gridcolor=C["border"], zerolinecolor=C["border"], dtick=5),
        yaxis=dict(title="年化波动率 (%)", gridcolor=C["border"], zerolinecolor=C["border"]),
        legend=dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    return fig


@st.cache_data(ttl=3600)
def fetch_fear_greed() -> dict:
    _zh = {
        "Extreme Fear":  "极度恐惧",
        "Fear":          "恐惧",
        "Neutral":       "中性",
        "Greed":         "贪婪",
        "Extreme Greed": "极度贪婪",
    }
    url = "https://api.alternative.me/fng/"
    try:
        import requests as _req
        resp = _req.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        raw = resp.json()["data"][0]
        val = int(raw["value"])
        cls = raw.get("value_classification", "")
        return {"value": val, "label_en": cls, "label_zh": _zh.get(cls, cls), "error": None}
    except Exception as e:
        return {"value": None, "label_en": None, "label_zh": "N/A", "error": str(e)}


def build_fg_gauge(fg: dict, C: dict) -> go.Figure:
    val = fg["value"] if fg["value"] is not None else 50
    label_zh = fg["label_zh"]

    if fg["value"] is None or val <= 25:
        bar_color = "#f43f5e"
    elif val <= 45:
        bar_color = "#f59e0b"
    elif val <= 55:
        bar_color = "#4a5568"
    elif val <= 75:
        bar_color = "#10b981"
    else:
        bar_color = "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={"x": [0, 1], "y": [0, 1]},
        number={
            "font": {"size": 30, "color": bar_color, "family": "Space Mono, monospace"},
            "suffix": "",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 25, 50, 75, 100],
                "ticktext": ["0", "25", "50", "75", "100"],
                "tickfont": {"size": 9, "color": "#475569"},
                "tickwidth": 1,
                "tickcolor": "#1e2d4a",
            },
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   25],  "color": "rgba(244,63,94,0.12)"},
                {"range": [25,  45],  "color": "rgba(245,158,11,0.12)"},
                {"range": [45,  55],  "color": "rgba(74,85,104,0.10)"},
                {"range": [55,  75],  "color": "rgba(16,185,129,0.12)"},
                {"range": [75, 100],  "color": "rgba(16,185,129,0.25)"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.85,
                "value": val,
            },
        },
    ))
    fig.update_layout(
        height=170,
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color="#64748b", size=11),
        margin=dict(l=8, r=8, t=28, b=0),
        annotations=[dict(
            text=label_zh,
            x=0.5, y=0.20,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12, color=bar_color, family="DM Sans, sans-serif"),
        )],
    )
    return fig


@st.cache_data(ttl=600)
def run_monte_carlo(
    last_close: float,
    daily_mu: float,
    daily_sigma: float,
    n_days: int = 30,
    n_sims: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_days, n_sims))
    log_ret = (daily_mu - 0.5 * daily_sigma ** 2) + daily_sigma * Z
    paths = np.vstack([
        np.full(n_sims, last_close),
        last_close * np.exp(np.cumsum(log_ret, axis=0)),
    ])  # shape: (n_days+1, n_sims)

    pcts = {p: np.percentile(paths, p, axis=1) for p in (5, 25, 50, 75, 95)}
    final = paths[-1]
    final_rets = (final - last_close) / last_close

    buckets = [
        ("< −15%",      final_rets < -0.15),
        ("−15% ~ −5%",  (final_rets >= -0.15) & (final_rets < -0.05)),
        ("−5% ~ +5%",   (final_rets >= -0.05) & (final_rets < 0.05)),
        ("+5% ~ +15%",  (final_rets >= 0.05)  & (final_rets < 0.15)),
        ("> +15%",      final_rets >= 0.15),
    ]
    probs = [(lbl, float(mask.sum() / n_sims)) for lbl, mask in buckets]

    return {
        "paths": paths,
        "pcts": pcts,
        "final_rets": final_rets,
        "probs": probs,
        "up_prob": float((final > last_close).sum() / n_sims),
        "S0": last_close,
        "n_days": n_days,
        "n_sims": n_sims,
    }


def build_mc_chart(mc: dict, ticker: str, C: dict) -> go.Figure:
    days = np.arange(mc["n_days"] + 1)
    S0, pcts = mc["S0"], mc["pcts"]

    fig = go.Figure()

    sample = mc["paths"][:, :200]
    for col in range(sample.shape[1]):
        fig.add_trace(go.Scatter(
            x=days, y=sample[:, col], mode="lines",
            line=dict(color="rgba(99,102,241,0.035)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    def _band(hi, lo, fill_color, name):
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([pcts[hi], pcts[lo][::-1]]),
            fill="toself", fillcolor=fill_color,
            line=dict(color="rgba(0,0,0,0)"),
            name=name, hoverinfo="skip",
        ))

    _band(95, 5,  "rgba(99,102,241,0.07)",  "5–95% 区间")
    _band(75, 25, "rgba(99,102,241,0.13)",  "25–75% 区间")

    for p, color, dash, name in [
        (5,  C["down"],   "dash",  "5th pct（悲观）"),
        (50, C["accent2"], "solid", "中位数"),
        (95, C["up"],     "dash",  "95th pct（乐观）"),
    ]:
        fig.add_trace(go.Scatter(
            x=days, y=pcts[p], name=name,
            line=dict(color=color, width=2 if p == 50 else 1.8, dash=dash),
        ))

    fig.add_hline(
        y=S0, line_dash="dot", line_color=C["dim"], opacity=0.8,
        annotation_text=f"当前 ${S0:.2f}",
        annotation_font_color=C["muted"], annotation_font_size=11,
    )

    fig.update_layout(
        height=340,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis=dict(title="未来天数", gridcolor=C["border"], zerolinecolor=C["border"], dtick=5),
        yaxis=dict(title="模拟价格 ($)", gridcolor=C["border"], zerolinecolor=C["border"]),
        legend=dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
                    orientation="h", y=1.12, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    return fig


def build_mc_dist_chart(mc: dict, C: dict) -> go.Figure:
    labels = [p[0] for p in mc["probs"]]
    values = [p[1] * 100 for p in mc["probs"]]
    colors = [C["down"], "#f97316", C["muted"], "#34d399", C["up"]]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family="Space Mono, monospace", size=12, color=C["text"]),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        height=240,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis=dict(
            title="概率 (%)", gridcolor=C["border"], zerolinecolor=C["border"],
            range=[0, max(values) * 1.35],
        ),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        margin=dict(l=0, r=50, t=0, b=0),
        showlegend=False,
    )
    return fig


@st.cache_data(ttl=600)
def fetch_and_optimize(tickers: tuple) -> dict:
    """Download 2yr prices for tickers and run PyPortfolioOpt max-Sharpe optimization."""
    try:
        from pypfopt import EfficientFrontier, risk_models, expected_returns
    except ImportError:
        return {"error": "pypfopt_missing"}

    try:
        raw = yf.download(list(tickers), period="2y", progress=False, auto_adjust=True)
    except YFRateLimitError:
        return {"error": "rate_limit"}
    except Exception as e:
        return {"error": str(e)}

    # Normalise columns regardless of yfinance version
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.dropna(how="all").ffill().bfill().dropna(how="any")
    prices = prices[[c for c in prices.columns if prices[c].notna().sum() >= 100]]

    if prices.shape[1] < 2 or len(prices) < 100:
        return {"error": "insufficient_data"}

    try:
        mu = expected_returns.mean_historical_return(prices)
        S  = risk_models.sample_cov(prices)

        # ── Max Sharpe ──────────────────────────────────────────────────────
        _n_assets = len(tickers)
        _w_bounds = (max(0.05, 1.0 / _n_assets * 0.5), 0.60)

        ef = EfficientFrontier(mu, S, weight_bounds=_w_bounds)
        ef.max_sharpe(risk_free_rate=0.05)
        weights  = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.05, verbose=False)

        # ── Min Volatility ─────────────────────────────────────────────────
        ef2 = EfficientFrontier(mu, S, weight_bounds=_w_bounds)
        ef2.min_volatility()
        mv_weights = ef2.clean_weights()
        mv_ret, mv_vol, mv_sharpe = ef2.portfolio_performance(risk_free_rate=0.05, verbose=False)

        # ── Efficient frontier trace ────────────────────────────────────────
        frontier = []
        for target in np.linspace(float(mu.min()), float(mu.max()), 60):
            try:
                ef_t = EfficientFrontier(mu, S, weight_bounds=_w_bounds)
                ef_t.efficient_return(target)
                r, v, _ = ef_t.portfolio_performance(risk_free_rate=0.05, verbose=False)
                frontier.append((float(v), float(r)))
            except Exception:
                pass

        # ── Random portfolios (background scatter) ──────────────────────────
        n = prices.shape[1]
        rng = np.random.default_rng(42)
        daily = prices.pct_change().dropna()
        ann_mu  = daily.mean().values * 252
        ann_cov = daily.cov().values * 252
        W = rng.dirichlet(np.ones(n), size=3000)
        r_rets  = W @ ann_mu
        r_vols  = np.sqrt(np.einsum("ij,jk,ik->i", W, ann_cov, W))
        r_sharpe = np.where(r_vols > 0, (r_rets - 0.05) / r_vols, 0.0)

        return {
            "weights":   dict(weights),
            "ret": float(ret), "vol": float(vol), "sharpe": float(sharpe),
            "mv_weights": dict(mv_weights),
            "mv_ret": float(mv_ret), "mv_vol": float(mv_vol), "mv_sharpe": float(mv_sharpe),
            "frontier":  frontier,
            "rand_vols":    r_vols.tolist(),
            "rand_rets":    r_rets.tolist(),
            "rand_sharpes": np.nan_to_num(r_sharpe).tolist(),
            "tickers": list(prices.columns),
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


def build_ef_chart(result: dict, C: dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result["rand_vols"], y=result["rand_rets"],
        mode="markers",
        marker=dict(
            size=3, opacity=0.45,
            color=result["rand_sharpes"],
            colorscale=[[0, C["surface"]], [0.45, "#3b4fd4"], [1, C["up"]]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Sharpe", font=dict(size=11, color=C["muted"])),
                thickness=10, len=0.65,
                tickfont=dict(size=10, color=C["muted"]),
            ),
            cmin=float(np.percentile(result["rand_sharpes"], 5)),
            cmax=float(np.percentile(result["rand_sharpes"], 95)),
        ),
        name="随机组合",
        hovertemplate="波动率: %{x:.1%}<br>收益率: %{y:.1%}<extra></extra>",
    ))

    if result["frontier"]:
        ef_v = [p[0] for p in result["frontier"]]
        ef_r = [p[1] for p in result["frontier"]]
        fig.add_trace(go.Scatter(
            x=ef_v, y=ef_r, mode="lines",
            line=dict(color=C["accent2"], width=2.5),
            name="有效前沿",
        ))

    fig.add_trace(go.Scatter(
        x=[result["mv_vol"]], y=[result["mv_ret"]],
        mode="markers+text",
        marker=dict(size=14, color=C["warn"], symbol="diamond",
                    line=dict(color=C["bg"], width=2)),
        text=["最小波动"], textposition="top right",
        textfont=dict(size=10, color=C["warn"]),
        name=f"最小波动  Sharpe {result['mv_sharpe']:.2f}",
    ))

    fig.add_trace(go.Scatter(
        x=[result["vol"]], y=[result["ret"]],
        mode="markers+text",
        marker=dict(size=16, color=C["up"], symbol="star",
                    line=dict(color=C["bg"], width=2)),
        text=["最优夏普"], textposition="top right",
        textfont=dict(size=10, color=C["up"]),
        name=f"最优夏普  Sharpe {result['sharpe']:.2f}",
    ))

    fig.update_layout(
        height=420,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis=dict(title="年化波动率", gridcolor=C["border"],
                   zerolinecolor=C["border"], tickformat=".0%"),
        yaxis=dict(title="年化预期收益率", gridcolor=C["border"],
                   zerolinecolor=C["border"], tickformat=".0%"),
        legend=dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
                    font=dict(size=11), x=0.01, y=0.99, xanchor="left"),
        margin=dict(l=0, r=20, t=10, b=0),
        hovermode="closest",
    )
    return fig


def build_weight_chart(result: dict, C: dict) -> go.Figure:
    palette = [C["accent"], C["up"], C["warn"], C["down"], C["accent2"], "#34d399"]
    items = sorted(
        [(k, v) for k, v in result["weights"].items() if v > 0.001],
        key=lambda x: -x[1],
    )
    labels = [x[0] for x in items]
    values = [x[1] for x in items]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.58,
        marker=dict(colors=palette[:len(labels)],
                    line=dict(color=C["bg"], width=3)),
        textinfo="label+percent",
        textfont=dict(family="Space Mono, monospace", size=12),
        hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        direction="clockwise", sort=True,
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        annotations=[dict(
            text=f"Sharpe<br><b>{result['sharpe']:.2f}</b>",
            x=0.5, y=0.5, font_size=13, showarrow=False,
            font=dict(color=C["accent2"], family="Space Mono, monospace"),
        )],
    )
    return fig


@st.cache_data(ttl=300)
def fetch_spy_6m() -> pd.Series:
    try:
        spy = yf.download("SPY", period="6mo", progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        return spy["Close"]
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=300)
def fetch_etf_full(symbol: str, period: str) -> pd.DataFrame:
    """Fetch any ETF OHLCV with MA20/MA50.
    Returns DataFrame [Close, MA20, MA50] or empty DataFrame on failure."""
    try:
        etf = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if etf.empty:
            return pd.DataFrame()
        if isinstance(etf.columns, pd.MultiIndex):
            etf.columns = etf.columns.get_level_values(0)
        out = etf[["Close"]].copy()
        out["MA20"] = out["Close"].rolling(20).mean()
        out["MA50"] = out["Close"].rolling(50).mean()
        return out
    except Exception:
        return pd.DataFrame()


def fetch_spy_full(period: str) -> pd.DataFrame:
    """Convenience wrapper preserving the old call signature."""
    return fetch_etf_full("SPY", period)


def get_sector_etf(info: dict) -> tuple[str | None, str | None]:
    """Map a yfinance ticker info dict to (etf_symbol, display_label).
    Returns (None, None) when no clean mapping exists."""
    industry = str(info.get("industry") or "").lower()
    sector   = str(info.get("sector")   or "").lower()

    # Industry-level overrides — more specific than the broad sector ETF
    if "semiconductor" in industry:                  return ("SOXX", "SOXX · 半导体")
    if "software" in industry:                        return ("IGV",  "IGV · 软件")
    if "biotech" in industry:                         return ("IBB",  "IBB · 生物科技")
    if "bank" in industry:                            return ("KRE",  "KRE · 区域银行")

    # Sector-level fallback (SPDR sector ETFs)
    if "technology" in sector:                        return ("XLK",  "XLK · 科技")
    if "communication" in sector:                     return ("XLC",  "XLC · 通信服务")
    if "financial" in sector:                         return ("XLF",  "XLF · 金融")
    if "healthcare" in sector or "health care" in sector:
        return ("XLV", "XLV · 医疗")
    if "consumer cyclical" in sector or "consumer discretionary" in sector:
        return ("XLY", "XLY · 可选消费")
    if "consumer defensive" in sector or "consumer staples" in sector:
        return ("XLP", "XLP · 必选消费")
    if "energy" in sector:                            return ("XLE",  "XLE · 能源")
    if "industrials" in sector or "industrial" in sector:
        return ("XLI", "XLI · 工业")
    if "utilities" in sector:                         return ("XLU",  "XLU · 公用事业")
    if "real estate" in sector:                       return ("XLRE", "XLRE · 房地产")
    if "basic materials" in sector or "materials" in sector:
        return ("XLB", "XLB · 原材料")
    return (None, None)


def compute_insider_adj(insider_rows: list[dict]) -> float:
    """Compute ±0.5 weighted_score adjustment from recent CEO/CFO insider trades.

    Rules:
    • Only transactions in the last 30 days count.
    • Only CEO / CFO roles (position contains "Chief Executive" or "Chief Financial").
    • Only trades of > 10 000 shares.
    • Net direction: if buys dominate → +0.5; if sells dominate → -0.5; else 0.0.
    """
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
    buy_shares = 0
    sell_shares = 0
    for row in insider_rows:
        try:
            tx_date = pd.Timestamp(row.get("日期", ""))
        except Exception:
            continue
        if tx_date < cutoff:
            continue
        pos = row.get("职位", "").lower()
        if "chief executive" not in pos and "chief financial" not in pos:
            continue
        shares = row.get("shares_raw", 0)
        if shares <= 10_000:
            continue
        if row.get("类型") == "买入":
            buy_shares += shares
        elif row.get("类型") == "卖出":
            sell_shares += shares

    if buy_shares > sell_shares and buy_shares > 0:
        return 0.5
    if sell_shares > buy_shares and sell_shares > 0:
        return -0.5
    return 0.0


def compute_risk_metrics(df: pd.DataFrame, spy_close: pd.Series, beta_from_info: float | None) -> dict:
    daily = df["Close"].pct_change().dropna()
    ann_vol = float(daily.std() * np.sqrt(252))

    # Beta: prefer info value; recalculate from price data as fallback
    if beta_from_info and not np.isnan(float(beta_from_info)):
        beta = float(beta_from_info)
        corr = None
    else:
        beta = None
        corr = None

    if not spy_close.empty:
        _stock_s = (daily.iloc[:, 0] if isinstance(daily, pd.DataFrame) else daily).rename("stock")
        _spy_s   = spy_close.pct_change().dropna()
        _spy_s   = (_spy_s.iloc[:, 0] if isinstance(_spy_s, pd.DataFrame) else _spy_s).rename("spy")
        aligned  = pd.concat([_stock_s, _spy_s], axis=1, join="inner")
        if len(aligned) >= 20:
            corr = float(aligned.corr().iloc[0, 1])
            if beta is None:
                cov = aligned.cov().iloc[0, 1]
                var_spy = float(aligned["spy"].var())
                beta = float(cov / var_spy) if var_spy > 0 else None

    return {"ann_vol": ann_vol, "beta": beta, "corr": corr}


def build_comparison_chart(df: pd.DataFrame, spy_close: pd.Series, ticker: str, C: dict) -> go.Figure:
    cutoff = df.index[-1] - pd.DateOffset(months=6)
    stock_6m = df["Close"][df.index >= cutoff]
    stock_norm = stock_6m / float(stock_6m.iloc[0])
    spy_norm = pd.Series(dtype=float)

    fig = go.Figure()

    if not spy_close.empty:
        spy_aligned = spy_close.reindex(stock_6m.index, method="ffill").dropna()
        spy_norm = spy_aligned / float(spy_aligned.iloc[0])
        fig.add_trace(go.Scatter(
            x=spy_norm.index, y=spy_norm.values, name="SPY",
            line=dict(color=C["muted"], width=1.5, dash="dot"),
        ))

    fig.add_trace(go.Scatter(
        x=stock_norm.index, y=stock_norm.values, name=ticker,
        line=dict(color=C["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
    ))

    final_ret = float(stock_norm.iloc[-1] - 1)
    spy_ret = float(spy_norm.iloc[-1] - 1) if not spy_norm.empty else None
    subtitle = f"{ticker} {final_ret:+.1%}"
    if spy_ret is not None:
        subtitle += f"  vs  SPY {spy_ret:+.1%}"

    fig.update_layout(
        height=260,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], title="净值 (起始=1)", tickformat=".2f"),
        legend=dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
                    orientation="h", y=1.12, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        title=dict(text=subtitle, font=dict(size=13, color=C["muted"]), x=0, xanchor="left", pad=dict(b=4)),
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color=C["border"], opacity=0.8)
    return fig


def build_chart(df: pd.DataFrame, ticker: str, C: dict) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=("价格 & 均线 & 布林带", "MACD", "RSI")
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="K线",
        increasing_line_color=C["up"],
        decreasing_line_color=C["down"],
        increasing_fillcolor=C["up"],
        decreasing_fillcolor=C["down"],
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB上轨",
        line=dict(color=C["accent"], width=1, dash="dot"), opacity=0.6), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB下轨",
        line=dict(color=C["accent"], width=1, dash="dot"), opacity=0.6,
        fill="tonexty", fillcolor="rgba(99,102,241,0.05)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20",
        line=dict(color=C["warn"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50",
        line=dict(color=C["blue"], width=1.5)), row=1, col=1)

    macd_colors = [C["up"] if v >= 0 else C["down"] for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD柱",
        marker_color=macd_colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
        line=dict(color=C["blue"], width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
        line=dict(color=C["warn"], width=1.5)), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
        line=dict(color=C["accent2"], width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=C["down"], opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=C["up"], opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=680,
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
            font=dict(size=11), orientation="h", y=1.02
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=i, col=1)
        fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"], row=i, col=1)

    return fig


# ── ML Price Prediction ───────────────────────────────────────────────────────

@st.cache_data(ttl=86400)  # daily cache: same ticker + same day tunes only once
def run_ml_prediction(feat_rows: tuple, close_tuple: tuple, dates_tuple: tuple,
                      ticker: str = "", today_str: str = "",
                      atr_tuple: tuple = (),
                      tp: float = 0.05, sl: float = 0.03) -> dict:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.utils.class_weight import compute_sample_weight
        import xgboost as xgb
        import lightgbm as lgb
    except ImportError as e:
        return {"error": f"missing_lib:{e}"}

    HORIZON  = 5
    MIN_ROWS = 100
    N_SPLITS = 5
    EMBARGO  = 5

    try:
        X      = np.array(feat_rows)
        closes = np.array(close_tuple)
        n      = len(X)
        if n < MIN_ROWS + HORIZON:
            return {"error": "insufficient_data"}

        # ── Triple-barrier labels — ATR-based dynamic barriers ────────────────
        # Upper barrier = 2 × ATR14/Close; Lower barrier = 1.5 × ATR14/Close.
        # Falls back to fixed tp/sl when ATR unavailable.
        # Hit upper first → label 2 (止盈); hit lower first → label 0 (止损);
        # neither within HORIZON days → label 1 (无信号).
        valid_n = n - HORIZON
        X_v = X[:valid_n]
        atr_pct = np.array(atr_tuple, dtype=float) if atr_tuple else np.full(n, np.nan)

        labels = np.ones(valid_n, dtype=int)   # default: 1 = 无信号
        for idx in range(valid_n):
            entry   = closes[idx]
            _atr_i  = atr_pct[idx] if idx < len(atr_pct) else np.nan
            _tp_bar = 2.0  * _atr_i  if not np.isnan(_atr_i) else tp
            _sl_bar = 1.5  * _atr_i  if not np.isnan(_atr_i) else sl
            for h in range(1, HORIZON + 1):
                ret = (closes[idx + h] - entry) / entry
                if ret >= _tp_bar:
                    labels[idx] = 2   # 止盈
                    break
                if ret <= -_sl_bar:
                    labels[idx] = 0   # 止损
                    break

        # Time-decay sample weights: exp(-lambda * age_in_days), lambda=0.005
        _DECAY   = 0.005
        _dates_v = np.array([pd.Timestamp(d) for d in list(dates_tuple)[:valid_n]])
        _age     = np.array([(_dates_v[-1] - d).days for d in _dates_v], dtype=float)
        weights  = np.exp(-_DECAY * _age)   # shape (valid_n,), newest=1.0

        # Feature names
        _feat_names = [
            "RSI", "RSI变化", "MACD", "MACD柱", "MACD柱变化",
            "布林%B", "ATR", "MA20偏差", "MA50偏差",
            "1日动量", "3日动量", "5日动量", "10日动量", "20日动量",
            "成交量变化1日", "成交量变化5日", "VWAP偏差",
        ]
        _n_feats = X_v.shape[1]
        _cols = _feat_names[:_n_feats]
        _df = lambda arr: pd.DataFrame(arr, columns=_cols)

        # ── Embargo helper ────────────────────────────────────────────────────
        def _embargoed_splits(cv, X, embargo=EMBARGO):
            for tr, te in cv.split(X):
                cutoff = te[0] - embargo
                tr_emb = tr[tr < cutoff]
                if len(tr_emb) >= 20:
                    yield tr_emb, te

        # ── Inner-CV classification accuracy ──────────────────────────────────
        inner_cv = TimeSeriesSplit(n_splits=3)
        rng = np.random.default_rng(42)

        def _cv_acc(estimator, X_raw, ys, cv, as_df=False, sw=None):
            scores = []
            for tr, te in _embargoed_splits(cv, X_raw):
                if len(tr) < 20:
                    continue
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(X_raw[tr])
                Xte_s = sc.transform(X_raw[te])
                Xtr_ = _df(Xtr_s) if as_df else Xtr_s
                Xte_ = _df(Xte_s) if as_df else Xte_s
                # Class-balanced weights combined with time-decay
                _cls_w = compute_sample_weight("balanced", ys[tr])
                _eff_w = (sw[tr] * _cls_w) if sw is not None else _cls_w
                estimator.fit(Xtr_, ys[tr], sample_weight=_eff_w)
                scores.append(float(np.mean(estimator.predict(Xte_) == ys[te])))
            return float(np.mean(scores)) if scores else 0.0

        # Search spaces
        _xgb_space = {
            "n_estimators":  [100, 150, 200],
            "max_depth":     [3, 4, 5],
            "learning_rate": [0.05, 0.10, 0.15],
            "subsample":     [0.8, 0.9],
        }
        _lgb_space = {
            "n_estimators":  [100, 150, 200],
            "num_leaves":    [15, 31, 63],
            "learning_rate": [0.05, 0.10, 0.15],
        }

        def _tune_fold(X_raw, ys, sw=None):
            """Return best (C, rf_p, xgb_p, lgb_p) using only X_raw/ys."""
            best_C = max(
                [0.01, 0.1, 1.0, 10.0, 100.0],
                key=lambda c: _cv_acc(
                    LogisticRegression(C=c, max_iter=500, random_state=42,
                                       solver="lbfgs"),
                    X_raw, ys, inner_cv, sw=sw,
                ),
            )
            rf_p = {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 4}
            _bs = -1.0
            for n_est in [100, 150]:
                for depth in [4, 6]:
                    p = {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": 4}
                    s = _cv_acc(RandomForestClassifier(**p, random_state=42, n_jobs=1),
                                X_raw, ys, inner_cv, sw=sw)
                    if s > _bs:
                        _bs, rf_p = s, p
            xgb_p = {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.8}
            _bs = -1.0
            for _ in range(8):
                p = {k: v[int(rng.integers(len(v)))] for k, v in _xgb_space.items()}
                s = _cv_acc(xgb.XGBClassifier(**p, random_state=42, verbosity=0,
                                               n_jobs=1, eval_metric="mlogloss"),
                            X_raw, ys, inner_cv, sw=sw)
                if s > _bs:
                    _bs, xgb_p = s, p
            lgb_p = {"n_estimators": 150, "num_leaves": 31, "learning_rate": 0.10}
            _bs = -1.0
            for _ in range(8):
                p = {k: v[int(rng.integers(len(v)))] for k, v in _lgb_space.items()}
                s = _cv_acc(lgb.LGBMClassifier(**p, random_state=42, verbose=-1, n_jobs=1),
                            X_raw, ys, inner_cv, as_df=True, sw=sw)
                if s > _bs:
                    _bs, lgb_p = s, p
            return best_C, rf_p, xgb_p, lgb_p

        # ── Walk-forward evaluation ───────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        outer_splits = list(_embargoed_splits(tscv, X_v))

        fold_accs = {m: [] for m in
                     ["LogReg", "RandomForest", "XGBoost", "LightGBM", "Ensemble"]}

        last_fold_params = None
        for fold_tr, fold_te in outer_splits:
            if len(fold_tr) < 30 or len(fold_te) < 30:
                continue

            w_fold = weights[fold_tr] * compute_sample_weight("balanced", labels[fold_tr])
            f_C, f_rf, f_xgb, f_lgb = _tune_fold(X_v[fold_tr], labels[fold_tr], sw=w_fold)

            sc = StandardScaler()
            Xtr = sc.fit_transform(X_v[fold_tr])
            Xte = sc.transform(X_v[fold_te])
            lbl_tr, lbl_te = labels[fold_tr], labels[fold_te]
            Xtr_df, Xte_df = _df(Xtr), _df(Xte)

            clfs = {
                "LogReg":       LogisticRegression(C=f_C, max_iter=500, random_state=42,
                                                   solver="lbfgs"),
                "RandomForest": RandomForestClassifier(**f_rf, random_state=42, n_jobs=1),
                "XGBoost":      xgb.XGBClassifier(**f_xgb, random_state=42, verbosity=0,
                                                   n_jobs=1, eval_metric="mlogloss"),
                "LightGBM":     lgb.LGBMClassifier(**f_lgb, random_state=42, verbose=-1, n_jobs=1),
            }
            preds = {}
            for name, clf in clfs.items():
                Xtr_ = Xtr_df if name == "LightGBM" else Xtr
                Xte_ = Xte_df if name == "LightGBM" else Xte
                clf.fit(Xtr_, lbl_tr, sample_weight=w_fold)
                preds[name] = clf.predict(Xte_)
            # Majority-vote ensemble
            _ens_arr = np.array(list(preds.values()))
            preds["Ensemble"] = np.array([
                max(set(col.tolist()), key=col.tolist().count)
                for col in _ens_arr.T
            ])
            for name, p in preds.items():
                fold_accs[name].append(float(np.mean(p == lbl_te)))

            last_fold_params = {"C": f_C, "rf": f_rf, "xgb": f_xgb, "lgb": f_lgb,
                                 "tr": fold_tr, "te": fold_te}

        model_avg_acc = {k: float(np.mean(v)) if v else 0.0
                         for k, v in fold_accs.items()}

        # ── Global tuning + final retrain on all data ─────────────────────────
        w_all = weights * compute_sample_weight("balanced", labels)
        best_C, best_rf_p, best_xgb_p, best_lgb_p = _tune_fold(X_v, labels, sw=w_all)

        scaler2  = StandardScaler()
        X_all_s  = scaler2.fit_transform(X_v)
        x_now    = scaler2.transform(X_v[[-1]])
        X_all_df = _df(X_all_s)
        x_now_df = _df(x_now)

        lr_f  = LogisticRegression(C=best_C, max_iter=500, random_state=42,
                                   solver="lbfgs").fit(X_all_s, labels, sample_weight=w_all)
        rf_f  = RandomForestClassifier(**best_rf_p, random_state=42, n_jobs=1).fit(
                                   X_all_s, labels, sample_weight=w_all)
        xgb_f = xgb.XGBClassifier(**best_xgb_p, random_state=42, verbosity=0,
                                   n_jobs=1, eval_metric="mlogloss").fit(
                                   X_all_s, labels, sample_weight=w_all)
        lgb_f = lgb.LGBMClassifier(**best_lgb_p, random_state=42,
                                   verbose=-1, n_jobs=1).fit(
                                   X_all_df, labels, sample_weight=w_all)

        # Extract P(止盈)=P(1), P(止损)=P(-1) for each classifier
        def _extract_proba(clf, x):
            p = clf.predict_proba(x)[0]
            cls = list(clf.classes_)
            p_sl      = float(p[cls.index(0)]) if 0 in cls else 0.0
            p_neutral = float(p[cls.index(1)]) if 1 in cls else 0.0
            p_tp      = float(p[cls.index(2)]) if 2 in cls else 0.0
            return p_sl, p_neutral, p_tp

        lr_sl,  lr_neutral,  lr_tp  = _extract_proba(lr_f,  x_now)
        rf_sl,  rf_neutral,  rf_tp  = _extract_proba(rf_f,  x_now)
        xgb_sl, xgb_neutral, xgb_tp = _extract_proba(xgb_f, x_now)
        lgb_sl, lgb_neutral, lgb_tp = _extract_proba(lgb_f, x_now_df)

        ens_tp      = (lr_tp  + rf_tp  + xgb_tp  + lgb_tp)  / 4
        ens_sl      = (lr_sl  + rf_sl  + xgb_sl  + lgb_sl)  / 4
        ens_neutral = (lr_neutral + rf_neutral + xgb_neutral + lgb_neutral) / 4

        model_probs = {
            "LogReg":       {"tp": lr_tp,  "sl": lr_sl,  "neutral": lr_neutral},
            "RandomForest": {"tp": rf_tp,  "sl": rf_sl,  "neutral": rf_neutral},
            "XGBoost":      {"tp": xgb_tp, "sl": xgb_sl, "neutral": xgb_neutral},
            "LightGBM":     {"tp": lgb_tp, "sl": lgb_sl, "neutral": lgb_neutral},
            "Ensemble":     {"tp": ens_tp, "sl": ens_sl, "neutral": ens_neutral},
        }

        # Feature importance from RF
        importance = rf_f.feature_importances_[:_n_feats].tolist()
        feat_names = _cols

        # ── OOS from last fold — predict_proba on test set ────────────────────
        last_tr = last_fold_params["tr"] if last_fold_params else outer_splits[-1][0]
        last_te = last_fold_params["te"] if last_fold_params else outer_splits[-1][1]
        _lf_C   = last_fold_params["C"]   if last_fold_params else best_C
        _lf_rf  = last_fold_params["rf"]  if last_fold_params else best_rf_p
        _lf_xgb = last_fold_params["xgb"] if last_fold_params else best_xgb_p
        _lf_lgb = last_fold_params["lgb"] if last_fold_params else best_lgb_p

        sc_oos  = StandardScaler()
        Xtr_oos = sc_oos.fit_transform(X_v[last_tr])
        Xte_oos = sc_oos.transform(X_v[last_te])
        Xtr_oos_df, Xte_oos_df = _df(Xtr_oos), _df(Xte_oos)
        lbl_oos_tr = labels[last_tr]

        def _oos_proba(clf, Xtr, Xte, lbl_tr, sw=None, as_df=False):
            _cls_w = compute_sample_weight("balanced", lbl_tr)
            _eff_w = (sw * _cls_w) if sw is not None else _cls_w
            clf.fit(_df(Xtr) if as_df else Xtr, lbl_tr, sample_weight=_eff_w)
            proba = clf.predict_proba(_df(Xte) if as_df else Xte)
            cls = list(clf.classes_)
            tp_i = cls.index(2) if 2 in cls else None
            sl_i = cls.index(0) if 0 in cls else None
            p_tp = proba[:, tp_i].tolist() if tp_i is not None else [0.0] * len(Xte)
            p_sl = proba[:, sl_i].tolist() if sl_i is not None else [0.0] * len(Xte)
            return p_tp, p_sl

        oos_tp, oos_sl = {}, {}
        _oos_clfs = {
            "LogReg":       LogisticRegression(C=_lf_C, max_iter=500, random_state=42,
                                               solver="lbfgs"),
            "RandomForest": RandomForestClassifier(**_lf_rf, random_state=42, n_jobs=1),
            "XGBoost":      xgb.XGBClassifier(**_lf_xgb, random_state=42, verbosity=0,
                                               n_jobs=1, eval_metric="mlogloss"),
        }
        w_oos = weights[last_tr]
        for name, clf in _oos_clfs.items():
            tp_p, sl_p = _oos_proba(clf, Xtr_oos, Xte_oos, lbl_oos_tr, sw=w_oos)
            oos_tp[name] = tp_p
            oos_sl[name] = sl_p
        lgb_tp_oos, lgb_sl_oos = _oos_proba(
            lgb.LGBMClassifier(**_lf_lgb, random_state=42, verbose=-1, n_jobs=1),
            Xtr_oos, Xte_oos, lbl_oos_tr, sw=w_oos, as_df=True,
        )
        oos_tp["LightGBM"] = lgb_tp_oos
        oos_sl["LightGBM"] = lgb_sl_oos
        _models4 = ["LogReg", "RandomForest", "XGBoost", "LightGBM"]
        oos_tp["Ensemble"] = [(a + b + c + d) / 4
                              for a, b, c, d in zip(*[oos_tp[m] for m in _models4])]
        oos_sl["Ensemble"] = [(a + b + c + d) / 4
                              for a, b, c, d in zip(*[oos_sl[m] for m in _models4])]

        # Current ATR: last non-NaN value in the full atr_pct array (most recent bar)
        _valid_atr = atr_pct[~np.isnan(atr_pct)]
        _cur_atr_v = float(_valid_atr[-1]) if len(_valid_atr) > 0 else None

        return {
            "cur_close":       float(closes[-1]),
            "ens_tp":          ens_tp,
            "ens_sl":          ens_sl,
            "ens_neutral":     ens_neutral,
            "model_probs":     model_probs,
            "model_avg_acc":   model_avg_acc,
            "model_fold_accs": {k: v for k, v in fold_accs.items()},
            "best_params":     {"C": best_C, "rf": best_rf_p,
                                "xgb": best_xgb_p, "lgb": best_lgb_p},
            "feat_importance": list(zip(feat_names, importance)),
            "test_dates":      list(dates_tuple)[last_te[0]: last_te[-1] + 1],
            "actual_labels":   [int({0: -1, 1: 0, 2: 1}[v]) for v in labels[last_te].tolist()],
            "oos_tp":          oos_tp,
            "oos_sl":          oos_sl,
            "label_counts": {
                "止盈": int((labels == 2).sum()),
                "止损": int((labels == 0).sum()),
                "中立": int((labels == 1).sum()),
            },
            "horizon": HORIZON, "n_total": valid_n, "n_splits": N_SPLITS,
            "embargo": EMBARGO,
            "cur_atr_pct":  _cur_atr_v,
            "cur_tp_bar":   float(2.0 * _cur_atr_v) if _cur_atr_v is not None else tp,
            "cur_sl_bar":   float(1.5 * _cur_atr_v) if _cur_atr_v is not None else sl,
            "atr_based":    len(atr_tuple) > 0 and _cur_atr_v is not None,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


def build_ml_prob_chart(ml: dict, C: dict) -> go.Figure:
    """Horizontal stacked bar showing TP / Neutral / SL probabilities per model."""
    models = ["LogReg", "RandomForest", "XGBoost", "LightGBM", "Ensemble"]
    probs  = ml["model_probs"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="止盈概率", orientation="h",
        y=models,
        x=[probs[m]["tp"] * 100 for m in models],
        marker_color=C["up"],
        text=[f"{probs[m]['tp']:.1%}" for m in models],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="%{y} 止盈: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="中立概率", orientation="h",
        y=models,
        x=[probs[m]["neutral"] * 100 for m in models],
        marker_color=C["border"],
        text=[f"{probs[m]['neutral']:.1%}" for m in models],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="%{y} 中立: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="止损概率", orientation="h",
        y=models,
        x=[probs[m]["sl"] * 100 for m in models],
        marker_color=C["down"],
        text=[f"{probs[m]['sl']:.1%}" for m in models],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="%{y} 止损: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        height=260,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis=dict(title="概率 (%)", gridcolor=C["border"],
                   zerolinecolor=C["border"], range=[0, 100]),
        yaxis=dict(gridcolor=C["border"]),
        legend=dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
                    orientation="h", y=1.14, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


def build_ml_importance_chart(ml: dict, C: dict) -> go.Figure:
    feats = sorted(ml["feat_importance"], key=lambda x: x[1])
    names = [f[0] for f in feats]
    vals  = [f[1] * 100 for f in feats]

    bar_colors = [C["accent"] if v >= max(vals) * 0.6 else C["border"] for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(family="Space Mono, monospace", size=11, color=C["text"]),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis=dict(title="重要性 (%)", gridcolor=C["border"],
                   zerolinecolor=C["border"], range=[0, max(vals) * 1.40]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        margin=dict(l=0, r=55, t=0, b=0),
        showlegend=False,
    )
    return fig


def build_ml_backtest_chart(ml: dict, C: dict) -> go.Figure:
    """OOS chart: predicted TP probability lines + actual label outcome markers."""
    test_dates   = [pd.Timestamp(d) for d in ml["test_dates"]]
    oos_tp       = ml.get("oos_tp", {})
    actual_lbls  = ml.get("actual_labels", [])

    fig = go.Figure()

    # Actual outcome: vertical tick markers at top/middle/bottom (100/50/0)
    _lbl_cfg = {1: (C["up"], "止盈", 95), -1: (C["down"], "止损", 5), 0: (C["muted"], "中立", 50)}
    for lval, (lcolor, lname, ly) in _lbl_cfg.items():
        idx_list = [i for i, v in enumerate(actual_lbls) if v == lval]
        if idx_list:
            fig.add_trace(go.Scatter(
                x=[test_dates[i] for i in idx_list],
                y=[ly] * len(idx_list),
                mode="markers",
                marker=dict(symbol="line-ns-open", size=8,
                            color=lcolor,
                            line=dict(color=lcolor, width=2)),
                name=f"实际:{lname}",
            ))

    _oos_styles = [
        ("LogReg",       C["warn"],    "dash"),
        ("RandomForest", C["up"],      "dot"),
        ("XGBoost",      C["accent2"], "dashdot"),
        ("LightGBM",     C["blue"],    "dot"),
        ("Ensemble",     C["down"],    "solid"),
    ]
    for name, color, dash in _oos_styles:
        if name in oos_tp:
            fig.add_trace(go.Scatter(
                x=test_dates, y=[v * 100 for v in oos_tp[name]],
                name=f"{name} 止盈概率",
                line=dict(color=color, width=1.5 if name != "Ensemble" else 2.5, dash=dash),
            ))

    fig.add_hline(y=33.3, line_color=C["border"], opacity=0.6, line_dash="dash",
                  annotation_text="随机基线", annotation_font_size=10)

    fig.update_layout(
        height=240,
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="DM Sans, sans-serif", color=C["muted"], size=12),
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
                   title="止盈概率 (%)", range=[0, 100]),
        legend=dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
                    orientation="h", y=1.12, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 参数配置")
    ticker = st.text_input("股票代码", value="AAPL", placeholder="AAPL / TSLA / NVDA").upper().strip()
    period = st.selectbox("回看周期", ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "7y", "10y", "max"], index=2)
    api_key = st.text_input("API Key (Gemini 或 DeepSeek)", type="password",
                            placeholder="AIza... 或 sk-...",
                            help="Gemini Key 以 AIza 开头；DeepSeek Key 以 sk- 开头（自动识别）")

    st.markdown("---")
    st.markdown("### 回测参数")
    holding_days = st.slider("最长持仓天数", min_value=3, max_value=20, value=5, step=1)
    stop_loss_pct = st.slider("止损比例", min_value=1, max_value=10, value=3, step=1, format="%d%%")
    take_profit_pct = st.slider("止盈比例", min_value=1, max_value=20, value=5, step=1, format="%d%%")
    slippage_pct = st.slider("滑点", min_value=0.0, max_value=1.0, value=0.1, step=0.05, format="%.2f%%")
    commission_pct = st.slider("手续费（单边）", min_value=0.0, max_value=0.5, value=0.1, step=0.05, format="%.2f%%")
    st.markdown("**时间止损**")
    time_stop_enabled = st.toggle("启用时间止损", value=True)
    time_stop_days = st.slider("时间止损天数", min_value=3, max_value=10, value=5, step=1,
                               disabled=not time_stop_enabled)
    time_stop_min_pnl_pct = st.slider("最低盈利要求", min_value=0.0, max_value=2.0, value=0.5,
                                      step=0.1, format="%.1f%%", disabled=not time_stop_enabled)

    run_btn = st.button("开始分析")

    st.markdown("---")
    st.markdown("**快捷股票**")
    cols = st.columns(2)
    quick = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL"]
    for i, q in enumerate(quick):
        if cols[i % 2].button(q, key=f"q_{q}"):
            ticker = q
            run_btn = True

    st.markdown("---")
    st.markdown("### 多股票扫描")
    with st.form("scanner_form"):
        _scan_input = st.text_input(
            "最多10只股票（逗号分隔）",
            placeholder="AAPL, TSLA, NVDA, MSFT",
            value="AAPL, TSLA, NVDA, MSFT",
        )
        _scan_btn = st.form_submit_button("扫描信号", use_container_width=True)

    if _scan_btn:
        _scan_tickers = list(dict.fromkeys(
            t.strip().upper() for t in _scan_input.split(",") if t.strip()
        ))[:10]
        with st.spinner(f"扫描 {len(_scan_tickers)} 只股票..."):
            # Compute SPY bear flag once (shared across all tickers in this scan)
            _scan_spy = fetch_spy_full("6mo")
            _scan_spy_bear = False
            if not _scan_spy.empty and {"MA20", "MA50"}.issubset(_scan_spy.columns):
                _last_spy = _scan_spy.dropna().iloc[-1] if not _scan_spy.dropna().empty else None
                if _last_spy is not None:
                    _scan_spy_bear = float(_last_spy["MA20"]) < float(_last_spy["MA50"])
            _scan_raw = [scan_ticker(t, spy_bear=_scan_spy_bear) for t in _scan_tickers]
        _scan_raw.sort(key=lambda r: (r.get("sort_key", 1), r.get("ticker", "")))
        st.session_state["scan_results"] = _scan_raw

    if "scan_results" in st.session_state:
        _rows = st.session_state["scan_results"]
        _sig_color = {"BUY": "#10b981", "HOLD": "#f59e0b", "SELL": "#f43f5e"}
        _sig_zh    = {"BUY": "买入", "HOLD": "观望", "SELL": "卖出"}

        _header = (
            "<table style='width:100%;border-collapse:collapse;"
            "font-size:11px;font-family:DM Sans,sans-serif;'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:3px 4px;color:#64748b;font-weight:500;'>代码</th>"
            "<th style='text-align:right;padding:3px 4px;color:#64748b;font-weight:500;'>价格</th>"
            "<th style='text-align:center;padding:3px 4px;color:#64748b;font-weight:500;'>信号</th>"
            "<th style='text-align:right;padding:3px 4px;color:#64748b;font-weight:500;'>RSI</th>"
            "<th style='text-align:right;padding:3px 4px;color:#64748b;font-weight:500;'>涨跌</th>"
            "</tr></thead><tbody>"
        )
        _body = ""
        for _r in _rows:
            if _r.get("error"):
                _body += (
                    f"<tr><td style='padding:3px 4px;color:#64748b;' colspan='5'>"
                    f"{_r['ticker']} — 数据获取失败</td></tr>"
                )
                continue
            _sc  = _sig_color.get(_r["signal"], "#64748b")
            _sz  = _sig_zh.get(_r["signal"], _r["signal"])
            _chg = _r["chg_pct"]
            _cc  = "#10b981" if _chg >= 0 else "#f43f5e"
            _rsi_c = "#f43f5e" if _r["rsi"] > 70 else "#10b981" if _r["rsi"] < 35 else "#94a3b8"
            _body += (
                f"<tr style='border-top:1px solid #1e2d4a;'>"
                f"<td style='padding:4px 4px;font-weight:600;color:#e2e8f0;"
                f"font-family:Space Mono,monospace;'>{_r['ticker']}</td>"
                f"<td style='padding:4px 4px;text-align:right;color:#e2e8f0;"
                f"font-family:Space Mono,monospace;'>${_r['price']:.2f}</td>"
                f"<td style='padding:4px 4px;text-align:center;'>"
                f"<span style='background:{_sc}22;color:{_sc};border-radius:4px;"
                f"padding:1px 6px;font-weight:600;font-size:10px;'>{_sz}</span></td>"
                f"<td style='padding:4px 4px;text-align:right;color:{_rsi_c};"
                f"font-family:Space Mono,monospace;'>{_r['rsi']:.1f}</td>"
                f"<td style='padding:4px 4px;text-align:right;color:{_cc};"
                f"font-family:Space Mono,monospace;'>{_chg:+.2f}%</td>"
                f"</tr>"
            )
        st.markdown(_header + _body + "</tbody></table>", unsafe_allow_html=True)
        _n_buy  = sum(1 for r in _rows if r.get("signal") == "BUY")
        _n_sell = sum(1 for r in _rows if r.get("signal") == "SELL")
        st.markdown(
            f"<div style='font-size:10px;color:#64748b;margin-top:6px;'>"
            f"买入 <span style='color:#10b981;font-weight:600;'>{_n_buy}</span> &nbsp;·&nbsp; "
            f"卖出 <span style='color:#f43f5e;font-weight:600;'>{_n_sell}</span> &nbsp;·&nbsp; "
            f"观望 <span style='color:#f59e0b;font-weight:600;'>{len(_rows)-_n_buy-_n_sell}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#4a5568; line-height:1.6;'>
    本工具仅供学习研究，不构成任何投资建议。股市有风险，投资须谨慎。
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='header-banner'>
  <div style='font-family:Space Mono,monospace;font-size:11px;color:{C["muted"]};letter-spacing:2px;margin-bottom:6px;'>QUANTAI TERMINAL</div>
  <h1 style='margin:0;font-size:24px;color:{C["text"]};font-family:Space Mono,monospace;font-weight:700;'>股票量化分析系统</h1>
  <p style='margin:6px 0 0;color:{C["muted"]};font-size:13px;'>
    技术指标 · AI 解读 · 买卖信号 &nbsp;·&nbsp; Yahoo Finance &nbsp;·&nbsp; Claude AI
  </p>
</div>
""", unsafe_allow_html=True)

if run_btn:
    st.session_state["analysis_active"] = True
    # Clear cached AI results so the new ticker/period gets fresh analysis
    for _k in list(st.session_state.keys()):
        if _k.startswith("_sentiment_") or _k.startswith("_analysis_"):
            del st.session_state[_k]

if not st.session_state.get("analysis_active"):
    st.markdown(f"""
    <div style='text-align:center; padding:80px 0; color:{C["muted"]};'>
      <div style='font-family:Space Mono,monospace;font-size:13px;letter-spacing:2px;margin-bottom:12px;color:{C["dim"]};'>QUANTAI TERMINAL</div>
      <div style='font-family:Space Mono,monospace;font-size:16px;color:{C["text"]};'>在左侧输入股票代码，点击开始分析</div>
      <div style='font-size:13px; margin-top:8px;color:{C["dim"]};'>支持所有美股代码：AAPL · TSLA · NVDA · MSFT ···</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Data fetch ────────────────────────────────────────────────────────────────
try:
    with st.spinner(f"正在拉取 {ticker} 数据..."):
        df = fetch_data(ticker, period)
except YFRateLimitError:
    st.error("⏱️ Yahoo Finance 请求频率已达上限，请等待 1–2 分钟后重试。")
    st.stop()
except Exception as e:
    st.error(f"❌ 数据拉取失败：{e}")
    st.stop()

if df.empty:
    st.error(f"❌ 找不到股票代码 **{ticker}**，请检查拼写后重试。")
    st.stop()

if len(df) < 20:
    st.error(f"❌ **{ticker}** 数据不足（仅 {len(df)} 行），请切换至更长的回看周期（建议 3mo 或以上）。")
    st.stop()

try:
    info = fetch_ticker_info(ticker)
    if not info:
        raise ValueError("empty response")
except YFRateLimitError:
    info = {}
    st.warning("⚠️ Yahoo Finance 频率限制，基本面数据暂时无法加载，稍后刷新可恢复。")
except Exception:
    info = {}
    st.warning("⚠️ 无法获取基本面数据，相关字段将显示 N/A。")

company_name = info.get("longName", ticker)

# ── Header metrics ────────────────────────────────────────────────────────────
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2])
change = last_close - prev_close
change_pct = change / prev_close * 100
vol = float(df["Volume"].iloc[-1])
high52 = info.get("fiftyTwoWeekHigh") or float(df["Close"].max())
low52 = info.get("fiftyTwoWeekLow") or float(df["Close"].min())

st.markdown(f"""
<div style='margin-bottom:8px;'>
  <span class='ticker-tag'>{ticker}</span>
  <span style='font-size:20px; font-weight:600;'>{company_name}</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("收盘价", f"${last_close:.2f}", f"{change_pct:+.2f}%")
c2.metric("日涨跌", f"${change:+.2f}", "")
c3.metric("成交量", f"{vol/1e6:.1f}M", "")
c4.metric("52周高", f"${high52:.2f}", "")
c5.metric("52周低", f"${low52:.2f}", "")

# ── Fundamentals ─────────────────────────────────────────────────────────────
def _fmt_num(val, prefix="", suffix="", decimals=2):
    """Format large numbers into readable strings; return 'N/A' if missing."""
    if val is None or (isinstance(val, float) and (val != val)):
        return None
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    if abs(val) >= 1e12:
        return f"{prefix}{val/1e12:.{decimals}f}T{suffix}"
    if abs(val) >= 1e9:
        return f"{prefix}{val/1e9:.{decimals}f}B{suffix}"
    if abs(val) >= 1e6:
        return f"{prefix}{val/1e6:.{decimals}f}M{suffix}"
    return f"{prefix}{val:.{decimals}f}{suffix}"

def _fmt_plain(val, decimals=2, suffix=""):
    if val is None or (isinstance(val, float) and (val != val)):
        return None
    try:
        return f"{float(val):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return None

fund_fields = [
    ("市盈率 P/E",    _fmt_plain(info.get("trailingPE"), decimals=1)),
    ("市值",          _fmt_num(info.get("marketCap"), prefix="$")),
    ("营收 TTM",      _fmt_num(info.get("totalRevenue"), prefix="$")),
    ("净利润 TTM",    _fmt_num(info.get("netIncomeToCommon"), prefix="$")),
    ("负债率 D/E",    _fmt_plain(info.get("debtToEquity"), decimals=2)),
    ("Beta",          _fmt_plain(info.get("beta"), decimals=2)),
    ("股息率",        _fmt_plain(info.get("dividendYield") and info["dividendYield"] * 100, decimals=2, suffix="%")),
]

fund_cols = st.columns(len(fund_fields))
for col, (label, val) in zip(fund_cols, fund_fields):
    display = f"<div class='fund-value'>{val}</div>" if val else "<div class='fund-na'>N/A</div>"
    col.markdown(f"""
    <div class='fund-card'>
      <div class='fund-label'>{label}</div>
      {display}
    </div>""", unsafe_allow_html=True)

# ── Risk Assessment ──────────────────────────────────────────────────────────
spy_close = fetch_spy_6m()
risk = compute_risk_metrics(df, spy_close, info.get("beta"))

def _risk_level(vol, beta):
    score = 0
    if vol is not None:
        score += 2 if vol > 0.40 else (1 if vol > 0.20 else 0)
    if beta is not None:
        score += 2 if abs(beta) > 1.5 else (1 if abs(beta) > 1.0 else 0)
    if score >= 3:
        return "高风险", "#f43f5e", "rgba(244,63,94,0.12)"
    elif score >= 1:
        return "中风险", "#f59e0b", "rgba(245,158,11,0.12)"
    else:
        return "低风险", "#10b981", "rgba(16,185,129,0.12)"

rl_label, rl_color, rl_bg = _risk_level(risk["ann_vol"], risk["beta"])

rc1, rc2, rc3, rc4 = st.columns(4)
risk_cards = [
    (rc1, "年化波动率", f"{risk['ann_vol']:.1%}" if risk["ann_vol"] is not None else "N/A",
     ("高" if risk["ann_vol"] and risk["ann_vol"] > 0.40 else "中" if risk["ann_vol"] and risk["ann_vol"] > 0.20 else "低"),
     ("#f43f5e" if risk["ann_vol"] and risk["ann_vol"] > 0.40 else "#f59e0b" if risk["ann_vol"] and risk["ann_vol"] > 0.20 else "#10b981")),
    (rc2, "Beta 值", f"{risk['beta']:.2f}" if risk["beta"] is not None else "N/A",
     ("高波动" if risk["beta"] and abs(risk["beta"]) > 1.5 else "中波动" if risk["beta"] and abs(risk["beta"]) > 1.0 else "低波动"),
     ("#f43f5e" if risk["beta"] and abs(risk["beta"]) > 1.5 else "#f59e0b" if risk["beta"] and abs(risk["beta"]) > 1.0 else "#10b981")),
    (rc3, "与SPY相关性", f"{risk['corr']:.2f}" if risk["corr"] is not None else "N/A",
     ("强相关" if risk["corr"] and abs(risk["corr"]) > 0.7 else "中相关" if risk["corr"] and abs(risk["corr"]) > 0.4 else "弱相关"),
     (C["dim"] if risk["corr"] and abs(risk["corr"]) > 0.7 else C["accent2"] if risk["corr"] and abs(risk["corr"]) > 0.4 else C["up"])),
    (rc4, "综合风险等级", rl_label, "", rl_color),
]
for col, lbl, val, badge, color in risk_cards:
    badge_html = f"<div class='risk-badge' style='background:{color}22;color:{color};'>{badge}</div>" if badge else ""
    col.markdown(f"""
    <div class='risk-card' style='border-color:{color}44;'>
      <div class='risk-label'>{lbl}</div>
      <div class='risk-value' style='color:{color};'>{val}</div>
      {badge_html}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Prepare ML features & pre-compute OOS predictions (used by signal + backtest) ──
# df["Close"] is already split/dividend-adjusted (fetch_data uses auto_adjust=True),
# so all price-based features below are free from stock-split contamination.
_close_arr = df["Close"].values.astype(float)
_high_arr  = df["High"].values.astype(float)
_low_arr   = df["Low"].values.astype(float)
_vol_arr   = df["Volume"].values.astype(float)
_n_ml      = len(_close_arr)
_rsi_arr   = df["RSI"].values.astype(float)
_macd_arr  = df["MACD"].values.astype(float)
_macd_h_arr = df["MACD_Hist"].values.astype(float)
_bb_u_arr  = df["BB_Upper"].values.astype(float)
_bb_l_arr  = df["BB_Lower"].values.astype(float)
_ma20_arr  = df["MA20"].values.astype(float)
_ma50_arr  = df["MA50"].values.astype(float)

_lc = np.log(np.where(_close_arr > 0, _close_arr, 1.0))

# RSI & change
_rsi_n    = _rsi_arr / 100
_rsi_chg  = np.full(_n_ml, np.nan); _rsi_chg[1:] = _rsi_arr[1:] - _rsi_arr[:-1]

# MACD (price-normalized) & hist change
_macd_n   = np.where(_close_arr > 0, _macd_arr / _close_arr, 0.0)
_macd_h_n = np.where(_close_arr > 0, _macd_h_arr / _close_arr, 0.0)
_macd_h_chg = np.full(_n_ml, np.nan); _macd_h_chg[1:] = _macd_h_arr[1:] - _macd_h_arr[:-1]

# Bollinger %B
_bb_width = np.where(_bb_u_arr - _bb_l_arr > 0, _bb_u_arr - _bb_l_arr, 1.0)
_bb_pctb  = (_close_arr - _bb_l_arr) / _bb_width

# ATR_ratio: 14-day rolling mean of (TR / Close), fully dimensionless
_prev_close = np.roll(_close_arr, 1)
_prev_close[0] = _close_arr[0]
_tr_ratio = np.maximum(
    (_high_arr - _low_arr) / np.where(_close_arr > 0, _close_arr, 1.0),
    np.maximum(
        np.abs(_high_arr - _prev_close) / np.where(_close_arr > 0, _close_arr, 1.0),
        np.abs(_low_arr  - _prev_close) / np.where(_close_arr > 0, _close_arr, 1.0),
    )
)
_atr_n = np.full(_n_ml, np.nan)
for _i in range(13, _n_ml):
    _atr_n[_i] = _tr_ratio[_i - 13: _i + 1].mean()

# MA deviation (%, not ratio)
_ma20_dev = np.where(_close_arr > 0, (_close_arr - _ma20_arr) / _close_arr, np.nan)
_ma50_dev = np.where(_close_arr > 0, (_close_arr - _ma50_arr) / _close_arr, np.nan)

# Momentum: 1/3/5/10/20-day log returns
def _logret(arr, lag):
    r = np.full(_n_ml, np.nan)
    if _n_ml > lag:
        r[lag:] = arr[lag:] - arr[:-lag]
    return r

_ret1  = _logret(_lc, 1)
_ret3  = _logret(_lc, 3)
_ret5  = _logret(_lc, 5)
_ret10 = _logret(_lc, 10)
_ret20 = _logret(_lc, 20)

# Volume change: 1-day and 5-day log change
_lv = np.log(np.where(_vol_arr > 0, _vol_arr, 1.0))
_vol_chg1 = _logret(_lv, 1)
_vol_chg5 = _logret(_lv, 5)

# VWAP deviation: 20-day rolling VWAP vs close
_cv = _close_arr * _vol_arr
_vwap20 = np.full(_n_ml, np.nan)
for _i in range(19, _n_ml):
    _sv = _vol_arr[_i - 19: _i + 1].sum()
    _vwap20[_i] = _cv[_i - 19: _i + 1].sum() / _sv if _sv > 0 else _close_arr[_i]
_vwap_dev = np.where(_close_arr > 0, (_close_arr - _vwap20) / _close_arr, np.nan)

# Stack all 17 features
_ml_feat = np.column_stack([
    _rsi_n, _rsi_chg, _macd_n, _macd_h_n, _macd_h_chg,
    _bb_pctb, _atr_n, _ma20_dev, _ma50_dev,
    _ret1, _ret3, _ret5, _ret10, _ret20,
    _vol_chg1, _vol_chg5, _vwap_dev,
])

# Rolling Z-score normalization (window=20, min 5 obs) — removes non-stationarity
_zwin = 20
_ml_feat_df_z = pd.DataFrame(_ml_feat)
_roll_mean_z  = _ml_feat_df_z.rolling(_zwin, min_periods=5).mean().values
_roll_std_z   = _ml_feat_df_z.rolling(_zwin, min_periods=5).std().values
_roll_std_z   = np.where(_roll_std_z < 1e-9, np.nan, _roll_std_z)
_ml_feat = np.where(
    np.isnan(_ml_feat) | np.isnan(_roll_mean_z) | np.isnan(_roll_std_z),
    np.nan,
    (_ml_feat - _roll_mean_z) / _roll_std_z,
)

_ml_mask  = ~np.any(np.isnan(_ml_feat) | np.isinf(_ml_feat), axis=1)
_ml_feat  = _ml_feat[_ml_mask]
_ml_close = _close_arr[_ml_mask]
_ml_dates = df.index[_ml_mask]
# Raw ATR14/Close ratio (pre-Z-score) for dynamic triple-barrier barriers
_ml_atr   = _atr_n[_ml_mask]   # ATR as fraction of close, same mask

# ── Beta adaptive strategy ────────────────────────────────────────────────────
_beta_raw = info.get("beta")
try:
    _beta_val: float | None = float(_beta_raw) if _beta_raw is not None else None
except (TypeError, ValueError):
    _beta_val = None

if _beta_val is not None and _beta_val < 0.8:
    _beta_mode    = "稳健模式"
    _beta_mode_c  = C["accent2"]   # calm color for low-vol
    _uptrend_thr  = 1.0
    _range_thr    = 1.5
    _rsi_weight   = 1
elif _beta_val is not None and _beta_val > 1.2:
    _beta_mode    = "动量模式"
    _beta_mode_c  = C["up"]
    _uptrend_thr  = 1.5
    _range_thr    = 2.0
    _rsi_weight   = 2
else:
    _beta_mode    = None           # no badge shown for mid-range beta
    _beta_mode_c  = C["muted"]
    _uptrend_thr  = 1.5
    _range_thr    = 2.0
    _rsi_weight   = 2

# ── Pre-fetch news sentiment (so signal system + backtest can use it) ────────
# Skip entirely if no API Key — sentiment_score stays None and bonus is 0
_sentiment_for_signal: float | None = None
if api_key:
    _senti_key = f"_sentiment_{ticker}"
    if _senti_key not in st.session_state:
        with st.spinner("分析新闻情绪..."):
            _news_for_senti = fetch_news(ticker)
            _hl = [n["title"] for n in _news_for_senti] if _news_for_senti else []
            if _hl:
                st.session_state[_senti_key] = get_news_sentiment(ticker, _hl, api_key)
    _cached_senti = st.session_state.get(_senti_key)
    if _cached_senti and "error" not in _cached_senti:
        _sentiment_for_signal = float(_cached_senti.get("score", 0))

# ── SPY data for beta-tiered penalty + RS_SPY display ────────────────────────
# (Sector ETF helpers are kept for future use — see get_sector_etf / fetch_etf_full —
#  but they no longer feed the score. Attribution analysis showed sector RS hurt Sharpe.)
_spy_for_signal = fetch_spy_full(period)
_spy_bear_now = False
_spy_regime_now = "—"
_rs_spy_now: float | None = None
if not _spy_for_signal.empty and {"MA20", "MA50", "Close"}.issubset(_spy_for_signal.columns):
    _spy_aligned_live = _spy_for_signal.reindex(df.index, method="ffill")
    _spy_drop = _spy_aligned_live.dropna(subset=["MA20", "MA50"])
    if not _spy_drop.empty:
        _last_spy = _spy_drop.iloc[-1]
        _spy_bear_now = float(_last_spy["MA20"]) < float(_last_spy["MA50"])
        _spy_regime_now = "UPTREND" if not _spy_bear_now else "DOWNTREND"
    _rs_spy_now = compute_rs_spread(df["Close"], _spy_aligned_live["Close"], idx=-1, window=20)

# ── Signal + Fear & Greed ─────────────────────────────────────────────────────
signal, reasons, regime_info = get_signal(
    df,
    uptrend_thr=_uptrend_thr,
    range_thr=_range_thr,
    rsi_weight=_rsi_weight,
    spy_bear=_spy_bear_now,
    sentiment_score=_sentiment_for_signal,
    rs_spy=_rs_spy_now,
    beta=_beta_val,
)
fg = fetch_fear_greed()

signal_styles = {
    "BUY":  ("signal-buy",  "BUY · 买入",  C["up"]),
    "SELL": ("signal-sell", "SELL · 卖出", C["down"]),
    "HOLD": ("signal-hold", "HOLD · 观望", C["muted"]),
}
css_class, label, color = signal_styles[signal]

# Regime display helpers
_regime_zh  = {"UPTREND": "上升趋势", "DOWNTREND": "下降趋势", "RANGE": "震荡区间"}
_regime_col = {"UPTREND": C["up"],    "DOWNTREND": C["down"],  "RANGE": C["muted"]}
_cur_regime = regime_info["current"]
_rc         = _regime_col[_cur_regime]
_rz         = _regime_zh[_cur_regime]
_slope_pct  = regime_info["current_slope"] * 100
_strength   = regime_info["strength"]
_str_zh     = "强" if _strength > 3 else "中" if _strength > 1.5 else "弱"
_raw_score  = regime_info["raw_score"]
_w_score    = regime_info["weighted_score"]
_threshold  = regime_info["buy_threshold"]
_thr_txt    = "禁止买入" if _threshold is None else f"{_threshold:.1f}"
_score_col  = C["up"] if _w_score >= (_threshold or 999) else C["warn"] if _w_score > 0 else C["down"]

# Beta mode badge HTML (empty string when beta is in mid range)
_beta_val_txt = f"{_beta_val:.2f}" if _beta_val is not None else "N/A"
_c_dim = C["dim"]
if _beta_mode:
    _beta_badge_html = (
        f"<div style='margin-top:7px; display:inline-flex; align-items:center; gap:5px;"
        f"  padding:3px 8px; border-radius:12px;"
        f"  background:{_beta_mode_c}18; border:1px solid {_beta_mode_c}50;'>"
        f"  <span style='font-size:11px; font-weight:700; color:{_beta_mode_c};'>{_beta_mode}</span>"
        f"  <span style='font-size:10px; color:{_c_dim};'>β={_beta_val_txt}</span>"
        f"</div>"
    )
else:
    _beta_badge_html = (
        f"<div style='margin-top:7px; font-size:10px; color:{_c_dim};'>"
        f"β={_beta_val_txt}"
        f"</div>"
    )

# ── Cross-asset display strings (used in signal box) ──────────────────────────
def _regime_color_html(regime_str):
    if regime_str == "UPTREND":   return C["up"], "上涨"
    if regime_str == "DOWNTREND": return C["down"], "下跌"
    return C["muted"], "—"

_spy_col_v, _spy_zh = _regime_color_html(_spy_regime_now)

def _rs_disp(rs_val):
    if rs_val is None:
        return "—", C["muted"]
    col = C["up"] if rs_val > 4 else C["down"] if rs_val < -4 else C["muted"]
    return f"{rs_val:+.1f}pp", col

_rs_spy_txt, _rs_spy_col = _rs_disp(_rs_spy_now)

# Beta tier display
_beta_tier_lbl_zh, _beta_tier_key = _beta_tier_label(_beta_val)
_tier_col = {"high": C["up"], "neutral": C["accent2"], "defensive": C["warn"], "unknown": C["muted"]}[_beta_tier_key]
_spy_pen_now = _spy_penalty_for_beta(_beta_val)
_pen_txt = f"过滤权重 {_spy_pen_now:+.1f}分" if _spy_pen_now != 0 else "跳过 SPY 过滤"

_cross_asset_html = (
    f"<div style='margin-top:8px; padding:7px 9px; background:{C['surface']};"
    f" border:1px solid {C['border']}; border-radius:6px; font-size:10px; line-height:1.7;'>"
    f"<div><span style='color:{C['dim']};'>Market (SPY)</span> &nbsp;"
    f"<span style='color:{_spy_col_v}; font-weight:600;'>{_spy_zh}</span></div>"
    f"<div><span style='color:{C['dim']};'>当前 Beta 分段</span> &nbsp;"
    f"<span style='color:{_tier_col}; font-weight:600;'>{_beta_tier_lbl_zh}</span>"
    f" &nbsp;<span style='color:{C['dim']}; font-size:9px;'>· {_pen_txt}</span></div>"
    f"<div><span style='color:{C['dim']};'>RS vs SPY</span> &nbsp;"
    f"<span style='font-family:Space Mono,monospace; color:{_rs_spy_col}; font-weight:600;'>{_rs_spy_txt}</span>"
    f" &nbsp;<span style='color:{C['dim']}; font-size:9px;'>· 仅参考</span></div>"
    f"</div>"
)

col_sig, col_reasons, col_fg = st.columns([1, 2, 1])
with col_sig:
    st.markdown(f"""
    <div class='{css_class}'>
      <div style='font-size:22px; font-weight:700; color:{color};'>{label}</div>
      <div style='font-size:11px; color:{C["muted"]}; margin-top:6px;'>综合技术信号</div>
    </div>
    <div style='margin-top:10px; padding:8px 10px; background:{_rc}12;
                border:1px solid {_rc}40; border-radius:8px;'>
      <div style='font-size:12px; font-weight:600; color:{_rc};'>{_rz}</div>
      <div style='font-size:10px; color:{C["dim"]}; margin-top:3px;'>
        MA50斜率 <span style='font-family:Space Mono,monospace;color:{_rc};'>{_slope_pct:+.3f}%/日</span>
        &nbsp;·&nbsp; 强度 <span style='color:{_rc};'>{_str_zh}</span>
      </div>
      <div style='margin-top:6px; display:flex; gap:8px; flex-wrap:wrap;'>
        <span style='font-size:10px; color:{C["dim"]};'>原始分
          <span style='font-family:Space Mono,monospace; color:{_score_col};
                       font-weight:600;'>{_raw_score:+d}</span>
        </span>
        <span style='font-size:10px; color:{C["dim"]};'>加权分
          <span style='font-family:Space Mono,monospace; color:{_score_col};
                       font-weight:600;'>{_w_score:+.1f}</span>
        </span>
        <span style='font-size:10px; color:{C["dim"]};'>触发线
          <span style='font-family:Space Mono,monospace; color:{C["accent2"]};
                       font-weight:600;'>{_thr_txt}</span>
        </span>
      </div>
      {_beta_badge_html}
    </div>
    {_cross_asset_html}
    """, unsafe_allow_html=True)

with col_reasons:
    st.markdown("**信号依据**")
    for r in reasons:
        st.markdown(f"- {r}")

with col_fg:
    _fg_src = "N/A" if fg["error"] else "alternative.me"
    _fg_muted = C["muted"]
    _fg_dim = C["dim"]
    st.markdown(
        f"<div style='font-size:11px;color:{_fg_muted};margin-bottom:2px;'>"
        f"市场恐慌贪婪指数 &nbsp;<span style='color:{_fg_dim};font-size:10px;'>{_fg_src}</span></div>",
        unsafe_allow_html=True,
    )
    if fg["value"] is None:
        st.markdown(
            f"<div style='font-size:28px;font-weight:700;color:{_fg_muted};padding:32px 0;text-align:center;'>N/A</div>",
            unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(build_fg_gauge(fg, C), width="stretch")

st.markdown("---")

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown("### 技术图表")
fig = build_chart(df, ticker, C)
st.plotly_chart(fig, width="stretch")

# ── Market Comparison ────────────────────────────────────────────────────────
st.markdown("### 大盘对比（近6个月收益率）")
if spy_close.empty:
    st.warning("无法加载 SPY 数据，大盘对比图暂不可用。")
else:
    cmp_fig = build_comparison_chart(df, spy_close, ticker, C)
    st.plotly_chart(cmp_fig, width="stretch")

# ── Backtest ─────────────────────────────────────────────────────────────────
_bt_muted = C["muted"]
_bt_down = C["down"]
_bt_up = C["up"]
st.markdown(
    f"### 策略回测 "
    f"<span style='font-size:13px; color:{_bt_muted}; font-family:DM Sans,sans-serif; font-weight:400;'>"
    f"持仓 {holding_days}日 &nbsp;·&nbsp; "
    f"止损 <span style='color:{_bt_down};'>-{stop_loss_pct}%</span> &nbsp;·&nbsp; "
    f"止盈 <span style='color:{_bt_up};'>+{take_profit_pct}%</span> &nbsp;·&nbsp; "
    f"滑点 {slippage_pct:.2f}% &nbsp;·&nbsp; 手续费 {commission_pct:.2f}%"
    f"</span>",
    unsafe_allow_html=True,
)

with st.spinner("回测计算中..."):
    _spy_for_bt = _spy_for_signal if "_spy_for_signal" in dir() else fetch_spy_full(period)
    bt = run_backtest(
        df,
        holding_days=holding_days,
        stop_loss=stop_loss_pct / 100,
        take_profit=take_profit_pct / 100,
        slippage=slippage_pct / 100,
        commission=commission_pct / 100,
        time_stop_enabled=time_stop_enabled,
        time_stop_days=time_stop_days,
        time_stop_min_pnl=time_stop_min_pnl_pct / 100,
        uptrend_thr=_uptrend_thr,
        range_thr=_range_thr,
        rsi_weight=_rsi_weight,
        spy_full=_spy_for_bt,
        beta=_beta_val,
    )

if bt["metrics"] is None:
    st.warning("⚠️ 当前周期内未产生任何交易信号，请切换更长的回看周期（建议 1y 或 2y）。")
else:
    m = bt["metrics"]

    # ── Metric cards ──────────────────────────────────────────────────────────
    def _color(val, good_positive=True):
        if good_positive:
            return "#10b981" if val >= 0 else "#f43f5e"
        return "#f43f5e" if val >= 0 else "#10b981"

    # SPY metrics over the same date window as df
    _spy_sharpe       = None
    _spy_total_return = None
    _excess_return    = None
    _info_ratio       = None
    if not spy_close.empty:
        _spy_aligned = spy_close.reindex(df.index, method="ffill").dropna()
        if len(_spy_aligned) > 5:
            # Annualised Sharpe
            _spy_dr  = _spy_aligned.pct_change().dropna()
            _spy_std = float(_spy_dr.std())
            if _spy_std > 0:
                _spy_sharpe = float(_spy_dr.mean() / _spy_std * np.sqrt(252))

            # SPY total return over period → Excess Return
            _spy_total_return = float(_spy_aligned.iloc[-1] / _spy_aligned.iloc[0] - 1)
            _excess_return    = m["total_return"] - _spy_total_return

            # Information Ratio: per-trade excess returns vs SPY
            if not bt["trades"].empty:
                _ex_list = []
                for _, _tr in bt["trades"].iterrows():
                    _p0 = spy_close.asof(_tr["entry_date"])
                    _p1 = spy_close.asof(_tr["exit_date"])
                    if pd.notna(_p0) and pd.notna(_p1) and float(_p0) > 0:
                        _spy_tr = (float(_p1) - float(_p0)) / float(_p0)
                        _ex_list.append(float(_tr["return"]) - _spy_tr)
                if len(_ex_list) > 1:
                    _ex_arr = np.array(_ex_list)
                    _ex_std = float(_ex_arr.std())
                    if _ex_std > 0:
                        _info_ratio = float(_ex_arr.mean() / _ex_std)

    wr_color = "#10b981" if m["win_rate"] >= 0.5 else "#f43f5e"
    tr_color = _color(m["total_return"])
    dd_color = "#f43f5e" if m["max_drawdown"] < -0.1 else "#f59e0b" if m["max_drawdown"] < -0.05 else "#10b981"
    bh_color = _color(m["bh_return"])

    _sh_beats_spy = _spy_sharpe is not None and m["sharpe"] > _spy_sharpe
    sh_color = (
        "#10b981" if _sh_beats_spy
        else "#f59e0b" if m["sharpe"] >= 0
        else "#f43f5e"
    )
    _sh_sub = (
        f"SPY同期 {_spy_sharpe:.2f}  ·  持均 {m['avg_hold_days']:.1f}日"
        if _spy_sharpe is not None
        else f"年化，持均 {m['avg_hold_days']:.1f}日"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "胜率", f"{m['win_rate']:.1%}", f"{m['total_trades']} 笔交易", wr_color),
        (c2, "总收益率", f"{m['total_return']:+.2%}", f"买入持有 {m['bh_return']:+.2%}", tr_color),
        (c3, "最大回撤", f"{m['max_drawdown']:.2%}", "策略期间峰谷跌幅", dd_color),
        (c4, "夏普比率", f"{m['sharpe']:.2f}", _sh_sub, sh_color),
        (c5, "平均单笔", f"{m['avg_return']:+.2%}", "每笔交易平均收益", _color(m["avg_return"])),
    ]
    for col, label, val, sub, color in cards:
        col.markdown(f"""
        <div class='bt-card'>
          <div class='bt-label'>{label}</div>
          <div class='bt-value' style='color:{color};'>{val}</div>
          <div class='bt-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    # ── Second row: Excess Return · Information Ratio · Downside Deviation ────
    _dd = m.get("downside_dev", 0.0)

    _er_val   = f"{_excess_return:+.2%}" if _excess_return is not None else "N/A"
    _er_color = (_color(_excess_return) if _excess_return is not None else C["muted"])
    _er_sub   = (f"vs SPY同期 {_spy_total_return:+.2%}"
                 if _spy_total_return is not None else "无SPY数据")

    _ir_val   = f"{_info_ratio:.2f}" if _info_ratio is not None else "N/A"
    _ir_color = (
        C["up"]   if _info_ratio is not None and _info_ratio > 0.5  else
        C["warn"] if _info_ratio is not None and _info_ratio >= 0   else
        C["down"] if _info_ratio is not None                        else C["muted"]
    )
    _ir_sub   = "逐笔超额收益均值 / 标准差"

    _dd_color = "#f43f5e" if _dd > 0.15 else "#f59e0b" if _dd > 0.08 else "#10b981"
    _dd_sub   = "年化，仅计算负收益波动"

    ex_c, ir_c, dd_c, _pad = st.columns([1, 1, 1, 2])
    _row2 = [
        (ex_c, "超额收益 (α)",  _er_val,            _er_sub,  _er_color),
        (ir_c, "信息比率 (IR)", _ir_val,            _ir_sub,  _ir_color),
        (dd_c, "下行偏差",      f"{_dd:.2%}",       _dd_sub,  _dd_color),
    ]
    for col, label, val, sub, color in _row2:
        col.markdown(f"""
        <div class='bt-card'>
          <div class='bt-label'>{label}</div>
          <div class='bt-value' style='color:{color};'>{val}</div>
          <div class='bt-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Equity curve ──────────────────────────────────────────────────────────
    bt_fig = build_backtest_chart(bt["equity"], df, bt["trades"], C)
    st.plotly_chart(bt_fig, width="stretch")

    with st.expander("查看交易记录"):
        tdf = bt["trades"].copy()
        tdf["return"]   = tdf["return"].map(lambda x: f"{x:+.2%}")
        tdf["position"] = tdf["position"].map(lambda x: f"{x:.0%}")
        tdf = tdf[["entry_date", "exit_date", "entry_price", "exit_price",
                   "return", "position", "days", "exit_reason"]]
        tdf.columns = ["买入日期", "卖出日期", "买入价", "卖出价", "收益率", "仓位", "持仓天数", "退出原因"]

        def _reason_style(v):
            if not isinstance(v, str):
                return ""
            if v == "时间止损":
                return "color:#f59e0b;font-weight:600"
            if v.startswith("止损"):
                return "color:#f43f5e;font-weight:600"
            if v.startswith("止盈"):
                return "color:#10b981;font-weight:600"
            return ""

        def _pos_style(v):
            if not isinstance(v, str):
                return ""
            if v == "150%":
                return "color:#10b981;font-weight:600"
            if v == "50%":
                return "color:#f59e0b;font-weight:600"
            return ""

        st.dataframe(
            tdf.style
               .map(
                   lambda v: "color:#10b981" if isinstance(v, str) and v.startswith("+") else
                             ("color:#f43f5e" if isinstance(v, str) and v.startswith("-") else ""),
                   subset=["收益率"],
               )
               .map(_reason_style, subset=["退出原因"])
               .map(_pos_style, subset=["仓位"]),
            width="stretch",
        )

    # ── Monte Carlo Robustness Test ───────────────────────────────────────────
    _bt_muted2 = C["muted"]
    st.markdown(
        f"<div style='font-size:14px;font-weight:600;color:{C['text']};margin:18px 0 6px;'>"
        "策略稳健性测试"
        f"<span style='font-size:11px;font-weight:400;color:{_bt_muted2};margin-left:8px;'>"
        "Monte Carlo 噪声注入 · 1000 次模拟 · 每笔扰动 ±0.5%</span></div>",
        unsafe_allow_html=True,
    )

    with st.spinner("正在运行稳健性模拟（1000 次）..."):
        _rets_tuple = tuple(float(r) for r in bt["trades"]["return"])
        rb = run_robustness_test(
            _rets_tuple,
            avg_hold=m["avg_hold_days"],
            n_sims=1000,
            noise_std=0.005,
        )

    if rb.get("error") == "insufficient_trades":
        st.info("ℹ️ 交易笔数不足（< 3），无法进行稳健性测试。")
    elif rb.get("error"):
        st.warning(f"稳健性测试失败：{rb['error']}")
    else:
        _actual_sh = m["sharpe"]

        # Robustness verdict
        _p50  = rb["p50"]
        _p5   = rb["p5"]
        _p95  = rb["p95"]
        _ppos = rb["pct_positive"]
        _ci_width = _p95 - _p5

        if _p5 > 0:
            _verdict     = "稳健"
            _verdict_c   = C["up"]
            _verdict_tip = "90% 置信区间均为正，策略在噪声下仍可靠"
        elif _p50 > 0 and _ppos >= 0.7:
            _verdict     = "中性"
            _verdict_c   = C["warn"]
            _verdict_tip = "中位数为正但区间跨零，具有一定随机性"
        else:
            _verdict     = "脆弱"
            _verdict_c   = C["down"]
            _verdict_tip = "策略在随机扰动下大幅退化，可能依赖少数运气单"

        # ── Metric row ──────────────────────────────────────────────────
        _rb1, _rb2, _rb3, _rb4, _rb5 = st.columns(5)
        _rb_sh_color = lambda v: C["up"] if v > 0.5 else C["warn"] if v >= 0 else C["down"]
        _rb_cards = [
            (_rb1, "中位夏普 (P50)",
             f"{_p50:.2f}",
             f"实际夏普 {_actual_sh:.2f}",
             _rb_sh_color(_p50)),
            (_rb2, "5th 分位夏普",
             f"{_p5:.2f}",
             "最差 5% 场景",
             C["down"] if _p5 < 0 else C["warn"]),
            (_rb3, "95th 分位夏普",
             f"{_p95:.2f}",
             "最好 5% 场景",
             C["up"]),
            (_rb4, "90% 置信区间宽度",
             f"{_ci_width:.2f}",
             "越窄越稳定",
             C["warn"] if _ci_width > 1.5 else C["up"]),
            (_rb5, "稳健性评级",
             _verdict,
             _verdict_tip,
             _verdict_c),
        ]
        for _col, _lbl, _val, _sub, _clr in _rb_cards:
            _col.markdown(f"""
            <div class='bt-card'>
              <div class='bt-label'>{_lbl}</div>
              <div class='bt-value' style='color:{_clr};'>{_val}</div>
              <div class='bt-sub'>{_sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Histogram ───────────────────────────────────────────────────
        _rb_chart_col, _rb_note_col = st.columns([3, 1])
        with _rb_chart_col:
            st.plotly_chart(
                build_robustness_chart(rb, _actual_sh, C),
                width="stretch",
            )
        with _rb_note_col:
            _rb_dim  = C["dim"]
            _rb_text = C["text"]
            _rb_ppos_c = C["up"] if _ppos >= 0.7 else C["warn"]
            st.markdown(
                f"<div style='font-size:11px;color:{_rb_dim};line-height:1.8;padding-top:12px;'>"
                f"<b style='color:{_rb_text};'>测试方法</b><br>"
                "对每笔历史收益加入"
                "<span style='font-family:Space Mono,monospace;'> N(0, 0.5%) </span>"
                "随机扰动，重复 1000 次，<br>每次重算年化夏普。"
                "<br><br>"
                f"<b style='color:{_rb_text};'>正 Sharpe 占比</b><br>"
                f"<span style='font-family:Space Mono,monospace;color:{_rb_ppos_c};'>"
                f"{_ppos:.1%}</span> 的模拟结果 > 0"
                "</div>",
                unsafe_allow_html=True,
            )

st.markdown("---")

# ── Monte Carlo / GARCH / ML (tabs) ──────────────────────────────────────────
_daily_ret = df["Close"].pct_change().dropna()
_mu  = round(float(_daily_ret.mean()), 8)
_sig = round(float(_daily_ret.std()),  8)

mc_tab, garch_tab, ml_tab = st.tabs([
    "蒙特卡洛模拟（30天·1000条路径）",
    "GARCH 波动率预测",
    "ML 价格预测（5日）",
])

with mc_tab:
    with st.spinner("正在运行 1000 条路径模拟..."):
        mc = run_monte_carlo(last_close, _mu, _sig)

    p5_price  = mc["pcts"][5][-1]
    p50_price = mc["pcts"][50][-1]
    p95_price = mc["pcts"][95][-1]
    up_prob   = mc["up_prob"]

    def _mc_color(price):
        return "#10b981" if price >= last_close else "#f43f5e"

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc_cards = [
        (mc1, "悲观预期（5th）",  f"${p5_price:.2f}",  f"{(p5_price-last_close)/last_close:+.1%}",  _mc_color(p5_price)),
        (mc2, "中位数（50th）",   f"${p50_price:.2f}", f"{(p50_price-last_close)/last_close:+.1%}", _mc_color(p50_price)),
        (mc3, "乐观预期（95th）", f"${p95_price:.2f}", f"{(p95_price-last_close)/last_close:+.1%}", _mc_color(p95_price)),
        (mc4, "30日上涨概率",    f"{up_prob:.1%}",     f"基于 {mc['n_sims']} 次模拟",
         "#10b981" if up_prob >= 0.5 else "#f43f5e"),
    ]
    for col, lbl, val, sub, color in mc_cards:
        col.markdown(f"""
        <div class='mc-card'>
          <div class='mc-label'>{lbl}</div>
          <div class='mc-value' style='color:{color};'>{val}</div>
          <div class='mc-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ch_col, dist_col = st.columns([3, 2])
    with ch_col:
        st.plotly_chart(build_mc_chart(mc, ticker, C), width="stretch")
    with dist_col:
        st.markdown(
            f"<div style='font-size:12px;color:{C['muted']};margin-bottom:8px;'>30日后收益区间概率分布</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_mc_dist_chart(mc, C), width="stretch")

with garch_tab:
    with st.spinner("正在滚动拟合 GARCH(1,1)（样本外，多次 refit）..."):
        _ret_tuple   = tuple(_daily_ret.values.round(8))
        _dates_tuple = tuple(_daily_ret.index.strftime("%Y-%m-%d"))
        g = run_garch_forecast(_ret_tuple, _dates_tuple)

    if g.get("error") == "arch_missing":
        st.error("❌ 请安装 arch 库：`pip install arch`")
    elif g.get("error"):
        st.error(f"❌ GARCH 拟合失败：{g['error']}")
    else:
        # ── Metric cards ──────────────────────────────────────────────────
        g1, g2, g3, g4 = st.columns(4)
        _risk_c = g["risk_color"]
        _g_cards = [
            (g1, "当前条件波动率",    f"{g['current_vol']:.1f}%",     "样本外预测（年化）", "#a78bfa"),
            (g2, "30日预测均值",      f"{g['avg_fc_annual']:.1f}%",   "预测年化波动率均值",     _risk_c),
            (g3, "波动率风险评级",    g["risk_level"],                 "",                       _risk_c),
            (g4, "持续性 α+β",        f"{g['persistence']:.4f}",      "越接近1波动率衰减越慢",  "#f59e0b"),
        ]
        for col, lbl, val, sub, color in _g_cards:
            col.markdown(f"""
            <div class='mc-card'>
              <div class='mc-label'>{lbl}</div>
              <div class='mc-value' style='color:{color};'>{val}</div>
              <div class='mc-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts side by side ───────────────────────────────────────────
        hist_col, fc_col = st.columns(2)
        with hist_col:
            st.markdown(
                f"<div style='font-size:12px;color:{C['muted']};margin-bottom:4px;'>"
                f"已实现波动率 vs GARCH 样本外预测（窗口 {g['roll_window']}日 · 步长 {g['stride']}日 · refit {g['n_refits']} 次）</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(build_garch_hist_chart(g, C), width="stretch")
        with fc_col:
            st.markdown(
                f"<div style='font-size:12px;color:{C['muted']};margin-bottom:4px;'>"
                "未来 30 天年化波动率预测</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(build_garch_forecast_chart(g, C), width="stretch")

        # ── Model params ──────────────────────────────────────────────────
        _g_dim = C["dim"]
        st.markdown(
            f"<div style='font-size:11px;color:{_g_dim};margin-top:4px;'>"
            f"最终窗口 GARCH(1,1) 参数 &nbsp;·&nbsp; "
            f"α (ARCH) = {g['alpha']:.5f} &nbsp;·&nbsp; "
            f"β (GARCH) = {g['beta']:.5f} &nbsp;·&nbsp; "
            f"持续性 α+β = {g['persistence']:.5f} &nbsp;·&nbsp; "
            f"滚动窗口 {g['roll_window']}日 · 步长 {g['stride']}日 · "
            f"历史曲线为严格样本外预测（共 refit {g['n_refits']} 次）"
            f"</div>",
            unsafe_allow_html=True,
        )

with ml_tab:
    with st.spinner("正在训练集成模型（LogReg · RandomForest · XGBoost · LightGBM）— 三壁障分类..."):
        _ml_feat_tup  = tuple(map(tuple, _ml_feat.round(8)))
        _ml_close_tup = tuple(_ml_close.round(4))
        _ml_dates_tup = tuple(_ml_dates.strftime("%Y-%m-%d"))
        _ml_atr_tup   = tuple(_ml_atr.round(6))
        ml = run_ml_prediction(
            _ml_feat_tup, _ml_close_tup, _ml_dates_tup,
            ticker=ticker,
            today_str=str(datetime.today().date()),
            atr_tuple=_ml_atr_tup,
        )

    if ml.get("error") and "missing_lib" in str(ml.get("error", "")):
        st.error(f"缺少依赖库：{ml['error']}  请运行 `pip install xgboost lightgbm scikit-learn`")
    elif ml.get("error") == "insufficient_data":
        st.warning("数据不足（需 ≥ 100 行有效特征），请切换至 1y 或 2y 周期。")
    elif ml.get("error"):
        st.error(f"模型训练失败：{ml['error']}")
    else:
        _avg        = ml["model_avg_acc"]
        _probs      = ml["model_probs"]
        _ens_tp     = ml["ens_tp"]
        _ens_sl     = ml["ens_sl"]
        _ens_acc    = _avg.get("Ensemble", 0.0)
        _lbl_cnt    = ml.get("label_counts", {})
        _cur_atr    = ml.get("cur_atr_pct")        # ATR/Close ratio, may be None
        _cur_tp_bar = ml.get("cur_tp_bar", 0.05)
        _cur_sl_bar = ml.get("cur_sl_bar", 0.03)
        _atr_based  = ml.get("atr_based", False)

        _acc_color  = lambda a: C["up"] if a >= 0.45 else C["warn"] if a >= 0.35 else C["down"]
        _prob_color = lambda p: C["up"] if p >= 0.4 else C["warn"] if p >= 0.25 else C["muted"]

        # ── ATR info banner ───────────────────────────────────────────────
        if _atr_based and _cur_atr is not None:
            _atr_pct_str = f"{_cur_atr:.2%}"
            _tp_str      = f"{_cur_tp_bar:.2%}"
            _sl_str      = f"{_cur_sl_bar:.2%}"
            st.markdown(
                f"<div style='font-size:12px;color:{C['muted']};padding:6px 10px;"
                f"border-left:3px solid {C['accent2']};margin-bottom:12px;'>"
                f"动态壁障（ATR-based）&nbsp;·&nbsp; "
                f"当前 ATR14 = <b style='color:{C['text']};'>{_atr_pct_str}</b>"
                f"（占收盘价百分比）&nbsp;·&nbsp; "
                f"上壁障（止盈）= 2× ATR = <b style='color:{C['up']};'>+{_tp_str}</b>&nbsp;·&nbsp; "
                f"下壁障（止损）= 1.5× ATR = <b style='color:{C['down']};'>-{_sl_str}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Summary metric cards ──────────────────────────────────────────
        ml1, ml2, ml3, ml4 = st.columns(4)
        _ml_cards = [
            (ml1, "止盈概率（集成）",
             f"{_ens_tp:.1%}",
             f"上壁障 +{_cur_tp_bar:.1%}（2×ATR）",
             _prob_color(_ens_tp)),
            (ml2, "止损概率（集成）",
             f"{_ens_sl:.1%}",
             f"下壁障 -{_cur_sl_bar:.1%}（1.5×ATR）",
             C["down"] if _ens_sl >= 0.3 else C["muted"]),
            (ml3, "分类准确率（参考）",
             f"{_ens_acc:.1%}",
             f"Walk-Forward {ml['n_splits']} 折均值", _acc_color(_ens_acc)),
            (ml4, "样本分布",
             f"{_lbl_cnt.get('止盈', 0)} / {_lbl_cnt.get('中立', 0)} / {_lbl_cnt.get('止损', 0)}",
             f"止盈 / 中立 / 止损  （共 {ml['n_total']} 条）", C["accent2"]),
        ]
        for col, lbl, val, sub, color in _ml_cards:
            col.markdown(f"""
            <div class='mc-card'>
              <div class='mc-label'>{lbl}</div>
              <div class='mc-value' style='color:{color};'>{val}</div>
              <div class='mc-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Model probability + accuracy table ────────────────────────────
        _fold_data  = ml["model_fold_accs"]
        _table_rows = []
        for _mname in ["LogReg", "RandomForest", "XGBoost", "LightGBM", "Ensemble"]:
            _folds = _fold_data.get(_mname, [])
            _mp = _probs.get(_mname, {})
            _row = {
                "模型":       _mname,
                "止盈概率":   f"{_mp.get('tp', 0):.1%}",
                "止损概率":   f"{_mp.get('sl', 0):.1%}",
                "中立概率":   f"{_mp.get('neutral', 0):.1%}",
                "分类准确率": f"{_avg.get(_mname, 0):.1%}",
            }
            for _fi, _fv in enumerate(_folds):
                _row[f"Fold {_fi+1}"] = f"{_fv:.1%}"
            _table_rows.append(_row)
        _acc_df = pd.DataFrame(_table_rows).set_index("模型")

        st.markdown(
            f"<div style='font-size:12px;color:{C['muted']};margin-bottom:6px;'>"
            f"各模型三壁障分类概率 · Walk-Forward 分类准确率（上壁障 2×ATR / 下壁障 1.5×ATR / 横壁障 {ml['horizon']} 日）</div>",
            unsafe_allow_html=True,
        )

        def _style_cell(v):
            if not isinstance(v, str) or not v.endswith("%"):
                return ""
            try:
                num = float(v.rstrip("%"))
                # detect probability columns by range (0–100 displayed as X.X%)
                return (f"color:{C['up']};font-weight:600" if num >= 40
                        else f"color:{C['down']}" if num < 20 else "")
            except ValueError:
                return ""

        st.dataframe(_acc_df.style.map(_style_cell), width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability chart + feature importance ────────────────────────
        fc_col, imp_col = st.columns([3, 2])
        with fc_col:
            st.markdown(
                f"<div style='font-size:12px;color:{C['muted']};margin-bottom:4px;'>"
                f"各模型三壁障概率分布（上壁障 +{_cur_tp_bar:.1%} · 下壁障 -{_cur_sl_bar:.1%} · 横壁障 {ml['horizon']} 日）</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(build_ml_prob_chart(ml, C), width="stretch")
        with imp_col:
            st.markdown(
                f"<div style='font-size:12px;color:{C['muted']};margin-bottom:4px;'>"
                "随机森林特征重要性（17维）</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(build_ml_importance_chart(ml, C), width="stretch")

        # ── OOS backtest ──────────────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:12px;color:{C['muted']};margin-bottom:4px;'>"
            "样本外回测：各模型止盈概率预测 vs 实际三壁障结果</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_ml_backtest_chart(ml, C), width="stretch")

        _ml_dim = C["dim"]
        st.markdown(
            f"<div style='font-size:11px;color:{_ml_dim};margin-top:4px;'>"
            f"标签：三壁障法（上壁障 2×ATR14 / 下壁障 1.5×ATR14 / 横壁障 {ml['horizon']} 日）&nbsp;·&nbsp; "
            f"特征（17维·滚动Z分·已调权价格）：RSI · MACD · 布林%B · ATR · MA偏差 · 多周期动量 · 成交量变化 · VWAP偏差 &nbsp;·&nbsp; "
            f"模型：LogReg + RandomForest + XGBoost + LightGBM &nbsp;·&nbsp; "
            f"验证：Walk-Forward {ml['n_splits']} 折（Embargo {ml['embargo']}日）&nbsp;·&nbsp; "
            f"超参数：自动调优（内层 3 折）</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Portfolio Optimization ────────────────────────────────────────────────────
_palette = ["#6366f1", "#10b981", "#f59e0b", "#f43f5e", "#a78bfa", "#34d399"]

with st.expander("投资组合优化（最大化夏普比率）", expanded=True):

    with st.form("portfolio_form"):
        po_raw = st.text_input(
            "输入 2–6 只股票代码（逗号分隔，使用 2 年历史数据）",
            value=st.session_state.get("po_input", f"{ticker}, MSFT, GOOGL"),
            placeholder="AAPL, MSFT, GOOGL, NVDA",
        )
        po_submitted = st.form_submit_button("开始优化", width="stretch")

    if po_submitted:
        _cleaned = tuple(dict.fromkeys(
            t.strip().upper() for t in po_raw.split(",") if t.strip()
        ))
        if len(_cleaned) < 2:
            st.warning("⚠️ 请至少输入 2 只股票。")
        elif len(_cleaned) > 6:
            st.warning("⚠️ 最多支持 6 只股票，请减少数量。")
        else:
            st.session_state["po_input"] = po_raw
            with st.spinner(f"正在下载 {', '.join(_cleaned)} 历史数据并计算最优权重..."):
                st.session_state["po_result"] = fetch_and_optimize(_cleaned)

    if "po_result" in st.session_state:
        _po = st.session_state["po_result"]
        _err = _po.get("error")

        if _err == "pypfopt_missing":
            st.error("❌ 未安装 PyPortfolioOpt，请运行：`pip install PyPortfolioOpt`")
        elif _err == "rate_limit":
            st.error("⏱️ Yahoo Finance 频率限制，请稍后重试。")
        elif _err == "insufficient_data":
            st.error("❌ 历史数据不足（需 ≥ 100 个交易日），请检查股票代码。")
        elif _err:
            st.error(f"❌ 优化失败：{_err}")
        else:
            # ── Summary metric cards ──────────────────────────────────────
            pm1, pm2, pm3, pm4 = st.columns(4)
            _n_active = sum(1 for v in _po["weights"].values() if v > 0.01)
            _po_cards = [
                (pm1, "预期年化收益",   f"{_po['ret']:+.2%}",   "最优夏普组合",   "#10b981" if _po["ret"] > 0 else "#f43f5e"),
                (pm2, "预期年化波动率", f"{_po['vol']:.2%}",    "年化标准差",     "#f59e0b"),
                (pm3, "预期夏普比率",   f"{_po['sharpe']:.2f}", "无风险利率 5%",  "#a78bfa"),
                (pm4, "有效配置资产",   f"{_n_active} 只",      f"共 {len(_po['tickers'])} 只输入", C["muted"]),
            ]
            for col, lbl, val, sub, color in _po_cards:
                col.markdown(f"""
                <div class='bt-card'>
                  <div class='bt-label'>{lbl}</div>
                  <div class='bt-value' style='color:{color};'>{val}</div>
                  <div class='bt-sub'>{sub}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Per-ticker weight cards ────────────────────────────────────
            _active = sorted(
                [(k, v) for k, v in _po["weights"].items() if v > 0.001],
                key=lambda x: -x[1],
            )
            _wt_cols = st.columns(max(len(_active), 1))
            for i, (tkr, wt) in enumerate(_active):
                _wt_cols[i].markdown(f"""
                <div class='po-ticker-card' style='border-top-color:{_palette[i]};'>
                  <div class='bt-label'>{tkr}</div>
                  <div class='bt-value' style='color:{_palette[i]};font-size:26px;'>{wt:.1%}</div>
                  <div class='bt-sub'>建议配比</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Charts ────────────────────────────────────────────────────
            ef_col, wt_col = st.columns([3, 2])
            with ef_col:
                st.markdown(
                    f"<div style='font-size:12px;color:{C['muted']};margin-bottom:4px;'>"
                    "有效前沿（散点颜色 = 夏普比率，绿星 = 最优，黄钻 = 最小波动）</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(build_ef_chart(_po, C), width="stretch")
            with wt_col:
                st.markdown(
                    f"<div style='font-size:12px;color:{C['muted']};margin-bottom:4px;'>"
                    "最优权重分配</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(build_weight_chart(_po, C), width="stretch")

            # ── Min-vol comparison ─────────────────────────────────────────
            with st.expander("对比：最小波动组合"):
                _mv_active = sorted(
                    [(k, v) for k, v in _po["mv_weights"].items() if v > 0.001],
                    key=lambda x: -x[1],
                )
                _mv_cols = st.columns(max(len(_mv_active), 1))
                for i, (tkr, wt) in enumerate(_mv_active):
                    _mv_cols[i].markdown(f"""
                    <div class='po-ticker-card' style='border-top-color:{_palette[i]};'>
                      <div class='bt-label'>{tkr}</div>
                      <div class='bt-value' style='color:{_palette[i]};font-size:22px;'>{wt:.1%}</div>
                      <div class='bt-sub'>最小波动配比</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-size:12px;color:{C['muted']};margin-top:12px;'>"
                    f"预期收益 {_po['mv_ret']:+.2%} &nbsp;·&nbsp; "
                    f"波动率 {_po['mv_vol']:.2%} &nbsp;·&nbsp; "
                    f"夏普 {_po['mv_sharpe']:.2f}</div>",
                    unsafe_allow_html=True,
                )

st.markdown("---")

# ── Earnings & Insider Trading ────────────────────────────────────────────────
st.markdown("### 财报 & 内部人交易")

with st.spinner("获取财报和内部人数据..."):
    ei = fetch_earnings_and_insider(ticker)

if ei.get("error"):
    st.warning(f"⚠️ 数据获取失败：{ei['error']}")
else:
    earn = ei["earnings"]
    earn_col, insider_col = st.columns([1, 2])

    with earn_col:
        # Next earnings date
        if earn["next_date"]:
            days = earn["days_until"]
            days_str = (f"还有 {days} 天" if days is not None and days >= 0
                        else ("已过" if days is not None else ""))
            date_color = C["warn"] if days is not None and days <= 30 else C["text"]
            st.markdown(f"""
            <div class='bt-card' style='text-align:left;padding:18px 20px;'>
              <div style='font-size:11px;color:{C["muted"]};margin-bottom:6px;'>下次财报日期</div>
              <div style='font-family:Space Mono,monospace;font-size:22px;
                          font-weight:700;color:{date_color};'>{earn["next_date"]}</div>
              <div style='font-size:12px;color:{C["warn"]};margin-top:4px;'>{days_str}</div>
              <hr style='border-color:{C["border"]};margin:12px 0;'>
              <div style='font-size:11px;color:{C["muted"]};margin-bottom:4px;'>EPS 预期</div>
              <div style='font-family:Space Mono,monospace;font-size:18px;
                          font-weight:700;color:{C["accent2"]};'>
                {"${:.2f}".format(float(earn["eps_avg"])) if earn["eps_avg"] is not None else "N/A"}
              </div>
              <div style='font-size:11px;color:{C["dim"]};margin-top:2px;'>
                低 {"${:.2f}".format(float(earn["eps_low"])) if earn["eps_low"] is not None else "—"}
                &nbsp;·&nbsp;
                高 {"${:.2f}".format(float(earn["eps_high"])) if earn["eps_high"] is not None else "—"}
              </div>
              <hr style='border-color:{C["border"]};margin:12px 0;'>
              <div style='font-size:11px;color:{C["muted"]};margin-bottom:4px;'>营收预期</div>
              <div style='font-family:Space Mono,monospace;font-size:16px;
                          font-weight:700;color:{C["text"]};'>{earn["rev_avg"]}</div>
              <div style='font-size:11px;color:{C["dim"]};margin-top:2px;'>
                低 {earn["rev_low"]} &nbsp;·&nbsp; 高 {earn["rev_high"]}
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("暂无财报日期数据。")

    with insider_col:
        if ei["insider"]:
            insider_df = pd.DataFrame(ei["insider"])

            def _style_row(row):
                if row["类型"] == "买入":
                    bg = "background-color:rgba(16,185,129,0.10)"
                elif row["类型"] == "卖出":
                    bg = "background-color:rgba(244,63,94,0.08)"
                else:
                    bg = ""
                return [bg] * len(row)

            def _style_type(val):
                if val == "买入":
                    return "color:#10b981;font-weight:600"
                if val == "卖出":
                    return "color:#f43f5e;font-weight:600"
                return f"color:{C['muted']}"

            styled = (
                insider_df.style
                .apply(_style_row, axis=1)
                .map(_style_type, subset=["类型"])
            )
            st.markdown(
                f"<div style='font-size:12px;color:{C['muted']};margin-bottom:6px;'>"
                "最近 10 条内部人交易记录</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(styled, width="stretch", hide_index=True)
        else:
            st.info("暂无内部人交易记录。")

st.markdown("---")

# ── News Sentiment ────────────────────────────────────────────────────────────
st.markdown("### 新闻情绪分析")

with st.spinner("抓取最新新闻..."):
    news_items = fetch_news(ticker)

if not news_items:
    st.warning("暂无新闻数据。")
else:
    # ── Sentiment score (only with API key) ───────────────────────────────────
    if api_key:
        _senti_key = f"_sentiment_{ticker}"
        if _senti_key not in st.session_state:
            with st.spinner("Gemini 正在分析情绪..."):
                headlines = [n["title"] for n in news_items]
                _senti_result = get_news_sentiment(ticker, headlines, api_key)
                st.session_state[_senti_key] = _senti_result  # cache both success and error
        sentiment = st.session_state[_senti_key]
        if "error" in sentiment:
            st.error(f"情绪分析失败：{sentiment['error']}")
        else:
                sc = sentiment["score"]

                if sc >= 40:
                    sc_color, sc_label = "#10b981", "偏多 · 积极"
                elif sc >= 10:
                    sc_color, sc_label = "#34d399", "中性偏多"
                elif sc >= -10:
                    sc_color, sc_label = C["dim"], "中性"
                elif sc >= -40:
                    sc_color, sc_label = "#f97316", "中性偏空"
                else:
                    sc_color, sc_label = "#f43f5e", "偏空 · 消极"

                bar_pct = abs(sc) / 2
                if sc >= 0:
                    bar_css = f"position:absolute;left:50%;width:{bar_pct}%;height:100%;background:{sc_color};border-radius:0 4px 4px 0;"
                else:
                    bar_css = f"position:absolute;right:50%;width:{bar_pct}%;height:100%;background:{sc_color};border-radius:4px 0 0 4px;"

                st.markdown(f"""
                <div class='sentiment-wrap'>
                  <div class='sentiment-score-box'>
                    <div class='sentiment-score-num' style='color:{sc_color};'>{sc:+d}</div>
                    <div class='sentiment-score-label'>情绪得分</div>
                  </div>
                  <div class='sentiment-right'>
                    <div style='font-size:14px;font-weight:600;color:{sc_color};margin-bottom:8px;'>{sc_label}</div>
                    <div class='sentiment-bar-track'>
                      <div style='position:absolute;left:50%;width:1px;height:100%;background:#334155;'></div>
                      <div style='{bar_css}'></div>
                    </div>
                    <div style='display:flex;justify-content:space-between;font-size:10px;color:#475569;'>
                      <span>-100 极度悲观</span><span>0</span><span>+100 极度乐观</span>
                    </div>
                    <div class='sentiment-reason'>💬 {sentiment["reason"]}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("输入 API Key 后启用情绪分析（-100 到 +100 评分）。")

    # ── News headlines ────────────────────────────────────────────────────────
    for n in news_items:
        url = n["url"]
        title_html = f'<a href="{url}" target="_blank">{n["title"]}</a>' if url else n["title"]
        meta = " · ".join(filter(None, [n["publisher"], n["pub_time"]]))
        st.markdown(f"""
        <div class='news-item'>
          <div class='news-dot'></div>
          <div>
            {title_html}
            <div class='news-meta'>{meta}</div>
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── AI Analysis ───────────────────────────────────────────────────────────────
st.markdown("### AI 深度分析")

if not api_key:
    rule_report = generate_rule_report(ticker, df, signal, info)
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
      <span style='background:{C["border"]};color:{C["muted"]};font-size:11px;padding:3px 10px;
                   border-radius:4px;font-family:Space Mono,monospace;'>规则驱动</span>
      <span style='font-size:12px;color:{C["dim"]};'>输入 API Key 后切换为 Gemini 深度分析</span>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"<div class='ai-report'>{rule_report}</div>", unsafe_allow_html=True)
else:
    _analysis_key = f"_analysis_{ticker}_{period}"
    if _analysis_key not in st.session_state:
        with st.spinner("Gemini 正在分析中..."):
            st.session_state[_analysis_key] = get_ai_analysis(ticker, df, signal, reasons, api_key)
    report = st.session_state[_analysis_key]
    if not report or report.startswith("AI 分析暂时不可用") or report.startswith("未提供") or report.startswith(_GEMINI_RATE_LIMIT_MSG):
        st.error(report)
        rule_report = generate_rule_report(ticker, df, signal, info)
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
          <span style='background:{C["border"]};color:{C["muted"]};font-size:11px;padding:3px 10px;
                       border-radius:4px;font-family:Space Mono,monospace;'>规则驱动（回退）</span>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-report'>{rule_report}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
          <span style='background:{C["surface"]};color:{C["accent2"]};font-size:11px;padding:3px 10px;
                       border:1px solid {C["border"]};border-radius:4px;font-family:Space Mono,monospace;'>Gemini AI</span>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-report'>{report}</div>", unsafe_allow_html=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("查看原始数据"):
    show_df = df[["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI", "MACD"]].tail(30)
    show_df = show_df.round(3)
    st.dataframe(show_df, width="stretch")

st.markdown(f"""
<div style='text-align:center; padding:32px 0 16px; color:{C["dim"]}; font-size:12px; font-family:Space Mono,monospace; letter-spacing:0.5px;'>
  QUANTAI · 技术分析仅供学习研究 · 不构成投资建议 · 数据延迟15分钟
</div>
""", unsafe_allow_html=True)
