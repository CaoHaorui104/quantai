import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
from datetime import datetime, timedelta
import time

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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

.main { background-color: #0a0e1a; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    letter-spacing: -0.5px;
}

.stApp { background: #0a0e1a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid #1e2d4a;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 16px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    padding: 10px 24px;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Input */
.stTextInput > div > div > input,
.stSelectbox > div > div > select {
    background: #0f1629 !important;
    border: 1px solid #1e2d4a !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif;
}

/* Signal box */
.signal-buy {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
}
.signal-sell {
    background: linear-gradient(135deg, #4c0519, #881337);
    border: 1px solid #f43f5e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
}
.signal-hold {
    background: linear-gradient(135deg, #1c1917, #292524);
    border: 1px solid #a8a29e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
}

/* AI report */
.ai-report {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-left: 3px solid #6366f1;
    border-radius: 12px;
    padding: 24px;
    font-size: 15px;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* Portfolio weight ticker cards */
.po-ticker-card {
    background: #0f1629;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    border-top: 3px solid #6366f1;
    border-left: 1px solid #1e2d4a;
    border-right: 1px solid #1e2d4a;
    border-bottom: 1px solid #1e2d4a;
}

/* Monte Carlo key metric cards */
.mc-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.mc-label { font-size: 11px; color: #64748b; margin-bottom: 6px; }
.mc-value {
    font-family: 'Space Mono', monospace;
    font-size: 18px;
    font-weight: 700;
}
.mc-sub { font-size: 10px; color: #475569; margin-top: 4px; }

/* Risk assessment cards */
.risk-card {
    background: #0f1629;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
    border: 1px solid #1e2d4a;
}
.risk-label { font-size: 11px; color: #64748b; margin-bottom: 6px; }
.risk-value {
    font-family: 'Space Mono', monospace;
    font-size: 20px;
    font-weight: 700;
}
.risk-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-top: 6px;
}

/* Fundamental data cards */
.fund-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.fund-label {
    font-size: 11px;
    color: #64748b;
    margin-bottom: 6px;
    letter-spacing: 0.3px;
}
.fund-value {
    font-family: 'Space Mono', monospace;
    font-size: 16px;
    font-weight: 700;
    color: #e2e8f0;
}
.fund-na {
    font-family: 'Space Mono', monospace;
    font-size: 16px;
    color: #334155;
}

/* News items */
.news-item {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: border-color 0.2s;
}
.news-item:hover { border-color: #3b82f6; }
.news-item a {
    color: #e2e8f0;
    text-decoration: none;
    font-size: 14px;
    line-height: 1.5;
    flex: 1;
}
.news-item a:hover { color: #93c5fd; }
.news-meta {
    font-size: 11px;
    color: #475569;
    margin-top: 4px;
    white-space: nowrap;
}
.news-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #334155;
    margin-top: 6px;
    flex-shrink: 0;
}

/* Sentiment gauge */
.sentiment-wrap {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    margin-bottom: 16px;
}
.sentiment-score-box {
    text-align: center;
    min-width: 90px;
}
.sentiment-score-num {
    font-family: 'Space Mono', monospace;
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
}
.sentiment-score-label {
    font-size: 10px;
    color: #64748b;
    margin-top: 4px;
    letter-spacing: 0.5px;
}
.sentiment-right { flex: 1; }
.sentiment-bar-track {
    background: #1e2d4a;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    position: relative;
    margin-bottom: 6px;
    overflow: hidden;
}
.sentiment-reason {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 8px;
}

/* Backtest metric card */
.bt-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.bt-card .bt-label {
    font-size: 11px;
    color: #64748b;
    font-family: 'DM Sans', sans-serif;
    margin-bottom: 6px;
}
.bt-card .bt-value {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
}
.bt-card .bt-sub {
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
}
.ticker-tag {
    display: inline-block;
    background: #1e2d4a;
    border: 1px solid #3b82f6;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #93c5fd;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)


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


def compute_signal_scores(df: pd.DataFrame) -> pd.Series:
    """Vectorized signal score for every row in df (requires pre-computed indicators)."""
    scores = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        last = df.iloc[i]
        prev = df.iloc[i - 1]
        if pd.isna(last["RSI"]) or pd.isna(last["MACD"]) or pd.isna(last["BB_Upper"]):
            continue
        s = 0
        rsi = last["RSI"]
        if rsi < 35:
            s += 2
        elif rsi > 70:
            s -= 2
        if last["MACD"] > last["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
            s += 2
        elif last["MACD"] < last["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
            s -= 2
        if pd.notna(last.get("MA20")) and pd.notna(last.get("MA50")):
            if last["MA20"] > last["MA50"] and prev["MA20"] <= prev["MA50"]:
                s += 1
            elif last["MA20"] < last["MA50"] and prev["MA20"] >= prev["MA50"]:
                s -= 1
        if last["Close"] < last["BB_Lower"]:
            s += 1
        elif last["Close"] > last["BB_Upper"]:
            s -= 1
        scores.iloc[i] = s
    return scores


def run_backtest(
    df: pd.DataFrame,
    holding_days: int = 5,
    stop_loss: float = 0.03,
    take_profit: float = 0.05,
    slippage: float = 0.001,
    commission: float = 0.001,
) -> dict:
    """
    Simulate the signal strategy on historical data.
    Entry on BUY (score >= 2) at next-day close.
    Exit priority: stop-loss → take-profit → sell signal → max holding days.
    Slippage and commission are applied on both entry and exit legs.
    """
    scores = compute_signal_scores(df)
    closes = df["Close"].values
    dates = df.index
    n = len(df)

    trades = []
    in_trade = False
    entry_idx = entry_price = None

    # Round-trip cost: buy slippage + sell slippage + 2 × commission
    cost = 2 * slippage + 2 * commission

    for i in range(1, n - 1):
        if not in_trade:
            if scores.iloc[i] >= 2:
                entry_idx = i + 1
                entry_price = float(closes[entry_idx]) * (1 + slippage + commission)
                in_trade = True
        else:
            pnl = (float(closes[i]) - entry_price) / entry_price
            hold = i - entry_idx + 1

            if pnl <= -stop_loss:
                reason = f"止损 -{stop_loss:.0%}"
            elif pnl >= take_profit:
                reason = f"止盈 +{take_profit:.0%}"
            elif scores.iloc[i] <= -2:
                reason = "信号卖出"
            elif hold >= holding_days:
                reason = f"持有{holding_days}日"
            else:
                continue

            # SL/TP triggers on current bar close; signal exits on next-day close
            if reason.startswith("止"):
                exit_idx = i
                exit_price = float(closes[i]) * (1 - slippage - commission)
            else:
                exit_idx = min(i + 1, n - 1)
                exit_price = float(closes[exit_idx]) * (1 - slippage - commission)

            ret = (exit_price - entry_price) / entry_price
            trades.append({
                "entry_date": dates[entry_idx],
                "exit_date": dates[exit_idx],
                "entry_price": round(float(closes[entry_idx]), 2),
                "exit_price": round(float(closes[exit_idx]), 2),
                "return": ret,
                "days": exit_idx - entry_idx,
                "exit_reason": reason,
            })
            in_trade = False

    # Close any open position at end
    if in_trade and entry_idx is not None:
        exit_price = float(closes[-1]) * (1 - slippage - commission)
        ret = (exit_price - entry_price) / entry_price
        trades.append({
            "entry_date": dates[entry_idx],
            "exit_date": dates[-1],
            "entry_price": round(float(closes[entry_idx]), 2),
            "exit_price": round(float(closes[-1]), 2),
            "return": ret,
            "days": n - 1 - entry_idx,
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

    metrics = {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "total_return": total_return,
        "bh_return": bh_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "avg_return": float(rets.mean()),
        "avg_hold_days": avg_hold,
    }
    return {"trades": trades_df, "metrics": metrics, "equity": equity_series}


def build_backtest_chart(equity: pd.Series, df: pd.DataFrame, trades_df: pd.DataFrame) -> go.Figure:
    bh = df["Close"] / float(df["Close"].iloc[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bh.index, y=bh.values, name="买入持有",
        line=dict(color="#475569", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, name="策略净值",
        line=dict(color="#6366f1", width=2.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
        mode="lines+markers",
        marker=dict(size=5, color="#a78bfa"),
    ))

    # Mark winning / losing trades on the equity curve
    if not trades_df.empty:
        wins = trades_df[trades_df["return"] > 0]
        losses = trades_df[trades_df["return"] <= 0]
        win_equity = equity.reindex(wins["exit_date"])
        loss_equity = equity.reindex(losses["exit_date"])
        if not win_equity.empty:
            fig.add_trace(go.Scatter(
                x=wins["exit_date"], y=win_equity.values, mode="markers", name="盈利交易",
                marker=dict(color="#10b981", size=9, symbol="triangle-up"),
            ))
        if not loss_equity.empty:
            fig.add_trace(go.Scatter(
                x=losses["exit_date"], y=loss_equity.values, mode="markers", name="亏损交易",
                marker=dict(color="#f43f5e", size=9, symbol="triangle-down"),
            ))

    fig.update_layout(
        height=320,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        xaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", title="净值 (起始=1)"),
        legend=dict(bgcolor="#0f1629", bordercolor="#1e2d4a", borderwidth=1,
                    orientation="h", y=1.08, font=dict(size=11)),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )
    return fig


def get_signal(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Simple rule-based signal generator."""
    reasons = []
    score = 0

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # RSI
    rsi = last["RSI"]
    if rsi < 35:
        score += 2
        reasons.append(f"RSI={rsi:.1f} → 超卖区间，有反弹预期")
    elif rsi > 70:
        score -= 2
        reasons.append(f"RSI={rsi:.1f} → 超买区间，存在回调风险")
    else:
        reasons.append(f"RSI={rsi:.1f} → 中性区间")

    # MACD crossover
    if last["MACD"] > last["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        score += 2
        reasons.append("MACD 金叉 → 短期看涨信号")
    elif last["MACD"] < last["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        score -= 2
        reasons.append("MACD 死叉 → 短期看跌信号")

    # MA crossover (MA20 vs MA50)
    if "MA20" in df.columns and "MA50" in df.columns:
        if last["MA20"] > last["MA50"] and prev["MA20"] <= prev["MA50"]:
            score += 1
            reasons.append("均线金叉 (MA20>MA50) → 中期趋势转强")
        elif last["MA20"] < last["MA50"] and prev["MA20"] >= prev["MA50"]:
            score -= 1
            reasons.append("均线死叉 (MA20<MA50) → 中期趋势转弱")

    # Price vs Bollinger
    if last["Close"] < last["BB_Lower"]:
        score += 1
        reasons.append("价格跌破布林带下轨 → 短期超卖")
    elif last["Close"] > last["BB_Upper"]:
        score -= 1
        reasons.append("价格突破布林带上轨 → 短期超买")

    if score >= 2:
        return "BUY", reasons
    elif score <= -2:
        return "SELL", reasons
    else:
        return "HOLD", reasons


@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    # Let YFRateLimitError and network errors propagate to the caller for friendly UI handling
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
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
                        "日期":    str(row.get("Start Date", ""))[:10],
                        "内部人":  str(row.get("Insider", "")).title(),
                        "职位":    str(row.get("Position", "")),
                        "类型":    _tx_type(str(row.get("Text", ""))),
                        "股数":    f"{shares:,}",
                        "交易额":  val_str,
                    })
        except Exception:
            pass

        return {"earnings": earnings, "insider": insider_rows, "error": None}
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


def get_news_sentiment(ticker: str, headlines: list[str], api_key: str) -> dict:
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    prompt = f"""你是一位专业金融分析师。根据以下 {ticker} 最新新闻标题，评估整体市场情绪。

{numbered}

严格按以下格式输出，不要有任何其他内容：
SCORE: <-100到+100的整数，-100极度悲观，0中性，+100极度乐观>
REASON: <15字以内的一句话理由>"""

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    score, reason = 0, "分析完成"
    for line in text.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = max(-100, min(100, int(line.split(":", 1)[1].strip())))
            except ValueError:
                pass
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return {"score": score, "reason": reason}


def generate_rule_report(
    ticker: str,
    df: pd.DataFrame,
    signal: str,
    info: dict,
) -> str:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    rsi = float(last["RSI"])
    macd = float(last["MACD"])
    macd_sig = float(last["MACD_Signal"])
    close = float(last["Close"])
    bb_upper = float(last["BB_Upper"])
    bb_lower = float(last["BB_Lower"])
    bb_mid = float(last["BB_Mid"])
    price_20d_pct = (close - float(df["Close"].iloc[-20])) / float(df["Close"].iloc[-20]) * 100

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
    last = df.iloc[-1]
    price_change = ((df["Close"].iloc[-1] - df["Close"].iloc[-20]) / df["Close"].iloc[-20] * 100)

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

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


@st.cache_data(ttl=600)
def run_garch_forecast(returns_tuple: tuple, dates_tuple: tuple, n_forecast: int = 30) -> dict:
    try:
        from arch import arch_model
    except ImportError:
        return {"error": "arch_missing"}
    try:
        returns_scaled = np.array(returns_tuple) * 100  # scale for numerical stability
        model = arch_model(returns_scaled, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
        fitted = model.fit(disp="off", show_warning=False)

        # Historical conditional volatility (annualised %)
        cond_vol_pct = pd.Series(fitted.conditional_volatility) / 100 * np.sqrt(252) * 100

        # 20-day rolling realised volatility (annualised %)
        raw = pd.Series(np.array(returns_tuple))
        realized_pct = raw.rolling(20).std() * np.sqrt(252) * 100

        # 30-day forecast
        fc = fitted.forecast(horizon=n_forecast, reindex=False)
        fc_var = fc.variance.iloc[-1].values          # shape (n_forecast,)
        fc_vol_daily_pct = np.sqrt(fc_var) / 100 * 100   # daily vol %
        fc_vol_annual_pct = fc_vol_daily_pct * np.sqrt(252)

        avg_fc = float(fc_vol_annual_pct.mean())
        if avg_fc < 15:
            risk_level, risk_color = "低风险",   "#10b981"
        elif avg_fc < 30:
            risk_level, risk_color = "中等风险", "#f59e0b"
        elif avg_fc < 50:
            risk_level, risk_color = "高风险",   "#f97316"
        else:
            risk_level, risk_color = "极高风险", "#f43f5e"

        p = fitted.params
        alpha = float(p.get("alpha[1]", 0))
        beta  = float(p.get("beta[1]",  0))

        return {
            "dates":          list(dates_tuple),
            "cond_vol_pct":   cond_vol_pct.tolist(),
            "realized_pct":   realized_pct.fillna(0).tolist(),
            "fc_vol_annual":  fc_vol_annual_pct.tolist(),
            "fc_vol_daily":   fc_vol_daily_pct.tolist(),
            "current_vol":    float(cond_vol_pct.iloc[-1]),
            "avg_fc_annual":  avg_fc,
            "risk_level":     risk_level,
            "risk_color":     risk_color,
            "alpha":  alpha,
            "beta":   beta,
            "persistence": alpha + beta,
            "error":  None,
        }
    except Exception as e:
        return {"error": str(e)}


def build_garch_hist_chart(g: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["dates"], y=g["realized_pct"],
        name="已实现波动率（20日）",
        line=dict(color="#64748b", width=1.5),
        opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        x=g["dates"], y=g["cond_vol_pct"],
        name="GARCH 条件波动率",
        line=dict(color="#a78bfa", width=2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.07)",
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        xaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", title="年化波动率 (%)"),
        legend=dict(bgcolor="#0f1629", bordercolor="#1e2d4a", borderwidth=1,
                    orientation="h", y=1.12, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    return fig


def build_garch_forecast_chart(g: dict) -> go.Figure:
    days = list(range(1, len(g["fc_vol_annual"]) + 1))
    color = g["risk_color"]
    fig = go.Figure()
    fig.add_hline(
        y=g["current_vol"],
        line_dash="dot", line_color="#475569", opacity=0.8,
        annotation_text=f"当前 {g['current_vol']:.1f}%",
        annotation_font_color="#64748b", annotation_font_size=10,
    )
    fig.add_trace(go.Scatter(
        x=days, y=g["fc_vol_annual"],
        name="预测年化波动率",
        line=dict(color=color, width=2.5),
        fill="tozeroy", fillcolor=f"rgba(99,102,241,0.07)",
        mode="lines+markers",
        marker=dict(size=4, color=color),
        hovertemplate="第 %{x} 天: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        xaxis=dict(title="未来天数", gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", dtick=5),
        yaxis=dict(title="年化波动率 (%)", gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
        legend=dict(bgcolor="#0f1629", bordercolor="#1e2d4a", borderwidth=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
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


def build_mc_chart(mc: dict, ticker: str) -> go.Figure:
    days = np.arange(mc["n_days"] + 1)
    S0, pcts = mc["S0"], mc["pcts"]

    fig = go.Figure()

    # Background paths sample (200)
    sample = mc["paths"][:, :200]
    for col in range(sample.shape[1]):
        fig.add_trace(go.Scatter(
            x=days, y=sample[:, col], mode="lines",
            line=dict(color="rgba(99,102,241,0.035)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Shaded bands
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

    # Three key percentile lines
    for p, color, dash, name in [
        (5,  "#f43f5e", "dash",  "5th pct（悲观）"),
        (50, "#a78bfa", "solid", "中位数"),
        (95, "#10b981", "dash",  "95th pct（乐观）"),
    ]:
        fig.add_trace(go.Scatter(
            x=days, y=pcts[p], name=name,
            line=dict(color=color, width=2 if p == 50 else 1.8, dash=dash),
        ))

    fig.add_hline(
        y=S0, line_dash="dot", line_color="#475569", opacity=0.8,
        annotation_text=f"当前 ${S0:.2f}",
        annotation_font_color="#64748b", annotation_font_size=11,
    )

    fig.update_layout(
        height=340,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        xaxis=dict(title="未来天数", gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", dtick=5),
        yaxis=dict(title="模拟价格 ($)", gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
        legend=dict(bgcolor="#0f1629", bordercolor="#1e2d4a", borderwidth=1,
                    orientation="h", y=1.12, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    return fig


def build_mc_dist_chart(mc: dict) -> go.Figure:
    labels = [p[0] for p in mc["probs"]]
    values = [p[1] * 100 for p in mc["probs"]]
    colors = ["#f43f5e", "#f97316", "#64748b", "#34d399", "#10b981"]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family="Space Mono, monospace", size=12, color="#e2e8f0"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        height=240,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        xaxis=dict(
            title="概率 (%)", gridcolor="#1e2d4a", zerolinecolor="#1e2d4a",
            range=[0, max(values) * 1.35],
        ),
        yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
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
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef.max_sharpe(risk_free_rate=0.05)
        weights  = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.05, verbose=False)

        # ── Min Volatility ─────────────────────────────────────────────────
        ef2 = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef2.min_volatility()
        mv_weights = ef2.clean_weights()
        mv_ret, mv_vol, mv_sharpe = ef2.portfolio_performance(risk_free_rate=0.05, verbose=False)

        # ── Efficient frontier trace ────────────────────────────────────────
        frontier = []
        for target in np.linspace(float(mu.min()), float(mu.max()), 60):
            try:
                ef_t = EfficientFrontier(mu, S, weight_bounds=(0, 1))
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


def build_ef_chart(result: dict) -> go.Figure:
    fig = go.Figure()

    # Random portfolio scatter (coloured by Sharpe)
    fig.add_trace(go.Scatter(
        x=result["rand_vols"], y=result["rand_rets"],
        mode="markers",
        marker=dict(
            size=3, opacity=0.45,
            color=result["rand_sharpes"],
            colorscale=[[0, "#0f1629"], [0.45, "#3b4fd4"], [1, "#10b981"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Sharpe", font=dict(size=11, color="#64748b")),
                thickness=10, len=0.65,
                tickfont=dict(size=10, color="#64748b"),
            ),
            cmin=float(np.percentile(result["rand_sharpes"], 5)),
            cmax=float(np.percentile(result["rand_sharpes"], 95)),
        ),
        name="随机组合",
        hovertemplate="波动率: %{x:.1%}<br>收益率: %{y:.1%}<extra></extra>",
    ))

    # Efficient frontier curve
    if result["frontier"]:
        ef_v = [p[0] for p in result["frontier"]]
        ef_r = [p[1] for p in result["frontier"]]
        fig.add_trace(go.Scatter(
            x=ef_v, y=ef_r, mode="lines",
            line=dict(color="#a78bfa", width=2.5),
            name="有效前沿",
        ))

    # Min-vol portfolio
    fig.add_trace(go.Scatter(
        x=[result["mv_vol"]], y=[result["mv_ret"]],
        mode="markers+text",
        marker=dict(size=14, color="#f59e0b", symbol="diamond",
                    line=dict(color="#0a0e1a", width=2)),
        text=["最小波动"], textposition="top right",
        textfont=dict(size=10, color="#f59e0b"),
        name=f"最小波动  Sharpe {result['mv_sharpe']:.2f}",
    ))

    # Max-Sharpe portfolio
    fig.add_trace(go.Scatter(
        x=[result["vol"]], y=[result["ret"]],
        mode="markers+text",
        marker=dict(size=16, color="#10b981", symbol="star",
                    line=dict(color="#0a0e1a", width=2)),
        text=["最优夏普"], textposition="top right",
        textfont=dict(size=10, color="#10b981"),
        name=f"最优夏普  Sharpe {result['sharpe']:.2f}",
    ))

    fig.update_layout(
        height=420,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        xaxis=dict(title="年化波动率", gridcolor="#1e2d4a",
                   zerolinecolor="#1e2d4a", tickformat=".0%"),
        yaxis=dict(title="年化预期收益率", gridcolor="#1e2d4a",
                   zerolinecolor="#1e2d4a", tickformat=".0%"),
        legend=dict(bgcolor="#0f1629", bordercolor="#1e2d4a", borderwidth=1,
                    font=dict(size=11), x=0.01, y=0.99, xanchor="left"),
        margin=dict(l=0, r=20, t=10, b=0),
        hovermode="closest",
    )
    return fig


def build_weight_chart(result: dict) -> go.Figure:
    palette = ["#6366f1", "#10b981", "#f59e0b", "#f43f5e", "#a78bfa", "#34d399"]
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
                    line=dict(color="#0a0e1a", width=3)),
        textinfo="label+percent",
        textfont=dict(family="Space Mono, monospace", size=12),
        hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        direction="clockwise", sort=True,
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor="#0a0e1a",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        annotations=[dict(
            text=f"Sharpe<br><b>{result['sharpe']:.2f}</b>",
            x=0.5, y=0.5, font_size=13, showarrow=False,
            font=dict(color="#a78bfa", family="Space Mono, monospace"),
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
        spy_daily = spy_close.pct_change().dropna()
        aligned = pd.concat([daily, spy_daily], axis=1, join="inner")
        aligned.columns = ["stock", "spy"]
        if len(aligned) >= 20:
            corr = float(aligned.corr().iloc[0, 1])
            if beta is None:
                cov = aligned.cov().iloc[0, 1]
                var_spy = float(aligned["spy"].var())
                beta = float(cov / var_spy) if var_spy > 0 else None

    return {"ann_vol": ann_vol, "beta": beta, "corr": corr}


def build_comparison_chart(df: pd.DataFrame, spy_close: pd.Series, ticker: str) -> go.Figure:
    # Trim stock to last 6 months
    cutoff = df.index[-1] - pd.DateOffset(months=6)
    stock_6m = df["Close"][df.index >= cutoff]

    stock_norm = stock_6m / float(stock_6m.iloc[0])

    fig = go.Figure()

    if not spy_close.empty:
        spy_aligned = spy_close.reindex(stock_6m.index, method="ffill").dropna()
        spy_norm = spy_aligned / float(spy_aligned.iloc[0])
        fig.add_trace(go.Scatter(
            x=spy_norm.index, y=spy_norm.values, name="SPY",
            line=dict(color="#64748b", width=1.5, dash="dot"),
        ))

    stock_color = "#6366f1"
    fig.add_trace(go.Scatter(
        x=stock_norm.index, y=stock_norm.values, name=ticker,
        line=dict(color=stock_color, width=2.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
    ))

    final_ret = float(stock_norm.iloc[-1] - 1)
    spy_ret = float(spy_norm.iloc[-1] - 1) if not spy_close.empty and len(spy_norm) else None
    subtitle = f"{ticker} {final_ret:+.1%}"
    if spy_ret is not None:
        subtitle += f"  vs  SPY {spy_ret:+.1%}"

    fig.update_layout(
        height=260,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        xaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", title="净值 (起始=1)", tickformat=".2f"),
        legend=dict(bgcolor="#0f1629", bordercolor="#1e2d4a", borderwidth=1,
                    orientation="h", y=1.12, font=dict(size=11)),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        title=dict(text=subtitle, font=dict(size=13, color="#94a3b8"), x=0, xanchor="left", pad=dict(b=4)),
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#334155", opacity=0.6)
    return fig


def build_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=("价格 & 均线 & 布林带", "MACD", "RSI")
    )

    # ── Row 1: Candlestick + BB + MA ─────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="K线",
        increasing_line_color="#10b981",
        decreasing_line_color="#f43f5e",
        increasing_fillcolor="#10b981",
        decreasing_fillcolor="#f43f5e",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB上轨",
        line=dict(color="#6366f1", width=1, dash="dot"), opacity=0.6), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB下轨",
        line=dict(color="#6366f1", width=1, dash="dot"), opacity=0.6,
        fill="tonexty", fillcolor="rgba(99,102,241,0.05)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20",
        line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50",
        line=dict(color="#3b82f6", width=1.5)), row=1, col=1)

    # ── Row 2: MACD ───────────────────────────────────────────────────────────
    colors = ["#10b981" if v >= 0 else "#f43f5e" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD柱",
        marker_color=colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
        line=dict(color="#3b82f6", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
        line=dict(color="#f59e0b", width=1.5)), row=2, col=1)

    # ── Row 3: RSI ────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#a78bfa", width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f43f5e", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#10b981", opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=680,
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            bgcolor="#0f1629", bordercolor="#1e2d4a", borderwidth=1,
            font=dict(size=11), orientation="h", y=1.02
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", row=i, col=1)
        fig.update_yaxes(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", row=i, col=1)

    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 参数配置")
    ticker = st.text_input("股票代码", value="AAPL", placeholder="AAPL / TSLA / NVDA").upper().strip()
    period = st.selectbox("回看周期", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")

    st.markdown("---")
    st.markdown("### 🔁 回测参数")
    holding_days = st.slider("最长持仓天数", min_value=3, max_value=20, value=5, step=1)
    stop_loss_pct = st.slider("止损比例", min_value=1, max_value=10, value=3, step=1, format="%d%%")
    take_profit_pct = st.slider("止盈比例", min_value=1, max_value=20, value=5, step=1, format="%d%%")
    slippage_pct = st.slider("滑点", min_value=0.0, max_value=1.0, value=0.1, step=0.05, format="%.2f%%")
    commission_pct = st.slider("手续费（单边）", min_value=0.0, max_value=0.5, value=0.1, step=0.05, format="%.2f%%")

    run_btn = st.button("🚀 开始分析")

    st.markdown("---")
    st.markdown("**快捷股票**")
    cols = st.columns(2)
    quick = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL"]
    for i, q in enumerate(quick):
        if cols[i % 2].button(q, key=f"q_{q}"):
            ticker = q
            run_btn = True

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#4a5568; line-height:1.6;'>
    ⚠️ 本工具仅供学习研究，不构成任何投资建议。股市有风险，投资须谨慎。
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='header-banner'>
  <h1 style='margin:0; font-size:28px; color:#e2e8f0;'>📈 QuantAI · 股票分析系统</h1>
  <p style='margin:8px 0 0; color:#64748b; font-size:14px;'>
    技术指标 · AI解读 · 买卖信号 &nbsp;|&nbsp; 数据来源：Yahoo Finance &nbsp;|&nbsp; AI：Claude
  </p>
</div>
""", unsafe_allow_html=True)

if run_btn:
    st.session_state["analysis_active"] = True

if not st.session_state.get("analysis_active"):
    st.markdown("""
    <div style='text-align:center; padding:80px 0; color:#4a5568;'>
      <div style='font-size:48px; margin-bottom:16px;'>🔍</div>
      <div style='font-family: Space Mono, monospace; font-size:16px;'>在左侧输入股票代码，点击开始分析</div>
      <div style='font-size:13px; margin-top:8px;'>支持所有美股代码：AAPL · TSLA · NVDA · MSFT ···</div>
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
     ("#94a3b8" if risk["corr"] and abs(risk["corr"]) > 0.7 else "#a78bfa" if risk["corr"] and abs(risk["corr"]) > 0.4 else "#34d399")),
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

# ── Signal ────────────────────────────────────────────────────────────────────
signal, reasons = get_signal(df)

signal_styles = {
    "BUY":  ("signal-buy",  "🟢 BUY · 买入",  "#10b981"),
    "SELL": ("signal-sell", "🔴 SELL · 卖出", "#f43f5e"),
    "HOLD": ("signal-hold", "⚪ HOLD · 观望", "#a8a29e"),
}
css_class, label, color = signal_styles[signal]

col_sig, col_reasons = st.columns([1, 2])
with col_sig:
    st.markdown(f"""
    <div class='{css_class}'>
      <div style='font-size:22px; font-weight:700; color:{color};'>{label}</div>
      <div style='font-size:11px; color:#94a3b8; margin-top:6px;'>综合技术信号</div>
    </div>
    """, unsafe_allow_html=True)

with col_reasons:
    st.markdown("**信号依据**")
    for r in reasons:
        st.markdown(f"- {r}")

st.markdown("---")

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown("### 📊 技术图表")
fig = build_chart(df, ticker)
st.plotly_chart(fig, width="stretch")

# ── Market Comparison ────────────────────────────────────────────────────────
st.markdown("### 📈 大盘对比（近6个月收益率）")
if spy_close.empty:
    st.warning("⚠️ 无法加载 SPY 数据，大盘对比图暂不可用。")
else:
    cmp_fig = build_comparison_chart(df, spy_close, ticker)
    st.plotly_chart(cmp_fig, width="stretch")

# ── Backtest ─────────────────────────────────────────────────────────────────
st.markdown(
    f"### 🔁 策略回测 "
    f"<span style='font-size:13px; color:#64748b; font-family:DM Sans,sans-serif; font-weight:400;'>"
    f"持仓 {holding_days}日 &nbsp;·&nbsp; "
    f"止损 <span style='color:#f43f5e;'>-{stop_loss_pct}%</span> &nbsp;·&nbsp; "
    f"止盈 <span style='color:#10b981;'>+{take_profit_pct}%</span> &nbsp;·&nbsp; "
    f"滑点 {slippage_pct:.2f}% &nbsp;·&nbsp; 手续费 {commission_pct:.2f}%"
    f"</span>",
    unsafe_allow_html=True,
)

with st.spinner("回测计算中..."):
    bt = run_backtest(
        df,
        holding_days=holding_days,
        stop_loss=stop_loss_pct / 100,
        take_profit=take_profit_pct / 100,
        slippage=slippage_pct / 100,
        commission=commission_pct / 100,
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

    wr_color = "#10b981" if m["win_rate"] >= 0.5 else "#f43f5e"
    tr_color = _color(m["total_return"])
    dd_color = "#f43f5e" if m["max_drawdown"] < -0.1 else "#f59e0b" if m["max_drawdown"] < -0.05 else "#10b981"
    sh_color = "#10b981" if m["sharpe"] >= 1 else "#f59e0b" if m["sharpe"] >= 0 else "#f43f5e"
    bh_color = _color(m["bh_return"])

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "胜率", f"{m['win_rate']:.1%}", f"{m['total_trades']} 笔交易", wr_color),
        (c2, "总收益率", f"{m['total_return']:+.2%}", f"买入持有 {m['bh_return']:+.2%}", tr_color),
        (c3, "最大回撤", f"{m['max_drawdown']:.2%}", "策略期间峰谷跌幅", dd_color),
        (c4, "夏普比率", f"{m['sharpe']:.2f}", f"年化，持均 {m['avg_hold_days']:.1f}日", sh_color),
        (c5, "平均单笔", f"{m['avg_return']:+.2%}", "每笔交易平均收益", _color(m["avg_return"])),
    ]
    for col, label, val, sub, color in cards:
        col.markdown(f"""
        <div class='bt-card'>
          <div class='bt-label'>{label}</div>
          <div class='bt-value' style='color:{color};'>{val}</div>
          <div class='bt-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Equity curve ──────────────────────────────────────────────────────────
    bt_fig = build_backtest_chart(bt["equity"], df, bt["trades"])
    st.plotly_chart(bt_fig, width="stretch")

    # ── Trade log ─────────────────────────────────────────────────────────────
    with st.expander("📋 查看交易记录"):
        tdf = bt["trades"].copy()
        tdf["return"] = tdf["return"].map(lambda x: f"{x:+.2%}")
        tdf.columns = ["买入日期", "卖出日期", "买入价", "卖出价", "收益率", "持仓天数", "退出原因"]
        st.dataframe(tdf.style.map(
            lambda v: "color: #10b981" if isinstance(v, str) and v.startswith("+") else
                      ("color: #f43f5e" if isinstance(v, str) and v.startswith("-") else ""),
            subset=["收益率"]
        ), width="stretch")

st.markdown("---")

# ── Monte Carlo & GARCH (tabs) ────────────────────────────────────────────────
_daily_ret = df["Close"].pct_change().dropna()
_mu  = round(float(_daily_ret.mean()), 8)
_sig = round(float(_daily_ret.std()),  8)

mc_tab, garch_tab = st.tabs(["🎲 蒙特卡洛模拟（30天·1000条路径）", "📊 GARCH 波动率预测"])

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
        st.plotly_chart(build_mc_chart(mc, ticker), width="stretch")
    with dist_col:
        st.markdown(
            "<div style='font-size:12px;color:#64748b;margin-bottom:8px;'>30日后收益区间概率分布</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_mc_dist_chart(mc), width="stretch")

with garch_tab:
    with st.spinner("正在拟合 GARCH(1,1) 模型..."):
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
            (g1, "当前条件波动率",    f"{g['current_vol']:.1f}%",     "GARCH 最新估计（年化）", "#a78bfa"),
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
                "<div style='font-size:12px;color:#64748b;margin-bottom:4px;'>"
                "历史波动率 vs GARCH 条件波动率</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(build_garch_hist_chart(g), width="stretch")
        with fc_col:
            st.markdown(
                "<div style='font-size:12px;color:#64748b;margin-bottom:4px;'>"
                "未来 30 天年化波动率预测</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(build_garch_forecast_chart(g), width="stretch")

        # ── Model params ──────────────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:11px;color:#475569;margin-top:4px;'>"
            f"GARCH(1,1) 参数 &nbsp;·&nbsp; "
            f"α (ARCH) = {g['alpha']:.5f} &nbsp;·&nbsp; "
            f"β (GARCH) = {g['beta']:.5f} &nbsp;·&nbsp; "
            f"持续性 α+β = {g['persistence']:.5f}"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Portfolio Optimization ────────────────────────────────────────────────────
_palette = ["#6366f1", "#10b981", "#f59e0b", "#f43f5e", "#a78bfa", "#34d399"]

with st.expander("📐 投资组合优化（最大化夏普比率）", expanded=True):

    # st.form batches all widget changes; only the submit button triggers a rerun,
    # so this section never resets the rest of the page.
    with st.form("portfolio_form"):
        po_raw = st.text_input(
            "输入 2–6 只股票代码（逗号分隔，使用 2 年历史数据）",
            value=st.session_state.get("po_input", f"{ticker}, MSFT, GOOGL"),
            placeholder="AAPL, MSFT, GOOGL, NVDA",
        )
        po_submitted = st.form_submit_button("🔧 开始优化", width="stretch")

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
                (pm4, "有效配置资产",   f"{_n_active} 只",      f"共 {len(_po['tickers'])} 只输入", "#64748b"),
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
                    "<div style='font-size:12px;color:#64748b;margin-bottom:4px;'>"
                    "有效前沿（散点颜色 = 夏普比率，绿星 = 最优，黄钻 = 最小波动）</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(build_ef_chart(_po), width="stretch")
            with wt_col:
                st.markdown(
                    "<div style='font-size:12px;color:#64748b;margin-bottom:4px;'>"
                    "最优权重分配</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(build_weight_chart(_po), width="stretch")

            # ── Min-vol comparison ─────────────────────────────────────────
            with st.expander("📋 对比：最小波动组合"):
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
                    f"<div style='font-size:12px;color:#64748b;margin-top:12px;'>"
                    f"预期收益 {_po['mv_ret']:+.2%} &nbsp;·&nbsp; "
                    f"波动率 {_po['mv_vol']:.2%} &nbsp;·&nbsp; "
                    f"夏普 {_po['mv_sharpe']:.2f}</div>",
                    unsafe_allow_html=True,
                )

st.markdown("---")

# ── Earnings & Insider Trading ────────────────────────────────────────────────
st.markdown("### 📅 财报 & 内部人交易")

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
            date_color = ("#f59e0b" if days is not None and days <= 30
                          else "#e2e8f0")
            st.markdown(f"""
            <div class='bt-card' style='text-align:left;padding:18px 20px;'>
              <div style='font-size:11px;color:#64748b;margin-bottom:6px;'>下次财报日期</div>
              <div style='font-family:Space Mono,monospace;font-size:22px;
                          font-weight:700;color:{date_color};'>{earn["next_date"]}</div>
              <div style='font-size:12px;color:#f59e0b;margin-top:4px;'>{days_str}</div>
              <hr style='border-color:#1e2d4a;margin:12px 0;'>
              <div style='font-size:11px;color:#64748b;margin-bottom:4px;'>EPS 预期</div>
              <div style='font-family:Space Mono,monospace;font-size:18px;
                          font-weight:700;color:#a78bfa;'>
                {"${:.2f}".format(float(earn["eps_avg"])) if earn["eps_avg"] is not None else "N/A"}
              </div>
              <div style='font-size:11px;color:#475569;margin-top:2px;'>
                低 {"${:.2f}".format(float(earn["eps_low"])) if earn["eps_low"] is not None else "—"}
                &nbsp;·&nbsp;
                高 {"${:.2f}".format(float(earn["eps_high"])) if earn["eps_high"] is not None else "—"}
              </div>
              <hr style='border-color:#1e2d4a;margin:12px 0;'>
              <div style='font-size:11px;color:#64748b;margin-bottom:4px;'>营收预期</div>
              <div style='font-family:Space Mono,monospace;font-size:16px;
                          font-weight:700;color:#e2e8f0;'>{earn["rev_avg"]}</div>
              <div style='font-size:11px;color:#475569;margin-top:2px;'>
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
                return "color:#64748b"

            styled = (
                insider_df.style
                .apply(_style_row, axis=1)
                .map(_style_type, subset=["类型"])
            )
            st.markdown(
                "<div style='font-size:12px;color:#64748b;margin-bottom:6px;'>"
                "最近 10 条内部人交易记录</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(styled, width="stretch", hide_index=True)
        else:
            st.info("暂无内部人交易记录。")

st.markdown("---")

# ── News Sentiment ────────────────────────────────────────────────────────────
st.markdown("### 📰 新闻情绪分析")

with st.spinner("抓取最新新闻..."):
    news_items = fetch_news(ticker)

if not news_items:
    st.warning("暂无新闻数据。")
else:
    # ── Sentiment score (only with API key) ───────────────────────────────────
    if api_key:
        with st.spinner("Claude 正在分析情绪..."):
            try:
                headlines = [n["title"] for n in news_items]
                sentiment = get_news_sentiment(ticker, headlines, api_key)
                sc = sentiment["score"]

                if sc >= 40:
                    sc_color, sc_label = "#10b981", "偏多 · 积极"
                elif sc >= 10:
                    sc_color, sc_label = "#34d399", "中性偏多"
                elif sc >= -10:
                    sc_color, sc_label = "#94a3b8", "中性"
                elif sc >= -40:
                    sc_color, sc_label = "#f97316", "中性偏空"
                else:
                    sc_color, sc_label = "#f43f5e", "偏空 · 消极"

                # Bar: left half = negative zone, right half = positive zone
                # Fill from center outward
                bar_pct = abs(sc) / 2   # 0–50% of total width from center
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
            except Exception as e:
                st.error(f"情绪分析失败：{e}")
    else:
        st.info("💡 输入 API Key 后启用情绪分析（-100 到 +100 评分）。")

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
st.markdown("### 🤖 AI 深度分析")

if not api_key:
    rule_report = generate_rule_report(ticker, df, signal, info)
    st.markdown("""
    <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
      <span style='background:#1e2d4a;color:#94a3b8;font-size:11px;padding:3px 10px;
                   border-radius:4px;font-family:Space Mono,monospace;'>规则驱动</span>
      <span style='font-size:12px;color:#475569;'>输入 API Key 后切换为 Claude 深度分析</span>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"<div class='ai-report'>{rule_report}</div>", unsafe_allow_html=True)
else:
    with st.spinner("Claude 正在分析中..."):
        try:
            report = get_ai_analysis(ticker, df, signal, reasons, api_key)
            st.markdown("""
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
              <span style='background:#1e1b4b;color:#a78bfa;font-size:11px;padding:3px 10px;
                           border-radius:4px;font-family:Space Mono,monospace;'>Claude AI</span>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"<div class='ai-report'>{report}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Claude 分析失败：{e}")
            rule_report = generate_rule_report(ticker, df, signal, info)
            st.markdown("""
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
              <span style='background:#1e2d4a;color:#94a3b8;font-size:11px;padding:3px 10px;
                           border-radius:4px;font-family:Space Mono,monospace;'>规则驱动（回退）</span>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"<div class='ai-report'>{rule_report}</div>", unsafe_allow_html=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("📋 查看原始数据"):
    show_df = df[["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI", "MACD"]].tail(30)
    show_df = show_df.round(3)
    st.dataframe(show_df, width="stretch")

st.markdown("""
<div style='text-align:center; padding:32px 0 16px; color:#334155; font-size:12px;'>
  QuantAI · 技术分析仅供学习研究 · 不构成投资建议 · 数据延迟15分钟
</div>
""", unsafe_allow_html=True)
