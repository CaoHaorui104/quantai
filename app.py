import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
from datetime import datetime, timedelta
import time

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
) -> dict:
    """
    Simulate the signal strategy on historical data.
    Entry on BUY (score >= 2) at next-day close.
    Exit priority: stop-loss → take-profit → sell signal → max holding days.
    """
    scores = compute_signal_scores(df)
    closes = df["Close"].values
    dates = df.index
    n = len(df)

    trades = []
    in_trade = False
    entry_idx = entry_price = None

    for i in range(1, n - 1):
        if not in_trade:
            if scores.iloc[i] >= 2:
                entry_idx = i + 1
                entry_price = float(closes[entry_idx])
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
                exit_price = float(closes[i])
            else:
                exit_idx = min(i + 1, n - 1)
                exit_price = float(closes[exit_idx])

            ret = (exit_price - entry_price) / entry_price
            trades.append({
                "entry_date": dates[entry_idx],
                "exit_date": dates[exit_idx],
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return": ret,
                "days": exit_idx - entry_idx,
                "exit_reason": reason,
            })
            in_trade = False

    # Close any open position at end
    if in_trade and entry_idx is not None:
        exit_price = float(closes[-1])
        ret = (exit_price - entry_price) / entry_price
        trades.append({
            "entry_date": dates[entry_idx],
            "exit_date": dates[-1],
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
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
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        return df
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(df["Close"])
    return df


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

if not run_btn:
    st.markdown("""
    <div style='text-align:center; padding:80px 0; color:#4a5568;'>
      <div style='font-size:48px; margin-bottom:16px;'>🔍</div>
      <div style='font-family: Space Mono, monospace; font-size:16px;'>在左侧输入股票代码，点击开始分析</div>
      <div style='font-size:13px; margin-top:8px;'>支持所有美股代码：AAPL · TSLA · NVDA · MSFT ···</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Data fetch ────────────────────────────────────────────────────────────────
with st.spinner(f"正在拉取 {ticker} 数据..."):
    df = fetch_data(ticker, period)

if df.empty:
    st.error(f"❌ 找不到股票代码 **{ticker}**，请检查后重试。")
    st.stop()

info = yf.Ticker(ticker).info
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
st.plotly_chart(fig, use_container_width=True)

# ── Backtest ─────────────────────────────────────────────────────────────────
st.markdown(
    f"### 🔁 策略回测 "
    f"<span style='font-size:13px; color:#64748b; font-family:DM Sans,sans-serif; font-weight:400;'>"
    f"持仓 {holding_days}日 &nbsp;·&nbsp; "
    f"止损 <span style='color:#f43f5e;'>-{stop_loss_pct}%</span> &nbsp;·&nbsp; "
    f"止盈 <span style='color:#10b981;'>+{take_profit_pct}%</span>"
    f"</span>",
    unsafe_allow_html=True,
)

with st.spinner("回测计算中..."):
    bt = run_backtest(
        df,
        holding_days=holding_days,
        stop_loss=stop_loss_pct / 100,
        take_profit=take_profit_pct / 100,
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
    st.plotly_chart(bt_fig, use_container_width=True)

    # ── Trade log ─────────────────────────────────────────────────────────────
    with st.expander("📋 查看交易记录"):
        tdf = bt["trades"].copy()
        tdf["return"] = tdf["return"].map(lambda x: f"{x:+.2%}")
        tdf.columns = ["买入日期", "卖出日期", "买入价", "卖出价", "收益率", "持仓天数", "退出原因"]
        st.dataframe(tdf.style.map(
            lambda v: "color: #10b981" if isinstance(v, str) and v.startswith("+") else
                      ("color: #f43f5e" if isinstance(v, str) and v.startswith("-") else ""),
            subset=["收益率"]
        ), use_container_width=True)

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
    st.info("💡 在左侧输入 Anthropic API Key 即可获取 AI 智能分析报告。")
else:
    with st.spinner("Claude 正在分析中..."):
        try:
            report = get_ai_analysis(ticker, df, signal, reasons, api_key)
            st.markdown(f"<div class='ai-report'>{report}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"AI 分析失败：{e}")

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("📋 查看原始数据"):
    show_df = df[["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI", "MACD"]].tail(30)
    show_df = show_df.round(3)
    st.dataframe(show_df, use_container_width=True)

st.markdown("""
<div style='text-align:center; padding:32px 0 16px; color:#334155; font-size:12px;'>
  QuantAI · 技术分析仅供学习研究 · 不构成投资建议 · 数据延迟15分钟
</div>
""", unsafe_allow_html=True)
