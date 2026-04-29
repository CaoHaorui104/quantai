# QuantAI · Stock Analyzer

A Streamlit-based stock analysis dashboard powered by Claude AI. Combines real-time market data, technical indicators, rule-based trade signals, a backtesting engine, and news sentiment analysis in a single dark-themed interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![Claude](https://img.shields.io/badge/AI-Claude_Haiku%2FOpus-purple)

---

## Features

| Module | Description |
|--------|-------------|
| **Live Quote** | Real-time close price, daily change, volume, 52-week high/low |
| **Technical Indicators** | RSI (14), MACD (12/26/9), Bollinger Bands (20), MA20 / MA50 |
| **Trade Signal** | Rule-based BUY / SELL / HOLD score combining all indicators |
| **Backtesting** | Simulate the signal strategy on historical data with configurable stop-loss, take-profit, and max holding days |
| **News Sentiment** | Fetches latest headlines via yfinance; Claude scores overall sentiment from −100 to +100 |
| **AI Analysis** | Claude Opus generates a concise technical report with risk notes and trading ideas |

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) *(optional — all non-AI features work without one)*

---

## Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd quantai

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install streamlit yfinance pandas numpy plotly anthropic
```

---

## Usage

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Sidebar controls

| Control | Description |
|---------|-------------|
| **Ticker** | Any US stock symbol — `AAPL`, `TSLA`, `NVDA`, … |
| **Lookback period** | `1mo` / `3mo` / `6mo` / `1y` / `2y` |
| **Anthropic API Key** | Enables AI analysis and news sentiment scoring |
| **Max holding days** | Backtest exit after N days (3 – 20) |
| **Stop-loss %** | Exit when position drops this much (1 – 10%) |
| **Take-profit %** | Exit when position gains this much (1 – 20%) |

---

## How the signal system works

Each bar is scored across four independent rules:

```
RSI < 35            → +2   (oversold)
RSI > 70            → −2   (overbought)
MACD golden cross   → +2
MACD death cross    → −2
MA20/MA50 cross     → ±1
Price vs BB bands   → ±1
```

`score ≥ 2` → **BUY** · `score ≤ −2` → **SELL** · otherwise → **HOLD**

---

## Backtesting logic

- **Entry**: BUY signal fires → enter at next-day close
- **Exit priority**: stop-loss → take-profit → sell signal → max holding days
- **Reported metrics**: win rate, total return vs buy-and-hold, max drawdown, annualized Sharpe ratio

---

## Project structure

```
app.py          # Single-file Streamlit application
README.md
```

---

## Disclaimer

This tool is for educational and research purposes only. Nothing in this application constitutes financial advice. Always do your own due diligence before making investment decisions.
