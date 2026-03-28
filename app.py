"""
Apple Stock Prediction Analysis
Comprehensive Streamlit dashboard — data sourced from AAPL.csv
Models: Linear Regression, Random Forest, XGBoost, ARIMA, LSTM, CNN
GPU: XGBoost via CUDA device; LSTM/CNN via PyTorch CUDA
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TF startup noise

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# ── Optional GPU-accelerated libraries ────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    GPU_AVAILABLE   = torch.cuda.is_available()
    GPU_NAME        = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "CPU only"
    DEVICE          = torch.device("cuda" if GPU_AVAILABLE else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE   = False
    GPU_NAME        = "PyTorch not installed"
    DEVICE          = None

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apple Stock Prediction Analysis",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1c2130 0%, #252d3d 100%);
    border: 1px solid #2e3a4e;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 6px;
}
.metric-label  { font-size: 0.78rem; color: #c8d0db; letter-spacing:.04em; text-transform:uppercase; }
.metric-value  { font-size: 1.65rem; font-weight: 700; color: #ffffff; margin: 4px 0; }
.delta-pos     { color: #38d983; font-size: 0.85rem; }
.delta-neg     { color: #f56565; font-size: 0.85rem; }
.delta-neutral { color: #a0aec0; font-size: 0.85rem; }
.section-title {
    font-size: 1.35rem; font-weight: 700; color: #ffffff;
    margin: 18px 0 10px 0; padding-bottom: 6px;
    border-bottom: 2px solid #4a5568;
}
[data-testid="stMetricLabel"] p { color: #c8d0db !important; font-weight: 600 !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"]   { color: #ffffff !important; font-weight: 700 !important; }
.tab-title { font-size: 1.7rem; font-weight: 800; color: #2E2F2F; margin-bottom: 18px; }
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div   { gap: 4px; }
div[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: transparent; border-radius: 8px;
    padding: 10px 14px !important; width: 100%; cursor: pointer;
    font-size: 0.88rem !important; color: #a0aec0 !important;
    border: 1px solid transparent; transition: background .15s, color .15s;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: #2d3748 !important; color: #e2e8f0 !important;
}
hr { border-color: #2e3a4e; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
C = dict(
    blue="#007AFF", green="#34C759", red="#FF3B30",
    orange="#FF9500", purple="#AF52DE", teal="#5AC8FA",
    text="#e2e8f0", muted="#a0aec0",
)
QUALITATIVE  = px.colors.qualitative.Set1
MONTH_NAMES  = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ── CSV path (same directory as this script) ───────────────────────────────────
_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AAPL.csv")

# ── Key Apple milestones ───────────────────────────────────────────────────────
AAPL_MILESTONES = [
    {"date": "1984-01-24", "label": "Macintosh",         "short": "Mac"},
    {"date": "1985-09-16", "label": "Steve Jobs ousted",  "short": "Jobs out"},
    {"date": "1997-09-16", "label": "Steve Jobs returns", "short": "Jobs back"},
    {"date": "2001-01-09", "label": "iTunes",             "short": "iTunes"},
    {"date": "2001-10-23", "label": "iPod",               "short": "iPod"},
    {"date": "2007-06-29", "label": "iPhone",             "short": "iPhone"},
    {"date": "2011-08-24", "label": "Tim Cook becomes CEO","short": "Cook CEO"},
    {"date": "2014-09-09", "label": "iPhone 6 & Watch",   "short": "iPhone 6"},
    {"date": "2018-08-02", "label": "$1T Market Cap",     "short": "$1T"},
    {"date": "2021-03-09", "label": "COVID recovery pop", "short": "COVID pop"},
]

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False) #tell streamlit to cache the dataset
def load_aapl() -> pd.DataFrame:
    """Load AAPL data from the local CSV file."""
    df = pd.read_csv(_CSV, parse_dates=["Date"], index_col="Date") #open csv file and converts Date into datetime format
    df.index = pd.to_datetime(df.index).tz_localize(None)
    # Rename 'Adj Close' → 'Adj_Close' to avoid space issues; keep 'Close' as main price
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna() #remove rows with missing values
    df.sort_index(inplace=True) #sort data with Date index
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_peers(tickers: tuple, start: str) -> dict:
    """Download peer stock data from Yahoo Finance; skip tickers that fail."""
    out = {}
    for t in tickers:
        try:
            raw = yf.download(t, start=start,
                              end=datetime.today().strftime("%Y-%m-%d"),
                              auto_adjust=True, progress=False, timeout=15)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            out[t] = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            pass   # silently skip unavailable tickers
    return out

# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for p in [20, 50, 100, 200]:
        d[f"SMA_{p}"] = d["Close"].rolling(p).mean()
        d[f"EMA_{p}"] = d["Close"].ewm(span=p, adjust=False).mean()

    # RSI-14
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.where(loss != 0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    e12 = d["Close"].ewm(span=12, adjust=False).mean()
    e26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]        = e12 - e26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"]   = d["MACD"] - d["MACD_Signal"]

    # Bollinger Bands (20)
    d["BB_Mid"]   = d["Close"].rolling(20).mean()
    bb_std        = d["Close"].rolling(20).std()
    d["BB_Upper"] = d["BB_Mid"] + 2 * bb_std
    d["BB_Lower"] = d["BB_Mid"] - 2 * bb_std
    d["BB_Width"] = (d["BB_Upper"] - d["BB_Lower"]) / d["BB_Mid"].replace(0, np.nan)

    # Returns
    d["Daily_Return"]  = d["Close"].pct_change()
    d["Log_Return"]    = np.log(d["Close"] / d["Close"].shift(1))
    d["Volatility_21"] = d["Daily_Return"].rolling(21).std() * np.sqrt(252)

    # ATR-14
    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev_close).abs(),
        (d["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    return d


def build_ml_features(df: pd.DataFrame):
    d = add_indicators(df.copy())
    for lag in [1, 2, 3, 5, 10, 21]:
        d[f"Lag_{lag}"]    = d["Close"].shift(lag)
    for w in [5, 10, 20]:
        d[f"RollMean_{w}"] = d["Close"].rolling(w).mean()
        d[f"RollStd_{w}"]  = d["Close"].rolling(w).std()
    d["DayOfWeek"] = d.index.dayofweek
    d["Month"]     = d.index.month
    d["Target"]    = d["Close"].shift(-1)
    d.dropna(inplace=True)
    # All columns that are NOT raw OHLCV, returns, or the target
    exclude = {"Open", "High", "Low", "Volume", "Target",
               "Daily_Return", "Log_Return", "Adj_Close"}
    feat_cols = [c for c in d.columns if c not in exclude]
    return d, feat_cols

# ══════════════════════════════════════════════════════════════════════════════
# MODEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def eval_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE": mape}


# ── Sequence builder for LSTM / CNN ───────────────────────────────────────────
def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """X_seq[k] = X[k:k+lookback],  y_seq[k] = y[k+lookback-1]."""
    n     = len(X) - lookback + 1
    X_seq = np.stack([X[k: k + lookback] for k in range(n)])
    y_seq = y[lookback - 1:]
    return X_seq.astype(np.float32), y_seq.astype(np.float32)


# ── PyTorch model definitions ──────────────────────────────────────────────────
if TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):
        def __init__(self, n_feat, hidden=128, n_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.fc1  = nn.Linear(hidden, 64)
            self.fc2  = nn.Linear(64, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.drop(out[:, -1, :])
            return self.fc2(self.relu(self.fc1(out))).squeeze(-1)

    class _CNNNet(nn.Module):
        def __init__(self, n_feat, seq_len, filters=64, kernel=3):
            super().__init__()
            pad = kernel // 2
            self.conv1 = nn.Conv1d(n_feat, filters,       kernel, padding=pad)
            self.conv2 = nn.Conv1d(filters, filters // 2, kernel, padding=pad)
            self.pool  = nn.MaxPool1d(2)
            self.drop  = nn.Dropout(0.2)
            self.relu  = nn.ReLU()
            with torch.no_grad():
                d = torch.zeros(1, n_feat, seq_len)
                d = self.pool(self.relu(self.conv1(d)))
                d = self.pool(self.relu(self.conv2(d)))
                flat = int(d.view(1, -1).shape[1])
            self.fc1 = nn.Linear(flat, 64)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):               # x: (B, T, F)
            x = x.permute(0, 2, 1)         # → (B, F, T)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.drop(x.flatten(1))
            return self.fc2(self.relu(self.fc1(x))).squeeze(-1)


def train_pytorch(model, X_tr, y_tr, X_te,
                  epochs=50, batch_size=64, lr=1e-3,
                  progress_cb=None):
    """Train PyTorch model on GPU (or CPU) and return (model, y_pred)."""
    model.to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    ds   = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    dl   = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        if progress_cb:
            progress_cb(ep + 1, ep_loss / len(X_tr))

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te).to(DEVICE)).cpu().numpy()
    return model, preds


def fit_arima(series: pd.Series, order=(5, 1, 0), test_frac=0.2):
    n       = int(len(series) * (1 - test_frac))
    train   = series.iloc[:n]
    test    = series.iloc[n:]
    history = list(train)
    preds   = []
    prog    = st.progress(0, text="Running ARIMA rolling forecast…")
    for i, val in enumerate(test):
        m = ARIMA(history, order=order).fit()
        preds.append(float(m.forecast()[0]))
        history.append(float(val))
        prog.progress((i + 1) / len(test))
    prog.empty()
    preds   = np.array(preds)
    metrics = eval_metrics(test.values, preds)
    return preds, test.values, test.index, metrics

# ══════════════════════════════════════════════════════════════════════════════
# CHART LAYOUT HELPER  (deep-merges xaxis / yaxis dicts)
# ══════════════════════════════════════════════════════════════════════════════

def chart_layout(**kw) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=C["text"],
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(gridcolor="#2d3748", showgrid=True),
        yaxis=dict(gridcolor="#2d3748", showgrid=True),
    )
    # Deep-merge xaxis / yaxis so callers only override specific keys
    for ax in ("xaxis", "yaxis"):
        if ax in kw and isinstance(kw[ax], dict):
            merged = {**base[ax], **kw.pop(ax)}
            kw[ax] = merged
    base.update(kw)
    return base

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def tab_overview(df: pd.DataFrame):
    st.markdown('<p class="tab-title">📊 Market Overview</p>', unsafe_allow_html=True)

    latest  = df.iloc[-1]
    prev    = df.iloc[-2]
    change  = float(latest["Close"] - prev["Close"])
    pct_chg = change / float(prev["Close"]) * 100

    yr_df    = df[df.index.year == datetime.today().year]
    ytd_open = float(yr_df.iloc[0]["Close"]) if not yr_df.empty else float(df.iloc[0]["Close"])
    ytd_ret  = (float(latest["Close"]) - ytd_open) / ytd_open * 100
    high_52  = float(df["Close"].tail(252).max())
    low_52   = float(df["Close"].tail(252).min())
    vol_30   = float(df["Volume"].tail(30).mean())
    years    = (df.index[-1] - df.index[0]).days / 365.25
    cagr     = ((float(latest["Close"]) / float(df.iloc[0]["Close"])) ** (1 / years) - 1) * 100

    def _card(label, value, delta_html=""):
        return (f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'{delta_html}</div>')

    cols  = st.columns(6)
    sign  = "▲" if change >= 0 else "▼"
    d_cls = "delta-pos" if change >= 0 else "delta-neg"
    cols[0].markdown(_card("Current Price", f"${float(latest['Close']):.2f}",
        f'<div class="{d_cls}">{sign} ${abs(change):.2f} ({pct_chg:+.2f}%)</div>'),
        unsafe_allow_html=True)
    cols[1].markdown(_card("52-Week High", f"${high_52:.2f}",
        f'<div class="delta-neg">{(float(latest["Close"])-high_52)/high_52*100:.1f}% from high</div>'),
        unsafe_allow_html=True)
    cols[2].markdown(_card("52-Week Low", f"${low_52:.2f}",
        f'<div class="delta-pos">{(float(latest["Close"])-low_52)/low_52*100:.1f}% from low</div>'),
        unsafe_allow_html=True)
    cols[3].markdown(_card("Avg Volume (30d)", f"{vol_30/1e6:.1f}M",
        '<div class="delta-neutral">shares / day</div>'), unsafe_allow_html=True)
    ytd_cls = "delta-pos" if ytd_ret >= 0 else "delta-neg"
    cols[4].markdown(_card("YTD Return", f"{ytd_ret:+.1f}%",
        f'<div class="{ytd_cls}">Year to date</div>'), unsafe_allow_html=True)
    cagr_cls = "delta-pos" if cagr >= 0 else "delta-neg"
    cols[5].markdown(_card("Years of Data", f"{years:.1f} yrs",
        f'<div class="{cagr_cls}">CAGR {cagr:+.1f}%</div>'), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_chart, c_stats = st.columns([3, 1])
    with c_chart:
        st.markdown('<p class="section-title">All-Time Price History</p>', unsafe_allow_html=True)
        ov_c1, ov_c2 = st.columns([3, 1])
        tf_opt  = ov_c1.selectbox("Timeframe", ["1Y","3Y","5Y","10Y","Max"], index=2, key="ov_tf")
        ov_ms   = ov_c2.checkbox("Show Milestones", value=True, key="ov_ms")
        tf_map  = {"1Y":252,"3Y":756,"5Y":1260,"10Y":2520,"Max":len(df)}
        plot_df = df.tail(tf_map[tf_opt])
        fig = go.Figure(go.Scatter(
            x=plot_df.index, y=plot_df["Close"],
            fill="tozeroy", fillcolor="rgba(0,122,255,0.08)",
            line=dict(color=C["blue"], width=2), name="Close",
        ))
        if ov_ms:
            _add_milestones(fig, plot_df)
        fig.update_layout(**chart_layout(height=380, showlegend=False,
            yaxis=dict(title="Price (USD)"),
            xaxis=dict(rangeslider=dict(visible=True))))
        st.plotly_chart(fig, use_container_width=True)

    with c_stats:
        st.markdown('<p class="section-title">Key Statistics</p>', unsafe_allow_html=True)
        all_hi  = float(df["Close"].max())
        all_lo  = float(df["Close"].min())
        tot_ret = (float(latest["Close"]) - float(df.iloc[0]["Close"])) / float(df.iloc[0]["Close"]) * 100
        max_dd  = float(((df["Close"] / df["Close"].cummax()) - 1).min() * 100)
        for lbl, val in [
            ("All-Time High",  f"${all_hi:.2f}"),
            ("All-Time Low",   f"${all_lo:.2f}"),
            ("Total Return",   f"{tot_ret:.0f}%"),
            ("CAGR",           f"{cagr:.2f}%"),
            ("Max Drawdown",   f"{max_dd:.1f}%"),
            ("Data Since",     df.index[0].strftime("%b %Y")),
            ("Years of Data",  f"{years:.1f}"),
            ("Trading Days",   f"{len(df):,}"),
            ("Avg Daily Vol",  f"{float(df['Volume'].mean())/1e6:.0f}M"),
        ]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:7px 0;'
                f'border-bottom:1px solid #2e3a4e;">'
                f'<span style="color:#000000;font-size:.82rem;">{lbl}</span>'
                f'<span style="color:#47484A;font-weight:600;font-size:.88rem;">{val}</span>'
                f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA
# ══════════════════════════════════════════════════════════════════════════════

_MS_COLORS = [
    "#f6c90e","#38d983","#ff6b6b","#a78bfa",
    "#fb923c","#38bdf8","#f472b6","#34d399",
    "#facc15","#60a5fa",
]

def _add_milestones(fig: go.Figure, fdf: pd.DataFrame) -> list:
    """Add dotted vlines + rotated labels for milestones within fdf's date range."""
    visible = [
        m for m in AAPL_MILESTONES
        if fdf.index[0] <= pd.Timestamp(m["date"]) <= fdf.index[-1]
    ]
    for i, ms in enumerate(visible):
        clr = _MS_COLORS[i % len(_MS_COLORS)]
        ts  = pd.Timestamp(ms["date"])
        fig.add_vline(x=ts, line_width=1.2, line_dash="dot", line_color=clr)
        fig.add_annotation(
            x=ts, y=1.0, yref="paper",
            text=ms["short"], showarrow=False,
            textangle=-90, font=dict(size=9, color=clr),
            xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0.45)", borderpad=2,
        )
    return visible


def tab_eda(df: pd.DataFrame):
    st.markdown('<p class="tab-title">🔍 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    df_ind = add_indicators(df)

    menu_col, content_col = st.columns([1, 4])
    EDA_VIEWS = [
        "📈 Price Trend","🕯️ Candlestick","🔥 Correlation Heatmap",
        "📦 Volume Analysis","〰️ Moving Averages","📊 Returns Distribution",
        "⚡ Volatility","🌊 Seasonality","📅 Year-over-Year",
    ]
    with menu_col:
        st.markdown("#### Analysis View")
        view = st.radio("", EDA_VIEWS, key="eda_nav", label_visibility="collapsed")

    with content_col:
        d1, d2 = st.columns(2)
        start_d = d1.date_input("From", value=pd.Timestamp("2010-01-01"),
                                min_value=df_ind.index[0].date(),
                                max_value=df_ind.index[-1].date(), key="eda_from")
        end_d   = d2.date_input("To",   value=df_ind.index[-1].date(),
                                min_value=df_ind.index[0].date(),
                                max_value=df_ind.index[-1].date(), key="eda_to")
        fdf = df_ind.loc[pd.Timestamp(start_d):pd.Timestamp(end_d)]
        if fdf.empty:
            st.warning("No data for the selected date range.")
            return
        st.markdown("---")
        show_milestones = st.checkbox("📍 Show Key Milestones", value=True, key="eda_ms")
        if show_milestones:
            _vis_ms = [m for m in AAPL_MILESTONES
                       if fdf.index[0] <= pd.Timestamp(m["date"]) <= fdf.index[-1]]
            if _vis_ms:
                with st.expander("Milestone details", expanded=False):
                    _mc = st.columns(2)
                    for _i, _m in enumerate(_vis_ms):
                        _mc[_i % 2].markdown(
                            f"<span style='color:{_MS_COLORS[_i % len(_MS_COLORS)]}'>●</span> "
                            f"**{_m['date']}** — {_m['label']}",
                            unsafe_allow_html=True,
                        )
        st.markdown("")

        # ── PRICE TREND ────────────────────────────────────────────────────
        if view == "📈 Price Trend":
            st.markdown('<p class="section-title">Price Trend Analysis</p>', unsafe_allow_html=True)
            pc1, pc2 = st.columns([2, 1])
            price_col = pc1.selectbox("Price Field", ["Close","Open","High","Low"], key="pt_field")
            log_scale = pc2.checkbox("Logarithmic Scale", key="pt_log")

            fig = go.Figure(go.Scatter(
                x=fdf.index, y=fdf[price_col],
                fill="tozeroy", fillcolor="rgba(0,122,255,0.07)",
                line=dict(color=C["blue"], width=1.8), name=price_col,
            ))
            if show_milestones:
                _add_milestones(fig, fdf)
            fig.update_layout(**chart_layout(
                height=520,
                yaxis_type="log" if log_scale else "linear",
                yaxis=dict(title="Price (USD)"),
                xaxis=dict(rangeslider=dict(visible=True)),
                title=f"AAPL {price_col} Price",
            ))
            st.plotly_chart(fig, use_container_width=True)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Min",     f"${fdf[price_col].min():.2f}")
            s2.metric("Max",     f"${fdf[price_col].max():.2f}")
            s3.metric("Mean",    f"${fdf[price_col].mean():.2f}")
            s4.metric("Std Dev", f"${fdf[price_col].std():.2f}")

        # ── CANDLESTICK ────────────────────────────────────────────────────
        elif view == "🕯️ Candlestick":
            st.markdown('<p class="section-title">Candlestick Chart</p>', unsafe_allow_html=True)
            show_vol = st.checkbox("Show Volume", value=True, key="cs_vol")
            if show_vol:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.7,0.3], vertical_spacing=0.03)
            else:
                fig = go.Figure()
            candle = go.Candlestick(
                x=fdf.index, open=fdf["Open"], high=fdf["High"],
                low=fdf["Low"],  close=fdf["Close"],
                increasing_line_color=C["green"],
                decreasing_line_color=C["red"], name="AAPL",
            )
            if show_vol:
                fig.add_trace(candle, row=1, col=1)
                pct_chgs = fdf["Close"].pct_change().fillna(0)
                vol_colors = [C["green"] if r >= 0 else C["red"] for r in pct_chgs]
                fig.add_trace(go.Bar(x=fdf.index, y=fdf["Volume"],
                                     marker_color=vol_colors, opacity=0.65,
                                     name="Volume"), row=2, col=1)
            else:
                fig.add_trace(candle)
            if show_milestones:
                _add_milestones(fig, fdf)
            fig.update_layout(**chart_layout(
                height=560, showlegend=False,
                xaxis=dict(rangeslider=dict(visible=False)),
                yaxis=dict(title="Price (USD)"),
            ))
            fig.update_xaxes(gridcolor="#2d3748")
            fig.update_yaxes(gridcolor="#2d3748")
            st.plotly_chart(fig, use_container_width=True)

        # ── VOLUME ANALYSIS ────────────────────────────────────────────────
        elif view == "📦 Volume Analysis":
            st.markdown('<p class="section-title">Volume Analysis</p>', unsafe_allow_html=True)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.55,0.45], vertical_spacing=0.04,
                                subplot_titles=["Close Price","Volume"])
            fig.add_trace(go.Scatter(x=fdf.index, y=fdf["Close"],
                                     line=dict(color=C["blue"], width=1.8),
                                     name="Close"), row=1, col=1)
            vol_clr = [C["green"] if r >= 0 else C["red"]
                       for r in fdf["Close"].pct_change().fillna(0)]
            fig.add_trace(go.Bar(x=fdf.index, y=fdf["Volume"],
                                 marker_color=vol_clr, opacity=0.65,
                                 name="Volume"), row=2, col=1)
            if show_milestones:
                _add_milestones(fig, fdf)
            fig.update_layout(**chart_layout(height=560, hovermode="x unified"))
            fig.update_xaxes(gridcolor="#2d3748")
            fig.update_yaxes(gridcolor="#2d3748")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<p class="section-title">Volume Distribution</p>', unsafe_allow_html=True)
            fig2 = px.histogram(fdf, x="Volume", nbins=60,
                                color_discrete_sequence=[C["blue"]])
            fig2.update_layout(**chart_layout(height=300))
            st.plotly_chart(fig2, use_container_width=True)

        # ── MOVING AVERAGES ────────────────────────────────────────────────
        elif view == "〰️ Moving Averages":
            st.markdown('<p class="section-title">Moving Averages</p>', unsafe_allow_html=True)
            mc1, mc2   = st.columns(2)
            ma_type    = mc1.radio("Type", ["SMA","EMA","Both"], horizontal=True, key="ma_t")
            ma_periods = mc2.multiselect("Periods", [20,50,100,200],
                                         default=[20,50,200], key="ma_p")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fdf.index, y=fdf["Close"],
                                     line=dict(color="#7f8ea3", width=1),
                                     name="Close", opacity=0.5))
            pal = [C["blue"], C["green"], C["orange"], C["purple"]]
            for i, p in enumerate(sorted(ma_periods)):
                clr = pal[i % len(pal)]
                # use pre-computed columns (rolling on full history) so values
                # exist from the very start of the filtered date range
                if ma_type in ("SMA", "Both"):
                    sma_col = f"SMA_{p}"
                    sma_y = fdf[sma_col] if sma_col in fdf.columns else fdf["Close"].rolling(p).mean()
                    fig.add_trace(go.Scatter(x=fdf.index, y=sma_y,
                                             line=dict(color=clr, width=2), name=f"SMA {p}"))
                if ma_type in ("EMA", "Both"):
                    ema_col = f"EMA_{p}"
                    ema_y = fdf[ema_col] if ema_col in fdf.columns else fdf["Close"].ewm(span=p).mean()
                    fig.add_trace(go.Scatter(x=fdf.index, y=ema_y,
                                             line=dict(color=clr, width=2, dash="dot"),
                                             name=f"EMA {p}"))
            if show_milestones:
                _add_milestones(fig, fdf)
            fig.update_layout(**chart_layout(height=500, title="Moving Averages",
                yaxis=dict(title="Price (USD)")))
            st.plotly_chart(fig, use_container_width=True)

        # ── CORRELATION HEATMAP ────────────────────────────────────────────
        elif view == "🔥 Correlation Heatmap":
            st.markdown('<p class="section-title">Correlation Heatmap</p>', unsafe_allow_html=True)
            avail = ["Open","High","Low","Close","Volume",
                     "Daily_Return","BB_Width",
                     "Volatility_21","ATR","SMA_20","SMA_50","SMA_200"]
            avail = [c for c in avail if c in fdf.columns]
            sel = st.multiselect("Features", avail,
                default=[c for c in ["Open","High","Low","Close","Volume",
                                     "Daily_Return","Volatility_21"] if c in avail],
                key="hm_sel")
            if len(sel) >= 2:
                corr = fdf[sel].dropna().corr()
                fig  = px.imshow(corr, color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                                 title="Feature Correlation Matrix")
                fig.update_layout(**chart_layout(height=560))
                st.plotly_chart(fig, use_container_width=True)
                mask  = np.triu(np.ones_like(corr, dtype=bool), k=1)
                pairs = (corr.where(mask).stack()
                         .sort_values(key=abs, ascending=False)
                         .reset_index())
                pairs.columns = ["Feature A","Feature B","Correlation"]
                st.markdown("#### Strongest Correlations")
                st.dataframe(pairs.head(12), hide_index=True, use_container_width=True)
            else:
                st.info("Select at least 2 features.")

        # ── RETURNS DISTRIBUTION ───────────────────────────────────────────
        elif view == "📊 Returns Distribution":
            st.markdown('<p class="section-title">Returns Analysis</p>', unsafe_allow_html=True)
            rets = fdf["Close"].pct_change().dropna()

            mu, sig  = float(rets.mean()), float(rets.std())
            x_norm   = np.linspace(float(rets.min()), float(rets.max()), 200)
            bin_w    = (float(rets.max()) - float(rets.min())) / 100
            y_norm   = scipy_stats.norm.pdf(x_norm, mu, sig) * len(rets) * bin_w
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(x=rets*100, nbinsx=100,
                                        marker_color=C["blue"], opacity=0.7,
                                        name="Daily Returns"))
            fig1.add_trace(go.Scatter(x=x_norm*100, y=y_norm,
                                      line=dict(color=C["red"], width=2),
                                      name="Normal Fit"))
            fig1.update_layout(**chart_layout(height=350, title="Daily Returns (%)",
                xaxis=dict(title="Return (%)"), yaxis=dict(title="Count")))
            st.plotly_chart(fig1, use_container_width=True)

            cum  = (1 + rets).cumprod() - 1
            fig3 = go.Figure(go.Scatter(x=cum.index, y=cum*100,
                                        fill="tozeroy",
                                        fillcolor="rgba(0,122,255,0.07)",
                                        line=dict(color=C["blue"]), name="Cumulative"))
            if show_milestones:
                _add_milestones(fig3, fdf)
            fig3.update_layout(**chart_layout(height=340, title="Cumulative Returns",
                yaxis=dict(title="Return (%)")))
            st.plotly_chart(fig3, use_container_width=True)

            sk     = float(scipy_stats.skew(rets))
            ku     = float(scipy_stats.kurtosis(rets))
            sharpe = float(rets.mean()) / float(rets.std()) * np.sqrt(252)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Mean Daily",   f"{float(rets.mean())*100:.3f}%")
            m2.metric("Std Dev",      f"{float(rets.std())*100:.2f}%")
            m3.metric("Skewness",     f"{sk:.3f}")
            m4.metric("Kurtosis",     f"{ku:.3f}")
            m5.metric("Sharpe (ann)", f"{sharpe:.2f}")

        # ── VOLATILITY ─────────────────────────────────────────────────────
        elif view == "⚡ Volatility":
            st.markdown('<p class="section-title">Volatility Analysis</p>', unsafe_allow_html=True)
            win  = st.slider("Rolling Window (days)", 5, 252, 21, key="vw")
            rvol = fdf["Close"].pct_change().rolling(win).std() * np.sqrt(252) * 100
            fig  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.5,0.5], vertical_spacing=0.04,
                                 subplot_titles=["Close Price",
                                     f"Rolling {win}-day Annualised Volatility (%)"])
            fig.add_trace(go.Scatter(x=fdf.index, y=fdf["Close"],
                                     line=dict(color=C["blue"]), name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=rvol.index, y=rvol,
                                     fill="tozeroy", fillcolor="rgba(255,59,48,0.12)",
                                     line=dict(color=C["red"]),
                                     name="Volatility (%)"), row=2, col=1)
            if show_milestones:
                _add_milestones(fig, fdf)
            fig.update_layout(**chart_layout(height=560, showlegend=False))
            fig.update_xaxes(gridcolor="#2d3748")
            fig.update_yaxes(gridcolor="#2d3748")
            st.plotly_chart(fig, use_container_width=True)


        # ── SEASONALITY ────────────────────────────────────────────────────
        elif view == "🌊 Seasonality":
            st.markdown('<p class="section-title">Seasonality Analysis</p>', unsafe_allow_html=True)

            mo_prices = fdf["Close"].resample("ME").last()
            mo_ret    = mo_prices.pct_change() * 100
            mo_df     = pd.DataFrame({
                "Year":  mo_ret.index.year,
                "Month": mo_ret.index.month,
                "Ret":   mo_ret.values,
            }).dropna()

            if mo_df.empty:
                st.info("Not enough data for seasonality — widen the date range.")
            else:
                pivot = mo_df.pivot_table(values="Ret", index="Year", columns="Month")
                pivot.columns = [MONTH_NAMES[m] for m in pivot.columns]

                fig1 = px.imshow(pivot, color_continuous_scale="RdYlGn",
                                 text_auto=".1f", aspect="auto",
                                 title="Monthly Returns Heatmap (%)")
                fig1.update_layout(**chart_layout(height=520))
                st.plotly_chart(fig1, use_container_width=True)

                avg_mo = mo_df.groupby("Month")["Ret"].mean()
                avg_mo.index = [MONTH_NAMES[m] for m in avg_mo.index]
                fig2 = go.Figure(go.Bar(
                    x=avg_mo.index, y=avg_mo.values,
                    marker_color=[C["green"] if v >= 0 else C["red"] for v in avg_mo.values],
                ))
                fig2.add_hline(y=0, line_color=C["muted"], line_width=1)
                fig2.update_layout(**chart_layout(height=340,
                    title="Average Monthly Returns (%)",
                    yaxis=dict(title="Avg Return (%)")))
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("#### Seasonal Decomposition (last 500 trading days)")
                recent = fdf["Close"].tail(500)
                if len(recent) >= 60:
                    try:
                        decomp  = seasonal_decompose(recent, model="multiplicative",
                                                     period=21, extrapolate_trend="freq")
                        comp_map = {
                            "Observed": (decomp.observed, C["blue"]),
                            "Trend":    (decomp.trend,    C["green"]),
                            "Seasonal": (decomp.seasonal, C["orange"]),
                            "Residual": (decomp.resid,    C["red"]),
                        }
                        fig3 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                             subplot_titles=list(comp_map.keys()),
                                             vertical_spacing=0.04)
                        for i, (name, (series, clr)) in enumerate(comp_map.items(), 1):
                            fig3.add_trace(go.Scatter(x=series.index, y=series.values,
                                line=dict(color=clr, width=1.4), name=name), row=i, col=1)
                        fig3.update_layout(**chart_layout(height=720, showlegend=False))
                        fig3.update_xaxes(gridcolor="#2d3748")
                        fig3.update_yaxes(gridcolor="#2d3748")
                        st.plotly_chart(fig3, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Decomposition skipped: {e}")

        # ── YEAR-OVER-YEAR ─────────────────────────────────────────────────
        elif view == "📅 Year-over-Year":
            st.markdown('<p class="section-title">Year-over-Year Analysis</p>', unsafe_allow_html=True)

            annual = df["Close"].resample("YE").last().pct_change() * 100
            annual.index = annual.index.year
            annual = annual.dropna()

            fig1 = go.Figure(go.Bar(
                x=annual.index.astype(str), y=annual.values,                          # year labels on x, % return on y
                marker_color=[C["green"] if v >= 0 else C["red"] for v in annual.values],  # green for gain, red for loss
            ))
            fig1.add_hline(y=0, line_color=C["muted"], line_width=1)                  # zero baseline reference line
            # ── Milestone overlays on annual bar chart ──────────────────────
            if show_milestones:
                for _i, _ms in enumerate(AAPL_MILESTONES):
                    _yr = int(_ms["date"][:4])                                         # extract milestone year from date string
                    if _yr in annual.index:                                            # only mark years that exist in the chart
                        _clr = _MS_COLORS[_i % len(_MS_COLORS)]                       # pick accent colour from palette
                        fig1.add_vline(x=str(_yr), line_width=1.5,
                                       line_dash="dot", line_color=_clr)               # dotted vertical line at milestone year bar
                        fig1.add_annotation(
                            x=str(_yr), y=1.0, yref="paper",
                            text=_ms["short"], showarrow=False,
                            textangle=-90, font=dict(size=9, color=_clr),
                            xanchor="left", yanchor="top",
                            bgcolor="rgba(0,0,0,0.45)", borderpad=2,                   # semi-transparent label box
                        )
            fig1.update_layout(**chart_layout(height=400, title="Annual Returns (%)",
                xaxis=dict(title="Year"), yaxis=dict(title="Return (%)")))
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("#### Normalised Year Path (base = 0%)")
            cur_year  = datetime.today().year
            yoy_years = st.multiselect("Select Years",
                list(range(2010, cur_year + 1)),
                default=[2020,2021,2022,2023,2024], key="yoy_sel")
            fig2 = go.Figure()
            for i, yr in enumerate(sorted(yoy_years)):
                yd = df[df.index.year == yr]["Close"]
                if not yd.empty:
                    norm = (yd / float(yd.iloc[0]) - 1) * 100
                    fig2.add_trace(go.Scatter(
                        x=list(range(len(norm))), y=norm.values,
                        name=str(yr),
                        line=dict(color=QUALITATIVE[i % len(QUALITATIVE)], width=2),
                    ))
            fig2.add_hline(y=0, line_color=C["muted"], line_dash="dash", line_width=1)
            fig2.update_layout(**chart_layout(height=440, title="Return from Year Start (%)",
                xaxis=dict(title="Trading Day of Year"),
                yaxis=dict(title="Return (%)")))
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_INFO = {
    "Linear Regression": {
        "desc":     "Fits a linear mapping from engineered features to next-day close price.",
        "pros":     ["Very fast", "Fully interpretable", "Strong baseline"],
        "cons":     ["Assumes linearity", "Prone to underfitting"],
        "best_for": "Baseline benchmark and trend understanding",
        "gpu":      False, "seq": False,
    },
    "Random Forest": {
        "desc":     "Ensemble of decision trees that captures non-linear patterns robustly.",
        "pros":     ["Non-linear", "Feature importance", "Robust to outliers"],
        "cons":     ["Slower to train", "Less interpretable"],
        "best_for": "Complex non-linear feature relationships",
        "gpu":      False, "seq": False,
    },
    "XGBoost": {
        "desc":     "Extreme Gradient Boosting — iterative boosted trees. "
                    "Uses CUDA kernel on GPU for significantly faster training.",
        "pros":     ["GPU-accelerated (CUDA)", "Very high accuracy", "Built-in regularisation"],
        "cons":     ["Many hyperparameters", "Needs careful tuning"],
        "best_for": "Maximum accuracy with GPU speed",
        "gpu":      True, "seq": False,
    },
    "ARIMA": {
        "desc":     "Statistical time-series model using autoregression and moving averages.",
        "pros":     ["Purpose-built for time series", "Interpretable coefficients"],
        "cons":     ["Assumes stationarity", "Rolling forecast is slow"],
        "best_for": "Short-horizon statistical forecasting",
        "gpu":      False, "seq": False,
    },
    "LSTM": {
        "desc":     "Long Short-Term Memory recurrent network. Learns temporal dependencies "
                    "over sequences of past trading days. Runs on GPU via PyTorch CUDA.",
        "pros":     ["GPU-accelerated (PyTorch CUDA)", "Captures long-range time patterns",
                     "State-of-the-art for sequences"],
        "cons":     ["Needs sufficient data", "Longer training time"],
        "best_for": "Learning multi-step temporal patterns in price series",
        "gpu":      True, "seq": True,
    },
    "CNN": {
        "desc":     "1-D Convolutional Neural Network that extracts local temporal patterns "
                    "via sliding filters over the price sequence. GPU-accelerated.",
        "pros":     ["GPU-accelerated (PyTorch CUDA)", "Fast training", "Parallel feature extraction"],
        "cons":     ["Fixed receptive field", "Less memory than LSTM"],
        "best_for": "Extracting short-range local price patterns",
        "gpu":      True, "seq": True,
    },
}


def _gpu_badge():
    if GPU_AVAILABLE:
        st.success(f"🚀 GPU active — {GPU_NAME}", icon=None)
    elif TORCH_AVAILABLE:
        st.warning("⚠️ PyTorch installed but no CUDA GPU found — using CPU.")
    else:
        st.info("ℹ️ PyTorch not installed — LSTM/CNN unavailable. Run:  "
                "`pip install torch --index-url https://download.pytorch.org/whl/cu124`")


def tab_model_selection(df: pd.DataFrame):
    st.markdown('<p class="tab-title">🤖 Model Selection & Training</p>', unsafe_allow_html=True)

    _gpu_badge()
    st.markdown("")

    cfg_col, info_col = st.columns([2, 1])

    with cfg_col:
        st.markdown("### Configure & Train")
        model_name = st.selectbox("Algorithm", list(MODEL_INFO.keys()), key="model_name")
        info       = MODEL_INFO[model_name]

        # Warn if GPU model but no GPU
        if info["gpu"] and not GPU_AVAILABLE:
            st.warning(f"⚠️ {model_name} is GPU-accelerated but no GPU detected — will run on CPU (slower).")

        st.markdown("#### Data Window")
        period_label = st.selectbox(
            "Training data",
            ["Last 3 Years", "Last 5 Years", "Last 10 Years", "All Available"],
            index=1, key="data_win",
        )
        period_rows = {
            "Last 3 Years":    3 * 252,
            "Last 5 Years":    5 * 252,
            "Last 10 Years":  10 * 252,
            "All Available":  len(df),
        }

        st.markdown("#### Hyperparameters")
        params = {}

        if model_name == "ARIMA":
            ac1, ac2, ac3 = st.columns(3)
            params["p"]         = int(ac1.number_input("p (AR)", 0, 10, 5, key="ar_p"))
            params["d"]         = int(ac2.number_input("d (I)",  0,  3, 1, key="ar_d"))
            params["q"]         = int(ac3.number_input("q (MA)", 0, 10, 0, key="ar_q"))
            params["test_frac"] = st.slider("Test Size (%)", 5, 30, 15, key="ar_ts") / 100
            st.warning("⚠️ ARIMA rolling forecast is slow — uses the last 2 years of data.")

        elif model_name == "Random Forest":
            params["test_frac"] = st.slider("Test Size (%)", 5, 40, 20, key="ml_ts") / 100
            rc1, rc2 = st.columns(2)
            params["n_est"]     = int(rc1.slider("Trees",     50, 500, 200, 50, key="rf_n"))
            params["max_depth"] = int(rc2.slider("Max Depth",  2,  30,  10,     key="rf_d"))

        elif model_name == "XGBoost":
            if not XGB_AVAILABLE:
                st.error("XGBoost not installed. Run: `pip install xgboost`")
            params["test_frac"] = st.slider("Test Size (%)", 5, 40, 20, key="xgb_ts") / 100
            x1, x2, x3 = st.columns(3)
            params["n_est"]  = int(x1.slider("Estimators",    100, 1000, 400, 50,   key="xgb_n"))
            params["depth"]  = int(x2.slider("Max Depth",       3,   10,   6,       key="xgb_d"))
            params["lr"]     = float(x3.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01, key="xgb_lr"))

        elif model_name in ("LSTM", "CNN"):
            if not TORCH_AVAILABLE:
                st.error("PyTorch not installed. Run:  "
                         "`pip install torch --index-url https://download.pytorch.org/whl/cu124`")
            params["test_frac"] = st.slider("Test Size (%)", 5, 40, 20, key="seq_ts") / 100
            s1, s2, s3 = st.columns(3)
            params["lookback"]   = int(s1.slider("Lookback (days)", 20, 120, 60, key="seq_lb"))
            params["epochs"]     = int(s2.slider("Epochs",          10, 100,  50, key="seq_ep"))
            params["batch_size"] = int(s3.selectbox("Batch Size", [32, 64, 128, 256],
                                                     index=1, key="seq_bs"))
            if model_name == "LSTM":
                l1, l2 = st.columns(2)
                params["hidden"]  = int(l1.selectbox("Hidden Units", [64, 128, 256],
                                                      index=1, key="lstm_h"))
                params["dropout"] = float(l2.slider("Dropout", 0.1, 0.5, 0.2, 0.05, key="lstm_dr"))
            else:  # CNN
                c1, c2 = st.columns(2)
                params["filters"] = int(c1.selectbox("Filters", [32, 64, 128], index=1, key="cnn_f"))
                params["kernel"]  = int(c2.selectbox("Kernel Size", [3, 5, 7], key="cnn_k"))

        else:  # Linear Regression
            params["test_frac"] = st.slider("Test Size (%)", 5, 40, 20, key="lr_ts") / 100

        train_btn = st.button("🚀 Train Model", type="primary",
                              use_container_width=True, key="train_btn")

    with info_col:
        st.markdown("### About This Model")
        gpu_tag = "🟢 GPU" if info["gpu"] else "⚪ CPU"
        seq_tag = "📋 Sequence" if info["seq"] else "📊 Tabular"
        st.markdown(f"**{gpu_tag}** &nbsp;|&nbsp; **{seq_tag}**")
        st.info(info["desc"])
        st.markdown("**Advantages**")
        for p in info["pros"]:  st.markdown(f"• {p}")
        st.markdown("**Limitations**")
        for c in info["cons"]:  st.markdown(f"• {c}")
        st.markdown(f"**Best for:** {info['best_for']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING LOGIC
    # ═══════════════════════════════════════════════════════════════════════════
    if train_btn:
        try:
            # ── ARIMA ──────────────────────────────────────────────────────────
            if model_name == "ARIMA":
                with st.spinner("Running ARIMA rolling forecast…"):
                    series = df["Close"].tail(2 * 252)
                    preds, actuals, idx, metrics = fit_arima(
                        series,
                        order=(params["p"], params["d"], params["q"]),
                        test_frac=params["test_frac"],
                    )
                st.session_state["model_results"] = dict(
                    kind="arima",
                    model_name=f"ARIMA({params['p']},{params['d']},{params['q']})",
                    preds=preds, actuals=actuals, test_idx=idx, metrics=metrics,
                )

            # ── LSTM / CNN (PyTorch) ────────────────────────────────────────────
            elif model_name in ("LSTM", "CNN"):
                if not TORCH_AVAILABLE:
                    st.error("PyTorch is required for LSTM/CNN."); st.stop()
                n_rows               = period_rows[period_label]
                data_feat, feat_cols = build_ml_features(df.tail(n_rows))
                X_raw = data_feat[feat_cols].values.astype(np.float32)
                y_raw = data_feat["Target"].values.astype(np.float32)
                lb    = params["lookback"]

                # Scale before sequencing (fit on train only)
                sp_raw = int(len(X_raw) * (1 - params["test_frac"]))
                scaler = MinMaxScaler()
                X_sc   = np.vstack([
                    scaler.fit_transform(X_raw[:sp_raw]),
                    scaler.transform(X_raw[sp_raw:]),
                ])
                y_sc_min, y_sc_rng = y_raw[:sp_raw].min(), y_raw[:sp_raw].ptp()
                y_norm = (y_raw - y_sc_min) / (y_sc_rng + 1e-8)

                X_seq, y_seq = create_sequences(X_sc, y_norm, lb)
                sp_seq = int(len(X_seq) * (1 - params["test_frac"]))
                X_tr, X_te = X_seq[:sp_seq], X_seq[sp_seq:]
                y_tr, y_te = y_seq[:sp_seq], y_seq[sp_seq:]

                n_feat = X_seq.shape[2]
                if model_name == "LSTM":
                    net = _LSTMNet(n_feat,
                                   hidden=params["hidden"],
                                   dropout=params["dropout"])
                else:
                    net = _CNNNet(n_feat, lb,
                                  filters=params["filters"],
                                  kernel=params["kernel"])

                # Live epoch progress bar
                ep_bar  = st.progress(0, text="Initialising GPU…" if GPU_AVAILABLE else "Training…")
                ep_text = st.empty()
                device_label = f"🚀 {GPU_NAME}" if GPU_AVAILABLE else "🖥️ CPU"
                ep_text.caption(f"Running on {device_label}")

                def _cb(ep, loss):
                    ep_bar.progress(ep / params["epochs"],
                                    text=f"Epoch {ep}/{params['epochs']}  |  "
                                         f"Loss: {loss:.5f}  |  {device_label}")

                net, y_pred_norm = train_pytorch(
                    net, X_tr, y_tr, X_te,
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    progress_cb=_cb,
                )
                ep_bar.empty(); ep_text.empty()

                # Denormalise
                y_pred   = y_pred_norm * y_sc_rng + y_sc_min
                y_te_raw = y_te       * y_sc_rng + y_sc_min
                metrics  = eval_metrics(y_te_raw, y_pred)

                # Align test dates: sequences end at index lb-1+k in data_feat
                test_dates = data_feat.index[lb - 1 + sp_seq: lb - 1 + sp_seq + len(y_pred)]

                st.session_state["model_results"] = dict(
                    kind="seq", model_name=model_name,
                    preds=y_pred, actuals=y_te_raw,
                    test_idx=test_dates, metrics=metrics,
                    model=net, scaler=scaler,
                    y_sc_min=y_sc_min, y_sc_rng=y_sc_rng,
                    feat_cols=feat_cols, lookback=lb,
                )

            # ── XGBoost (CUDA) ─────────────────────────────────────────────────
            elif model_name == "XGBoost":
                if not XGB_AVAILABLE:
                    st.error("XGBoost not installed."); st.stop()
                with st.spinner(f"Training XGBoost {'on GPU 🚀' if GPU_AVAILABLE else 'on CPU'}…"):
                    n_rows               = period_rows[period_label]
                    data_feat, feat_cols = build_ml_features(df.tail(n_rows))
                    X = data_feat[feat_cols].values
                    y = data_feat["Target"].values
                    sp = int(len(X) * (1 - params["test_frac"]))
                    X_tr, X_te = X[:sp], X[sp:]
                    y_tr, y_te = y[:sp], y[sp:]
                    scaler    = MinMaxScaler()
                    X_tr_s    = scaler.fit_transform(X_tr)
                    X_te_s    = scaler.transform(X_te)
                    device_xgb = "cuda" if GPU_AVAILABLE else "cpu"
                    mdl = xgb.XGBRegressor(
                        n_estimators=params["n_est"],
                        max_depth=params["depth"],
                        learning_rate=params["lr"],
                        device=device_xgb,
                        tree_method="hist",
                        random_state=42,
                        verbosity=0,
                    )
                    mdl.fit(X_tr_s, y_tr)
                    y_pred  = mdl.predict(X_te_s)
                    metrics = eval_metrics(y_te, y_pred)
                    st.session_state["model_results"] = dict(
                        kind="xgb", model_name=f"XGBoost ({device_xgb.upper()})",
                        preds=y_pred, actuals=y_te,
                        test_idx=data_feat.index[sp:], metrics=metrics,
                        model=mdl, scaler=scaler, feat_cols=feat_cols,
                    )

            # ── Linear Regression / Random Forest (CPU sklearn) ────────────────
            else:
                with st.spinner(f"Training {model_name}…"):
                    n_rows               = period_rows[period_label]
                    data_feat, feat_cols = build_ml_features(df.tail(n_rows))
                    X = data_feat[feat_cols].values
                    y = data_feat["Target"].values
                    sp = int(len(X) * (1 - params["test_frac"]))
                    X_tr, X_te = X[:sp], X[sp:]
                    y_tr, y_te = y[:sp], y[sp:]
                    scaler    = MinMaxScaler()
                    X_tr_s    = scaler.fit_transform(X_tr)
                    X_te_s    = scaler.transform(X_te)
                    if model_name == "Linear Regression":
                        mdl = LinearRegression()
                    else:  # Random Forest
                        mdl = RandomForestRegressor(
                            n_estimators=params.get("n_est", 200),
                            max_depth=params.get("max_depth", 10),
                            random_state=42, n_jobs=-1,
                        )
                    mdl.fit(X_tr_s, y_tr)
                    y_pred  = mdl.predict(X_te_s)
                    metrics = eval_metrics(y_te, y_pred)
                    st.session_state["model_results"] = dict(
                        kind="ml", model_name=model_name,
                        preds=y_pred, actuals=y_te,
                        test_idx=data_feat.index[sp:], metrics=metrics,
                        model=mdl, scaler=scaler, feat_cols=feat_cols,
                    )

            st.success(f"✅ {model_name} trained successfully!")

        except Exception as exc:
            import traceback
            st.error(f"Training failed: {exc}")
            st.code(traceback.format_exc())

    # ── Quick preview ──────────────────────────────────────────────────────────
    if "model_results" in st.session_state:
        res = st.session_state["model_results"]
        st.markdown("---")
        st.markdown(f"### Results Preview — {res['model_name']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"${res['metrics']['RMSE']:.2f}")
        m2.metric("MAE",  f"${res['metrics']['MAE']:.2f}")
        m3.metric("R²",   f"{res['metrics']['R²']:.4f}")
        m4.metric("MAPE", f"{res['metrics']['MAPE']:.2f}%")

        xi  = list(range(len(res["actuals"])))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xi, y=res["actuals"],
                                 line=dict(color=C["blue"], width=2), name="Actual"))
        fig.add_trace(go.Scatter(x=xi, y=res["preds"],
                                 line=dict(color=C["orange"], width=2, dash="dash"),
                                 name="Predicted"))
        fig.update_layout(**chart_layout(height=350,
            title="Actual vs Predicted (test set)",
            xaxis=dict(title="Test Sample"), yaxis=dict(title="Price (USD)")))
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Head to the **Prediction** tab for full analysis and future forecasts.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def tab_prediction(df: pd.DataFrame):
    st.markdown('<p class="tab-title">📈 Prediction & Forecasting</p>', unsafe_allow_html=True)

    if "model_results" not in st.session_state:
        st.warning("⚠️ No model trained yet. Go to **Model Selection** and train a model first.")
        return

    res = st.session_state["model_results"]
    st.success(f"Active model: **{res['model_name']}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE",  f"${res['metrics']['RMSE']:.2f}", help="Root Mean Squared Error")
    m2.metric("MAE",   f"${res['metrics']['MAE']:.2f}",  help="Mean Absolute Error")
    m3.metric("R²",    f"{res['metrics']['R²']:.4f}",    help="Coefficient of Determination")
    m4.metric("MAPE",  f"{res['metrics']['MAPE']:.2f}%", help="Mean Absolute Percentage Error")
    st.markdown("---")

    # ── Actual vs Predicted ────────────────────────────────────────────────────
    st.markdown("### Actual vs Predicted — Test Period")
    preds   = np.asarray(res["preds"])
    actuals = np.asarray(res["actuals"])
    test_idx = res["test_idx"]

    # Build x-axis: prefer dates, fall back to integers
    if hasattr(test_idx, "__len__") and len(test_idx) == len(preds):
        x_axis = list(test_idx)
    else:
        x_axis = list(range(len(preds)))

    resid  = actuals - preds
    upper  = preds + np.abs(resid)
    lower  = preds - np.abs(resid)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis + x_axis[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself", fillcolor="rgba(255,149,0,.07)",
        line=dict(color="rgba(0,0,0,0)"), name="Error Band",
    ))
    fig.add_trace(go.Scatter(x=x_axis, y=actuals,
                             line=dict(color=C["blue"], width=2), name="Actual"))
    fig.add_trace(go.Scatter(x=x_axis, y=preds,
                             line=dict(color=C["orange"], width=2, dash="dash"),
                             name="Predicted"))
    fig.update_layout(**chart_layout(height=460,
        yaxis=dict(title="Price (USD)")))
    st.plotly_chart(fig, use_container_width=True)

    # ── Residuals ──────────────────────────────────────────────────────────────
    r1, r2 = st.columns(2)
    with r1:
        fig2 = go.Figure(go.Scatter(x=x_axis, y=resid, mode="markers",
                                    marker=dict(color=C["red"], size=3, opacity=0.5)))
        fig2.add_hline(y=0, line_color=C["muted"], line_width=1)
        fig2.update_layout(**chart_layout(height=300, title="Residuals",
            yaxis=dict(title="Residual ($)")))
        st.plotly_chart(fig2, use_container_width=True)
    with r2:
        fig3 = px.histogram(x=resid, nbins=60,
                            color_discrete_sequence=[C["purple"]],
                            title="Residuals Distribution")
        fig3.update_layout(**chart_layout(height=300))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Future Forecast (ML models only) ──────────────────────────────────────
    if res["kind"] == "ml" and "model" in res:
        st.markdown("---")
        st.markdown("### Future Price Forecast")
        fc_days = st.slider("Forecast Horizon (trading days)", 5, 120, 30, key="fc_days")

        if st.button("🔮 Generate Forecast", type="primary", key="fc_btn"):
            with st.spinner("Generating forecast…"):
                try:
                    data_feat, _ = build_ml_features(df)
                    mdl    = res["model"]
                    scaler = res["scaler"]
                    fcols  = res["feat_cols"]

                    # Use only columns present after feature building
                    avail_fcols = [c for c in fcols if c in data_feat.columns]
                    last_feat   = data_feat[avail_fcols].iloc[-1:].values.astype(float)
                    last_s      = scaler.transform(last_feat)

                    cur_price = float(df["Close"].iloc[-1])
                    forecast  = []
                    feat_buf  = last_s.copy()

                    for _ in range(fc_days):
                        pred = float(mdl.predict(feat_buf)[0])
                        forecast.append(pred)
                        feat_buf = np.roll(feat_buf, -1, axis=1)
                        feat_buf[0, -1] = pred / cur_price

                    fc_dates  = pd.bdate_range(
                        start=df.index[-1] + timedelta(days=1),
                        periods=fc_days,
                    )
                    daily_std = float(df["Close"].pct_change().std()) * cur_price
                    ci_upper  = [f + daily_std * np.sqrt(i+1) for i, f in enumerate(forecast)]
                    ci_lower  = [f - daily_std * np.sqrt(i+1) for i, f in enumerate(forecast)]

                    hist_ctx = df["Close"].tail(90)
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(x=hist_ctx.index, y=hist_ctx.values,
                        line=dict(color=C["blue"], width=2), name="Historical"))
                    fig4.add_trace(go.Scatter(
                        x=list(fc_dates) + list(fc_dates)[::-1],
                        y=ci_upper + ci_lower[::-1],
                        fill="toself", fillcolor="rgba(52,199,89,.10)",
                        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
                    ))
                    fig4.add_trace(go.Scatter(x=fc_dates, y=forecast,
                        line=dict(color=C["green"], width=2.5, dash="dash"),
                        mode="lines+markers", marker=dict(size=5), name="Forecast"))
                    fig4.add_vline(x=df.index[-1], line_color=C["muted"],
                                   line_dash="dash",
                                   annotation_text="Today",
                                   annotation_position="top right")
                    fig4.update_layout(**chart_layout(height=460,
                        title=f"AAPL Forecast — Next {fc_days} Trading Days",
                        yaxis=dict(title="Price (USD)")))
                    st.plotly_chart(fig4, use_container_width=True)

                    fc_df = pd.DataFrame({
                        "Date":            fc_dates.strftime("%Y-%m-%d"),
                        "Forecast ($)":    [f"${p:.2f}" for p in forecast],
                        "Lower Bound ($)": [f"${p:.2f}" for p in ci_lower],
                        "Upper Bound ($)": [f"${p:.2f}" for p in ci_upper],
                        "Expected Δ":      [f"{(p/cur_price-1)*100:+.2f}%" for p in forecast],
                    })
                    st.dataframe(fc_df, hide_index=True, use_container_width=True)

                except Exception as exc:
                    st.error(f"Forecast failed: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – COMPARATIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

PEERS = {
    "MSFT":"Microsoft", "GOOGL":"Alphabet", "AMZN":"Amazon",
    "NVDA":"NVIDIA",    "META":"Meta",       "TSLA":"Tesla",
    "SPY":"S&P 500 ETF","QQQ":"Nasdaq ETF",
}


def tab_comparative(df: pd.DataFrame):
    st.markdown('<p class="tab-title">⚖️ Comparative Analysis</p>', unsafe_allow_html=True)
    sub1, sub2 = st.tabs(["📊 Peer Stock Comparison", "🏆 Model Benchmark"])

    # ── PEER COMPARISON ────────────────────────────────────────────────────────
    with sub1:
        st.markdown("### AAPL vs Peer Stocks")
        st.info("ℹ️ Peer data is fetched from Yahoo Finance — charts populate only when online.")

        sel_peers = st.multiselect("Select peers", list(PEERS.keys()),
                                   default=["MSFT","GOOGL","NVDA","SPY"], key="peers")
        cp_period = st.selectbox("Period", ["1Y","3Y","5Y","10Y"], index=1, key="cp_per")
        period_start = {"1Y":"2023-01-01","3Y":"2021-01-01",
                        "5Y":"2019-01-01","10Y":"2014-01-01"}[cp_period]

        # Slice AAPL from CSV to match comparison period
        aapl_slice = df[df.index >= pd.Timestamp(period_start)][["Open","High","Low","Close","Volume"]]

        if not sel_peers:
            # Show AAPL-only chart
            fig0 = go.Figure(go.Scatter(x=aapl_slice.index, y=aapl_slice["Close"],
                                        line=dict(color=C["blue"], width=2), name="AAPL"))
            fig0.update_layout(**chart_layout(height=400, title="AAPL Price",
                yaxis=dict(title="Price (USD)")))
            st.plotly_chart(fig0, use_container_width=True)
        else:
            with st.spinner("Loading peer data from Yahoo Finance…"):
                p_data = load_peers(tuple(sel_peers), period_start)

            # Combine AAPL (CSV) with peers (yfinance)
            all_data = {"AAPL": aapl_slice, **p_data}
            all_t    = list(all_data.keys())

            # ── Normalised price
            st.markdown("#### Normalised Performance (Base = 100)")
            fig1 = go.Figure()
            for i, t in enumerate(all_t):
                pr = all_data[t]["Close"].dropna()
                if pr.empty:
                    continue
                norm = pr / float(pr.iloc[0]) * 100
                fig1.add_trace(go.Scatter(
                    x=norm.index, y=norm.values, name=t,
                    line=dict(color=QUALITATIVE[i % len(QUALITATIVE)],
                              width=3 if t == "AAPL" else 1.5),
                ))
            if not fig1.data:
                st.warning("No peer data available — check internet connection.")
            else:
                fig1.add_hline(y=100, line_color=C["muted"], line_dash="dash", line_width=1)
                fig1.update_layout(**chart_layout(height=500,
                    yaxis=dict(title="Normalised Price")))
                st.plotly_chart(fig1, use_container_width=True)

            # ── Metrics table
            st.markdown("#### Performance Metrics")
            rows = []
            for t in all_t:
                pr = all_data[t]["Close"].dropna()
                if len(pr) < 2:
                    continue
                dr     = pr.pct_change().dropna()
                tot    = (float(pr.iloc[-1]) / float(pr.iloc[0]) - 1) * 100
                vol    = float(dr.std()) * np.sqrt(252) * 100
                sharpe = float(dr.mean()) / max(float(dr.std()), 1e-8) * np.sqrt(252)
                mdd    = float(((pr / pr.cummax()) - 1).min() * 100)
                rows.append({
                    "Ticker":          t,
                    "Name":            "Apple Inc." if t == "AAPL" else PEERS.get(t, t),
                    "Total Return":    f"{tot:.1f}%",
                    "Ann. Volatility": f"{vol:.1f}%",
                    "Sharpe Ratio":    f"{sharpe:.2f}",
                    "Max Drawdown":    f"{mdd:.1f}%",
                    "Latest Price":    f"${float(pr.iloc[-1]):.2f}",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # ── Return correlation matrix
            st.markdown("#### Return Correlation Matrix")
            ret_dict = {}
            for t in all_t:
                pr = all_data[t]["Close"].dropna()
                if len(pr) > 1:
                    ret_dict[t] = pr.pct_change()
            if len(ret_dict) >= 2:
                ret_df = pd.DataFrame(ret_dict).dropna()
                corr   = ret_df.corr()
                fig2   = px.imshow(corr, color_continuous_scale="RdBu_r",
                                   zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                                   title="Pairwise Return Correlations")
                fig2.update_layout(**chart_layout(height=460))
                st.plotly_chart(fig2, use_container_width=True)

            # ── Rolling beta vs SPY (only when SPY loaded)
            if "SPY" in all_data and len(all_data["SPY"]["Close"].dropna()) > 60:
                st.markdown("#### Rolling 60-day Beta vs S&P 500")
                spy_r  = all_data["SPY"]["Close"].pct_change()
                aapl_r = all_data["AAPL"]["Close"].pct_change()
                common = spy_r.index.intersection(aapl_r.index)
                beta   = (aapl_r[common].rolling(60).cov(spy_r[common])
                          / spy_r[common].rolling(60).var())
                fig3 = go.Figure(go.Scatter(x=beta.index, y=beta.values,
                                            line=dict(color=C["purple"], width=2),
                                            name="Beta (60d)"))
                fig3.add_hline(y=1, line_color=C["muted"], line_dash="dash",
                               annotation_text="β = 1")
                fig3.update_layout(**chart_layout(height=350,
                    yaxis=dict(title="Rolling Beta")))
                st.plotly_chart(fig3, use_container_width=True)

    # ── MODEL BENCHMARK ────────────────────────────────────────────────────────
    with sub2:
        st.markdown("### Automated Model Benchmark (last 5 years, 80/20 split)")
        st.info("Trains Linear Regression, Ridge, Random Forest, Gradient Boosting and SVR "
                "on identical data splits for a fair comparison.")

        if st.button("🏋️ Run Benchmark", type="primary", key="bench_btn"):
            with st.spinner("Training all models… please wait."):
                try:
                    data_feat, feat_cols = build_ml_features(df.tail(5 * 252))
                    X  = data_feat[feat_cols].values
                    y  = data_feat["Target"].values
                    sp = int(len(X) * 0.80)
                    X_tr, X_te = X[:sp], X[sp:]
                    y_tr, y_te = y[:sp], y[sp:]

                    scaler = MinMaxScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)

                    candidates = {
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression":  Ridge(alpha=1.0),
                        "Random Forest":     RandomForestRegressor(n_estimators=150,
                                                                    random_state=42, n_jobs=-1),
                        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150,
                                                                        random_state=42),
                        "SVR (RBF)":         SVR(kernel="rbf", C=10),
                    }
                    bench_met  = {}
                    bench_pred = {}
                    prog = st.progress(0, text="Training…")
                    for k, (name, mdl) in enumerate(candidates.items()):
                        mdl.fit(X_tr_s, y_tr)
                        yp = mdl.predict(X_te_s)
                        bench_met[name]  = eval_metrics(y_te, yp)
                        bench_pred[name] = yp
                        prog.progress((k + 1) / len(candidates),
                                      text=f"Trained: {name}")
                    prog.empty()

                    st.session_state["bench_met"]    = bench_met
                    st.session_state["bench_pred"]   = bench_pred
                    st.session_state["bench_actual"] = y_te
                    st.success("✅ Benchmark complete!")

                except Exception as exc:
                    st.error(f"Benchmark failed: {exc}")

        if "bench_met" in st.session_state:
            bench_met    = st.session_state["bench_met"]
            bench_pred   = st.session_state["bench_pred"]
            bench_actual = st.session_state["bench_actual"]

            bdf = pd.DataFrame(bench_met).T.round(4).sort_values("R²", ascending=False)
            st.markdown("#### Metrics Summary")
            st.dataframe(bdf, use_container_width=True)

            metrics_sel = ["RMSE", "MAE", "MAPE"]
            names       = list(bench_met.keys())
            fig4 = make_subplots(rows=1, cols=3, subplot_titles=metrics_sel)
            for j, metric in enumerate(metrics_sel, 1):
                vals = [bench_met[n][metric] for n in names]
                fig4.add_trace(go.Bar(
                    x=names, y=vals,
                    marker_color=[QUALITATIVE[i % len(QUALITATIVE)] for i in range(len(names))],
                    showlegend=False,
                ), row=1, col=j)
            fig4.update_layout(**chart_layout(height=380, title="Error Metrics by Model"))
            fig4.update_xaxes(tickangle=30, gridcolor="#2d3748")
            fig4.update_yaxes(gridcolor="#2d3748")
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown("#### All Models vs Actual (Test Period)")
            xi = list(range(len(bench_actual)))
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=xi, y=bench_actual,
                                      line=dict(color="#edf2f7", width=2.5),
                                      name="Actual"))
            for i, (name, yp) in enumerate(bench_pred.items()):
                fig5.add_trace(go.Scatter(x=xi, y=yp,
                    line=dict(color=QUALITATIVE[i % len(QUALITATIVE)],
                              width=1.5, dash="dash"),
                    name=name))
            fig5.update_layout(**chart_layout(height=500,
                xaxis=dict(title="Test Sample"),
                yaxis=dict(title="Price (USD)")))
            st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div style="text-align:center;padding:24px 0 8px 0;">
        <h1 style="font-size:2.4rem;font-weight:800;color:#edf2f7;letter-spacing:-.01em;">
            🍎 Apple Stock Prediction Analysis
        </h1>
        <p style="color:#7f8ea3;font-size:1.05rem;margin-top:4px;">
            Comprehensive historical analysis &amp; ML-powered forecasting for AAPL
            &nbsp;|&nbsp; 1980 – Present
        </p>
    </div><hr>
    """, unsafe_allow_html=True)

    with st.spinner("Loading AAPL data from CSV…"):
        df = load_aapl()

    if df.empty:
        st.error("Could not load AAPL.csv — make sure the file is in the same folder as app.py.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 EDA",
        "🤖 Model Selection",
        "📈 Prediction",
        "⚖️ Comparative Analysis",
    ])
    with tab1: tab_overview(df)
    with tab2: tab_eda(df)
    with tab3: tab_model_selection(df)
    with tab4: tab_prediction(df)
    with tab5: tab_comparative(df)


if __name__ == "__main__":
    main()
