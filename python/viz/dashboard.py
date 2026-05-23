"""
Streamlit dashboard for the statistical arbitrage engine.

5 tabs:
  1. Overview        — equity curve, Sharpe, drawdown
  2. Pairs           — cointegration status, rolling half-life
  3. Signals         — live z-score heatmap + open positions
  4. Risk            — VaR, correlation matrix, concentration
  5. Walk-Forward    — fold-by-fold performance table + distribution
"""
from __future__ import annotations

# Optional: silence Streamlit deprecation warnings during import
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stat-Arb Engine",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Stat-Arb Engine")
    st.markdown("---")

    data_dir = st.text_input("Parquet data directory", value="data/parquet")

    st.subheader("Strategy parameters")
    z_entry = st.slider("Z-entry threshold", 1.0, 4.0, 2.0, 0.1)
    z_exit = st.slider("Z-exit threshold", 0.1, 1.5, 0.5, 0.1)
    z_stop = st.slider("Z-stop threshold", 2.5, 6.0, 4.0, 0.25)
    kelly_cap = st.slider("Kelly cap (%)", 5, 50, 25, 5) / 100.0
    target_vol = st.slider("Target daily vol (%)", 0.5, 3.0, 1.0, 0.1) / 100.0

    run_backtest = st.button("Run Walk-Forward Backtest", type="primary")

    st.markdown("---")
    st.caption("Statistical Arbitrage Engine v0.1.0")


# ---------------------------------------------------------------------------
# Helper: load or generate data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_price_data(data_dir_str: str) -> Dict[str, pd.Series]:
    """Load price data from Parquet files or generate synthetic."""
    from python.backtest.walk_forward import _generate_synthetic_prices

    parquet_path = Path(data_dir_str)
    prices: Dict[str, pd.Series] = {}
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"]

    for sym in symbols:
        p = parquet_path / f"{sym}.parquet"
        if p.exists():
            import pyarrow.parquet as pq
            df = pq.read_table(str(p)).to_pandas()
            if "close" in df.columns:
                prices[sym] = df["close"]
            elif "open_time" in df.columns:
                df = df.set_index("open_time")
                prices[sym] = df["close"]
        else:
            prices = _generate_synthetic_prices(symbols, n_days=700)
            break

    return {k: v for k, v in prices.items() if not v.empty}


@st.cache_data(ttl=600, show_spinner="Running backtest...")
def run_walk_forward(
    z_entry: float,
    z_exit: float,
    z_stop: float,
    kelly_cap: float,
    target_vol: float,
) -> list:
    from python.backtest.walk_forward import WalkForwardBacktester, _generate_synthetic_prices

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    pairs = [
        ("BTCUSDT", "ETHUSDT"),
        ("BTCUSDT", "BNBUSDT"),
        ("ETHUSDT", "SOLUSDT"),
    ]
    price_data = _generate_synthetic_prices(symbols, n_days=700)

    wf = WalkForwardBacktester(
        z_entry=z_entry,
        z_exit=z_exit,
        z_stop=z_stop,
        kelly_cap=kelly_cap,
        target_daily_vol=target_vol,
    )
    return wf.run(price_data, pairs)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def equity_curve_fig(daily_returns: np.ndarray, title: str = "Equity Curve") -> go.Figure:
    equity = pd.Series(np.cumprod(1.0 + daily_returns))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode="lines", name="Portfolio", line=dict(color="#00c4b4", width=2)))
    peak = equity.cummax()
    fig.add_trace(go.Scatter(y=peak, mode="lines", name="Peak", line=dict(color="#888", dash="dash", width=1)))
    fig.update_layout(title=title, xaxis_title="Bars", yaxis_title="NAV", template="plotly_dark", height=350)
    return fig


def drawdown_fig(daily_returns: np.ndarray) -> go.Figure:
    equity = np.cumprod(1.0 + daily_returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=dd * 100, mode="lines", fill="tozeroy",
                             name="Drawdown (%)", line=dict(color="#ff4444", width=1)))
    fig.update_layout(title="Drawdown (%)", xaxis_title="Bars",
                      yaxis_title="Drawdown %", template="plotly_dark", height=250)
    return fig


def zscore_heatmap(z_matrix: pd.DataFrame) -> go.Figure:
    """z_matrix: (pairs x time) heatmap."""
    fig = go.Figure(go.Heatmap(
        z=z_matrix.values,
        x=z_matrix.columns.astype(str),
        y=z_matrix.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-4,
        zmax=4,
    ))
    fig.update_layout(title="Z-Score Heatmap", template="plotly_dark", height=300)
    return fig


def fold_bar_chart(fold_sharpes: List[float]) -> go.Figure:
    colors = ["#00c4b4" if s > 0 else "#ff4444" for s in fold_sharpes]
    fig = go.Figure(go.Bar(
        x=list(range(len(fold_sharpes))),
        y=fold_sharpes,
        marker_color=colors,
        name="Fold Sharpe",
    ))
    fig.update_layout(
        title="Walk-Forward Fold Sharpe Ratios",
        xaxis_title="Fold",
        yaxis_title="Sharpe",
        template="plotly_dark",
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Metric display helpers
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: Optional[str] = None) -> None:
    st.metric(label=label, value=value, delta=delta)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

price_data = load_price_data(data_dir)
fold_results = []

if run_backtest:
    with st.spinner("Running walk-forward backtest..."):
        fold_results = run_walk_forward(z_entry, z_exit, z_stop, kelly_cap, target_vol)
    st.success(f"Completed {len(fold_results)} folds.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Pairs", "Signals", "Risk", "Walk-Forward"
])

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------

with tab1:
    st.header("Portfolio Overview")

    if fold_results:
        all_rets = np.concatenate([f.daily_returns for f in fold_results])
        from python.backtest.metrics import compute_metrics
        all_trades = [t for f in fold_results for t in f.trades]
        m = compute_metrics(all_rets, all_trades)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Sharpe", f"{m['sharpe']:.2f}")
        with c2:
            metric_card("Sortino", f"{m['sortino']:.2f}")
        with c3:
            metric_card("Calmar", f"{m['calmar']:.2f}")
        with c4:
            metric_card("Max DD", f"{m['max_drawdown']:.1%}")
        with c5:
            metric_card("Ann. Return", f"{m['annualised_return']:.1%}")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(equity_curve_fig(all_rets), use_container_width=True)
        with col2:
            st.plotly_chart(drawdown_fig(all_rets), use_container_width=True)

        c6, c7, c8 = st.columns(3)
        with c6:
            metric_card("Total Trades", str(m["n_trades"]))
        with c7:
            metric_card("Win Rate", f"{m['win_rate']:.1%}")
        with c8:
            metric_card("Turnover/yr", f"{m['turnover']:.0f}")
    else:
        st.info("Click 'Run Walk-Forward Backtest' in the sidebar to generate results.")
        # Show price chart preview
        if price_data:
            sym = st.selectbox("Preview symbol", list(price_data.keys()))
            s = price_data[sym]
            fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines",
                                       line=dict(color="#00c4b4")))
            fig.update_layout(title=f"{sym} Close Price", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Pairs
# ---------------------------------------------------------------------------

with tab2:
    st.header("Cointegration Status")

    if price_data and len(price_data) >= 2:
        from python.cointegration.rolling_coint import RollingCointegrationEngine

        syms = list(price_data.keys())[:4]
        pairs_to_test = [
            (syms[i], syms[j])
            for i in range(len(syms))
            for j in range(i + 1, len(syms))
        ]

        with st.spinner("Running rolling cointegration..."):
            engine = RollingCointegrationEngine(window=126, step=21)
            coint_results = engine.run_all_pairs(price_data, pairs_to_test)

        rows = []
        for (sy, sx), ts in coint_results.items():
            if ts.history:
                latest = ts.history[-1]
                rows.append({
                    "Pair": f"{sy}/{sx}",
                    "Status": latest.status.value,
                    "EG p-value": f"{latest.eg_pvalue:.4f}",
                    "Johansen Rank": latest.johansen_rank,
                    "OLS Beta": f"{latest.beta_ols:.3f}",
                    "Half-life": f"{latest.half_life:.1f}" if np.isfinite(latest.half_life) else "inf",
                    "Hurst": f"{latest.hurst_exp:.3f}",
                    "Spread Vol": f"{latest.spread_vol:.4f}",
                })

        if rows:
            df_pairs = pd.DataFrame(rows)

            def _color_status(val: str) -> str:
                if val == "COINTEGRATED":
                    return "background-color: #004d3f; color: white"
                elif val == "WEAKLY_COINTEGRATED":
                    return "background-color: #3d3d00; color: white"
                return "background-color: #4d0000; color: white"

            styled = df_pairs.style.applymap(_color_status, subset=["Status"])
            st.dataframe(styled, use_container_width=True)
        else:
            st.warning("No pair results — ensure price data is loaded.")
    else:
        st.info("Need at least 2 symbols loaded.")

# ---------------------------------------------------------------------------
# Tab 3: Signals
# ---------------------------------------------------------------------------

with tab3:
    st.header("Signal Monitor")

    if price_data and len(price_data) >= 2:
        syms = list(price_data.keys())[:4]
        n_show = 60  # last N bars

        rows_z: Dict[str, list] = {}
        pairs_sig = [(syms[0], syms[1]), (syms[0], syms[2])] if len(syms) >= 3 else [(syms[0], syms[1])]

        for sy, sx in pairs_sig:
            log_y = np.log(price_data[sy].values[-200:])
            log_x = np.log(price_data[sx].values[-200:])
            from python.kalman.hedge_ratio import KalmanHedgeRatio as KF
            kf = KF()
            spreads_live = []
            for t in range(len(log_y)):
                r = kf.update(float(log_y[t]), float(log_x[t]))
                spreads_live.append(r["spread"])

            s_arr = np.array(spreads_live)
            z_arr = np.zeros(len(s_arr))
            for t in range(10, len(s_arr)):
                w = s_arr[max(0, t - 63):t]
                z_arr[t] = (s_arr[t] - w.mean()) / (w.std() + 1e-8)

            rows_z[f"{sy}/{sx}"] = z_arr[-n_show:].tolist()

        if rows_z:
            z_df = pd.DataFrame(rows_z).T
            st.plotly_chart(zscore_heatmap(z_df), use_container_width=True)

            # Table of current z-scores
            st.subheader("Current Z-Scores")
            current_z = {pair: vals[-1] for pair, vals in rows_z.items()}
            cz_df = pd.DataFrame.from_dict(
                current_z, orient="index", columns=["Z-Score"]
            )
            cz_df["Signal"] = cz_df["Z-Score"].apply(
                lambda z: "LONG" if z < -2.0 else "SHORT" if z > 2.0 else "FLAT"
            )
            st.dataframe(cz_df, use_container_width=True)
    else:
        st.info("Load price data to view signals.")

# ---------------------------------------------------------------------------
# Tab 4: Risk
# ---------------------------------------------------------------------------

with tab4:
    st.header("Risk Dashboard")

    if price_data and len(price_data) >= 2:
        # Return correlation matrix
        rets_dict = {}
        for sym, s in price_data.items():
            r = s.pct_change().dropna()
            rets_dict[sym] = r

        rets_df = pd.DataFrame(rets_dict).dropna()
        corr = rets_df.corr()

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        ))
        fig_corr.update_layout(
            title="Return Correlation Matrix",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        if fold_results:
            all_rets = np.concatenate([f.daily_returns for f in fold_results])
            var_95 = float(np.percentile(all_rets, 5))
            var_99 = float(np.percentile(all_rets, 1))
            cvar_95 = float(all_rets[all_rets <= var_95].mean())

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("VaR 95%", f"{-var_95:.2%}")
            with c2:
                metric_card("VaR 99%", f"{-var_99:.2%}")
            with c3:
                metric_card("CVaR 95%", f"{-cvar_95:.2%}")

            # Return distribution
            fig_hist = px.histogram(
                x=all_rets * 100,
                nbins=50,
                labels={"x": "Daily Return (%)"},
                title="Daily Return Distribution",
                template="plotly_dark",
            )
            fig_hist.add_vline(x=var_95 * 100, line_dash="dash",
                               line_color="orange", annotation_text="VaR 95%")
            fig_hist.add_vline(x=var_99 * 100, line_dash="dash",
                               line_color="red", annotation_text="VaR 99%")
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Load price data to view risk metrics.")

# ---------------------------------------------------------------------------
# Tab 5: Walk-Forward
# ---------------------------------------------------------------------------

with tab5:
    st.header("Walk-Forward Analysis")

    if fold_results:
        from python.backtest.statistical_tests import (
            bootstrap_confidence_interval,
            walk_forward_t_test,
        )

        # Fold table
        fold_rows = []
        for f in fold_results:
            fold_rows.append({
                "Fold": f.fold_id,
                "Test Period": f"{f.test_start.date()} → {f.test_end.date()}",
                "Sharpe": round(f.sharpe, 2),
                "Sortino": round(f.sortino, 2),
                "Calmar": round(f.calmar, 2),
                "Max DD": f"{f.max_drawdown:.1%}",
                "Ann. Return": f"{f.annualised_return:.1%}",
                "Trades": f.n_trades,
                "Win Rate": f"{f.win_rate:.1%}",
            })

        st.dataframe(pd.DataFrame(fold_rows), use_container_width=True)

        sharpes = [f.sharpe for f in fold_results]
        st.plotly_chart(fold_bar_chart(sharpes), use_container_width=True)

        # Statistical tests
        st.subheader("Statistical Significance")
        t_stat, p_val = walk_forward_t_test(sharpes)
        lo, hi = bootstrap_confidence_interval(np.array(sharpes))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Mean Fold Sharpe", f"{np.mean(sharpes):.2f}")
        with c2:
            metric_card("t-statistic", f"{t_stat:.2f}")
        with c3:
            metric_card("p-value", f"{p_val:.4f}")
        with c4:
            metric_card("95% CI Sharpe", f"[{lo:.2f}, {hi:.2f}]")

        # Sharpe distribution
        fig_sh = px.histogram(
            x=sharpes,
            nbins=max(len(sharpes) // 2, 5),
            labels={"x": "Fold Sharpe"},
            title="Fold Sharpe Distribution",
            template="plotly_dark",
        )
        fig_sh.add_vline(x=0, line_dash="solid", line_color="red")
        st.plotly_chart(fig_sh, use_container_width=True)

        from scipy.stats import kurtosis, skew

        from python.backtest.metrics import deflated_sharpe_ratio
        all_rets = np.concatenate([f.daily_returns for f in fold_results])
        dsr = deflated_sharpe_ratio(
            sharpe_obs=np.mean(sharpes),
            n_trials=len(fold_results),
            n_obs=len(all_rets),
            skew=float(skew(all_rets)),
            excess_kurt=float(kurtosis(all_rets)),
        )
        st.metric("Deflated Sharpe Ratio (DSR)", f"{dsr:.4f}",
                  help="P(true SR > 0 after multiple-testing adjustment). > 0.95 is strong evidence.")
    else:
        st.info("Run the backtest to view walk-forward results.")
