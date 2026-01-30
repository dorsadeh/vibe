"""
Leverage Simulator - Streamlit Application

Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import DataLoader
from src.simulation import (
    BacktestConfig,
    BacktestEngine,
    LeverageConfig,
    PortfolioConfig,
    RebalanceConfig,
)
from src.metrics import (
    compute_metrics,
    compute_benchmark_metrics,
    format_metrics,
    compare_metrics,
    calculate_rolling_sharpe,
    calculate_rolling_drawdown,
)

# Page config
st.set_page_config(
    page_title="Leverage Simulator",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Leverage Simulator")
st.markdown("*Research tool for simulating leveraged portfolio strategies*")


@st.cache_data(ttl=3600)
def load_data():
    """Load and cache market data."""
    loader = DataLoader(cache_dir="data")
    data = loader.load_aligned(
        symbols=["SPY", "GLD", "TLT", "BIL"],
        start_date="2004-01-01",
    )
    returns = loader.get_returns(data, symbols=["SPY", "GLD", "TLT", "BIL"])
    return data, returns


# Load data
with st.spinner("Loading market data..."):
    data, returns = load_data()

st.sidebar.header("Configuration")

# Date range
st.sidebar.subheader("Date Range")
min_date = data.index.min().date()
max_date = data.index.max().date()

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start", min_date, min_value=min_date, max_value=max_date)
end_date = col2.date_input("End", max_date, min_value=min_date, max_value=max_date)

# Portfolio mode
st.sidebar.subheader("Portfolio Mode")
portfolio_mode = st.sidebar.selectbox(
    "Allocation Strategy",
    ["fixed_weights", "momentum_rotation"],
    format_func=lambda x: "Fixed Weights" if x == "fixed_weights" else "Momentum Rotation"
)

if portfolio_mode == "fixed_weights":
    st.sidebar.markdown("**Asset Weights**")
    spy_weight = st.sidebar.slider("SPY", 0.0, 1.0, 0.6, 0.05)
    gld_weight = st.sidebar.slider("GLD", 0.0, 1.0, 0.2, 0.05)
    tlt_weight = st.sidebar.slider("TLT", 0.0, 1.0, 0.2, 0.05)

    # Normalize weights
    total = spy_weight + gld_weight + tlt_weight
    if total > 0:
        weights = {
            "SPY": spy_weight / total,
            "GLD": gld_weight / total,
            "TLT": tlt_weight / total,
        }
    else:
        weights = {"SPY": 1.0, "GLD": 0.0, "TLT": 0.0}

    st.sidebar.caption(f"Normalized: SPY={weights['SPY']:.0%}, GLD={weights['GLD']:.0%}, TLT={weights['TLT']:.0%}")
else:
    momentum_lookback = st.sidebar.slider("Momentum Lookback (days)", 21, 252, 126)
    momentum_top_n = st.sidebar.slider("Top N Assets", 1, 3, 1)

# Leverage settings
st.sidebar.subheader("Leverage Settings")
target_leverage = st.sidebar.slider("Target Leverage", 1.0, 2.5, 1.5, 0.1)
max_leverage = st.sidebar.slider("Max Leverage", 1.0, 3.0, 2.0, 0.1)
broker_spread = st.sidebar.slider("Broker Spread (%)", 0.0, 3.0, 1.5, 0.1) / 100
maintenance_margin = st.sidebar.slider("Maintenance Margin (%)", 10, 50, 25) / 100
risk_off_to_cash = st.sidebar.checkbox("Risk-Off to BIL", value=True)

# Leverage rule
st.sidebar.subheader("Leverage Rule")
signal_asset = st.sidebar.selectbox("Signal Asset", ["SPY", "GLD", "TLT"])

rule_templates = {
    "SMA Trend": f"({signal_asset}.close > {signal_asset}.SMA_200)",
    "SMA + RSI": f"({signal_asset}.close > {signal_asset}.SMA_200) AND ({signal_asset}.RSI_14 < 70)",
    "Golden Cross": f"({signal_asset}.SMA_50 > {signal_asset}.SMA_200)",
    "Golden Cross + RSI": f"({signal_asset}.SMA_50 > {signal_asset}.SMA_200) AND ({signal_asset}.RSI_14 < 70)",
    "Custom": "",
}

rule_template = st.sidebar.selectbox("Rule Template", list(rule_templates.keys()))

if rule_template == "Custom":
    leverage_rule = st.sidebar.text_area(
        "Custom Rule",
        f"({signal_asset}.close > {signal_asset}.SMA_200)",
        help="Use AND, OR, NOT. Available: close, SMA_20, SMA_50, SMA_200, RSI_14, EMA_12, EMA_26"
    )
else:
    leverage_rule = rule_templates[rule_template]
    st.sidebar.code(leverage_rule, language=None)

# Rebalancing
st.sidebar.subheader("Rebalancing")
rebalance_freq = st.sidebar.selectbox(
    "Frequency",
    ["daily", "weekly", "monthly", "63_days", "126_days", "252_days"],
    index=2,  # monthly default
)

# Initial equity
st.sidebar.subheader("Simulation")
initial_equity = st.sidebar.number_input("Initial Equity ($)", 10_000, 10_000_000, 100_000, 10_000)

# Run button
run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

# Main content
if run_backtest:
    # Build config
    if portfolio_mode == "fixed_weights":
        portfolio_config = PortfolioConfig(
            mode="fixed_weights",
            weights=weights,
            risky_assets=["SPY", "GLD", "TLT"],
            cash_asset="BIL",
        )
    else:
        portfolio_config = PortfolioConfig(
            mode="momentum_rotation",
            risky_assets=["SPY", "GLD", "TLT"],
            cash_asset="BIL",
            momentum_lookback=momentum_lookback,
            momentum_top_n=momentum_top_n,
        )

    config = BacktestConfig(
        initial_equity=initial_equity,
        start_date=str(start_date),
        end_date=str(end_date),
        leverage=LeverageConfig(
            max_leverage=max_leverage,
            broker_spread=broker_spread,
            day_count=360,
            maintenance_margin=maintenance_margin,
            risk_off_to_cash=risk_off_to_cash,
        ),
        portfolio=portfolio_config,
        rebalance=RebalanceConfig(frequency=rebalance_freq),
        signal_asset=signal_asset,
        leverage_rule=leverage_rule,
        target_leverage=target_leverage,
    )

    # Run backtest
    with st.spinner("Running backtest..."):
        try:
            engine = BacktestEngine(data, returns, config)
            result = engine.run()

            # Get rate series for Sharpe calculation
            rate_series = data[("_rates", "rate")].loc[result.equity_curve.index]

            # Compute metrics
            metrics = compute_metrics(
                equity_curve=result.equity_curve,
                leverage_series=result.leverage_series,
                cumulative_interest=result.cumulative_interest,
                margin_events=result.margin_events,
                risk_free_rate=rate_series,
            )

            # Compute benchmark (SPY buy-and-hold)
            spy_prices = data[(signal_asset, "adj_close")].loc[result.equity_curve.index]
            benchmark_metrics = compute_benchmark_metrics(spy_prices, initial_equity)

            st.success("Backtest completed!")

        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

    # Display results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Summary", "📈 Equity Curve", "📉 Drawdown", "⚙️ Leverage", "💰 Interest"
    ])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Return", f"{metrics.total_return*100:.1f}%")
        col2.metric("CAGR", f"{metrics.cagr*100:.2f}%")
        col3.metric("Max Drawdown", f"{metrics.max_drawdown*100:.1f}%")
        col4.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Volatility", f"{metrics.volatility*100:.1f}%")
        col2.metric("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")
        col3.metric("Interest Paid", f"${metrics.total_interest_paid:,.0f}")
        col4.metric("Margin Calls", f"{metrics.margin_calls}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Strategy vs Benchmark")
            comparison_df = pd.DataFrame({
                "Metric": ["CAGR", "Volatility", "Max Drawdown", "Sharpe", "Calmar"],
                "Strategy": [
                    f"{metrics.cagr*100:.2f}%",
                    f"{metrics.volatility*100:.1f}%",
                    f"{metrics.max_drawdown*100:.1f}%",
                    f"{metrics.sharpe_ratio:.2f}",
                    f"{metrics.calmar_ratio:.2f}",
                ],
                f"{signal_asset} B&H": [
                    f"{benchmark_metrics.cagr*100:.2f}%",
                    f"{benchmark_metrics.volatility*100:.1f}%",
                    f"{benchmark_metrics.max_drawdown*100:.1f}%",
                    f"{benchmark_metrics.sharpe_ratio:.2f}",
                    f"{benchmark_metrics.calmar_ratio:.2f}",
                ],
            })
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("### Leverage Statistics")
            lev_df = pd.DataFrame({
                "Metric": ["Avg Leverage", "% Time Leveraged", "% Time in Cash", "Margin Calls"],
                "Value": [
                    f"{metrics.avg_leverage:.2f}x",
                    f"{metrics.pct_time_leveraged*100:.1f}%",
                    f"{(1-metrics.pct_time_leveraged-(result.leverage_series==1).mean())*100:.1f}%",
                    f"{metrics.margin_calls}",
                ],
            })
            st.dataframe(lev_df, hide_index=True, use_container_width=True)

    with tab2:
        # Equity curve chart
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                          shared_xaxes=True, vertical_spacing=0.05)

        # Strategy equity
        fig.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                name="Strategy",
                line=dict(color="blue"),
            ),
            row=1, col=1
        )

        # Benchmark
        benchmark_equity = spy_prices / spy_prices.iloc[0] * initial_equity
        fig.add_trace(
            go.Scatter(
                x=benchmark_equity.index,
                y=benchmark_equity.values,
                name=f"{signal_asset} B&H",
                line=dict(color="gray", dash="dash"),
            ),
            row=1, col=1
        )

        # Leverage
        fig.add_trace(
            go.Scatter(
                x=result.leverage_series.index,
                y=result.leverage_series.values,
                name="Leverage",
                line=dict(color="orange"),
                fill="tozeroy",
            ),
            row=2, col=1
        )

        fig.update_layout(
            title="Portfolio Equity Curve",
            height=600,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Leverage", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Drawdown chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=result.drawdown_series.index,
                y=result.drawdown_series.values * 100,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="red"),
            )
        )

        fig.update_layout(
            title="Drawdown Over Time",
            yaxis_title="Drawdown (%)",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Rolling metrics
        st.markdown("### Rolling Metrics (252-day)")

        returns_series = result.equity_curve.pct_change().fillna(0)
        rolling_sharpe = calculate_rolling_sharpe(returns_series, window=252)
        rolling_dd = calculate_rolling_drawdown(result.equity_curve, window=252)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name="Rolling Sharpe",
                line=dict(color="green"),
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_dd.index,
                y=rolling_dd.values * 100,
                name="Rolling Max DD",
                line=dict(color="red"),
            ),
            row=2, col=1
        )

        fig.update_layout(height=500, showlegend=True)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Max Drawdown (%)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Leverage over time
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=result.leverage_series.index,
                y=result.leverage_series.values,
                name="Leverage Factor",
                fill="tozeroy",
                line=dict(color="orange"),
            )
        )

        fig.update_layout(
            title="Leverage Factor Over Time",
            yaxis_title="Leverage (x)",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Leverage distribution
        col1, col2 = st.columns(2)

        with col1:
            lev_counts = result.leverage_series.value_counts().sort_index()
            fig = go.Figure(data=[
                go.Bar(x=lev_counts.index.astype(str), y=lev_counts.values)
            ])
            fig.update_layout(title="Leverage Distribution", xaxis_title="Leverage", yaxis_title="Days")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Position weights over time
            st.markdown("### Asset Allocation")
            weights_df = result.weights.fillna(0)
            fig = go.Figure()
            for col in weights_df.columns:
                fig.add_trace(go.Scatter(
                    x=weights_df.index,
                    y=weights_df[col] * 100,
                    name=col,
                    stackgroup="one",
                ))
            fig.update_layout(title="Asset Weights Over Time", yaxis_title="Weight (%)")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        # Interest costs
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=result.cumulative_interest.index,
                    y=result.cumulative_interest.values,
                    name="Cumulative Interest",
                    fill="tozeroy",
                    line=dict(color="purple"),
                )
            )
            fig.update_layout(title="Cumulative Interest Paid", yaxis_title="Interest ($)", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Monthly interest
            monthly_interest = result.interest_series.resample("ME").sum()
            fig = go.Figure(data=[
                go.Bar(x=monthly_interest.index, y=monthly_interest.values)
            ])
            fig.update_layout(title="Monthly Interest Paid", yaxis_title="Interest ($)", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Interest summary
        st.markdown("### Interest Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Interest", f"${result.cumulative_interest.iloc[-1]:,.2f}")
        col2.metric("Avg Daily Interest", f"${result.interest_series.mean():.2f}")
        col3.metric("Interest as % of Return",
                   f"{result.cumulative_interest.iloc[-1] / (result.equity_curve.iloc[-1] - initial_equity) * 100:.1f}%"
                   if result.equity_curve.iloc[-1] > initial_equity else "N/A")

    # Export data
    st.markdown("---")
    st.markdown("### Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        equity_csv = result.equity_curve.to_csv()
        st.download_button(
            "📥 Download Equity Curve",
            equity_csv,
            "equity_curve.csv",
            "text/csv",
        )

    with col2:
        # Combine all series
        export_df = pd.DataFrame({
            "equity": result.equity_curve,
            "leverage": result.leverage_series,
            "daily_interest": result.interest_series,
            "cumulative_interest": result.cumulative_interest,
            "drawdown": result.drawdown_series,
        })
        export_csv = export_df.to_csv()
        st.download_button(
            "📥 Download Full Results",
            export_csv,
            "backtest_results.csv",
            "text/csv",
        )

    with col3:
        weights_csv = result.weights.to_csv()
        st.download_button(
            "📥 Download Positions",
            weights_csv,
            "positions.csv",
            "text/csv",
        )

else:
    # Show instructions when not running
    st.info("👈 Configure your strategy in the sidebar and click **Run Backtest** to start.")

    st.markdown("""
    ## How to Use

    1. **Date Range**: Select the backtest period (data available from 2007)
    2. **Portfolio Mode**:
       - *Fixed Weights*: Allocate fixed percentages to SPY, GLD, TLT
       - *Momentum Rotation*: Rotate into top-performing asset(s)
    3. **Leverage Settings**: Configure target leverage and risk parameters
    4. **Leverage Rule**: Define when leverage is applied based on technical indicators
    5. **Rebalancing**: Set how often to rebalance the portfolio

    ## Available Indicators for Rules

    - `{ASSET}.close` - Closing price
    - `{ASSET}.SMA_20`, `SMA_50`, `SMA_200` - Simple Moving Averages
    - `{ASSET}.EMA_12`, `EMA_26` - Exponential Moving Averages
    - `{ASSET}.RSI_14` - Relative Strength Index

    ## Example Rules

    - `(SPY.close > SPY.SMA_200)` - Price above 200-day SMA
    - `(SPY.SMA_50 > SPY.SMA_200)` - Golden Cross
    - `(SPY.close > SPY.SMA_200) AND (SPY.RSI_14 < 70)` - Trend following with RSI filter
    """)

    # Show current market data
    st.markdown("---")
    st.markdown("### Current Market Data")

    latest = data.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("SPY", f"${latest[('SPY', 'adj_close')]:.2f}")
    col2.metric("GLD", f"${latest[('GLD', 'adj_close')]:.2f}")
    col3.metric("TLT", f"${latest[('TLT', 'adj_close')]:.2f}")
    col4.metric("3M T-Bill Rate", f"{latest[('_rates', 'rate')]:.2f}%")
