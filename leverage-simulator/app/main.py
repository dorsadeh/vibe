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
from src.indicators import compute_indicators
from src.rules import RulesEngine
import re

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


def extract_indicators_from_rule(rule: str) -> dict:
    """
    Extract indicator names from a rule string.

    Returns dict with:
        - smas: list of SMA periods used (e.g., [50, 200])
        - emas: list of EMA periods used
        - rsi: RSI period if used, else None
        - uses_close: whether close price is used
    """
    # Find all indicator references like SPY.SMA_200, SPY.RSI_14, etc.
    pattern = r'(\w+)\.(SMA_(\d+)|EMA_(\d+)|RSI_(\d+)|close)'
    matches = re.findall(pattern, rule)

    smas = set()
    emas = set()
    rsi = None
    uses_close = False

    for match in matches:
        if match[1] == 'close':
            uses_close = True
        elif match[1].startswith('SMA_'):
            smas.add(int(match[2]))
        elif match[1].startswith('EMA_'):
            emas.add(int(match[3]))
        elif match[1].startswith('RSI_'):
            rsi = int(match[4])

    return {
        'smas': sorted(smas),
        'emas': sorted(emas),
        'rsi': rsi,
        'uses_close': uses_close,
    }


def find_crossovers(series1: pd.Series, series2: pd.Series) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Find crossover points between two series.

    Returns:
        Tuple of (cross_above_dates, cross_below_dates)
        - cross_above: series1 crosses above series2
        - cross_below: series1 crosses below series2
    """
    # series1 > series2
    above = series1 > series2

    # Find where it changes
    cross_above = above & ~above.shift(1).fillna(False)  # Was below, now above
    cross_below = ~above & above.shift(1).fillna(True)   # Was above, now below

    return cross_above[cross_above].index, cross_below[cross_below].index


def add_crossover_markers(fig, indicators: pd.DataFrame, signal_asset: str, rule_indicators: dict, row: int = 1):
    """
    Add crossover markers to the chart.

    Detects crossovers based on indicators used in the rule:
    - Price crossing SMAs
    - SMA crossing another SMA (e.g., Golden Cross)
    """
    price_col = f"{signal_asset}.close"
    price = indicators[price_col]

    markers_added = []

    # Check for price vs SMA crossovers
    if rule_indicators['uses_close'] and rule_indicators['smas']:
        for sma_period in rule_indicators['smas']:
            sma_col = f"{signal_asset}.SMA_{sma_period}"
            if sma_col in indicators.columns:
                sma = indicators[sma_col]
                cross_above, cross_below = find_crossovers(price, sma)

                # Add upward cross markers (green triangles)
                if len(cross_above) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=cross_above,
                            y=price.loc[cross_above],
                            mode="markers",
                            marker=dict(
                                symbol="triangle-up",
                                size=12,
                                color="#00E676",  # Bright green
                                line=dict(width=1, color="#FFFFFF"),
                            ),
                            name=f"Cross Above SMA({sma_period})",
                            hovertemplate=f"Cross Above SMA({sma_period})<br>%{{x}}<br>${{y:.2f}}<extra></extra>",
                        ),
                        row=row, col=1
                    )
                    markers_added.append(f"price > SMA({sma_period})")

                # Add downward cross markers (red triangles)
                if len(cross_below) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=cross_below,
                            y=price.loc[cross_below],
                            mode="markers",
                            marker=dict(
                                symbol="triangle-down",
                                size=12,
                                color="#FF5252",  # Bright red
                                line=dict(width=1, color="#FFFFFF"),
                            ),
                            name=f"Cross Below SMA({sma_period})",
                            hovertemplate=f"Cross Below SMA({sma_period})<br>%{{x}}<br>${{y:.2f}}<extra></extra>",
                        ),
                        row=row, col=1
                    )

    # Check for SMA vs SMA crossovers (Golden Cross / Death Cross)
    smas = rule_indicators['smas']
    if len(smas) >= 2:
        # Sort to get fast and slow SMAs
        smas_sorted = sorted(smas)
        for i, fast_period in enumerate(smas_sorted[:-1]):
            for slow_period in smas_sorted[i+1:]:
                fast_col = f"{signal_asset}.SMA_{fast_period}"
                slow_col = f"{signal_asset}.SMA_{slow_period}"

                if fast_col in indicators.columns and slow_col in indicators.columns:
                    fast_sma = indicators[fast_col]
                    slow_sma = indicators[slow_col]

                    cross_above, cross_below = find_crossovers(fast_sma, slow_sma)

                    # Golden Cross (fast crosses above slow)
                    if len(cross_above) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=cross_above,
                                y=fast_sma.loc[cross_above],
                                mode="markers",
                                marker=dict(
                                    symbol="star",
                                    size=16,
                                    color="#FFD700",  # Gold
                                    line=dict(width=1, color="#FFFFFF"),
                                ),
                                name=f"Golden Cross ({fast_period}/{slow_period})",
                                hovertemplate=f"Golden Cross SMA({fast_period})>SMA({slow_period})<br>%{{x}}<extra></extra>",
                            ),
                            row=row, col=1
                        )
                        markers_added.append(f"SMA({fast_period}) > SMA({slow_period})")

                    # Death Cross (fast crosses below slow)
                    if len(cross_below) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=cross_below,
                                y=fast_sma.loc[cross_below],
                                mode="markers",
                                marker=dict(
                                    symbol="x",
                                    size=12,
                                    color="#FF5252",  # Red
                                    line=dict(width=2, color="#FF5252"),
                                ),
                                name=f"Death Cross ({fast_period}/{slow_period})",
                                hovertemplate=f"Death Cross SMA({fast_period})<SMA({slow_period})<br>%{{x}}<extra></extra>",
                            ),
                            row=row, col=1
                        )

    return fig, markers_added


def add_leverage_regions(fig, leverage_series, row=1):
    """Add shaded regions for leverage ON periods."""
    # Find contiguous leverage ON regions
    is_leveraged = leverage_series > 1

    # Find transitions
    changes = is_leveraged.astype(int).diff().fillna(0)
    starts = leverage_series.index[changes == 1].tolist()
    ends = leverage_series.index[changes == -1].tolist()

    # Handle edge cases
    if is_leveraged.iloc[0]:
        starts = [leverage_series.index[0]] + starts
    if is_leveraged.iloc[-1]:
        ends = ends + [leverage_series.index[-1]]

    # Add shaded regions - using a teal/cyan that's visible on both light and dark
    for start, end in zip(starts, ends):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="#4DD0E1", opacity=0.15,  # Cyan for dark mode visibility
            layer="below", line_width=0,
            row=row, col=1,
        )

    return fig


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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Summary", "📈 Equity Curve", "🎯 Signals & Indicators", "📉 Drawdown", "⚙️ Leverage", "💰 Interest"
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
        # Extract which indicators are used in the rule
        rule_indicators = extract_indicators_from_rule(leverage_rule)

        # Compute indicators for the signal asset
        indicators = compute_indicators(
            data.loc[result.equity_curve.index],
            symbols=[signal_asset],
            sma_windows=config.sma_windows,
            ema_windows=config.ema_windows,
            rsi_period=config.rsi_period,
        )

        # Get the leverage signal
        rules_engine = RulesEngine(indicators)
        leverage_signal = rules_engine.evaluate(leverage_rule)

        # Determine subplot layout based on whether RSI is used
        has_rsi = rule_indicators['rsi'] is not None

        if has_rsi:
            fig = make_subplots(
                rows=4, cols=1,
                row_heights=[0.35, 0.15, 0.35, 0.15],
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    f"{signal_asset} Price & Indicators",
                    f"RSI({rule_indicators['rsi']})",
                    "Portfolio Equity",
                    "Leverage"
                )
            )
            equity_row = 3
            leverage_row = 4
        else:
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.45, 0.40, 0.15],
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    f"{signal_asset} Price & Indicators",
                    "Portfolio Equity",
                    "Leverage"
                )
            )
            equity_row = 2
            leverage_row = 3

        # === Row 1: Signal Asset Price with Indicators ===
        # Colors optimized for dark mode
        price_col = f"{signal_asset}.close"
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators[price_col],
                name=f"{signal_asset} Price",
                line=dict(color="#E0E0E0", width=1.5),  # Light gray for dark mode
            ),
            row=1, col=1
        )

        # Add SMAs used in rule (highlighted) and others (dimmed)
        # Bright colors for dark mode visibility
        sma_colors = {20: "#FFA726", 50: "#42A5F5", 200: "#EF5350"}  # Orange, Blue, Red
        for sma_period in config.sma_windows:
            sma_col = f"{signal_asset}.SMA_{sma_period}"
            if sma_col in indicators.columns:
                is_used = sma_period in rule_indicators['smas']
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators[sma_col],
                        name=f"SMA({sma_period})",
                        line=dict(
                            color=sma_colors.get(sma_period, "#CE93D8"),  # Light purple fallback
                            width=2.5 if is_used else 1,
                            dash="solid" if is_used else "dot",
                        ),
                        opacity=1.0 if is_used else 0.5,
                    ),
                    row=1, col=1
                )

        # Add EMAs if used in rule
        ema_colors = {12: "#66BB6A", 26: "#26A69A"}  # Green, Teal
        for ema_period in config.ema_windows:
            ema_col = f"{signal_asset}.EMA_{ema_period}"
            if ema_col in indicators.columns:
                is_used = ema_period in rule_indicators['emas']
                if is_used:
                    fig.add_trace(
                        go.Scatter(
                            x=indicators.index,
                            y=indicators[ema_col],
                            name=f"EMA({ema_period})",
                            line=dict(
                                color=ema_colors.get(ema_period, "#80DEEA"),  # Cyan fallback
                                width=2.5,
                            ),
                        ),
                        row=1, col=1
                    )

        # Add crossover markers (price crosses SMA, SMA crosses SMA)
        fig, markers_added = add_crossover_markers(fig, indicators, signal_asset, rule_indicators, row=1)

        # Add leverage ON shading to price chart
        fig = add_leverage_regions(fig, result.leverage_series, row=1)

        # === Row 2: RSI (if used) ===
        if has_rsi:
            rsi_col = f"{signal_asset}.RSI_{rule_indicators['rsi']}"
            if rsi_col in indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators[rsi_col],
                        name=f"RSI({rule_indicators['rsi']})",
                        line=dict(color="#BB86FC", width=1.5),  # Material purple for dark mode
                        showlegend=True,
                    ),
                    row=2, col=1
                )
                # Add overbought/oversold lines with dark mode colors
                fig.add_hline(y=70, line_dash="dash", line_color="#EF5350", opacity=0.7, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#66BB6A", opacity=0.7, row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="#9E9E9E", opacity=0.5, row=2, col=1)

        # === Equity Row: Portfolio Equity ===
        fig.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                name="Strategy Equity",
                line=dict(color="#2196F3", width=2),  # Bright blue
            ),
            row=equity_row, col=1
        )

        # Benchmark
        benchmark_equity = spy_prices / spy_prices.iloc[0] * initial_equity
        fig.add_trace(
            go.Scatter(
                x=benchmark_equity.index,
                y=benchmark_equity.values,
                name=f"{signal_asset} B&H",
                line=dict(color="#9E9E9E", dash="dash", width=1.5),  # Medium gray
            ),
            row=equity_row, col=1
        )

        # Add leverage shading to equity curve
        fig = add_leverage_regions(fig, result.leverage_series, row=equity_row)

        # === Leverage Row ===
        fig.add_trace(
            go.Scatter(
                x=result.leverage_series.index,
                y=result.leverage_series.values,
                name="Leverage",
                line=dict(color="#FFB74D"),  # Amber/orange
                fill="tozeroy",
                fillcolor="rgba(255, 183, 77, 0.4)",
            ),
            row=leverage_row, col=1
        )

        # Update layout
        fig.update_layout(
            height=900 if has_rsi else 750,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified",
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if has_rsi:
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=equity_row, col=1)
        fig.update_yaxes(title_text="Leverage", row=leverage_row, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Legend explanation
        st.caption("💡 Green shaded areas indicate periods when leverage is ON. Solid indicator lines are used in the rule; dotted lines are available but not used.")

    with tab3:
        # Signals & Indicators tab
        st.markdown("### Signal Asset Price with Indicators")
        st.markdown(f"**Rule:** `{leverage_rule}`")

        # Extract which indicators are used in the rule
        rule_indicators = extract_indicators_from_rule(leverage_rule)

        # Compute indicators for the signal asset
        indicators = compute_indicators(
            data.loc[result.equity_curve.index],
            symbols=[signal_asset],
            sma_windows=config.sma_windows,
            ema_windows=config.ema_windows,
            rsi_period=config.rsi_period,
        )

        # Get the leverage signal
        rules_engine = RulesEngine(indicators)
        leverage_signal = rules_engine.evaluate(leverage_rule)

        # Determine subplot layout based on whether RSI is used
        has_rsi = rule_indicators['rsi'] is not None
        if has_rsi:
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.6, 0.25, 0.15],
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f"{signal_asset} Price & Moving Averages", "RSI", "Leverage Signal")
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25],
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(f"{signal_asset} Price & Moving Averages", "Leverage Signal")
            )

        # Price line - dark mode optimized
        price_col = f"{signal_asset}.close"
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators[price_col],
                name="Price",
                line=dict(color="#E0E0E0", width=1.5),  # Light gray
            ),
            row=1, col=1
        )

        # Add SMAs used in rule (highlight) and others (dimmed) - dark mode colors
        sma_colors = {20: "#FFA726", 50: "#42A5F5", 200: "#EF5350"}
        for sma_period in config.sma_windows:
            sma_col = f"{signal_asset}.SMA_{sma_period}"
            if sma_col in indicators.columns:
                is_used = sma_period in rule_indicators['smas']
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators[sma_col],
                        name=f"SMA({sma_period})",
                        line=dict(
                            color=sma_colors.get(sma_period, "#CE93D8"),
                            width=2.5 if is_used else 1,
                            dash="solid" if is_used else "dot",
                        ),
                        opacity=1.0 if is_used else 0.5,
                    ),
                    row=1, col=1
                )

        # Add EMAs if used - dark mode colors
        ema_colors = {12: "#66BB6A", 26: "#26A69A"}
        for ema_period in config.ema_windows:
            ema_col = f"{signal_asset}.EMA_{ema_period}"
            if ema_col in indicators.columns:
                is_used = ema_period in rule_indicators['emas']
                if is_used:  # Only show if used in rule
                    fig.add_trace(
                        go.Scatter(
                            x=indicators.index,
                            y=indicators[ema_col],
                            name=f"EMA({ema_period})",
                            line=dict(
                                color=ema_colors.get(ema_period, "#80DEEA"),
                                width=2.5,
                            ),
                        ),
                        row=1, col=1
                    )

        # Add crossover markers
        fig, _ = add_crossover_markers(fig, indicators, signal_asset, rule_indicators, row=1)

        # Add leverage ON shading to price chart
        fig = add_leverage_regions(fig, result.leverage_series, row=1)

        # RSI subplot if used - dark mode colors
        if has_rsi:
            rsi_col = f"{signal_asset}.RSI_{rule_indicators['rsi']}"
            if rsi_col in indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators[rsi_col],
                        name=f"RSI({rule_indicators['rsi']})",
                        line=dict(color="#BB86FC", width=1.5),  # Material purple
                    ),
                    row=2, col=1
                )
                # Add overbought/oversold lines - dark mode colors
                fig.add_hline(y=70, line_dash="dash", line_color="#EF5350", opacity=0.7, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#66BB6A", opacity=0.7, row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="#9E9E9E", opacity=0.5, row=2, col=1)

            # Leverage signal on row 3
            signal_row = 3
        else:
            signal_row = 2

        # Leverage signal (binary) - dark mode colors
        fig.add_trace(
            go.Scatter(
                x=leverage_signal.index,
                y=leverage_signal.astype(int),
                name="Leverage ON",
                fill="tozeroy",
                line=dict(color="#4DD0E1", width=1),  # Cyan
                fillcolor="rgba(77, 208, 225, 0.4)",
            ),
            row=signal_row, col=1
        )

        # Update layout
        fig.update_layout(
            height=700 if has_rsi else 550,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified",
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if has_rsi:
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
            fig.update_yaxes(title_text="Signal", range=[-0.1, 1.1], row=3, col=1)
        else:
            fig.update_yaxes(title_text="Signal", range=[-0.1, 1.1], row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Signal statistics
        st.markdown("### Signal Statistics")
        col1, col2, col3, col4 = st.columns(4)

        signal_on_pct = leverage_signal.mean() * 100
        signal_changes = leverage_signal.astype(int).diff().abs().sum()

        col1.metric("Signal ON", f"{signal_on_pct:.1f}%")
        col2.metric("Signal OFF", f"{100 - signal_on_pct:.1f}%")
        col3.metric("Signal Changes", f"{int(signal_changes)}")
        col4.metric("Avg Days per State", f"{len(leverage_signal) / max(signal_changes, 1):.0f}")

        # Show indicator values at key points
        st.markdown("### Current Indicator Values")
        latest_idx = indicators.index[-1]

        indicator_values = {}
        indicator_values["Price"] = f"${indicators.loc[latest_idx, f'{signal_asset}.close']:.2f}"

        for sma in config.sma_windows:
            col = f"{signal_asset}.SMA_{sma}"
            if col in indicators.columns:
                val = indicators.loc[latest_idx, col]
                indicator_values[f"SMA({sma})"] = f"${val:.2f}"

        if has_rsi:
            rsi_col = f"{signal_asset}.RSI_{rule_indicators['rsi']}"
            if rsi_col in indicators.columns:
                indicator_values[f"RSI({rule_indicators['rsi']})"] = f"{indicators.loc[latest_idx, rsi_col]:.1f}"

        indicator_values["Signal"] = "ON" if leverage_signal.iloc[-1] else "OFF"

        st.dataframe(
            pd.DataFrame([indicator_values]),
            hide_index=True,
            use_container_width=True
        )

    with tab4:
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

    with tab5:
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

    with tab6:
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
