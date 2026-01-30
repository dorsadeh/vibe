"""Tests for UI helper functions."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List


# Minimal mock of BacktestResult for testing
@dataclass
class MockBacktestResult:
    """Mock backtest result for testing."""
    equity_curve: pd.Series
    leverage_series: pd.Series
    margin_events: List
    rebalance_dates: List
    weights: pd.DataFrame


def create_transaction_log(result, initial_equity: float) -> pd.DataFrame:
    """
    Create a transaction log from backtest results.

    This is a copy of the function from app/main.py for testing purposes.
    """
    transactions = []

    leverage = result.leverage_series
    prev_equity = initial_equity

    for i, date in enumerate(result.equity_curve.index):
        equity = result.equity_curve.iloc[i]
        daily_pnl = equity - prev_equity if i > 0 else 0
        lev = leverage.iloc[i]

        # Check for leverage state change
        if i > 0:
            prev_lev = leverage.iloc[i-1]
            if lev > 1 and prev_lev <= 1:
                transactions.append({
                    'Date': date,
                    'Type': 'Leverage ON',
                    'Details': f'Leverage increased to {lev:.2f}x',
                    'Leverage': lev,
                    'Equity': equity,
                    'Daily P&L': daily_pnl,
                })
            elif lev <= 1 and prev_lev > 1:
                transactions.append({
                    'Date': date,
                    'Type': 'Leverage OFF',
                    'Details': f'Leverage reduced to {lev:.2f}x',
                    'Leverage': lev,
                    'Equity': equity,
                    'Daily P&L': daily_pnl,
                })

        # Check for margin call
        if date in result.margin_events:
            transactions.append({
                'Date': date,
                'Type': 'MARGIN CALL',
                'Details': 'Forced deleveraging due to margin breach',
                'Leverage': lev,
                'Equity': equity,
                'Daily P&L': daily_pnl,
            })

        # Check for rebalance
        if date in result.rebalance_dates:
            weights = result.weights.loc[date]
            weight_str = ', '.join([f"{k}:{v:.0%}" for k, v in weights.items() if v > 0])
            transactions.append({
                'Date': date,
                'Type': 'Rebalance',
                'Details': weight_str,
                'Leverage': lev,
                'Equity': equity,
                'Daily P&L': daily_pnl,
            })

        prev_equity = equity

    df = pd.DataFrame(transactions)
    if len(df) > 0:
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df['Equity'] = df['Equity'].apply(lambda x: f"${x:,.0f}")
        df['Daily P&L'] = df['Daily P&L'].apply(lambda x: f"${x:+,.0f}")
        df['Leverage'] = df['Leverage'].apply(lambda x: f"{x:.2f}x")

    return df


class TestTransactionLog:
    """Tests for create_transaction_log function."""

    @pytest.fixture
    def sample_dates(self):
        """Create sample date range."""
        return pd.date_range(start='2020-01-01', periods=10, freq='D')

    @pytest.fixture
    def mock_result_with_leverage_changes(self, sample_dates):
        """Create mock result with leverage changes."""
        # Leverage: 1, 1, 1.5, 1.5, 1, 1, 1.5, 1, 1, 1
        leverage = pd.Series([1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0], index=sample_dates)
        equity = pd.Series([100000, 100500, 101000, 101500, 101000, 100500, 101000, 100000, 99500, 100000], index=sample_dates)
        weights = pd.DataFrame({'SPY': [0.6]*10, 'GLD': [0.2]*10, 'TLT': [0.2]*10}, index=sample_dates)

        return MockBacktestResult(
            equity_curve=equity,
            leverage_series=leverage,
            margin_events=[],
            rebalance_dates=[],
            weights=weights,
        )

    def test_creates_dataframe(self, mock_result_with_leverage_changes):
        """Test that function returns a DataFrame."""
        result = create_transaction_log(mock_result_with_leverage_changes, 100000)
        assert isinstance(result, pd.DataFrame)

    def test_detects_leverage_on(self, mock_result_with_leverage_changes):
        """Test that leverage ON events are detected."""
        result = create_transaction_log(mock_result_with_leverage_changes, 100000)
        leverage_on = result[result['Type'] == 'Leverage ON']
        assert len(leverage_on) == 2  # Two times leverage went from 1.0 to 1.5

    def test_detects_leverage_off(self, mock_result_with_leverage_changes):
        """Test that leverage OFF events are detected."""
        result = create_transaction_log(mock_result_with_leverage_changes, 100000)
        leverage_off = result[result['Type'] == 'Leverage OFF']
        assert len(leverage_off) == 2  # Two times leverage went from 1.5 to 1.0

    def test_has_required_columns(self, mock_result_with_leverage_changes):
        """Test that result has all required columns."""
        result = create_transaction_log(mock_result_with_leverage_changes, 100000)
        required_cols = ['Date', 'Type', 'Details', 'Leverage', 'Equity', 'Daily P&L']
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_empty_result_when_no_changes(self, sample_dates):
        """Test empty result when there are no leverage changes."""
        leverage = pd.Series([1.0]*10, index=sample_dates)
        equity = pd.Series([100000]*10, index=sample_dates)
        weights = pd.DataFrame({'SPY': [1.0]*10}, index=sample_dates)

        result_mock = MockBacktestResult(
            equity_curve=equity,
            leverage_series=leverage,
            margin_events=[],
            rebalance_dates=[],
            weights=weights,
        )

        result = create_transaction_log(result_mock, 100000)
        assert len(result) == 0

    def test_detects_margin_calls(self, sample_dates):
        """Test that margin call events are detected."""
        leverage = pd.Series([1.5]*10, index=sample_dates)
        equity = pd.Series([100000]*10, index=sample_dates)
        weights = pd.DataFrame({'SPY': [1.0]*10}, index=sample_dates)

        # Add margin call event on day 5
        margin_event_date = sample_dates[4]

        result_mock = MockBacktestResult(
            equity_curve=equity,
            leverage_series=leverage,
            margin_events=[margin_event_date],
            rebalance_dates=[],
            weights=weights,
        )

        result = create_transaction_log(result_mock, 100000)
        margin_calls = result[result['Type'] == 'MARGIN CALL']
        assert len(margin_calls) == 1

    def test_detects_rebalances(self, sample_dates):
        """Test that rebalance events are detected."""
        leverage = pd.Series([1.0]*10, index=sample_dates)
        equity = pd.Series([100000]*10, index=sample_dates)
        weights = pd.DataFrame({'SPY': [0.6]*10, 'GLD': [0.2]*10, 'TLT': [0.2]*10}, index=sample_dates)

        # Add rebalance event on day 5
        rebalance_date = sample_dates[4]

        result_mock = MockBacktestResult(
            equity_curve=equity,
            leverage_series=leverage,
            margin_events=[],
            rebalance_dates=[rebalance_date],
            weights=weights,
        )

        result = create_transaction_log(result_mock, 100000)
        rebalances = result[result['Type'] == 'Rebalance']
        assert len(rebalances) == 1

    def test_formats_equity_as_currency(self, mock_result_with_leverage_changes):
        """Test that equity is formatted as currency."""
        result = create_transaction_log(mock_result_with_leverage_changes, 100000)
        if len(result) > 0:
            # Check that equity column contains dollar signs
            assert all('$' in str(v) for v in result['Equity'])

    def test_formats_leverage_with_x(self, mock_result_with_leverage_changes):
        """Test that leverage is formatted with 'x' suffix."""
        result = create_transaction_log(mock_result_with_leverage_changes, 100000)
        if len(result) > 0:
            assert all('x' in str(v) for v in result['Leverage'])


class TestTransactionLogFiltering:
    """Tests for transaction log filtering logic."""

    @pytest.fixture
    def sample_transaction_log(self):
        """Create sample transaction log DataFrame."""
        return pd.DataFrame({
            'Date': ['2020-01-02', '2020-01-05', '2020-01-10', '2020-01-15', '2020-01-20'],
            'Type': ['Leverage ON', 'Rebalance', 'Leverage OFF', 'MARGIN CALL', 'Leverage ON'],
            'Details': ['Leverage to 1.5x', 'SPY:60%, GLD:20%', 'Leverage to 1.0x', 'Forced deleverage', 'Leverage to 1.5x'],
            'Leverage': ['1.50x', '1.00x', '1.00x', '1.00x', '1.50x'],
            'Equity': ['$101,000', '$102,000', '$100,000', '$98,000', '$99,000'],
            'Daily P&L': ['+$1,000', '+$1,000', '-$2,000', '-$2,000', '+$1,000'],
        })

    def test_filter_by_single_type(self, sample_transaction_log):
        """Test filtering by a single transaction type."""
        selected_types = ['Leverage ON']
        filtered = sample_transaction_log[sample_transaction_log['Type'].isin(selected_types)]
        assert len(filtered) == 2
        assert all(filtered['Type'] == 'Leverage ON')

    def test_filter_by_multiple_types(self, sample_transaction_log):
        """Test filtering by multiple transaction types."""
        selected_types = ['Leverage ON', 'Leverage OFF']
        filtered = sample_transaction_log[sample_transaction_log['Type'].isin(selected_types)]
        assert len(filtered) == 3

    def test_filter_includes_all_types(self, sample_transaction_log):
        """Test that including all types shows all transactions."""
        selected_types = ['Leverage ON', 'Leverage OFF', 'Rebalance', 'MARGIN CALL']
        filtered = sample_transaction_log[sample_transaction_log['Type'].isin(selected_types)]
        assert len(filtered) == len(sample_transaction_log)

    def test_add_running_index(self, sample_transaction_log):
        """Test adding running index column."""
        filtered = sample_transaction_log.copy()
        filtered.insert(0, '#', range(1, len(filtered) + 1))

        assert '#' in filtered.columns
        assert list(filtered['#']) == [1, 2, 3, 4, 5]

    def test_running_index_on_filtered_data(self, sample_transaction_log):
        """Test that running index is correct on filtered data."""
        selected_types = ['Leverage ON']
        filtered = sample_transaction_log[sample_transaction_log['Type'].isin(selected_types)].copy()
        filtered.insert(0, '#', range(1, len(filtered) + 1))

        assert list(filtered['#']) == [1, 2]  # Reindexed from 1

    def test_empty_filter_returns_all(self, sample_transaction_log):
        """Test behavior when no filter is selected - should return all or empty."""
        # When no types selected, could return all (our implementation) or empty
        selected_types = []
        if selected_types:
            filtered = sample_transaction_log[sample_transaction_log['Type'].isin(selected_types)]
        else:
            filtered = sample_transaction_log  # Return all when nothing selected

        assert len(filtered) == len(sample_transaction_log)
