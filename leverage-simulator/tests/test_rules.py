"""Tests for rules engine."""

import pytest
import pandas as pd
import numpy as np

from src.rules import RulesEngine, parse_rule, Lexer, Parser


@pytest.fixture
def sample_indicators():
    """Sample indicator DataFrame for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    return pd.DataFrame({
        "SPY.close": np.linspace(100, 150, 100),  # Trending up
        "SPY.SMA_200": np.linspace(90, 140, 100),  # Below close
        "SPY.SMA_50": np.linspace(95, 145, 100),   # Between close and SMA_200
        "SPY.RSI_14": np.linspace(30, 70, 100),    # Low to high
        "SPY.SMA_200_slope": np.linspace(-2, 3, 100),  # Negative to positive
    }, index=dates)


class TestLexer:
    """Tests for Lexer."""

    def test_simple_comparison(self):
        """Test tokenizing a simple comparison."""
        lexer = Lexer("SPY.close > 100")
        tokens = lexer.tokenize()

        assert tokens[0].value == "SPY.close"
        assert tokens[1].value == ">"
        assert tokens[2].value == "100"

    def test_complex_expression(self):
        """Test tokenizing a complex expression."""
        lexer = Lexer("(SPY.close > SPY.SMA_200) AND (SPY.RSI_14 < 70)")
        tokens = lexer.tokenize()

        expected_values = [
            "(", "SPY.close", ">", "SPY.SMA_200", ")",
            "AND",
            "(", "SPY.RSI_14", "<", "70", ")"
        ]

        for i, expected in enumerate(expected_values):
            assert tokens[i].value == expected, f"Token {i}: expected {expected}, got {tokens[i].value}"

    def test_all_comparators(self):
        """Test all comparison operators."""
        for op in [">", "<", ">=", "<=", "==", "!="]:
            lexer = Lexer(f"a {op} b")
            tokens = lexer.tokenize()
            assert tokens[1].value == op

    def test_not_operator(self):
        """Test NOT operator."""
        lexer = Lexer("NOT (a > b)")
        tokens = lexer.tokenize()
        assert tokens[0].value == "NOT"

    def test_negative_number(self):
        """Test negative numbers."""
        lexer = Lexer("a > -5")
        tokens = lexer.tokenize()
        assert tokens[2].value == "-5"

    def test_decimal_number(self):
        """Test decimal numbers."""
        lexer = Lexer("a > 3.14")
        tokens = lexer.tokenize()
        assert tokens[2].value == "3.14"


class TestParser:
    """Tests for Parser."""

    def test_simple_comparison(self):
        """Test parsing a simple comparison."""
        ast = parse_rule("SPY.close > 100")
        assert ast.operator == ">"
        assert ast.left.name == "SPY.close"
        assert ast.right.value == 100.0

    def test_and_expression(self):
        """Test parsing AND expression."""
        ast = parse_rule("(a > b) AND (c < d)")
        assert ast.operator == "AND"
        assert ast.left.operator == ">"
        assert ast.right.operator == "<"

    def test_or_expression(self):
        """Test parsing OR expression."""
        ast = parse_rule("(a > b) OR (c < d)")
        assert ast.operator == "OR"

    def test_not_expression(self):
        """Test parsing NOT expression."""
        ast = parse_rule("NOT (a > b)")
        assert hasattr(ast, "operand")  # NotNode
        assert ast.operand.operator == ">"

    def test_nested_expression(self):
        """Test parsing nested parentheses."""
        ast = parse_rule("((a > b) AND (c < d)) OR (e == f)")
        assert ast.operator == "OR"
        assert ast.left.operator == "AND"

    def test_chained_and(self):
        """Test chained AND expressions."""
        ast = parse_rule("(a > b) AND (c < d) AND (e == f)")
        # Should be left-associative: ((a>b) AND (c<d)) AND (e==f)
        assert ast.operator == "AND"
        assert ast.left.operator == "AND"


class TestRulesEngine:
    """Tests for RulesEngine."""

    def test_simple_greater_than(self, sample_indicators):
        """Test simple > comparison."""
        engine = RulesEngine(sample_indicators)
        result = engine.evaluate("SPY.close > 125")

        # close goes from 100 to 150, > 125 should be True for ~half
        assert result.dtype == bool
        assert result.sum() > 0
        assert result.sum() < len(result)

    def test_simple_less_than(self, sample_indicators):
        """Test simple < comparison."""
        engine = RulesEngine(sample_indicators)
        result = engine.evaluate("SPY.RSI_14 < 50")

        # RSI goes from 30 to 70, < 50 should be True for first half
        assert result.iloc[0] == True  # RSI starts at 30
        assert result.iloc[-1] == False  # RSI ends at 70

    def test_indicator_vs_indicator(self, sample_indicators):
        """Test comparing two indicators."""
        engine = RulesEngine(sample_indicators)
        result = engine.evaluate("SPY.close > SPY.SMA_200")

        # close is always above SMA_200 in our sample data
        assert result.all()

    def test_and_logic(self, sample_indicators):
        """Test AND logic."""
        engine = RulesEngine(sample_indicators)

        # Both conditions
        result = engine.evaluate("(SPY.close > SPY.SMA_200) AND (SPY.RSI_14 < 50)")

        # AND should be True only when both conditions are True
        # close > SMA_200 always True, RSI < 50 True for first half
        assert result.iloc[0] == True
        assert result.iloc[-1] == False

    def test_or_logic(self, sample_indicators):
        """Test OR logic."""
        engine = RulesEngine(sample_indicators)

        result = engine.evaluate("(SPY.RSI_14 < 40) OR (SPY.RSI_14 > 60)")

        # Should be True at start (RSI low) and end (RSI high)
        assert result.iloc[0] == True  # RSI = 30
        assert result.iloc[-1] == True  # RSI = 70
        # Middle should be False
        mid_idx = len(result) // 2
        assert result.iloc[mid_idx] == False  # RSI ~ 50

    def test_not_logic(self, sample_indicators):
        """Test NOT logic."""
        engine = RulesEngine(sample_indicators)

        result = engine.evaluate("NOT (SPY.RSI_14 > 50)")

        # NOT (RSI > 50) should be True when RSI <= 50
        assert result.iloc[0] == True  # RSI = 30
        assert result.iloc[-1] == False  # RSI = 70

    def test_complex_rule(self, sample_indicators):
        """Test complex rule with multiple conditions."""
        engine = RulesEngine(sample_indicators)

        rule = "(SPY.close > SPY.SMA_200) AND (SPY.RSI_14 < 70) AND NOT (SPY.SMA_200_slope < 0)"

        result = engine.evaluate(rule)

        # Should be True when:
        # - close > SMA_200 (always True in our data)
        # - RSI < 70 (True for most except very end)
        # - slope >= 0 (True for second half)
        # So True roughly in second half, excluding very end

        assert result.sum() > 0
        assert result.sum() < len(result)

    def test_evaluate_with_debug(self, sample_indicators):
        """Test debug output."""
        engine = RulesEngine(sample_indicators)
        signal, debug = engine.evaluate_with_debug("SPY.close > 125")

        assert "rule" in debug
        assert "true_count" in debug
        assert "false_count" in debug
        assert "true_pct" in debug
        assert debug["true_count"] + debug["false_count"] == len(sample_indicators)

    def test_available_indicators(self, sample_indicators):
        """Test listing available indicators."""
        engine = RulesEngine(sample_indicators)
        available = engine.available_indicators

        assert "SPY.close" in available
        assert "SPY.SMA_200" in available
        assert "SPY.RSI_14" in available

    def test_unknown_indicator_error(self, sample_indicators):
        """Test error on unknown indicator."""
        engine = RulesEngine(sample_indicators)

        with pytest.raises(ValueError, match="Unknown indicator"):
            engine.evaluate("UNKNOWN.close > 100")


class TestEdgeCases:
    """Test edge cases."""

    def test_lowercase_operators(self, sample_indicators):
        """Test lowercase and/or/not."""
        engine = RulesEngine(sample_indicators)

        result1 = engine.evaluate("(SPY.close > 100) and (SPY.close < 200)")
        result2 = engine.evaluate("(SPY.close > 100) AND (SPY.close < 200)")

        pd.testing.assert_series_equal(result1, result2)

    def test_equality_comparison(self, sample_indicators):
        """Test == and != comparisons."""
        engine = RulesEngine(sample_indicators)

        # These won't match exactly due to float comparison,
        # but should not raise errors
        result_eq = engine.evaluate("SPY.close == 100")
        result_neq = engine.evaluate("SPY.close != 100")

        assert result_eq.dtype == bool
        assert result_neq.dtype == bool

    def test_zero_comparison(self, sample_indicators):
        """Test comparison with zero."""
        engine = RulesEngine(sample_indicators)
        result = engine.evaluate("SPY.SMA_200_slope > 0")

        # Slope goes from -2 to 3, so should be True for second half
        assert result.iloc[0] == False
        assert result.iloc[-1] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
