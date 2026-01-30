"""
Rules engine for evaluating leverage conditions.

Supports expressions like:
    (SPY.close > SPY.SMA_200) AND (SPY.RSI_14 < 70) AND NOT (SPY.SMA_50 < SPY.SMA_200)

Grammar:
    expression := term ((AND | OR) term)*
    term       := NOT? factor
    factor     := comparison | '(' expression ')'
    comparison := operand comparator operand
    operand    := identifier | number
    identifier := SYMBOL.INDICATOR (e.g., SPY.close, SPY.SMA_200)
    comparator := '>' | '<' | '>=' | '<=' | '==' | '!='
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union
import re
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """Token types for rule parsing."""
    IDENTIFIER = auto()  # SPY.close, SPY.SMA_200
    NUMBER = auto()       # 70, 0.5
    COMPARATOR = auto()   # >, <, >=, <=, ==, !=
    AND = auto()
    OR = auto()
    NOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass
class Token:
    """A token in the rule expression."""
    type: TokenType
    value: str


class Lexer:
    """Tokenizer for rule expressions."""

    KEYWORDS = {
        "AND": TokenType.AND,
        "OR": TokenType.OR,
        "NOT": TokenType.NOT,
        "and": TokenType.AND,
        "or": TokenType.OR,
        "not": TokenType.NOT,
    }

    COMPARATORS = [">=", "<=", "==", "!=", ">", "<"]

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if text else None

    def advance(self):
        """Move to next character."""
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def skip_whitespace(self):
        """Skip whitespace characters."""
        while self.current_char and self.current_char.isspace():
            self.advance()

    def read_identifier(self) -> str:
        """Read an identifier (SYMBOL.INDICATOR or keyword)."""
        result = ""
        while self.current_char and (
            self.current_char.isalnum() or self.current_char in "._"
        ):
            result += self.current_char
            self.advance()
        return result

    def read_number(self) -> str:
        """Read a number (int or float)."""
        result = ""
        has_dot = False
        while self.current_char and (
            self.current_char.isdigit() or
            (self.current_char == "." and not has_dot) or
            (self.current_char == "-" and not result)
        ):
            if self.current_char == ".":
                has_dot = True
            result += self.current_char
            self.advance()
        return result

    def read_comparator(self) -> Optional[str]:
        """Try to read a comparator."""
        for comp in self.COMPARATORS:
            if self.text[self.pos:].startswith(comp):
                for _ in comp:
                    self.advance()
                return comp
        return None

    def get_next_token(self) -> Token:
        """Get the next token from input."""
        while self.current_char:
            # Skip whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            # Parentheses
            if self.current_char == "(":
                self.advance()
                return Token(TokenType.LPAREN, "(")

            if self.current_char == ")":
                self.advance()
                return Token(TokenType.RPAREN, ")")

            # Try comparator first (before identifiers that might start with >)
            comp = self.read_comparator()
            if comp:
                return Token(TokenType.COMPARATOR, comp)

            # Number (including negative)
            if self.current_char.isdigit() or (
                self.current_char == "-" and
                self.pos + 1 < len(self.text) and
                self.text[self.pos + 1].isdigit()
            ):
                return Token(TokenType.NUMBER, self.read_number())

            # Identifier or keyword
            if self.current_char.isalpha() or self.current_char == "_":
                value = self.read_identifier()
                if value in self.KEYWORDS:
                    return Token(self.KEYWORDS[value], value)
                return Token(TokenType.IDENTIFIER, value)

            raise ValueError(f"Unexpected character: {self.current_char}")

        return Token(TokenType.EOF, "")

    def tokenize(self) -> list[Token]:
        """Tokenize the entire input."""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens


# AST Node types
@dataclass
class NumberNode:
    value: float


@dataclass
class IdentifierNode:
    name: str


@dataclass
class ComparisonNode:
    left: Union["NumberNode", "IdentifierNode"]
    operator: str
    right: Union["NumberNode", "IdentifierNode"]


@dataclass
class NotNode:
    operand: "ASTNode"


@dataclass
class BinaryOpNode:
    left: "ASTNode"
    operator: str  # AND or OR
    right: "ASTNode"


ASTNode = Union[NumberNode, IdentifierNode, ComparisonNode, NotNode, BinaryOpNode]


class Parser:
    """
    Recursive descent parser for rule expressions.

    Grammar:
        expression := term ((AND | OR) term)*
        term       := NOT? factor
        factor     := comparison | '(' expression ')'
        comparison := operand comparator operand
        operand    := identifier | number
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0] if tokens else Token(TokenType.EOF, "")

    def advance(self):
        """Move to next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]
        else:
            self.current = Token(TokenType.EOF, "")

    def expect(self, token_type: TokenType):
        """Expect current token to be of given type."""
        if self.current.type != token_type:
            raise ValueError(
                f"Expected {token_type}, got {self.current.type} ({self.current.value})"
            )
        self.advance()

    def parse(self) -> ASTNode:
        """Parse the entire expression."""
        result = self.expression()
        if self.current.type != TokenType.EOF:
            raise ValueError(f"Unexpected token: {self.current.value}")
        return result

    def expression(self) -> ASTNode:
        """expression := term ((AND | OR) term)*"""
        left = self.term()

        while self.current.type in (TokenType.AND, TokenType.OR):
            op = self.current.value.upper()
            self.advance()
            right = self.term()
            left = BinaryOpNode(left, op, right)

        return left

    def term(self) -> ASTNode:
        """term := NOT? factor"""
        if self.current.type == TokenType.NOT:
            self.advance()
            operand = self.factor()
            return NotNode(operand)

        return self.factor()

    def factor(self) -> ASTNode:
        """factor := comparison | '(' expression ')'"""
        if self.current.type == TokenType.LPAREN:
            self.advance()
            result = self.expression()
            self.expect(TokenType.RPAREN)
            return result

        return self.comparison()

    def comparison(self) -> ASTNode:
        """comparison := operand comparator operand"""
        left = self.operand()

        if self.current.type != TokenType.COMPARATOR:
            raise ValueError(
                f"Expected comparator, got {self.current.type} ({self.current.value})"
            )

        op = self.current.value
        self.advance()

        right = self.operand()

        return ComparisonNode(left, op, right)

    def operand(self) -> Union[NumberNode, IdentifierNode]:
        """operand := identifier | number"""
        if self.current.type == TokenType.IDENTIFIER:
            name = self.current.value
            self.advance()
            return IdentifierNode(name)

        if self.current.type == TokenType.NUMBER:
            value = float(self.current.value)
            self.advance()
            return NumberNode(value)

        raise ValueError(
            f"Expected identifier or number, got {self.current.type} ({self.current.value})"
        )


class RuleEvaluator:
    """Evaluates parsed rule AST against indicator data."""

    def __init__(self, indicators: pd.DataFrame):
        """
        Initialize evaluator with indicator data.

        Args:
            indicators: DataFrame with columns like 'SPY.close', 'SPY.SMA_200', etc.
        """
        self.indicators = indicators

    def evaluate(self, ast: ASTNode) -> pd.Series:
        """
        Evaluate AST to produce a boolean Series.

        Args:
            ast: Parsed AST node

        Returns:
            Boolean Series (True = leverage ON, False = leverage OFF)
        """
        if isinstance(ast, NumberNode):
            return pd.Series(ast.value, index=self.indicators.index)

        if isinstance(ast, IdentifierNode):
            if ast.name not in self.indicators.columns:
                raise ValueError(f"Unknown indicator: {ast.name}")
            return self.indicators[ast.name]

        if isinstance(ast, ComparisonNode):
            left = self.evaluate(ast.left)
            right = self.evaluate(ast.right)

            if ast.operator == ">":
                return left > right
            elif ast.operator == "<":
                return left < right
            elif ast.operator == ">=":
                return left >= right
            elif ast.operator == "<=":
                return left <= right
            elif ast.operator == "==":
                return left == right
            elif ast.operator == "!=":
                return left != right
            else:
                raise ValueError(f"Unknown operator: {ast.operator}")

        if isinstance(ast, NotNode):
            operand = self.evaluate(ast.operand)
            return ~operand

        if isinstance(ast, BinaryOpNode):
            left = self.evaluate(ast.left)
            right = self.evaluate(ast.right)

            if ast.operator == "AND":
                return left & right
            elif ast.operator == "OR":
                return left | right
            else:
                raise ValueError(f"Unknown binary operator: {ast.operator}")

        raise ValueError(f"Unknown AST node type: {type(ast)}")


class RulesEngine:
    """
    Main interface for rule evaluation.

    Usage:
        engine = RulesEngine(indicators_df)
        signal = engine.evaluate("(SPY.close > SPY.SMA_200) AND (SPY.RSI_14 < 70)")
    """

    def __init__(self, indicators: pd.DataFrame):
        """
        Initialize rules engine.

        Args:
            indicators: DataFrame from compute_indicators()
        """
        self.indicators = indicators
        self.evaluator = RuleEvaluator(indicators)

    def parse(self, rule: str) -> ASTNode:
        """Parse a rule string into an AST."""
        lexer = Lexer(rule)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.parse()

    def evaluate(self, rule: str) -> pd.Series:
        """
        Evaluate a rule string against indicator data.

        Args:
            rule: Rule expression string

        Returns:
            Boolean Series (True = condition met, False = not met)
        """
        ast = self.parse(rule)
        return self.evaluator.evaluate(ast)

    def evaluate_with_debug(self, rule: str) -> tuple[pd.Series, dict]:
        """
        Evaluate rule with debug info.

        Returns:
            Tuple of (signal_series, debug_info_dict)
        """
        ast = self.parse(rule)
        signal = self.evaluator.evaluate(ast)

        debug_info = {
            "rule": rule,
            "ast": str(ast),
            "true_count": signal.sum(),
            "false_count": (~signal).sum(),
            "true_pct": signal.mean() * 100,
            "first_true": signal.idxmax() if signal.any() else None,
            "last_true": signal[::-1].idxmax() if signal.any() else None,
        }

        return signal, debug_info

    @property
    def available_indicators(self) -> list[str]:
        """List available indicator names for rules."""
        return list(self.indicators.columns)


def parse_rule(rule: str) -> ASTNode:
    """Convenience function to parse a rule string."""
    lexer = Lexer(rule)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


def evaluate_rule(rule: str, indicators: pd.DataFrame) -> pd.Series:
    """Convenience function to evaluate a rule."""
    engine = RulesEngine(indicators)
    return engine.evaluate(rule)
