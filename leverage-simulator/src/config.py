"""Configuration loading and validation."""

from pathlib import Path
from typing import Optional
import yaml

from .simulation import (
    BacktestConfig,
    LeverageConfig,
    PortfolioConfig,
    RebalanceConfig,
)


def load_config(config_path: str | Path) -> BacktestConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        BacktestConfig object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return parse_config(raw)


def parse_config(raw: dict) -> BacktestConfig:
    """
    Parse raw config dict into BacktestConfig.

    Args:
        raw: Raw config dictionary from YAML

    Returns:
        BacktestConfig object
    """
    # Parse leverage config
    lev_raw = raw.get("leverage", {})
    leverage_config = LeverageConfig(
        max_leverage=lev_raw.get("max_leverage", 1.5),
        broker_spread=lev_raw.get("broker_spread", 0.015),
        day_count=lev_raw.get("day_count", 360),
        maintenance_margin=lev_raw.get("maintenance_margin", 0.25),
        risk_off_to_cash=lev_raw.get("risk_off_to_cash", True),
    )

    # Parse portfolio config
    port_raw = raw.get("portfolio", {})
    portfolio_config = PortfolioConfig(
        mode=port_raw.get("mode", "fixed_weights"),
        weights=port_raw.get("weights", {"SPY": 0.6, "GLD": 0.2, "TLT": 0.2}),
        risky_assets=port_raw.get("risky_assets", ["SPY", "GLD", "TLT"]),
        cash_asset=port_raw.get("cash_asset", "BIL"),
        momentum_lookback=port_raw.get("momentum_lookback", 126),
        momentum_top_n=port_raw.get("momentum_top_n", 1),
    )

    # Parse rebalance config
    reb_raw = raw.get("rebalance", {})
    rebalance_config = RebalanceConfig(
        frequency=reb_raw.get("frequency", "monthly"),
        day_of_week=reb_raw.get("day_of_week", 0),
        day_of_month=reb_raw.get("day_of_month", 1),
    )

    # Build main config
    sim_raw = raw.get("simulation", {})

    return BacktestConfig(
        initial_equity=sim_raw.get("initial_equity", 100_000),
        start_date=raw.get("data", {}).get("start_date"),
        end_date=raw.get("data", {}).get("end_date"),
        leverage=leverage_config,
        portfolio=portfolio_config,
        rebalance=rebalance_config,
        signal_asset=lev_raw.get("signal_asset", "SPY"),
        leverage_rule=lev_raw.get("rule", "(SPY.close > SPY.SMA_200)"),
        target_leverage=lev_raw.get("target_leverage", 1.5),
        sma_windows=raw.get("indicators", {}).get("sma_windows", [20, 50, 200]),
        ema_windows=raw.get("indicators", {}).get("ema_windows", [12, 26]),
        rsi_period=raw.get("indicators", {}).get("rsi_period", 14),
        transaction_cost_bps=sim_raw.get("transaction_cost_bps", 0),
    )


def save_config(config: BacktestConfig, config_path: str | Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: BacktestConfig object
        config_path: Path to save YAML file
    """
    raw = {
        "data": {
            "assets": config.portfolio.risky_assets + [config.portfolio.cash_asset],
            "start_date": config.start_date,
            "end_date": config.end_date,
        },
        "portfolio": {
            "mode": config.portfolio.mode,
            "weights": config.portfolio.weights,
            "risky_assets": config.portfolio.risky_assets,
            "cash_asset": config.portfolio.cash_asset,
            "momentum_lookback": config.portfolio.momentum_lookback,
            "momentum_top_n": config.portfolio.momentum_top_n,
        },
        "rebalance": {
            "frequency": config.rebalance.frequency,
        },
        "leverage": {
            "signal_asset": config.signal_asset,
            "rule": config.leverage_rule,
            "target_leverage": config.target_leverage,
            "max_leverage": config.leverage.max_leverage,
            "broker_spread": config.leverage.broker_spread,
            "day_count": config.leverage.day_count,
            "maintenance_margin": config.leverage.maintenance_margin,
            "risk_off_to_cash": config.leverage.risk_off_to_cash,
        },
        "indicators": {
            "sma_windows": config.sma_windows,
            "ema_windows": config.ema_windows,
            "rsi_period": config.rsi_period,
        },
        "simulation": {
            "initial_equity": config.initial_equity,
            "transaction_cost_bps": config.transaction_cost_bps,
        },
    }

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
