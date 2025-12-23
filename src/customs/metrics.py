import numpy as np
import pandas as pd
import pandas.api.types

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2
MAX_VOLATILITY_RATIO = 1.2
TRADING_DAYS_PER_YEAR = 252


class ParticipantVisibleError(Exception):
    pass


def score_fn(
        solution: pd.DataFrame,
        submission: pd.DataFrame
) -> float:
    """
    Calculates a custom evaluation metric(volatility-adjusted Sharpe ratio).

    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.

    :returns:
        float: The calculated adjusted Sharpe ratio.
    """

    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError("Predictions must be numeric")

    solution = solution
    solution['position'] = submission['prediction']

    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution['position'].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution['position'].min()} exceeds minimum of {MIN_INVESTMENT}')

    solution['strategy_returns'] = (
        solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']
    )

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_returns = strategy_excess_cumulative ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    if strategy_std == 0:
        raise ParticipantVisibleError("Division by zero, strategy std is zero")
    sharpe = strategy_mean_excess_returns / strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    strategy_volatility = float(strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    # Calculate market return and volatility
    market_excess_return = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_return).prod()
    market_mean_excess_return = market_excess_cumulative ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    if market_volatility == 0:
        raise ParticipantVisibleError("Division by zero, market std is zero")

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - MAX_VOLATILITY_RATIO) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_returns) * 100 * TRADING_DAYS_PER_YEAR,
    )
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)


def score_fn_v2(
    predictions: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
) -> dict:
    # Validate inputs
    predictions = np.asarray(predictions)
    forward_returns = np.asarray(forward_returns)
    risk_free_rate = np.asarray(risk_free_rate)

    if len(predictions) != len(forward_returns) or len(predictions) != len(risk_free_rate):
        raise ValueError("All input arrays must have the same length")

    # Check position constraints
    if predictions.max() > MAX_INVESTMENT:
        raise ValueError(f"Position of {predictions.max()} exceeds maximum of {MAX_INVESTMENT}")
    if predictions.min() < MIN_INVESTMENT:
        raise ValueError(f"Position of {predictions.min()} below minimum of {MIN_INVESTMENT}")

    # Calculate strategy returns
    strategy_returns = risk_free_rate * (1 - predictions) + predictions * forward_returns

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = strategy_returns - risk_free_rate
    strategy_excess_cumulative = np.prod(1 + strategy_excess_returns)
    strategy_mean_excess_return = strategy_excess_cumulative ** (1 / len(predictions)) - 1
    strategy_std = np.std(strategy_returns, ddof=1)

    if strategy_std == 0:
        raise ValueError("Division by zero: strategy standard deviation is zero")

    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    strategy_volatility = float(strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    # Calculate market return and volatility
    market_excess_returns = forward_returns - risk_free_rate
    market_excess_cumulative = np.prod(1 + market_excess_returns)
    market_mean_excess_return = market_excess_cumulative ** (1 / len(predictions)) - 1
    market_std = np.std(forward_returns, ddof=1)

    market_volatility = float(market_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    if market_volatility == 0:
        raise ValueError("Division by zero: market standard deviation is zero")

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - MAX_VOLATILITY_RATIO)
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * TRADING_DAYS_PER_YEAR)
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    adjusted_sharpe = min(float(adjusted_sharpe), 1_000_000)

    return {
        "adjusted_sharpe": adjusted_sharpe,
        "sharpe": float(sharpe),
        "strategy_volatility": strategy_volatility,
        "market_volatility": market_volatility,
        "strategy_mean_excess_return": float(strategy_mean_excess_return * TRADING_DAYS_PER_YEAR * 100),
        "market_mean_excess_return": float(market_mean_excess_return * TRADING_DAYS_PER_YEAR * 100),
        "vol_penalty": float(vol_penalty),
        "return_penalty": float(return_penalty),
    }
