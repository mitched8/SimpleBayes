import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import mplfinance as mpf
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import logging
import scipy.stats
from utils.market_utils import (
    fetch_market_data,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_moving_averages,
    get_indicator_states,
    calculate_forward_returns
)
from utils.pattern_recognition import (
    detect_doji,
    detect_hammer,
    detect_engulfing,
    detect_morning_star,
    detect_evening_star,
    detect_pin_bar
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_fetching():
    """Test data fetching from local CSV files"""
    st.write("Testing data fetching from local CSV files...")
    
    # Use a date range we know exists in the data
    end_date = datetime(2010, 1, 1).date()
    start_date = end_date - pd.Timedelta(days=30)
    
    symbols = ["EURUSD"]  # We know this file exists
    results = {}
    
    for symbol in symbols:
        st.write(f"\nTesting with {symbol} from {start_date} to {end_date}")
        
        # Test our fetch_market_data function
        st.write("Testing fetch_market_data function...")
        data = fetch_market_data(symbol, "1d", start_date, end_date)
        
        if data is None or data.empty:
            st.warning(f"fetch_market_data returned empty data for {symbol}")
        else:
            st.success(f"fetch_market_data successful for {symbol}")
            st.write(f"Data shape: {data.shape}")
            st.write(f"Date range: {data.index.min()} to {data.index.max()}")
            
            # Display sample of the data
            st.write("Sample of the data:")
            st.dataframe(data.head())
            
            # Show basic statistics
            st.write("Basic statistics:")
            st.dataframe(data.describe())
        
        results[symbol] = {
            "success": not (data is None or data.empty),
            "rows": data.shape[0] if data is not None and not data.empty else 0
        }
    
    return results

def calculate_market_priors(data: pd.DataFrame) -> dict:
    """
    Calculate market priors from historical data using proper distributions.
    Returns dict with distribution parameters.
    """
    priors = {}
    
    # Calculate return statistics
    returns = data['Returns'].dropna()
    
    # Fit normal distribution to historical returns
    mu, std = scipy.stats.norm.fit(returns)
    priors['returns_dist'] = scipy.stats.norm(mu, std)
    priors['returns_mean'] = mu
    priors['returns_std'] = std
    
    # Calculate volatility (using rolling std of returns)
    volatility = returns.rolling(window=20).std().dropna()
    
    # Fit gamma distribution to volatility (always positive)
    alpha, loc, beta = scipy.stats.gamma.fit(volatility)
    priors['volatility_dist'] = scipy.stats.gamma(alpha, loc, beta)
    priors['volatility_mean'] = volatility.mean()
    priors['volatility_std'] = volatility.std()
    
    # Store raw data for plotting
    priors['returns_data'] = returns
    priors['volatility_data'] = volatility
    
    return priors

def calculate_likelihood(data: pd.DataFrame) -> dict:
    """
    Calculate likelihood from recent data.
    """
    likelihood = {}
    
    returns = data['Returns'].dropna()
    
    # Fit normal distribution to recent returns
    mu, std = scipy.stats.norm.fit(returns)
    likelihood['returns_dist'] = scipy.stats.norm(mu, std)
    likelihood['returns_mean'] = mu
    likelihood['returns_std'] = std
    
    # Calculate recent volatility
    volatility = returns.rolling(window=20).std().dropna()
    
    # Fit gamma distribution to volatility
    alpha, loc, beta = scipy.stats.gamma.fit(volatility)
    likelihood['volatility_dist'] = scipy.stats.gamma(alpha, loc, beta)
    
    # Store raw data for plotting
    likelihood['returns_data'] = returns
    likelihood['volatility_data'] = volatility
    
    return likelihood

def calculate_posterior(priors: dict, likelihood: dict, x_range: np.ndarray) -> dict:
    """
    Calculate posterior distributions using Bayes theorem.
    """
    posterior = {}
    
    # For returns (using grid approximation for posterior)
    prior_returns = priors['returns_dist'].pdf(x_range)
    likelihood_returns = likelihood['returns_dist'].pdf(x_range)
    posterior_returns = prior_returns * likelihood_returns
    posterior_returns /= np.trapz(posterior_returns, x_range)  # Normalize
    
    # Store posterior distribution and statistics
    posterior['returns_x'] = x_range
    posterior['returns_pdf'] = posterior_returns
    posterior['returns_mean'] = np.average(x_range, weights=posterior_returns)
    posterior['returns_std'] = np.sqrt(np.average((x_range - posterior['returns_mean'])**2, weights=posterior_returns))
    
    # Calculate credible intervals (95%)
    cumsum = np.cumsum(posterior_returns) / np.sum(posterior_returns)
    posterior['returns_ci_lower'] = x_range[np.searchsorted(cumsum, 0.025)]
    posterior['returns_ci_upper'] = x_range[np.searchsorted(cumsum, 0.975)]
    
    # Calculate probability of positive return
    positive_idx = x_range > 0
    posterior['prob_positive_return'] = np.trapz(posterior_returns[positive_idx], x_range[positive_idx])
    
    # Add volatility data if it's in the priors
    if 'volatility_data' in priors:
        posterior['volatility_mean'] = priors['volatility_mean']
        posterior['volatility_std'] = priors['volatility_std']
        
        # Store percentiles for volatility
        posterior['volatility_low'] = np.percentile(priors['volatility_data'], 25)
        posterior['volatility_median'] = np.percentile(priors['volatility_data'], 50)
        posterior['volatility_high'] = np.percentile(priors['volatility_data'], 75)
    
    return posterior

def calculate_rolling_bayesian_analysis(data: pd.DataFrame, window_size: int, condition_func) -> tuple:
    """
    Calculate Bayesian analysis using rolling windows and condition-based subsetting.
    """
    # Calculate returns if not already present
    if 'Returns' not in data.columns:
        data['Returns'] = data['Close'].pct_change()
    
    # Create rolling windows
    windows = []
    similar_periods_mask = np.zeros(len(data), dtype=bool)  # Track similar periods
    window_indices = []  # Track indices of similar windows
    
    for i in range(len(data) - window_size + 1):
        window = data.iloc[i:i + window_size]
        windows.append(window)
        if condition_func(window):
            similar_periods_mask[i:i + window_size] = True
            window_indices.append(i)
    
    # Get current window (most recent data)
    current_window = windows[-1]
    
    # Get similar windows (excluding current window)
    similar_windows = [windows[i] for i in window_indices[:-1]]  # Exclude last window if it matches
    similar_periods_count = len(similar_windows)
    total_periods = len(data)  # Total number of observations in dataset
    
    if not similar_windows:
        return None, None, None, 0, None, total_periods
    
    # Combine similar windows for prior
    historical_data = pd.concat(similar_windows)
    
    # Calculate priors from similar historical windows
    priors = calculate_market_priors(historical_data)
    
    # Calculate likelihood from current window
    likelihood = calculate_likelihood(current_window)
    
    # Define range for posterior calculation
    x_range = np.linspace(
        min(priors['returns_mean'] - 4*priors['returns_std'],
            likelihood['returns_mean'] - 4*likelihood['returns_std']),
        max(priors['returns_mean'] + 4*priors['returns_std'],
            likelihood['returns_mean'] + 4*likelihood['returns_std']),
        1000
    )
    
    # Calculate posterior
    posterior = calculate_posterior(priors, likelihood, x_range)
    
    return priors, likelihood, posterior, similar_periods_count, similar_periods_mask, total_periods

def weekly_return_condition(threshold: float):
    """Create a condition function based on weekly return threshold"""
    def condition(window: pd.DataFrame) -> bool:
        weekly_return = (window['Close'].iloc[-1] / window['Close'].iloc[-5] - 1) 
        return abs(weekly_return) >= threshold
    return condition

def volatility_condition(threshold: float):
    """Create a condition function based on volatility threshold"""
    def condition(window: pd.DataFrame) -> bool:
        volatility = window['Returns'].std()
        return volatility >= threshold
    return condition

def volatility_level_condition(threshold_low: float, threshold_high: float):
    """Create a condition function based on realized volatility level"""
    def condition(window: pd.DataFrame) -> bool:
        if 'Returns' not in window.columns:
            window['Returns'] = window['Close'].pct_change()
        
        # Calculate realized volatility (annualized)
        rv = window['Returns'].std() * np.sqrt(252)
        return threshold_low <= rv <= threshold_high
    return condition

def trend_condition(threshold: float):
    """Create a condition function based on trend strength"""
    def condition(window: pd.DataFrame) -> bool:
        trend = (window['Close'].iloc[-1] - window['Close'].iloc[0]) / window['Close'].iloc[0]
        return abs(trend) >= threshold
    return condition

def plot_distributions(priors: dict, likelihood: dict, posterior: dict) -> go.Figure:
    """
    Plot prior, likelihood, and posterior distributions.
    """
    fig = go.Figure()
    
    # Plot prior distribution
    fig.add_trace(go.Scatter(
        x=posterior['returns_x'],
        y=priors['returns_dist'].pdf(posterior['returns_x']),
        name='Prior',
        line=dict(color='blue', dash='dash')
    ))
    
    # Plot likelihood
    fig.add_trace(go.Scatter(
        x=posterior['returns_x'],
        y=likelihood['returns_dist'].pdf(posterior['returns_x']),
        name='Likelihood',
        line=dict(color='green', dash='dash')
    ))
    
    # Plot posterior
    fig.add_trace(go.Scatter(
        x=posterior['returns_x'],
        y=posterior['returns_pdf'],
        name='Posterior',
        line=dict(color='red')
    ))
    
    # Add credible intervals
    fig.add_vline(x=posterior['returns_ci_lower'], line_dash="dash", line_color="gray")
    fig.add_vline(x=posterior['returns_ci_upper'], line_dash="dash", line_color="gray")
    fig.add_vline(x=posterior['returns_mean'], line_color="red")
    
    fig.update_layout(
        title="Return Distributions",
        xaxis_title="Returns",
        yaxis_title="Density",
        showlegend=True
    )
    
    return fig

def calculate_distribution_stats(data: np.ndarray, weights: np.ndarray = None) -> dict:
    """Calculate detailed statistics for a distribution"""
    if weights is not None:
        weights = weights / np.sum(weights)  # Normalize weights
        mean = np.average(data, weights=weights)
        var = np.average((data - mean)**2, weights=weights)
        std = np.sqrt(var)
        
        # Calculate weighted percentiles
        cumsum = np.cumsum(weights)
        idx_25 = np.searchsorted(cumsum, 0.25)
        idx_50 = np.searchsorted(cumsum, 0.50)
        idx_75 = np.searchsorted(cumsum, 0.75)
        
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        q1 = sorted_data[idx_25]
        median = sorted_data[idx_50]
        q3 = sorted_data[idx_75]
        
        # Calculate skewness and kurtosis
        skewness = np.average(((data - mean)/std)**3, weights=weights)
        kurtosis = np.average(((data - mean)/std)**4, weights=weights)
        
        # Count of effective observations (sum of weights)
        count = len(data)
    else:
        mean = np.mean(data)
        std = np.std(data)
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        skewness = scipy.stats.skew(data)
        kurtosis = scipy.stats.kurtosis(data)
        count = len(data)
    
    return {
        'count': count,
        'mean': mean,
        'std': std,
        'median': median,
        'q1': q1,
        'q3': q3,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'iqr': q3 - q1
    }

def plot_time_series_with_conditions(data: pd.DataFrame, similar_periods_mask: np.ndarray) -> go.Figure:
    """Plot time series with highlighted similar periods"""
    fig = go.Figure()
    
    # Plot full price series
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Price',
        line=dict(color='gray', width=1)
    ))
    
    # Plot similar periods
    similar_data = data[similar_periods_mask].copy()
    fig.add_trace(go.Scatter(
        x=similar_data.index,
        y=similar_data['Close'],
        name='Similar Periods',
        mode='markers',
        marker=dict(
            color='red',
            size=6,
            symbol='circle'
        )
    ))
    
    fig.update_layout(
        title="Price Series with Similar Periods Highlighted",
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True
    )
    
    return fig

def display_distribution_comparison(priors: dict, posterior: dict, similar_periods_count: int, total_periods: int):
    """Display a comparison table of prior and posterior distribution statistics"""
    # Calculate stats for prior distribution
    prior_stats = calculate_distribution_stats(
        posterior['returns_x'],
        priors['returns_dist'].pdf(posterior['returns_x'])
    )
    prior_stats['count'] = total_periods  # Use total number of periods for prior
    
    # Calculate stats for posterior distribution
    posterior_stats = calculate_distribution_stats(
        posterior['returns_x'],
        posterior['returns_pdf']
    )
    posterior_stats['count'] = similar_periods_count  # Use number of similar periods for posterior
    
    # Create comparison DataFrame
    stats_df = pd.DataFrame({
        'Statistic': [
            'Number of Observations',
            'Mean Return',
            'Standard Deviation',
            'Median Return',
            'Q1 (25th percentile)',
            'Q3 (75th percentile)',
            'IQR',
            'Skewness',
            'Kurtosis'
        ],
        'Prior': [
            f"{prior_stats['count']:,d}",  # Format with thousands separator
            f"{prior_stats['mean']:.4%}",
            f"{prior_stats['std']:.4%}",
            f"{prior_stats['median']:.4%}",
            f"{prior_stats['q1']:.4%}",
            f"{prior_stats['q3']:.4%}",
            f"{prior_stats['iqr']:.4%}",
            f"{prior_stats['skewness']:.3f}",
            f"{prior_stats['kurtosis']:.3f}"
        ],
        'Posterior': [
            f"{posterior_stats['count']:,d}",  # Format with thousands separator
            f"{posterior_stats['mean']:.4%}",
            f"{posterior_stats['std']:.4%}",
            f"{posterior_stats['median']:.4%}",
            f"{posterior_stats['q1']:.4%}",
            f"{posterior_stats['q3']:.4%}",
            f"{posterior_stats['iqr']:.4%}",
            f"{posterior_stats['skewness']:.3f}",
            f"{posterior_stats['kurtosis']:.3f}"
        ]
    })
    
    return stats_df

def calculate_indicator_posteriors(data: pd.DataFrame, indicator_params: dict) -> dict:
    """Calculate Bayesian posteriors for each technical indicator state"""
    # Get indicator states
    states = get_indicator_states(
        data,
        rsi_period=indicator_params['rsi_period'],
        rsi_high=indicator_params['rsi_high'],
        rsi_low=indicator_params['rsi_low'],
        bb_period=indicator_params['bb_period'],
        bb_std=indicator_params['bb_std'],
        bb_threshold=indicator_params['bb_threshold'],
        macd_fast=indicator_params['macd_fast'],
        macd_slow=indicator_params['macd_slow'],
        macd_signal=indicator_params['macd_signal'],
        ma_short=indicator_params['ma_short'],
        ma_long=indicator_params['ma_long']
    )
    
    # Calculate forward returns
    forward_returns = calculate_forward_returns(data)
    
    # Calculate posteriors for each state
    posteriors = {}
    for state_name, state_mask in states.items():
        if state_mask.sum() > 0:  # Only if we have examples of this state
            state_returns = forward_returns[state_mask]['forward_5d'].dropna()
            
            if len(state_returns) > 0:
                # Calculate distribution parameters
                mean = state_returns.mean()
                std = state_returns.std()
                
                # Create posterior distribution
                posteriors[state_name] = {
                    'returns_mean': mean,
                    'returns_std': std,
                    'count': len(state_returns),
                    'prob_positive': (state_returns > 0).mean(),
                    'returns_dist': scipy.stats.norm(mean, std),
                    'returns_x': np.linspace(mean - 4*std, mean + 4*std, 1000),
                    'returns_pdf': scipy.stats.norm(mean, std).pdf(
                        np.linspace(mean - 4*std, mean + 4*std, 1000)
                    ),
                    'returns_ci_lower': mean - 1.96 * std,
                    'returns_ci_upper': mean + 1.96 * std
                }
    
    return posteriors

def display_indicator_comparison(priors: dict, indicator_posteriors: dict, total_periods: int) -> pd.DataFrame:
    """Display a comparison table of prior and indicator-based posterior distributions"""
    # Create x-range for distributions based on all available data
    all_means = [priors['returns_mean']]
    all_stds = [priors['returns_std']]
    
    for posterior in indicator_posteriors.values():
        all_means.append(posterior['returns_mean'])
        all_stds.append(posterior['returns_std'])
    
    # Create a common x-range that covers all distributions
    min_x = min(all_means) - 4 * max(all_stds)
    max_x = max(all_means) + 4 * max(all_stds)
    x_range = np.linspace(min_x, max_x, 1000)
    
    # Calculate prior stats using the common x-range
    prior_pdf = priors['returns_dist'].pdf(x_range)
    prior_stats = calculate_distribution_stats(x_range, prior_pdf)
    prior_stats['count'] = total_periods
    
    # Create DataFrame first with the Statistic column
    stats_df = pd.DataFrame({
        'Statistic': [
            'Number of Observations',
            'Mean Return',
            'Standard Deviation',
            'Median Return',
            'Q1 (25th percentile)',
            'Q3 (75th percentile)',
            'IQR',
            'Skewness',
            'Kurtosis'
        ]
    })
    
    # Add Prior column
    stats_df['Prior'] = [
        f"{prior_stats['count']:,d}",
        f"{prior_stats['mean']:.4%}",
        f"{prior_stats['std']:.4%}",
        f"{prior_stats['median']:.4%}",
        f"{prior_stats['q1']:.4%}",
        f"{prior_stats['q3']:.4%}",
        f"{prior_stats['iqr']:.4%}",
        f"{prior_stats['skewness']:.3f}",
        f"{prior_stats['kurtosis']:.3f}"
    ]
    
    # Add posterior stats for each indicator state
    for state_name, posterior in indicator_posteriors.items():
        column_name = state_name.replace('_', ' ').title()
        
        # Calculate stats for this posterior using the common x-range
        posterior_pdf = posterior['returns_dist'].pdf(x_range)
        stats = calculate_distribution_stats(x_range, posterior_pdf)
        stats['count'] = posterior['count']
        
        stats_df[column_name] = [
            f"{stats['count']:,d}",
            f"{stats['mean']:.4%}",
            f"{stats['std']:.4%}",
            f"{stats['median']:.4%}",
            f"{stats['q1']:.4%}",
            f"{stats['q3']:.4%}",
            f"{stats['iqr']:.4%}",
            f"{stats['skewness']:.3f}",
            f"{stats['kurtosis']:.3f}"
        ]
    
    return stats_df

def calculate_pattern_posteriors(data: pd.DataFrame, pattern_params: dict) -> dict:
    """Calculate Bayesian posteriors for each candlestick pattern"""
    # Initialize pattern detectors with parameters
    patterns = {
        'doji': lambda d: detect_doji(d, threshold=pattern_params['doji_threshold']),
        'hammer': lambda d: detect_hammer(d, min_body_ratio=pattern_params['hammer_ratio']),
        'bullish_engulfing': lambda d: [date for date in detect_engulfing(d) 
                                      if d.loc[date, 'Close'] > d.loc[date, 'Open']],
        'bearish_engulfing': lambda d: [date for date in detect_engulfing(d) 
                                      if d.loc[date, 'Close'] < d.loc[date, 'Open']],
        'morning_star': detect_morning_star,
        'evening_star': detect_evening_star,
        'pin_bar': lambda d: detect_pin_bar(d, min_body_ratio=pattern_params['pin_ratio'])
    }
    
    # Calculate forward returns
    forward_returns = calculate_forward_returns(data)
    
    # Calculate posteriors for each pattern
    posteriors = {}
    for pattern_name, detector in patterns.items():
        # Get pattern dates
        pattern_dates = detector(data)
        
        if pattern_dates:  # Only if we have examples of this pattern
            # Get returns following pattern occurrences
            pattern_returns = []
            for date in pattern_dates:
                try:
                    # Get the next available forward return
                    next_return = forward_returns.loc[date, 'forward_5d']
                    if pd.notna(next_return):
                        pattern_returns.append(next_return)
                except KeyError:
                    continue
            
            if pattern_returns:
                # Calculate distribution parameters
                returns_array = np.array(pattern_returns)
                mean = np.mean(returns_array)
                std = np.std(returns_array)
                
                # Create posterior distribution
                posteriors[pattern_name] = {
                    'returns_mean': mean,
                    'returns_std': std,
                    'count': len(pattern_returns),
                    'prob_positive': np.mean(returns_array > 0),
                    'returns_dist': scipy.stats.norm(mean, std),
                    'returns_ci_lower': mean - 1.96 * std,
                    'returns_ci_upper': mean + 1.96 * std
                }
    
    return posteriors

def plot_signals_on_price(data: pd.DataFrame, signal_dates: List[pd.Timestamp], signal_name: str) -> go.Figure:
    """Plot price chart with signal points highlighted"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add signal points
    if signal_dates:
        signal_prices = [data.loc[date, 'Close'] for date in signal_dates]
        fig.add_trace(go.Scatter(
            x=signal_dates,
            y=signal_prices,
            mode='markers',
            marker=dict(
                size=10,
                symbol='star',
                color='red'
            ),
            name=f'{signal_name} Signals'
        ))
    
    fig.update_layout(
        title=f"Price Chart with {signal_name} Signals",
        yaxis_title="Price",
        xaxis_title="Date"
    )
    
    return fig

def get_technical_signal_dates(data: pd.DataFrame, signal_type: str, indicator_params: dict) -> List[pd.Timestamp]:
    """Get dates when a specific technical signal was active"""
    states = get_indicator_states(data, **indicator_params)
    signal_mask = states.get(signal_type, pd.Series(False, index=data.index))
    return data.index[signal_mask].tolist()

def get_pattern_dates(data: pd.DataFrame, pattern_type: str, pattern_params: dict) -> List[pd.Timestamp]:
    """Get dates when a specific pattern occurred"""
    if pattern_type == 'doji':
        dates = detect_doji(data, threshold=pattern_params['doji_threshold'])
    elif pattern_type == 'hammer':
        dates = detect_hammer(data, min_body_ratio=pattern_params['hammer_ratio'])
    elif pattern_type == 'bullish_engulfing':
        dates = [d for d in detect_engulfing(data) 
                if data.loc[d, 'Close'] > data.loc[d, 'Open']]
    elif pattern_type == 'bearish_engulfing':
        dates = [d for d in detect_engulfing(data) 
                if data.loc[d, 'Close'] < data.loc[d, 'Open']]
    elif pattern_type == 'morning_star':
        dates = detect_morning_star(data)
    elif pattern_type == 'evening_star':
        dates = detect_evening_star(data)
    elif pattern_type == 'pin_bar':
        dates = detect_pin_bar(data, min_body_ratio=pattern_params['pin_ratio'])
    else:
        dates = []
    return dates

def calculate_realized_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Calculate realized volatility using rolling window
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    window : int
        Rolling window size in days
    annualize : bool
        Whether to annualize the volatility
    
    Returns:
    --------
    pd.Series
        Rolling realized volatility
    """
    # Calculate rolling variance
    rolling_var = returns.rolling(window=window).var()
    
    # Annualize if requested (252 trading days in a year)
    if annualize:
        annualized_var = rolling_var * 252
        realized_vol = np.sqrt(annualized_var)
        return realized_vol
    else:
        return np.sqrt(rolling_var)

def plot_realized_volatility(data: pd.DataFrame, window_sizes: List[int] = [20, 60, 120]) -> go.Figure:
    """
    Plot realized volatility for multiple rolling windows
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with returns column
    window_sizes : List[int]
        List of rolling window sizes in days
    
    Returns:
    --------
    go.Figure
        Plotly figure with realized volatility
    """
    if 'Returns' not in data.columns:
        data['Returns'] = data['Close'].pct_change()
    
    fig = go.Figure()
    
    # Calculate and plot realized volatility for each window size
    for window in window_sizes:
        rv = calculate_realized_volatility(data['Returns'], window=window)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=rv,
            name=f'{window}-day RV',
            line=dict(width=2)
        ))
    
    # Format the figure
    fig.update_layout(
        title='Realized Volatility (Annualized)',
        xaxis_title='Date',
        yaxis_title='Volatility',
        yaxis_tickformat='.1%',
        legend_title='Window Size',
        hovermode='x unified'
    )
    
    return fig

def plot_volatility_with_conditions(data: pd.DataFrame, similar_periods_mask: np.ndarray, window_size: int = 20) -> go.Figure:
    """Plot realized volatility with highlighted similar periods"""
    # Calculate realized volatility
    if 'Returns' not in data.columns:
        data['Returns'] = data['Close'].pct_change()
    
    realized_vol = calculate_realized_volatility(data['Returns'], window=window_size)
    
    fig = go.Figure()
    
    # Plot full volatility series
    fig.add_trace(go.Scatter(
        x=data.index,
        y=realized_vol,
        name='Realized Volatility',
        line=dict(color='gray', width=1)
    ))
    
    # Plot similar periods
    similar_data = data[similar_periods_mask].copy()
    if len(similar_data) > 0:
        # Get realized vol for similar periods
        similar_vol = realized_vol[similar_periods_mask]
        
        fig.add_trace(go.Scatter(
            x=similar_data.index,
            y=similar_vol,
            name='Similar Periods',
            mode='markers',
            marker=dict(
                color='red',
                size=6,
                symbol='circle'
            )
        ))
    
    # Add horizontal lines for percentiles
    vol_25 = realized_vol.quantile(0.25)
    vol_50 = realized_vol.quantile(0.50)
    vol_75 = realized_vol.quantile(0.75)
    
    fig.add_hline(y=vol_25, line_dash="dash", line_color="blue", annotation_text="25th %")
    fig.add_hline(y=vol_50, line_dash="dash", line_color="green", annotation_text="Median")
    fig.add_hline(y=vol_75, line_dash="dash", line_color="orange", annotation_text="75th %")
    
    fig.update_layout(
        title=f"Realized Volatility ({window_size}-day) with Similar Periods Highlighted",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat='.1%',
        showlegend=True
    )
    
    return fig

def render_financial_markets():
    """
    Main function to render the financial markets analysis page.
    """
    st.title("Bayesian Financial Market Analysis")
    
    # Add disclaimer
    st.warning("""
    ⚠️ Disclaimer: This tool is for educational and analytical purposes only.
    Not intended for direct trading advice. Past performance is not indicative of future results.
    """)
    
    # Test data fetching
    if st.button("Test Data Fetching"):
        results = test_data_fetching()
        st.subheader("Data Fetching Test Results")
        for symbol, result in results.items():
            st.write(f"{symbol}: {result}")
    
    # Data Selection Section
    st.header("Data Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        # Currency pair selection - simplified to just EURUSD
        currency_pairs = ["EURUSD"]
        selected_pair = st.selectbox(
            "Select Currency Pair",
            currency_pairs,
            index=0
        )
        
        # Time frame selection
        timeframes = ["1d", "1wk", "1mo"]
        selected_timeframe = st.selectbox(
            "Select Time Frame",
            timeframes,
            index=0
        )
    
    with col2:
        # Get data range from CSV first
        initial_data = fetch_market_data(
            symbol=selected_pair,
            timeframe=selected_timeframe,
            start_date=datetime(2009, 1, 1).date(),  # Use datetime directly
            end_date=datetime.now().date()
        )
        
        if initial_data is not None:
            min_date = initial_data.index.min().date()
            max_date = initial_data.index.max().date()
            
            # Date range selection with two separate controls
            col_start, col_end = st.columns(2)
            
            with col_start:
                start_date = st.date_input(
                    "From",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col_end:
                end_date = st.date_input(
                    "To",
                    value=max_date,
                    min_value=start_date,
                    max_value=max_date
                )
            
            # Convert to pandas Timestamps
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
        else:
            st.error("Could not determine data range from CSV file")
            return
    
    # Fetch full dataset
    full_data = fetch_market_data(
        symbol=selected_pair,
        timeframe=selected_timeframe,
        start_date=start_date,  # Already a Timestamp
        end_date=end_date      # Already a Timestamp
    )
    
    if full_data is not None:
        # Calculate returns if not already present
        if 'Returns' not in full_data.columns:
            full_data['Returns'] = full_data['Close'].pct_change()
        
        st.success(f"Data fetched successfully. Shape: {full_data.shape}")
        
        # Display basic market analysis
        st.header("Market Analysis")
        
        # Display basic data information
        st.subheader("Market Data Overview")
        st.dataframe(full_data.head())
        
        # Display basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(full_data.describe())
        
        # Display candlestick chart
        st.subheader("Price Chart")
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=full_data.index,
            open=full_data['Open'],
            high=full_data['High'],
            low=full_data['Low'],
            close=full_data['Close'],
            name='Price'
        ))
        
        fig.update_layout(
            title=f"{selected_pair} Price Chart",
            yaxis_title="Price",
            xaxis_title="Date"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bayesian Analysis Section
        st.header("Bayesian Analysis")
        
        # Analysis Parameters
        st.subheader("Analysis Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            window_size = st.slider(
                "Rolling Window Size (days)",
                min_value=5,
                max_value=90,
                value=30,
                step=5,
                help="Size of the rolling window for analysis"
            )
            
            condition_type = st.selectbox(
                "Condition Type",
                ["Weekly Return", "Volatility", "Trend", "Realized Volatility"],
                help="Type of condition to identify similar market periods"
            )
        
        with col2:
            if condition_type == "Realized Volatility":
                vol_threshold_low = st.slider(
                    "Volatility Range (Low)",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.05,
                    step=0.01,
                    format="%.2f",
                    help="Lower bound for realized volatility (annualized)"
                )
                
                vol_threshold_high = st.slider(
                    "Volatility Range (High)",
                    min_value=vol_threshold_low + 0.01,
                    max_value=0.5,
                    value=min(vol_threshold_low + 0.1, 0.5),
                    step=0.01,
                    format="%.2f",
                    help="Upper bound for realized volatility (annualized)"
                )
                
                threshold = None  # Not used for volatility range
            else:
                threshold = st.slider(
                    "Condition Threshold",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.02,
                    step=0.005,
                    format="%.3f",
                    help="Threshold for the selected condition"
                )
        
        # Create condition function based on user selection
        if condition_type == "Weekly Return":
            condition_func = weekly_return_condition(threshold)
            condition_description = f"Periods with weekly returns >= {threshold:.1%}"
        elif condition_type == "Volatility":
            condition_func = volatility_condition(threshold)
            condition_description = f"Periods with volatility >= {threshold:.1%}"
        elif condition_type == "Realized Volatility":
            condition_func = volatility_level_condition(vol_threshold_low, vol_threshold_high)
            condition_description = f"Periods with realized volatility between {vol_threshold_low:.1%} and {vol_threshold_high:.1%}"
        else:  # Trend
            condition_func = trend_condition(threshold)
            condition_description = f"Periods with trend strength >= {threshold:.1%}"
        
        st.write(f"Analyzing similar periods based on: {condition_description}")
        
        if full_data is not None and len(full_data) >= window_size:
            # Perform rolling Bayesian analysis
            priors, likelihood, posterior, similar_periods_count, similar_periods_mask, total_periods = calculate_rolling_bayesian_analysis(
                full_data,
                window_size,
                condition_func
            )
            
            if priors is not None:
                # Display number of similar periods found
                st.info(f"Found {similar_periods_count:,d} similar periods out of {total_periods:,d} total observations.")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Historical Priors")
                    st.write(f"Mean Return: {priors['returns_mean']:.4%}")
                    st.write(f"Return Volatility: {priors['returns_std']:.4%}")
                    if 'volatility_mean' in priors:
                        st.write(f"Mean Realized Vol: {priors['volatility_mean']:.4%}")
                
                with col2:
                    st.subheader("Recent Window")
                    st.write(f"Mean Return: {likelihood['returns_mean']:.4%}")
                    st.write(f"Return Volatility: {likelihood['returns_std']:.4%}")
                    if 'volatility_mean' in likelihood:
                        st.write(f"Recent Realized Vol: {likelihood['volatility_mean']:.4%}")
                
                with col3:
                    st.subheader("Posterior Estimates")
                    st.write(f"Expected Return: {posterior['returns_mean']:.4%}")
                    st.write(f"95% Credible Interval:")
                    st.write(f"[{posterior['returns_ci_lower']:.4%}, {posterior['returns_ci_upper']:.4%}]")
                    st.write(f"Prob. Positive: {posterior['prob_positive_return']:.1%}")
                
                # Add volatility analysis section if we have the data
                if 'volatility_mean' in posterior:
                    st.subheader("Volatility Analysis")
                    vol_col1, vol_col2, vol_col3 = st.columns(3)
                    
                    with vol_col1:
                        st.write("**Historical Volatility Stats**")
                        st.write(f"Mean: {posterior['volatility_mean']:.4%}")
                        st.write(f"Std Dev: {posterior['volatility_std']:.4%}")
                    
                    with vol_col2:
                        st.write("**Volatility Percentiles**")
                        st.write(f"25th: {posterior['volatility_low']:.4%}")
                        st.write(f"50th: {posterior['volatility_median']:.4%}")
                        st.write(f"75th: {posterior['volatility_high']:.4%}")
                    
                    with vol_col3:
                        st.write("**Volatility Prediction**")
                        # Determine volatility prediction based on recent volatility vs historical
                        if 'volatility_mean' in likelihood:
                            vol_ratio = likelihood['volatility_mean'] / posterior['volatility_mean']
                            
                            if vol_ratio < 0.8:
                                vol_pred = "Decreasing"
                                vol_color = "green"
                            elif vol_ratio > 1.2:
                                vol_pred = "Increasing"
                                vol_color = "red"
                            else:
                                vol_pred = "Stable"
                                vol_color = "orange"
                            
                            st.markdown(f"Trend: <span style='color:{vol_color};font-weight:bold'>{vol_pred}</span>", unsafe_allow_html=True)
                            st.write(f"Vol Ratio: {vol_ratio:.2f}x")
                
                # Plot distributions
                st.plotly_chart(plot_distributions(priors, likelihood, posterior))
                
                # Plot time series with similar periods
                st.plotly_chart(plot_time_series_with_conditions(full_data, similar_periods_mask))
                
                # Add volatility analysis plots
                if 'volatility_mean' in posterior and similar_periods_mask is not None:
                    st.subheader("Volatility Regime Analysis")
                    
                    # Plot realized volatility with similar periods highlighted
                    vol_plot = plot_volatility_with_conditions(full_data, similar_periods_mask, window_size=20)
                    st.plotly_chart(vol_plot, use_container_width=True)
                    
                    # Additional analysis for condition based on realized volatility
                    if condition_type == "Realized Volatility":
                        st.write(f"""
                        **Volatility Regime Interpretation:**
                        - You've selected periods with realized volatility between {vol_threshold_low:.1%} and {vol_threshold_high:.1%}
                        - This regime represents {similar_periods_count/total_periods:.1%} of historical data
                        - In this volatility regime, expected returns are {posterior['returns_mean']:.2%} with {posterior['prob_positive_return']:.1%} probability of positive returns
                        """)
                
                # Display detailed statistics comparison
                st.subheader("Distribution Statistics Comparison")
                stats_df = display_distribution_comparison(priors, posterior, similar_periods_count, total_periods)
                st.table(stats_df)
                
                # Add interpretation
                st.write("""
                **Key Statistics Interpretation:**
                - **Number of Observations**: Count of similar periods used in the analysis
                - **Mean/Median**: Center of the distribution, representing expected returns
                - **Standard Deviation**: Measure of volatility/uncertainty
                - **Q1/Q3**: Shows the range where 50% of returns fall
                - **IQR**: Interquartile range, showing spread of the middle 50% of data
                - **Skewness**: Measures asymmetry (0 is symmetric, >0 right-skewed, <0 left-skewed)
                - **Kurtosis**: Measures tail heaviness (3 is normal, >3 heavy-tailed, <3 light-tailed)
                """)
                
                # Trading Signals
                st.header("Trading Signals")
                signal_strength = abs(posterior['returns_mean']) / posterior['returns_std']
                
                if posterior['returns_mean'] > 0:
                    if posterior['prob_positive_return'] > 0.75:
                        signal = "Strong Buy"
                        signal_color = "green"
                    else:
                        signal = "Weak Buy"
                        signal_color = "lightgreen"
                else:
                    if posterior['prob_positive_return'] < 0.25:
                        signal = "Strong Sell"
                        signal_color = "red"
                    else:
                        signal = "Weak Sell"
                        signal_color = "pink"
                
                st.markdown(f"Signal: <span style='color:{signal_color};font-weight:bold'>{signal}</span>", unsafe_allow_html=True)
                st.write(f"Signal Strength: {signal_strength:.2f}")
                st.write(f"Probability of Positive Return: {posterior['prob_positive_return']:.1%}")
                
                # Add volatility-adjusted metrics
                if 'volatility_mean' in posterior:
                    st.subheader("Volatility-Adjusted Metrics")
                    
                    # Calculate Sharpe ratio (expected return / volatility)
                    sharpe = posterior['returns_mean'] / posterior['volatility_mean']
                    
                    # Calculate risk-adjusted return (Sortino-like: expected return / downside vol)
                    if 'returns_data' in priors:
                        downside_returns = priors['returns_data'][priors['returns_data'] < 0]
                        if len(downside_returns) > 0:
                            downside_vol = downside_returns.std() * np.sqrt(252)
                            sortino = posterior['returns_mean'] / downside_vol
                        else:
                            sortino = None
                    else:
                        sortino = None
                    
                    st.write(f"Sharpe Ratio (annualized): {sharpe:.2f}")
                    if sortino is not None:
                        st.write(f"Sortino Ratio (annualized): {sortino:.2f}")
                    
                    # Calculate volatility-adjusted position sizing
                    target_vol = 0.15  # 15% target portfolio volatility
                    position_sizing = target_vol / posterior['volatility_mean']
                    
                    st.write(f"Suggested Position Size: {min(position_sizing, 1.0):.2f}x (based on {target_vol:.0%} target volatility)")
                    
                    # Create a risk-reward chart
                    st.subheader("Risk-Reward Analysis")
                    
                    if 'returns_ci_lower' in posterior and 'returns_ci_upper' in posterior:
                        risk_reward_ratio = abs(posterior['returns_mean'] / posterior['returns_ci_lower'])
                        
                        st.write(f"""
                        **Risk-Reward Metrics:**
                        - Expected Return: {posterior['returns_mean']:.2%}
                        - Downside Risk (95% CI): {posterior['returns_ci_lower']:.2%}
                        - Upside Potential (95% CI): {posterior['returns_ci_upper']:.2%}
                        - Risk-Reward Ratio: {risk_reward_ratio:.2f}
                        """)
                        
                        # Only show trade recommendation if we have risk-reward data
                        st.write("**Trade Recommendation**")
                        
                        if posterior['returns_mean'] > 0 and risk_reward_ratio > 1.5:
                            st.markdown("<span style='color:green;font-weight:bold'>✅ Favorable risk-reward for long position</span>", unsafe_allow_html=True)
                        elif posterior['returns_mean'] < 0 and risk_reward_ratio > 1.5:
                            st.markdown("<span style='color:green;font-weight:bold'>✅ Favorable risk-reward for short position</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span style='color:red;font-weight:bold'>❌ Unfavorable risk-reward ratio</span>", unsafe_allow_html=True)
            else:
                st.warning("No similar periods found with the current conditions. Try adjusting the threshold.")
        else:
            st.error("Insufficient data for the selected window size.")
            
        # Add Technical Indicator Analysis
        st.header("Technical Indicator Analysis")
        
        # Technical Indicator Parameters
        st.subheader("Indicator Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI Parameters
            st.write("**RSI Parameters**")
            rsi_period = st.slider("RSI Period", 5, 30, 14)
            rsi_high = st.slider("RSI Overbought", 60, 90, 70)
            rsi_low = st.slider("RSI Oversold", 10, 40, 30)
            
            # MACD Parameters
            st.write("**MACD Parameters**")
            macd_fast = st.slider("MACD Fast Period", 5, 20, 12)
            macd_slow = st.slider("MACD Slow Period", 15, 40, 26)
            macd_signal = st.slider("MACD Signal Period", 5, 20, 9)
        
        with col2:
            # Bollinger Bands Parameters
            st.write("**Bollinger Bands Parameters**")
            bb_period = st.slider("BB Period", 5, 40, 20)
            bb_std = st.slider("BB Standard Deviations", 1.0, 3.0, 2.0, 0.1)
            bb_threshold = st.slider("BB Threshold", 0.01, 0.10, 0.05, 0.01)
            
            # Moving Average Parameters
            st.write("**Moving Average Parameters**")
            ma_short = st.slider("Short MA Period", 5, 50, 20)
            ma_long = st.slider("Long MA Period", 20, 100, 50)
        
        # Collect all parameters
        indicator_params = {
            'rsi_period': rsi_period,
            'rsi_high': rsi_high,
            'rsi_low': rsi_low,
            'bb_period': bb_period,
            'bb_std': bb_std,
            'bb_threshold': bb_threshold,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'ma_short': ma_short,
            'ma_long': ma_long
        }
        
        # Calculate indicator-based posteriors
        indicator_posteriors = calculate_indicator_posteriors(full_data, indicator_params)
        
        if indicator_posteriors:
            # Display indicator statistics comparison
            st.subheader("Technical Indicator Statistics Comparison")
            st.write("Comparing returns distribution across different technical conditions:")
            stats_df = display_indicator_comparison(priors, indicator_posteriors, total_periods)
            st.table(stats_df)
            
            # Add interpretation
            st.write("""
            **Technical Indicator Statistics Interpretation:**
            - Each column shows the return distribution when that indicator condition is met
            - Compare the mean returns and probabilities across different signals
            - Higher observation counts indicate more frequent signals
            - Lower standard deviations suggest more reliable signals
            - Skewness and kurtosis help identify potential outlier effects
            """)
            
            # Current Indicator States
            st.subheader("Current Technical Conditions")
            current_states = get_indicator_states(full_data.iloc[-20:], **indicator_params)
            active_signals = [
                state.replace('_', ' ').title()
                for state, mask in current_states.items()
                if mask.iloc[-1]
            ]
            
            if active_signals:
                st.write("**Active Signals:**")
                for signal in active_signals:
                    st.write(f"- {signal}")
                
                # Get the posterior stats for active signals
                active_posteriors = {
                    state: indicator_posteriors[state]
                    for state in current_states.keys()
                    if current_states[state].iloc[-1]
                    and state in indicator_posteriors
                }
                
                if active_posteriors:
                    st.write("**Expected Returns for Active Signals:**")
                    for state, posterior in active_posteriors.items():
                        st.write(f"- {state.replace('_', ' ').title()}: "
                               f"Mean: {posterior['returns_mean']:.2%}, "
                               f"Prob. Positive: {posterior['prob_positive']:.1%}")
            else:
                st.write("No technical conditions are currently active.")
            
            # Add signal visualization
            st.subheader("Technical Signal Visualization")
            signal_options = list(indicator_posteriors.keys())
            selected_signal = st.selectbox(
                "Select Technical Signal to Visualize",
                signal_options,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if selected_signal:
                signal_dates = get_technical_signal_dates(full_data, selected_signal, indicator_params)
                if signal_dates:
                    st.plotly_chart(
                        plot_signals_on_price(
                            full_data,
                            signal_dates,
                            selected_signal.replace('_', ' ').title()
                        ),
                        use_container_width=True
                    )
                    st.write(f"Number of {selected_signal.replace('_', ' ').title()} signals found: {len(signal_dates)}")
                else:
                    st.write("No signals found for the selected indicator")
        else:
            st.warning("Insufficient data to calculate indicator statistics.")
            
        # Add Candlestick Pattern Analysis
        st.header("Candlestick Pattern Analysis")
        
        # Pattern Parameters
        st.subheader("Pattern Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Doji Parameters
            st.write("**Doji Parameters**")
            doji_threshold = st.slider(
                "Doji Body/Shadow Ratio",
                0.01,
                0.2,
                0.1,
                0.01,
                help="Maximum ratio of body to shadow length for Doji pattern"
            )
            
            # Hammer Parameters
            st.write("**Hammer Parameters**")
            hammer_ratio = st.slider(
                "Hammer Body Ratio",
                0.1,
                0.5,
                0.3,
                0.05,
                help="Minimum ratio of body to total length for Hammer pattern"
            )
        
        with col2:
            # Pin Bar Parameters
            st.write("**Pin Bar Parameters**")
            pin_ratio = st.slider(
                "Pin Bar Body Ratio",
                0.1,
                0.5,
                0.3,
                0.05,
                help="Minimum ratio of body to total length for Pin Bar pattern"
            )
        
        # Collect pattern parameters
        pattern_params = {
            'doji_threshold': doji_threshold,
            'hammer_ratio': hammer_ratio,
            'pin_ratio': pin_ratio
        }
        
        # Calculate pattern-based posteriors
        pattern_posteriors = calculate_pattern_posteriors(full_data, pattern_params)
        
        if pattern_posteriors:
            # Display pattern statistics comparison
            st.subheader("Candlestick Pattern Statistics Comparison")
            st.write("Comparing returns distribution following different candlestick patterns:")
            stats_df = display_indicator_comparison(priors, pattern_posteriors, total_periods)
            st.table(stats_df)
            
            # Add interpretation
            st.write("""
            **Candlestick Pattern Statistics Interpretation:**
            - Each column shows the return distribution following that pattern
            - Compare the mean returns and probabilities across different patterns
            - Higher observation counts indicate more frequent patterns
            - Lower standard deviations suggest more reliable signals
            - Patterns with extreme skewness may indicate strong directional moves
            """)
            
            # Recent Pattern Analysis
            st.subheader("Recent Pattern Analysis")
            recent_data = full_data.iloc[-20:]  # Look at last 20 periods
            
            # Detect patterns in recent data
            recent_patterns = []
            for pattern_name in pattern_posteriors.keys():
                if pattern_name == 'doji':
                    dates = detect_doji(recent_data, threshold=pattern_params['doji_threshold'])
                elif pattern_name == 'hammer':
                    dates = detect_hammer(recent_data, min_body_ratio=pattern_params['hammer_ratio'])
                elif pattern_name == 'bullish_engulfing':
                    dates = [d for d in detect_engulfing(recent_data) 
                           if recent_data.loc[d, 'Close'] > recent_data.loc[d, 'Open']]
                elif pattern_name == 'bearish_engulfing':
                    dates = [d for d in detect_engulfing(recent_data) 
                           if recent_data.loc[d, 'Close'] < recent_data.loc[d, 'Open']]
                elif pattern_name == 'morning_star':
                    dates = detect_morning_star(recent_data)
                elif pattern_name == 'evening_star':
                    dates = detect_evening_star(recent_data)
                else:  # pin_bar
                    dates = detect_pin_bar(recent_data, min_body_ratio=pattern_params['pin_ratio'])
                
                if dates:
                    recent_patterns.extend([(d, pattern_name) for d in dates])
            
            if recent_patterns:
                st.write("**Recently Detected Patterns:**")
                # Sort by date
                recent_patterns.sort(key=lambda x: x[0])
                for date, pattern in recent_patterns:
                    pattern_stats = pattern_posteriors[pattern]
                    st.write(f"- {pattern.replace('_', ' ').title()} on {date.strftime('%Y-%m-%d')}: "
                           f"Mean: {pattern_stats['returns_mean']:.2%}, "
                           f"Prob. Positive: {pattern_stats['prob_positive']:.1%}")
            else:
                st.write("No patterns detected in recent data.")
            
            # Add pattern visualization
            st.subheader("Pattern Visualization")
            pattern_options = list(pattern_posteriors.keys())
            selected_pattern = st.selectbox(
                "Select Pattern to Visualize",
                pattern_options,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if selected_pattern:
                pattern_dates = get_pattern_dates(full_data, selected_pattern, pattern_params)
                if pattern_dates:
                    st.plotly_chart(
                        plot_signals_on_price(
                            full_data,
                            pattern_dates,
                            selected_pattern.replace('_', ' ').title()
                        ),
                        use_container_width=True
                    )
                    st.write(f"Number of {selected_pattern.replace('_', ' ').title()} patterns found: {len(pattern_dates)}")
                else:
                    st.write("No patterns found for the selected type")
        else:
            st.warning("Insufficient data to calculate pattern statistics.")
        
        # Add Realized Volatility Analysis
        st.header("Realized Volatility Analysis")
        
        # Volatility Parameters
        st.subheader("Volatility Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            primary_window = st.slider(
                "Primary Window Size (days)",
                min_value=5,
                max_value=252,
                value=20,
                step=5,
                help="Primary rolling window size for realized volatility calculation"
            )
            
            show_multiple_windows = st.checkbox(
                "Show Multiple Windows",
                value=True,
                help="Show realized volatility for multiple window sizes"
            )
        
        with col2:
            if show_multiple_windows:
                secondary_window = st.slider(
                    "Secondary Window Size (days)",
                    min_value=5,
                    max_value=252,
                    value=60,
                    step=5,
                    help="Secondary rolling window size for comparison"
                )
                
                tertiary_window = st.slider(
                    "Tertiary Window Size (days)",
                    min_value=5,
                    max_value=252,
                    value=120,
                    step=5,
                    help="Tertiary rolling window size for comparison"
                )
                
                window_sizes = [primary_window, secondary_window, tertiary_window]
            else:
                window_sizes = [primary_window]
        
        # Calculate and display realized volatility
        if 'Returns' not in full_data.columns:
            full_data['Returns'] = full_data['Close'].pct_change()
        
        # Calculate primary realized volatility
        rv = calculate_realized_volatility(full_data['Returns'], window=primary_window)
        
        # Display current realized volatility
        current_rv = rv.iloc[-1]
        st.subheader(f"Current {primary_window}-day Realized Volatility")
        st.write(f"**{current_rv:.2%}** (annualized)")
        
        # Display realized volatility stats
        st.subheader("Realized Volatility Statistics")
        
        rv_stats = pd.DataFrame({
            'Statistic': ['Current', 'Mean', 'Median', 'Min', 'Max', 'Std Dev', '25th Percentile', '75th Percentile'],
            'Value': [
                f"{current_rv:.2%}",
                f"{rv.mean():.2%}",
                f"{rv.median():.2%}",
                f"{rv.min():.2%}",
                f"{rv.max():.2%}",
                f"{rv.std():.2%}",
                f"{rv.quantile(0.25):.2%}",
                f"{rv.quantile(0.75):.2%}"
            ]
        })
        
        st.table(rv_stats)
        
        # Plot realized volatility
        st.subheader("Realized Volatility Time Series")
        st.plotly_chart(
            plot_realized_volatility(full_data, window_sizes),
            use_container_width=True
        )
        
        # Add explanation
        st.write("""
        **Realized Volatility Explanation:**
        - Calculated as the rolling standard deviation of daily returns
        - Annualized by multiplying by √252 (square root of trading days in a year)
        - Higher values indicate more market turbulence
        - Comparing different window sizes helps identify short vs. long-term volatility regimes
        """)
        
        # Add volatility regime analysis
        st.subheader("Volatility Regime Analysis")
        
        # Define volatility regimes based on percentiles
        low_threshold = rv.quantile(0.25)
        high_threshold = rv.quantile(0.75)
        
        if current_rv < low_threshold:
            regime = "Low Volatility"
            regime_color = "green"
        elif current_rv > high_threshold:
            regime = "High Volatility"
            regime_color = "red"
        else:
            regime = "Normal Volatility"
            regime_color = "orange"
        
        st.markdown(f"Current Regime: <span style='color:{regime_color};font-weight:bold'>{regime}</span>", unsafe_allow_html=True)
        
        # Display regime thresholds
        st.write(f"Low Volatility Threshold: {low_threshold:.2%}")
        st.write(f"High Volatility Threshold: {high_threshold:.2%}")
        
        # Calculate volatility percentile
        percentile = (rv < current_rv).mean() * 100
        st.write(f"Current Volatility Percentile: {percentile:.1f}%")
            
    else:
        st.error("Failed to fetch data. Please try different parameters.")

if __name__ == "__main__":
    render_financial_markets() 