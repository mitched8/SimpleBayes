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
            start_date=date(2009, 1, 1),  # Use a very early date
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
                ["Weekly Return", "Volatility", "Trend"],
                help="Type of condition to identify similar market periods"
            )
        
        with col2:
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
                    st.write(f"Volatility: {priors['returns_std']:.4%}")
                
                with col2:
                    st.subheader("Recent Window")
                    st.write(f"Mean Return: {likelihood['returns_mean']:.4%}")
                    st.write(f"Volatility: {likelihood['returns_std']:.4%}")
                
                with col3:
                    st.subheader("Posterior Estimates")
                    st.write(f"Expected Return: {posterior['returns_mean']:.4%}")
                    st.write(f"95% Credible Interval:")
                    st.write(f"[{posterior['returns_ci_lower']:.4%}, {posterior['returns_ci_upper']:.4%}]")
                    st.write(f"Prob. Positive: {posterior['prob_positive_return']:.1%}")
                
                # Plot distributions
                st.plotly_chart(plot_distributions(priors, likelihood, posterior))
                
                # Plot time series with similar periods
                st.plotly_chart(plot_time_series_with_conditions(full_data, similar_periods_mask))
                
                # Display detailed statistics comparison
                st.subheader("Distribution Statistics Comparison")
                st.write("Comparing the prior and posterior distributions:")
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
                    else:
                        signal = "Weak Buy"
                else:
                    if posterior['prob_positive_return'] < 0.25:
                        signal = "Strong Sell"
                    else:
                        signal = "Weak Sell"
                
                st.write(f"Signal: {signal}")
                st.write(f"Signal Strength: {signal_strength:.2f}")
                st.write(f"Probability of Positive Return: {posterior['prob_positive_return']:.1%}")
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
        else:
            st.warning("Insufficient data to calculate indicator statistics.")
            
    else:
        st.error("Failed to fetch data. Please try different parameters.")

if __name__ == "__main__":
    render_financial_markets() 