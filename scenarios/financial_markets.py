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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import utility functions
from utils.market_utils import (
    fetch_market_data,
    calculate_returns,
    identify_candlestick_patterns
)
from utils.pattern_recognition import (
    detect_doji,
    detect_hammer,
    detect_engulfing,
    detect_morning_star,
    detect_evening_star,
    detect_pin_bar
)

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
    Calculate market priors from historical data.
    Returns dict with mean, std, and other statistical properties.
    """
    priors = {}
    
    # Calculate return statistics
    returns = data['Returns'].dropna()
    priors['returns_mean'] = returns.mean()
    priors['returns_std'] = returns.std()
    priors['returns_skew'] = returns.skew()
    priors['returns_kurtosis'] = returns.kurtosis()
    
    # Calculate volatility (using rolling std of returns)
    volatility = returns.rolling(window=20).std().dropna()
    priors['volatility_mean'] = volatility.mean()
    priors['volatility_std'] = volatility.std()
    
    # Calculate trend strength (using rolling returns)
    rolling_returns = returns.rolling(window=20).mean().dropna()
    priors['trend_mean'] = rolling_returns.mean()
    priors['trend_std'] = rolling_returns.std()
    
    # Calculate probability of positive returns
    priors['prob_positive_return'] = (returns > 0).mean()
    
    return priors

def calculate_bayesian_analysis(historical_data: pd.DataFrame, recent_data: pd.DataFrame) -> dict:
    """
    Perform Bayesian analysis using historical priors and recent data.
    """
    # Calculate priors from historical data
    priors = calculate_market_priors(historical_data)
    
    # Calculate recent statistics
    recent_returns = recent_data['Returns'].dropna()
    recent_stats = {
        'returns_mean': recent_returns.mean(),
        'returns_std': recent_returns.std(),
        'prob_positive_return': (recent_returns > 0).mean()
    }
    
    # Calculate posterior probabilities using Bayesian updating
    n_historical = len(historical_data)
    n_recent = len(recent_data)
    
    # Weight recent data more heavily
    weight_recent = 0.7
    weight_historical = 0.3
    
    posterior = {}
    
    # Update return expectations
    posterior['expected_return'] = (
        weight_historical * priors['returns_mean'] +
        weight_recent * recent_stats['returns_mean']
    )
    
    # Update volatility expectations
    posterior['expected_volatility'] = (
        weight_historical * priors['returns_std'] +
        weight_recent * recent_stats['returns_std']
    )
    
    # Calculate probability of positive return
    posterior['prob_positive_return'] = (
        weight_historical * priors['prob_positive_return'] +
        weight_recent * recent_stats['prob_positive_return']
    )
    
    # Calculate credible intervals for returns
    posterior['return_ci_lower'] = posterior['expected_return'] - 1.96 * posterior['expected_volatility']
    posterior['return_ci_upper'] = posterior['expected_return'] + 1.96 * posterior['expected_volatility']
    
    return priors, recent_stats, posterior

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
        
        # Split data for Bayesian analysis
        split_date = end_date - timedelta(days=30)  # Last 30 days for recent data
        split_date = pd.Timestamp(split_date)  # Convert to pandas Timestamp
        
        historical_data = full_data[full_data.index < split_date].copy()
        recent_data = full_data[full_data.index >= split_date].copy()
        
        if not recent_data.empty and not historical_data.empty:
            # Perform Bayesian analysis
            priors, recent_stats, posterior = calculate_bayesian_analysis(historical_data, recent_data)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Historical Priors")
                st.write(f"Mean Return: {priors['returns_mean']:.4%}")
                st.write(f"Volatility: {priors['returns_std']:.4%}")
                st.write(f"Prob. Positive: {priors['prob_positive_return']:.1%}")
            
            with col2:
                st.subheader("Recent Statistics")
                st.write(f"Mean Return: {recent_stats['returns_mean']:.4%}")
                st.write(f"Volatility: {recent_stats['returns_std']:.4%}")
                st.write(f"Prob. Positive: {recent_stats['prob_positive_return']:.1%}")
            
            with col3:
                st.subheader("Posterior Estimates")
                st.write(f"Expected Return: {posterior['expected_return']:.4%}")
                st.write(f"Expected Volatility: {posterior['expected_volatility']:.4%}")
                st.write(f"Prob. Positive: {posterior['prob_positive_return']:.1%}")
            
            # Plot distributions
            st.subheader("Return Distributions")
            fig = go.Figure()
            
            # Historical returns distribution
            hist_returns = historical_data['Returns'].dropna()
            fig.add_trace(go.Histogram(
                x=hist_returns,
                name="Historical Returns",
                opacity=0.7,
                nbinsx=50,
                histnorm='probability'
            ))
            
            # Recent returns distribution
            recent_returns = recent_data['Returns'].dropna()
            fig.add_trace(go.Histogram(
                x=recent_returns,
                name="Recent Returns",
                opacity=0.7,
                nbinsx=30,
                histnorm='probability'
            ))
            
            fig.update_layout(
                title="Historical vs Recent Return Distributions",
                xaxis_title="Returns",
                yaxis_title="Probability",
                barmode='overlay'
            )
            
            st.plotly_chart(fig)
            
            # Trading Signals
            st.header("Trading Signals")
            signal_strength = abs(posterior['expected_return']) / posterior['expected_volatility']
            
            if posterior['expected_return'] > 0:
                if signal_strength > 0.5:
                    signal = "Strong Buy"
                else:
                    signal = "Weak Buy"
            else:
                if signal_strength > 0.5:
                    signal = "Strong Sell"
                else:
                    signal = "Weak Sell"
            
            st.write(f"Signal: {signal}")
            st.write(f"Signal Strength: {signal_strength:.2f}")
            st.write(f"95% Credible Interval: [{posterior['return_ci_lower']:.4%}, {posterior['return_ci_upper']:.4%}]")
            
        else:
            st.warning("Insufficient data for Bayesian analysis. Need both historical and recent data.")
            
    else:
        st.error("Failed to fetch data. Please try different parameters.")

if __name__ == "__main__":
    render_financial_markets() 