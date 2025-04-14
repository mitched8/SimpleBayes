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

def render_financial_markets():
    """
    Main function to render the financial markets analysis page.
    """
    st.title("Financial Markets Bayesian Analysis")
    
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
        # Date range selection
        end_date = datetime.now().date() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=30)  # Just 30 days for testing
        
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "Start Date",
                value=start_date,
                max_value=end_date
            )
        with col_end:
            end_date = st.date_input(
                "End Date",
                value=end_date,
                min_value=start_date
            )
    
    # Fetch and display data
    try:
        logger.info(f"Fetching data for {selected_pair} from {start_date} to {end_date}")
        data = fetch_market_data(
            symbol=selected_pair,
            timeframe=selected_timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if data is not None and not data.empty:
            logger.info(f"Data fetched successfully. Shape: {data.shape}")
            
            # Calculate technical indicators using TA-Lib
            data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
            data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
            data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(
                data['Close'],
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            
            # Display basic data information
            st.subheader("Market Data Overview")
            st.dataframe(data.head())
            
            # Display basic statistics
            st.subheader("Basic Statistics")
            st.dataframe(data.describe())
            
            # Display candlestick chart with technical indicators
            st.subheader("Price Chart")
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
            
            # Add SMA
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange')
            ))
            
            fig.update_layout(
                title=f"{selected_pair} Price Chart",
                yaxis_title="Price",
                xaxis_title="Date"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display technical indicators
            st.subheader("Technical Indicators")
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name='RSI'
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI (14)")
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                # MACD Chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    name='MACD'
                ))
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    name='Signal'
                ))
                fig_macd.add_trace(go.Bar(
                    x=data.index,
                    y=data['MACD_Hist'],
                    name='Histogram'
                ))
                fig_macd.update_layout(title="MACD")
                st.plotly_chart(fig_macd, use_container_width=True)
            
            # Pattern Recognition Section
            st.header("Pattern Recognition")
            patterns = identify_candlestick_patterns(data)
            
            # Display detected patterns
            st.subheader("Detected Patterns")
            for pattern, dates in patterns.items():
                if dates:
                    st.write(f"{pattern}: {len(dates)} occurrences")
                    st.write(f"Last occurrence: {dates[-1]}")
            
            # Bayesian Analysis Section
            st.header("Bayesian Analysis")
            # TODO: Implement Bayesian analysis components
            
        else:
            logger.warning(f"No data available for {selected_pair}")
            st.error("No data available for the selected parameters.")
            
    except Exception as e:
        logger.error(f"Error in render_financial_markets: {str(e)}", exc_info=True)
        st.error(f"Error fetching data: {str(e)}")

if __name__ == "__main__":
    render_financial_markets() 