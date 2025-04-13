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
    """Test function to verify data fetching works"""
    logger.info("Testing data fetching...")
    
    # Test with EURUSD
    symbol = "EURUSD=X"
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)  # Just 30 days for testing
    
    logger.info(f"Testing with {symbol} from {start_date} to {end_date}")
    
    try:
        # Test direct yfinance fetch
        logger.info("Testing direct yfinance fetch...")
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.max.time()),
            interval="1d"
        )
        logger.info(f"Direct fetch result - Shape: {data.shape if data is not None else 'None'}")
        if data is not None and not data.empty:
            logger.info(f"Data columns: {data.columns.tolist()}")
            logger.info(f"Data index range: {data.index.min()} to {data.index.max()}")
        
        # Test our fetch_market_data function
        logger.info("Testing fetch_market_data function...")
        data = fetch_market_data(
            symbol=symbol,
            timeframe="1d",
            start_date=start_date,
            end_date=end_date
        )
        logger.info(f"fetch_market_data result - Shape: {data.shape if data is not None else 'None'}")
        
        return data is not None and not data.empty
        
    except Exception as e:
        logger.error(f"Error in test_data_fetching: {str(e)}", exc_info=True)
        return False

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
        success = test_data_fetching()
        if success:
            st.success("Data fetching test successful!")
        else:
            st.error("Data fetching test failed. Check logs for details.")
    
    # Data Selection Section
    st.header("Data Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        # Currency pair selection
        currency_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
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