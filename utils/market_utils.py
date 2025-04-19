import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, Dict, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .pattern_recognition import (
    detect_doji,
    detect_hammer,
    detect_engulfing,
    detect_morning_star,
    detect_evening_star,
    detect_pin_bar
)

def fetch_market_data(
    symbol: str,
    timeframe: str = "1d",
    start_date: Optional[Union[date, pd.Timestamp]] = None,
    end_date: Optional[Union[date, pd.Timestamp]] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch market data from local CSV files.
    
    Args:
        symbol: The market symbol to fetch (e.g., "EURUSD")
        timeframe: The timeframe for the data (e.g., "1d", "1wk", "1mo")
        start_date: Start date for the data (date or Timestamp)
        end_date: End date for the data (date or Timestamp)
    
    Returns:
        DataFrame containing the market data or None if fetch fails
    """
    try:
        logger.info(f"Fetching data for {symbol} with timeframe {timeframe}")
        
        # Map timeframe to filename suffix
        timeframe_map = {
            "1d": "D1",
            "1h": "H1",
            "1m": "M1",
            "1mo": "MN"
        }
        
        # Remove =X suffix if present
        symbol = symbol.replace("=X", "")
        
        # Construct file path
        file_suffix = timeframe_map.get(timeframe, "D1")  # Default to D1 if timeframe not found
        file_path = f"data/market_data/{symbol}_{file_suffix}.csv"
        
        logger.info(f"Reading data from {file_path}")
        
        # Read CSV with proper column names
        data = pd.read_csv(file_path, 
                          names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread'],
                          parse_dates=['Datetime'])
        
        # Set datetime as index
        data.set_index('Datetime', inplace=True)
        
        # Filter by date range if provided
        if start_date is not None:
            start_datetime = pd.Timestamp(start_date)
            data = data[data.index >= start_datetime]
            
        if end_date is not None:
            end_datetime = pd.Timestamp(end_date)
            data = data[data.index <= end_datetime]
        
        if data.empty:
            logger.warning(f"No data found for {symbol} in date range")
            return None
            
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Data index range: {data.index.min()} to {data.index.max()}")
        
        # Add returns calculations
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close']).diff()
        
        logger.info("Successfully processed data")
        return data
    
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading data for {symbol}: {str(e)}", exc_info=True)
        return None

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various return metrics for the market data.
    
    Args:
        data: DataFrame containing market data
    
    Returns:
        DataFrame with additional return metrics
    """
    if data is None or data.empty:
        logger.warning("Empty or None data provided to calculate_returns")
        return data
    
    # Calculate different types of returns
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close']).diff()
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
    
    return data

def identify_candlestick_patterns(data: pd.DataFrame) -> Dict[str, List[datetime]]:
    """
    Identify common candlestick patterns in the market data.
    
    Args:
        data: DataFrame containing market data
    
    Returns:
        Dictionary mapping pattern names to their occurrence dates
    """
    if data is None or data.empty:
        logger.warning("Empty or None data provided to identify_candlestick_patterns")
        return {}
    
    patterns = {
        'Doji': detect_doji(data),
        'Hammer': detect_hammer(data),
        'Engulfing': detect_engulfing(data),
        'Morning_Star': detect_morning_star(data),
        'Evening_Star': detect_evening_star(data),
        'Pin_Bar': detect_pin_bar(data)
    }
    
    return patterns 

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI technical indicator"""
    close = data['Close']
    delta = close.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> tuple:
    """Calculate Bollinger Bands"""
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    middle_band = typical_price.rolling(window=period).mean()
    std_dev = typical_price.rolling(window=period).std()
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """Calculate MACD indicator"""
    close = data['Close']
    exp1 = close.ewm(span=fast_period, adjust=False).mean()
    exp2 = close.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_moving_averages(data: pd.DataFrame, short_period: int = 20, long_period: int = 50) -> tuple:
    """Calculate Moving Averages"""
    close = data['Close']
    short_ma = close.rolling(window=short_period).mean()
    long_ma = close.rolling(window=long_period).mean()
    
    # Calculate slopes (momentum)
    short_ma_slope = short_ma.diff(5)  # 5-period slope
    long_ma_slope = long_ma.diff(5)
    
    return short_ma, long_ma, short_ma_slope, long_ma_slope

def get_indicator_states(data: pd.DataFrame, 
                        rsi_period: int = 14, 
                        rsi_high: float = 70, 
                        rsi_low: float = 30,
                        bb_period: int = 20, 
                        bb_std: float = 2.0,
                        bb_threshold: float = 0.05,
                        macd_fast: int = 12, 
                        macd_slow: int = 26, 
                        macd_signal: int = 9,
                        ma_short: int = 20, 
                        ma_long: int = 50) -> dict:
    """
    Calculate all technical indicator states
    Returns a dictionary of boolean Series for each state
    """
    # Calculate all indicators
    rsi = calculate_rsi(data, rsi_period)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data, bb_period, bb_std)
    macd_line, signal_line, macd_hist = calculate_macd(data, macd_fast, macd_slow, macd_signal)
    short_ma, long_ma, short_slope, long_slope = calculate_moving_averages(data, ma_short, ma_long)
    
    # Define states
    states = {
        # RSI States
        'rsi_high': rsi > rsi_high,
        'rsi_low': rsi < rsi_low,
        
        # Bollinger Band States
        'bb_upper': (data['Close'] > bb_upper * (1 - bb_threshold)),
        'bb_lower': (data['Close'] < bb_lower * (1 + bb_threshold)),
        
        # MACD States
        'macd_bull': (macd_line > signal_line) & (macd_hist > 0) & (macd_hist > macd_hist.shift(1)),
        'macd_bear': (macd_line < signal_line) & (macd_hist < 0) & (macd_hist < macd_hist.shift(1)),
        
        # Moving Average States
        'ma_above': (data['Close'] > short_ma) & (short_slope > 0),
        'ma_below': (data['Close'] < short_ma) & (short_slope < 0)
    }
    
    return states

def calculate_forward_returns(data: pd.DataFrame, periods: list = [1, 5, 10, 20]) -> pd.DataFrame:
    """Calculate forward returns for multiple periods"""
    returns = pd.DataFrame(index=data.index)
    
    for period in periods:
        returns[f'forward_{period}d'] = (data['Close'].shift(-period) / data['Close'] - 1)
    
    return returns 