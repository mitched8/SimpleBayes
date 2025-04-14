import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, Dict, List
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
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch market data from local CSV files.
    
    Args:
        symbol: The market symbol to fetch (e.g., "EURUSD")
        timeframe: The timeframe for the data (e.g., "1d", "1wk", "1mo")
        start_date: Start date for the data
        end_date: End date for the data
    
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
            start_datetime = pd.Timestamp(datetime.combine(start_date, datetime.min.time()))
            data = data[data.index >= start_datetime]
            
        if end_date is not None:
            end_datetime = pd.Timestamp(datetime.combine(end_date, datetime.max.time()))
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