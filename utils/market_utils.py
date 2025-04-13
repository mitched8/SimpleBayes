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
    Fetch market data for a given symbol and timeframe.
    
    Args:
        symbol: The market symbol to fetch (e.g., "EURUSD=X")
        timeframe: The timeframe for the data (e.g., "1d", "1wk", "1mo")
        start_date: Start date for the data
        end_date: End date for the data
    
    Returns:
        DataFrame containing the market data or None if fetch fails
    """
    try:
        logger.info(f"Fetching data for {symbol} with timeframe {timeframe}")
        logger.info(f"Start date: {start_date}, End date: {end_date}")
        
        # If no dates provided, use default range
        if start_date is None:
            start_date = datetime.now().date() - pd.Timedelta(days=365)
        if end_date is None:
            end_date = datetime.now().date()
        
        # Convert dates to datetime for yfinance
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        logger.info(f"Converted dates - Start: {start_datetime}, End: {end_datetime}")
        
        # Fetch data from Yahoo Finance
        logger.info(f"Creating Ticker object for {symbol}")
        ticker = yf.Ticker(symbol)
        
        logger.info("Fetching history data...")
        data = ticker.history(
            start=start_datetime,
            end=end_datetime,
            interval=timeframe
        )
        
        logger.info(f"Data shape: {data.shape if data is not None else 'None'}")
        
        if data is None:
            logger.error("Data is None")
            return None
            
        if data.empty:
            logger.warning(f"No data found for {symbol} ({timeframe} {start_date} -> {end_date})")
            return None
        
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Data index range: {data.index.min()} to {data.index.max()}")
        
        # Add returns and other basic calculations
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close']).diff()
        
        logger.info("Successfully processed data")
        return data
    
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}", exc_info=True)
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