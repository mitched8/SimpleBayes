import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime

def detect_doji(data: pd.DataFrame, threshold: float = 0.1) -> List[datetime]:
    """
    Detect Doji candlestick patterns.
    A Doji occurs when the open and close prices are very close to each other.
    
    Args:
        data: DataFrame containing OHLCV data
        threshold: Maximum ratio of body to total range to be considered a Doji
    
    Returns:
        List of dates where Doji patterns were detected
    """
    if data is None or data.empty:
        return []
    
    doji_dates = []
    
    for idx in range(len(data)):
        row = data.iloc[idx]
        body_size = abs(row['Close'] - row['Open'])
        total_range = row['High'] - row['Low']
        
        if total_range > 0 and (body_size / total_range) < threshold:
            doji_dates.append(data.index[idx])
    
    return doji_dates

def detect_hammer(data: pd.DataFrame, min_body_ratio: float = 0.3) -> List[datetime]:
    """
    Detect Hammer candlestick patterns.
    A Hammer has a small body and a long lower shadow.
    
    Args:
        data: DataFrame containing OHLCV data
        min_body_ratio: Minimum ratio of body to total range
    
    Returns:
        List of dates where Hammer patterns were detected
    """
    if data is None or data.empty:
        return []
    
    hammer_dates = []
    
    for idx in range(len(data)):
        row = data.iloc[idx]
        body_size = abs(row['Close'] - row['Open'])
        total_range = row['High'] - row['Low']
        lower_shadow = min(row['Open'], row['Close']) - row['Low']
        upper_shadow = row['High'] - max(row['Open'], row['Close'])
        
        if (total_range > 0 and 
            body_size / total_range >= min_body_ratio and
            lower_shadow > 2 * body_size and
            upper_shadow < body_size):
            hammer_dates.append(data.index[idx])
    
    return hammer_dates

def detect_engulfing(data: pd.DataFrame) -> List[datetime]:
    """
    Detect Engulfing candlestick patterns.
    An Engulfing pattern occurs when a candle's body completely engulfs the previous candle's body.
    
    Args:
        data: DataFrame containing OHLCV data
    
    Returns:
        List of dates where Engulfing patterns were detected
    """
    if data is None or data.empty or len(data) < 2:
        return []
    
    engulfing_dates = []
    
    for idx in range(1, len(data)):
        prev_row = data.iloc[idx-1]
        curr_row = data.iloc[idx]
        
        # Bullish engulfing
        if (prev_row['Close'] < prev_row['Open'] and  # Previous candle is bearish
            curr_row['Open'] < prev_row['Close'] and  # Current open below previous close
            curr_row['Close'] > prev_row['Open']):    # Current close above previous open
            engulfing_dates.append(data.index[idx])
        
        # Bearish engulfing
        elif (prev_row['Close'] > prev_row['Open'] and  # Previous candle is bullish
              curr_row['Open'] > prev_row['Close'] and  # Current open above previous close
              curr_row['Close'] < prev_row['Open']):    # Current close below previous open
            engulfing_dates.append(data.index[idx])
    
    return engulfing_dates

def detect_morning_star(data: pd.DataFrame) -> List[datetime]:
    """
    Detect Morning Star candlestick patterns.
    A Morning Star is a three-candle pattern that signals a potential bullish reversal.
    
    Args:
        data: DataFrame containing OHLCV data
    
    Returns:
        List of dates where Morning Star patterns were detected
    """
    if data is None or data.empty or len(data) < 3:
        return []
    
    morning_star_dates = []
    
    for idx in range(2, len(data)):
        first = data.iloc[idx-2]
        second = data.iloc[idx-1]
        third = data.iloc[idx]
        
        # Check for morning star pattern
        if (first['Close'] < first['Open'] and  # First candle is bearish
            second['Open'] < first['Close'] and  # Second candle gaps down
            second['Close'] < second['Open'] and  # Second candle is bearish
            third['Open'] > second['Close'] and  # Third candle gaps up
            third['Close'] > third['Open'] and   # Third candle is bullish
            third['Close'] > (first['Open'] + first['Close']) / 2):  # Third candle closes above midpoint of first
            morning_star_dates.append(data.index[idx])
    
    return morning_star_dates

def detect_evening_star(data: pd.DataFrame) -> List[datetime]:
    """
    Detect Evening Star candlestick patterns.
    An Evening Star is a three-candle pattern that signals a potential bearish reversal.
    
    Args:
        data: DataFrame containing OHLCV data
    
    Returns:
        List of dates where Evening Star patterns were detected
    """
    if data is None or data.empty or len(data) < 3:
        return []
    
    evening_star_dates = []
    
    for idx in range(2, len(data)):
        first = data.iloc[idx-2]
        second = data.iloc[idx-1]
        third = data.iloc[idx]
        
        # Check for evening star pattern
        if (first['Close'] > first['Open'] and  # First candle is bullish
            second['Open'] > first['Close'] and  # Second candle gaps up
            second['Close'] > second['Open'] and  # Second candle is bullish
            third['Open'] < second['Close'] and  # Third candle gaps down
            third['Close'] < third['Open'] and   # Third candle is bearish
            third['Close'] < (first['Open'] + first['Close']) / 2):  # Third candle closes below midpoint of first
            evening_star_dates.append(data.index[idx])
    
    return evening_star_dates

def detect_pin_bar(data: pd.DataFrame, min_body_ratio: float = 0.3) -> List[datetime]:
    """
    Detect Pin Bar candlestick patterns.
    A Pin Bar has a small body and a long tail in one direction.
    
    Args:
        data: DataFrame containing OHLCV data
        min_body_ratio: Minimum ratio of body to total range
    
    Returns:
        List of dates where Pin Bar patterns were detected
    """
    if data is None or data.empty:
        return []
    
    pin_bar_dates = []
    
    for idx in range(len(data)):
        row = data.iloc[idx]
        body_size = abs(row['Close'] - row['Open'])
        total_range = row['High'] - row['Low']
        lower_shadow = min(row['Open'], row['Close']) - row['Low']
        upper_shadow = row['High'] - max(row['Open'], row['Close'])
        
        if total_range > 0:
            # Bullish pin bar
            if (body_size / total_range >= min_body_ratio and
                lower_shadow > 2 * body_size and
                upper_shadow < body_size):
                pin_bar_dates.append(data.index[idx])
            
            # Bearish pin bar
            elif (body_size / total_range >= min_body_ratio and
                  upper_shadow > 2 * body_size and
                  lower_shadow < body_size):
                pin_bar_dates.append(data.index[idx])
    
    return pin_bar_dates 