"""
Time Utilities Module

This module provides utility functions for converting between different time formats
used in running data analysis, including time strings, pace strings, and seconds.
"""

import pandas as pd


def time_str_to_seconds(time_str):
    """
    Convert time string in format 'HH:MM:SS.S' to total seconds.
    
    Args:
        time_str (str): Time string in format 'H:MM:SS.S'
        
    Returns:
        float: Total time in seconds, or 0 if invalid format
    """
    if pd.isna(time_str):
        return 0
    
    parts = str(time_str).split(":")
    if len(parts) != 3:
        return 0
    
    try:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    except (ValueError, TypeError):
        return 0


def format_seconds_to_time(seconds):
    """
    Format seconds as time string in format 'H:MM:SS.S'.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if pd.isna(seconds) or seconds == 0:
        return "0:00:00.0"
    
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}"


def pace_str_to_seconds(pace_str):
    """
    Convert pace string in format 'MM:SS' to seconds per kilometer/mile.
    
    Args:
        pace_str (str): Pace string in format 'MM:SS'
        
    Returns:
        int: Pace in seconds per distance unit, or 0 if invalid format
    """
    if pd.isna(pace_str):
        return 0
    
    parts = str(pace_str).split(":")
    if len(parts) != 2:
        return 0
    
    try:
        m, s = int(parts[0]), int(parts[1])
        return m * 60 + s
    except (ValueError, TypeError):
        return 0


def format_seconds_to_pace(seconds):
    """
    Format seconds per kilometer/mile as pace string in format 'MM:SS'.
    
    Args:
        seconds (float): Pace in seconds per distance unit
        
    Returns:
        str: Formatted pace string
    """
    if pd.isna(seconds) or seconds == 0:
        return "0:00"
    
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{int(s):02d}"