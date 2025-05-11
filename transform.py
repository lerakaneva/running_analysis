"""
Data Transformation Module

This module provides functions for transforming and aggregating running data,
including conversion of time/pace values and creating weekly summaries.
"""

import pandas as pd
import numpy as np
from time_utils import time_str_to_seconds, pace_str_to_seconds

# Columns that must be present in the input CSV file
REQUIRED_COLUMNS = ["Date", "Distance", "Time", "Avg HR", "Avg Pace"]


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw running data into a format suitable for analysis.
    
    This function converts string time/pace values to seconds, sets the date
    as index, and removes redundant columns.
    
    Args:
        df (pd.DataFrame): Raw running data DataFrame
        
    Returns:
        pd.DataFrame: Transformed DataFrame with calculated metrics
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date and time columns
    df["Date"] = pd.to_datetime(df["Date"])
    df["Time_seconds"] = df["Time"].apply(time_str_to_seconds)
    df["Avg_Pace_seconds"] = df["Avg Pace"].apply(pace_str_to_seconds)
    df["Avg_Pace_to_HR"] = df["Avg_Pace_seconds"].div(df["Avg HR"].replace(0, np.nan)).fillna(0)
    
    # Set date as index and remove original time columns
    df.set_index("Date", inplace=True)
    df.drop(columns=["Time", "Avg Pace"], inplace=True)
    
    return df


def create_weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a weekly summary of running data, aggregating by week ending on Sunday.
    
    Args:
        df (pd.DataFrame): DataFrame with running data indexed by date
        
    Returns:
        pd.DataFrame: Weekly summary with aggregated metrics
    """
    # Resample by week and aggregate metrics
    weekly = df.resample("W-SUN").agg({
        "Distance": "sum",
        "Time_seconds": "sum",
        "Avg_Pace_seconds": "mean",
        "Avg HR": "mean",
        "Avg_Pace_to_HR": "mean"
    }).rename(columns={
        "Distance": "total_distance", 
        "Time_seconds": "total_time_seconds", 
        "Avg HR": "Avg_HR"
    })

    # Calculate derived metrics
    weekly["run_count"] = df.resample("W-SUN").size()
    weekly["avg_distance_per_run"] = (
        weekly["total_distance"] / weekly["run_count"].replace(0, np.nan)
    )
    weekly["avg_time_seconds_per_run"] = (
        weekly["total_time_seconds"] / weekly["run_count"].replace(0, np.nan)
    )
    
    # Fill missing values with zeros and return
    return weekly.fillna(0)