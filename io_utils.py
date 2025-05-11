"""
Input/Output Utilities Module

This module provides functions for loading and saving running data files,
including CSV import with validation and summary report generation.
"""

import os
import pandas as pd
from time_utils import format_seconds_to_time, format_seconds_to_pace


def load_data(file_path: str, required_columns) -> pd.DataFrame:
    """
    Loads a CSV file and verifies that all required columns are present.

    Args:
        file_path (str): Path to the CSV file
        required_columns (list): List of required column names

    Returns:
        pd.DataFrame: A cleaned DataFrame or None if validation fails
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()  # remove accidental whitespace

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        return df

    except Exception as e:
        print(f"Error loading file '{file_path}': {e}")
        return None


def save_summary(df: pd.DataFrame, output_file="weekly_summary.csv", output_dir="."):
    """
    Saves a summary DataFrame to CSV with formatted time and pace columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing running summary data
        output_file (str): Name for the output file
        output_dir (str): Directory to save the output file
        
    Returns:
        str: Path to the saved CSV file
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Format time and pace columns for readability
    df["total_time_formatted"] = df["total_time_seconds"].apply(format_seconds_to_time)
    df["avg_time_per_run_formatted"] = df["avg_time_seconds_per_run"].apply(format_seconds_to_time)
    if "Avg_Pace_seconds" in df.columns:
        df["Avg_Pace_formatted"] = df["Avg_Pace_seconds"].apply(format_seconds_to_pace)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create full path
    output_path = os.path.join(output_dir, output_file)
    
    # Save to CSV
    df.reset_index().to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")
    
    return output_path