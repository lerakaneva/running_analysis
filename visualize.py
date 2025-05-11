"""
Visualization Module

This module provides functions for creating visualizations of running data metrics,
including distance, pace, heart rate, and time. It supports saving charts as both
PNG files and PDF reports in English and Russian.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
from time_utils import format_seconds_to_time, format_seconds_to_pace


def get_text_dictionary(language="en"):
    """
    Get dictionary of localized text strings for chart generation.
    
    Args:
        language (str): 'en' for English, 'ru' for Russian
        
    Returns:
        dict: Dictionary with localized text strings
    """
    text = {
        "en": {
            "total_distance": "Weekly Distance",
            "run_count": "Number of Runs",
            "total_time_seconds": "Weekly Time",
            "avg_distance_per_run": "Avg Distance/Run",
            "avg_time_seconds_per_run": "Avg Time/Run",
            "Avg_HR": "Average Heart Rate",
            "Avg_Pace_seconds": "Average Pace",
            "ylabel_km": "km",
            "ylabel_runs": "Runs per Week",
            "ylabel_minutes": "min",
            "ylabel_km_per_run": "km",
            "ylabel_min_per_run": "min",
            "ylabel_bpm": "bpm",
            "ylabel_pace": "min/km",
            "all_charts_saved": "All charts saved to PNG in '{0}' and PDF: {1}"
        },
        "ru": {
            "total_distance": "Недельный объём",
            "run_count": "Количество тренировок",
            "total_time_seconds": "Недельное время",
            "avg_distance_per_run": "Средняя дистанция",
            "avg_time_seconds_per_run": "Среднее время тренировки",
            "Avg_HR": "Средний пульс",
            "Avg_Pace_seconds": "Средний темп",
            "ylabel_km": "км",
            "ylabel_runs": "Тренировок в неделю",
            "ylabel_minutes": "мин",
            "ylabel_km_per_run": "км",
            "ylabel_min_per_run": "мин",
            "ylabel_bpm": "уд/мин",
            "ylabel_pace": "мин/км",
            "all_charts_saved": "Все графики сохранены в PNG в папке '{0}' и PDF: {1}"
        }
    }
    
    return text.get(language, text["en"])


def setup_time_axis(ax, index):
    """
    Configure time axis formatting based on date range.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to configure
        index (pd.DatetimeIndex): DatetimeIndex containing the date range
    """
    span = index.max() - index.min()
    ax.xaxis.set_major_locator(
        mdates.MonthLocator() if span.days > 180 else mdates.WeekdayLocator(byweekday=6)
    )
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %Y" if span.days > 180 else "%d %b")
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def plot_metric(
    df, column, title, ylabel, color, formatter=None, invert=False, filename=None
):
    """
    Plot a single metric as a bar chart and save as PNG.
    
    Args:
        df (pd.DataFrame): DataFrame with time series data indexed by date
        column (str): Column name to plot
        title (str): Chart title
        ylabel (str): Y-axis label
        color (str): Bar color
        formatter (callable, optional): Function to format y-axis labels
        invert (bool): Whether to invert the y-axis (for pace)
        filename (str, optional): Output filename for PNG
        
    Returns:
        matplotlib.figure.Figure or None: The figure object if successful, None if column not found
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in data.")
        return None

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="W-SUN")
    data = df.reindex(full_index).copy()
    data["run_count"] = data["run_count"].fillna(0)

    y = data[column] / 60 if column.endswith("_seconds") else data[column]
    bar_width = pd.Timedelta(days=6.5)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(data.index, y, width=bar_width, color=color, edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    setup_time_axis(ax, data.index)

    if formatter:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter))
    if invert:
        ax.invert_yaxis()

    plt.tight_layout()

    # Save as PNG
    if filename:
        fig.savefig(filename)
        print(f"PNG saved: {filename}")

    return fig


def plot_all_metrics(summary, output_dir="charts", language="en"):
    """
    Plot all available metrics and save them as individual PNGs and a combined PDF.
    
    Args:
        summary (pd.DataFrame): DataFrame with weekly running summary data
        output_dir (str): Directory to save output files
        language (str): 'en' for English, 'ru' for Russian
        
    Returns:
        str: Path to the generated PDF file
    """
    # Formatters for time and pace
    def tf(y, pos): return format_seconds_to_time(y * 60)
    def pf(y, pos): return format_seconds_to_pace(y)
    
    # Get localized text
    txt = get_text_dictionary(language)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics with their properties
    metrics = [
        ("total_distance", txt["total_distance"], txt["ylabel_km"], "skyblue", None, False),
        ("run_count", txt["run_count"], txt["ylabel_runs"], "gray", None, False),
        ("total_time_seconds", txt["total_time_seconds"], txt["ylabel_minutes"], "salmon", tf, False),
        ("avg_distance_per_run", txt["avg_distance_per_run"], txt["ylabel_km_per_run"], "green", None, False),
        ("avg_time_seconds_per_run", txt["avg_time_seconds_per_run"], txt["ylabel_min_per_run"], "purple", tf, False),
        ("Avg_HR", txt["Avg_HR"], txt["ylabel_bpm"], "red", None, False),
        ("Avg_Pace_seconds", txt["Avg_Pace_seconds"], txt["ylabel_pace"], "orange", pf, True),
    ]
    
    # Full path for PDF
    pdf_path = os.path.join(output_dir, f"weekly_metrics_{language}.pdf")
    
    # Generate and save all charts
    with PdfPages(pdf_path) as pdf:
        for col, title, ylabel, color, fmt, inv in metrics:
            if col in summary.columns:
                png_filename = os.path.join(output_dir, f"{col}.png")
                fig = plot_metric(
                    summary,
                    column=col,
                    title=title,
                    ylabel=ylabel,
                    color=color,
                    formatter=fmt,
                    invert=inv,
                    filename=png_filename
                )
                if fig:
                    pdf.savefig(fig)
                    plt.close(fig)
    
    print(txt["all_charts_saved"].format(output_dir, pdf_path))
    return pdf_path