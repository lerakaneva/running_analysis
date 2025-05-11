"""
Running Analysis Module

This module provides tools for analyzing running performance data,
creating statistical models, and generating comprehensive PDF reports.
It supports multiple languages (English and Russian) and includes
interpretations of statistical findings.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import matplotlib.ticker as mticker
from time_utils import pace_str_to_seconds, format_seconds_to_pace
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import textwrap


def get_interpretation(coef, p_value, is_hr=True, language="en"):
    """
    Generate human-readable interpretation of regression coefficients.

    Args:
        coef (float): Coefficient value from regression model
        p_value (float): p-value for statistical significance
        is_hr (bool): Whether this is heart rate (True) or volume (False)
        language (str): 'en' for English, 'ru' for Russian

    Returns:
        str: Human-readable interpretation of the coefficient
    """
    # Set up templates based on language
    templates = {
        'en': {
            'hr_improve': "A 1 BPM higher heart rate is associated with a {:.2f} seconds/km faster pace.",
            'hr_worsen': "A 1 BPM higher heart rate is associated with a {:.2f} seconds/km slower pace.",
            'hr_no_effect': "There is no statistically significant relationship between heart rate and pace.",
            'vol_improve': "A 2.5 km higher weekly volume is associated with a {:.2f} seconds/km faster pace at the same heart rate.",
            'vol_worsen': "A 2.5 km higher weekly volume is associated with a {:.2f} seconds/km slower pace at the same heart rate.",
            'vol_no_effect': "There is no statistically significant relationship between training volume and pace."
        },
        'ru': {
            'hr_improve': "Более высокий пульс на 1 удар/мин связан с более быстрым темпом на {:.2f} сек/км.",
            'hr_worsen': "Более высокий пульс на 1 удар/мин связан с более медленным темпом на {:.2f} сек/км.",
            'hr_no_effect': "Статистически значимой связи между пульсом и темпом не обнаружено.",
            'vol_improve': "Более высокий недельный объем на 2.5 км связан с более быстрым темпом на {:.2f} сек/км при том же пульсе.",
            'vol_worsen': "Более высокий недельный объем на 2.5 км связан с более медленным темпом на {:.2f} сек/км при том же пульсе.",
            'vol_no_effect': "Статистически значимой связи между объемом тренировок и темпом не обнаружено."
        }
    }

    # Get the correct language templates
    lang = templates.get(language, templates["en"])

    if p_value < 0.05:  # Statistically significant
        if is_hr:  # Heart rate
            if coef < 0:
                return lang["hr_improve"].format(abs(coef))
            else:
                return lang["hr_worsen"].format(coef)
        else:  # Volume - convert from 30-day effect to weekly effect (divide by 4)
            # For volume, we show effect per 2.5 km weekly (10 km / 4 weeks)
            if coef < 0:
                # 30-day coefficient * 2.5 = effect of 2.5 km weekly increase
                return lang["vol_improve"].format(abs(coef * 2.5))
            else:
                return lang["vol_worsen"].format(coef * 2.5)
    else:  # Not statistically significant
        if is_hr:
            return lang["hr_no_effect"]
        else:
            return lang["vol_no_effect"]


def get_text_dictionary(language="en"):
    """
    Get dictionary of localized text strings for report generation.
    
    Args:
        language (str): 'en' for English, 'ru' for Russian
        
    Returns:
        dict: Dictionary with localized text strings
    """
    text = {
        "en": {
            "title": "Running Form Analysis",
            "regression_title": "Regression Analysis: Pace vs Heart Rate and Training Volume",
            "correlation_title": "Correlation Analysis",
            "hr_pace_title": "Pace vs Heart Rate",
            "volume_pace_title": "Pace vs 30-day Volume",
            "hr_volume_title": "Heart Rate and Training Volume Relationship",
            "contour_title": "Pace Zones by Heart Rate and Training Volume",
            "heatmap_title": "Average Pace by Heart Rate and Volume Ranges",
            "regression_summary": "Regression Summary",
            "correlation_coef": "Correlation coefficient between HR and training volume:",
            "model_warning": "Model Warnings:",
            "volume_coef": "Volume coefficient (30-day):",
            "hr_coef": "Heart rate coefficient:",
            "r_squared": "R-squared:",
            "p_values": "P-values:",
        },
        "ru": {
            "title": "Анализ беговой формы",
            "regression_title": "Регрессионный анализ: Темп и пульс, объем тренировок",
            "correlation_title": "Корреляционный анализ",
            "hr_pace_title": "Темп и пульс",
            "volume_pace_title": "Темп и объем тренировок (30 дней)",
            "hr_volume_title": "Взаимосвязь пульса и объема тренировок",
            "contour_title": "Зоны темпа по пульсу и объему тренировок",
            "heatmap_title": "Средний темп по диапазонам пульса и объема",
            "regression_summary": "Сводка регрессии",
            "correlation_coef": "Коэффициент корреляции между пульсом и объемом тренировок:",
            "model_warning": "Предупреждения модели:",
            "volume_coef": "Коэффициент объема (30 дней):",
            "hr_coef": "Коэффициент пульса:",
            "r_squared": "R-квадрат:",
            "p_values": "P-значения:",
        },
    }
    
    return text.get(language, text["en"])


def prepare_running_data(df):
    """
    Prepare and clean running data for analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with running data
        
    Returns:
        pd.DataFrame: Cleaned and prepared DataFrame with additional metrics
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure date column is properly formatted
    if df.index.name == "Date":
        df = df.reset_index()
    
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    
    # Check required columns
    required = ["Avg_Pace_seconds", "Avg HR", "Distance"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Clean data
    df = df[required + ["Date"]].dropna()
    df = df[(df["Avg_Pace_seconds"] > 0) & (df["Avg HR"] > 0) & (df["Distance"] > 0)]
    
    # Calculate accumulated volume (last 30 days for each run)
    df["distance_past_30d"] = [
        df[
            (df["Date"] < row["Date"])
            & (df["Date"] >= row["Date"] - pd.Timedelta(days=30))
        ]["Distance"].sum()
        for _, row in df.iterrows()
    ]
    
    # Keep only runs with training history
    df = df[df["distance_past_30d"] > 0]
    
    return df


def build_regression_model(df):
    """
    Build and fit a regression model predicting pace from HR and volume.
    
    Args:
        df (pd.DataFrame): Prepared DataFrame with running data
        
    Returns:
        tuple: (model, correlation) - fitted model and correlation matrix
    """
    # Create regression model
    X = df[["Avg HR", "distance_past_30d"]]
    y = df["Avg_Pace_seconds"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    # Calculate correlation between HR and volume
    corr = df[["Avg HR", "distance_past_30d"]].corr(method="pearson")
    
    return model, corr


def create_title_page(model, corr, language="en"):
    """
    Create the title page with regression summary for the PDF report.
    
    Args:
        model: Fitted statsmodels regression model
        corr: Correlation matrix
        language (str): 'en' for English, 'ru' for Russian
        
    Returns:
        matplotlib.figure.Figure: Figure for the title page
    """
    txt = get_text_dictionary(language)
    
    # Create figure
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    
    # Title
    plt.text(0.5, 0.97, txt["title"], ha="center", fontsize=24, fontweight="bold")
    plt.text(0.5, 0.92, f"{txt['regression_summary']}", ha="center", fontsize=18)
    
    # Model coefficients and statistics
    hr_coef = round(model.params["Avg HR"], 4)
    volume_coef = round(model.params["distance_past_30d"], 4)
    hr_p = round(model.pvalues["Avg HR"], 4)
    volume_p = round(model.pvalues["distance_past_30d"], 4)
    r_squared = round(model.rsquared, 4)
    corr_value = round(corr.iloc[0, 1], 4)
    
    # Get interpretations
    hr_interpretation = get_interpretation(
        hr_coef, hr_p, is_hr=True, language=language
    )
    volume_interpretation = get_interpretation(
        volume_coef, volume_p, is_hr=False, language=language
    )
    
    # Model warnings
    extra_txt = "None"
    if hasattr(model.summary(), "extra_txt"):
        extra_txt = model.summary().extra_txt
        
        # Clean up the extra_txt to make it more readable
        extra_txt = extra_txt.strip()
        # Remove extra newlines and keep formatting simple
        if "[1]" in extra_txt and "[2]" in extra_txt:
            lines = extra_txt.split("\n")
            cleaned_lines = []
            for line in lines:
                if line.strip() and not line.strip().startswith("Notes:"):
                    cleaned_lines.append(line.strip())
            extra_txt = "\n".join(cleaned_lines)
    
    # Format text for summary page with better width control
    stats_text = f"{txt['hr_coef']} {hr_coef} (p={hr_p})\n"
    
    # Add interpretation with text wrapping for better fit
    hr_lines = textwrap.fill(hr_interpretation, width=70).split("\n")
    for line in hr_lines:
        stats_text += line + "\n"
    
    stats_text += f"\n{txt['volume_coef']} {volume_coef} (p={volume_p})\n"
    
    # Add interpretation with text wrapping for better fit
    vol_lines = textwrap.fill(volume_interpretation, width=70).split("\n")
    for line in vol_lines:
        stats_text += line + "\n"
    
    stats_text += f"\n{txt['r_squared']} {r_squared}\n\n"
    stats_text += f"{txt['correlation_title']}\n"
    stats_text += f"{txt['correlation_coef']} {corr_value}\n\n"
    
    # Add warnings with text wrapping for better fit
    warnings_text = f"{txt['model_warning']}\n"
    if extra_txt != "None":
        for line in textwrap.fill(extra_txt, width=80).split("\n"):
            warnings_text += line + "\n"
    else:
        warnings_text += "None"
    
    # Place text on the page
    plt.text(0.1, 0.85, stats_text, va="top", fontsize=12)
    plt.text(0.1, 0.35, warnings_text, va="top", fontsize=10)
    
    # Date information
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    plt.text(0.5, 0.05, f"Generated: {date_str}", ha="center", fontsize=10)
    
    return fig


def create_basic_scatter_plots(df, language="en"):
    """
    Create basic scatter plots comparing pace with HR and volume.
    
    Args:
        df (pd.DataFrame): DataFrame with running data
        language (str): 'en' for English, 'ru' for Russian
        
    Returns:
        matplotlib.figure.Figure: Figure with scatter plots
    """
    txt = get_text_dictionary(language)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Pace vs HR plot
    sns.scatterplot(data=df, x="Avg HR", y="Avg_Pace_seconds", ax=axes[0])
    axes[0].set_title(txt["hr_pace_title"])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Avg Pace (sec/km)")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: format_seconds_to_pace(y))
    )
    
    # Pace vs Volume plot
    sns.scatterplot(
        data=df, x="distance_past_30d", y="Avg_Pace_seconds", ax=axes[1]
    )
    axes[1].set_title(txt["volume_pace_title"])
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Avg Pace (sec/km)")
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: format_seconds_to_pace(y))
    )
    
    plt.tight_layout()
    return fig


def create_hr_volume_scatter(df, language="en"):
    """
    Create HR vs volume scatter plot with pace colors.
    
    Args:
        df (pd.DataFrame): DataFrame with running data
        language (str): 'en' for English, 'ru' for Russian
        
    Returns:
        matplotlib.figure.Figure: Figure with scatter plot
    """
    txt = get_text_dictionary(language)
    
    fig = plt.figure(figsize=(11, 8))
    scatter = plt.scatter(
        df["Avg HR"],
        df["distance_past_30d"],
        c=df["Avg_Pace_seconds"],
        cmap="plasma",
        edgecolor="k",
        alpha=0.8,
        s=80,
    )
    plt.xlabel("Average Heart Rate (bpm)")
    plt.ylabel("Distance Last 30 Days (km)")
    
    # Create colorbar with pace formatter
    cbar = plt.colorbar(
        scatter,
        format=mticker.FuncFormatter(lambda v, pos: format_seconds_to_pace(v)),
    )
    cbar.set_label("Pace (min/km)")
    
    plt.title(txt["hr_volume_title"])
    plt.tight_layout()
    
    return fig


def create_contour_plot(df, language="en"):
    """
    Create contour plot showing pace zones by HR and volume.
    
    Args:
        df (pd.DataFrame): DataFrame with running data
        language (str): 'en' for English, 'ru' for Russian
        
    Returns:
        matplotlib.figure.Figure: Figure with contour plot
    """
    txt = get_text_dictionary(language)
    
    fig = plt.figure(figsize=(11, 8))
    
    # Create contour plot for pace zones
    tri = plt.tricontourf(
        df["Avg HR"],
        df["distance_past_30d"],
        df["Avg_Pace_seconds"],
        levels=30,
        cmap="plasma",
    )
    
    # Overlay scatter points
    plt.scatter(
        df["Avg HR"],
        df["distance_past_30d"],
        c=df["Avg_Pace_seconds"],
        cmap="plasma",
        edgecolor="k",
        s=50,
        alpha=0.5,
    )
    
    plt.xlabel("Avg HR (bpm)")
    plt.ylabel("Distance (last 30 days, km)")
    plt.title(txt["contour_title"])
    
    # Create colorbar with pace formatter
    cbar = plt.colorbar(
        tri, format=mticker.FuncFormatter(lambda v, pos: format_seconds_to_pace(v))
    )
    cbar.set_label("Avg Pace (min/km)")
    
    plt.tight_layout()
    
    return fig


def create_heatmap(df, language="en"):
    """
    Create binned heatmap of pace by HR and volume ranges.
    
    Args:
        df (pd.DataFrame): DataFrame with running data
        language (str): 'en' for English, 'ru' for Russian
        
    Returns:
        matplotlib.figure.Figure: Figure with heatmap
    """
    txt = get_text_dictionary(language)
    
    # Create bins for HR and volume
    df = df.copy()
    df["HR_bin"] = pd.cut(df["Avg HR"], bins=np.arange(120, 190, 5))
    df["Volume_bin"] = pd.cut(df["distance_past_30d"], bins=np.arange(0, 130, 10))
    
    # Create pivot table with binned data
    pivot = df.pivot_table(
        index="Volume_bin",
        columns="HR_bin",
        values="Avg_Pace_seconds",
        aggfunc="mean",
        observed=False,
    )
    
    fig = plt.figure(figsize=(11, 8))
    
    # Create heatmap with pace values
    sns.heatmap(
        pivot,
        annot=True,
        fmt="",
        cmap="plasma_r",
        linewidths=0.5,
        cbar_kws={"label": "Avg Pace (min/km)"},
        annot_kws={"size": 9},
    )
    
    # Convert numeric pace values to min:sec format
    for t in plt.gca().texts:
        try:
            val = float(t.get_text())
            t.set_text(format_seconds_to_pace(val))
        except:
            pass
    
    plt.title(txt["heatmap_title"])
    plt.xlabel("Avg HR (bpm)")
    plt.ylabel("30-day Volume (km)")
    plt.tight_layout()
    
    return fig


def save_pdf_report(figures, output_path):
    """
    Save a list of figures to a PDF report.
    
    Args:
        figures (list): List of matplotlib figures to include in the PDF
        output_path (str): Path to save the PDF file
        
    Returns:
        str: Path to the saved PDF file
    """
    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    
    return output_path


def analyze_running_form(df, output_pdf="running_analysis.pdf", language="en", output_dir="charts"):
    """
    Analyzes running form progress and creates a PDF report with charts and statistical analysis.

    Args:
        df (pd.DataFrame): DataFrame with running data containing ['Date', 'Avg_Pace_seconds', 'Avg HR', 'Distance']
        output_pdf (str): Name for the output PDF file
        language (str): 'en' for English, 'ru' for Russian
        output_dir (str): Directory to save the output PDF and charts

    Returns:
        str: Path to the generated PDF report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    df_prepared = prepare_running_data(df)
    
    # Build statistical model
    model, corr = build_regression_model(df_prepared)
    
    # Create figures for the report
    figures = [
        create_title_page(model, corr, language),
        create_basic_scatter_plots(df_prepared, language),
        create_hr_volume_scatter(df_prepared, language),
        create_contour_plot(df_prepared, language),
        create_heatmap(df_prepared, language)
    ]
    
    # Save figures to PDF
    pdf_path = os.path.join(output_dir, output_pdf)
    save_pdf_report(figures, pdf_path)
    
    print(f"Analysis PDF saved to: {pdf_path}")
    return pdf_path


def save_running_analysis_pdf(df, language="en", output_dir="charts"):
    """
    Analyze running data and save results to PDF.
    This function can be called from main.py.

    Args:
        df (pd.DataFrame): DataFrame with running data
        language (str): 'en' or 'ru'
        output_dir (str): Directory to save the output PDF and charts

    Returns:
        str: Path to generated PDF
    """
    filename = f"running_analysis_{language}.pdf"
    return analyze_running_form(df, filename, language, output_dir)


