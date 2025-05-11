"""
Running Analysis Tool

Entry point for analyzing running data and generating reports in multiple languages.
"""

import argparse
import os
from analysis import save_running_analysis_pdf
from io_utils import load_data, save_summary
from transform import transform_data, create_weekly_summary, REQUIRED_COLUMNS
from visualize import plot_all_metrics


def main(
    file_path: str,
    output_dir: str = "output",
    languages: list = ["en"],
):
    """
    Main function to analyze running data and generate reports.
    
    Args:
        file_path (str): Path to the input CSV file with running data
        output_dir (str): Directory to save all output files
        languages (list): List of language codes to generate reports in ("en", "ru")
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and validate data
    df = load_data(file_path, required_columns=REQUIRED_COLUMNS)
    if df is None:
        print(f"Failed to load data from {file_path}")
        return False
    
    # Transform data and create weekly summary
    transformed_df = transform_data(df)
    summary = create_weekly_summary(transformed_df)
    
    # Save summary to CSV
    save_summary(summary, output_dir=output_dir)
    
    # Generate reports for each requested language
    for language in languages:
        # Generate weekly metrics charts
        plot_all_metrics(summary, output_dir=output_dir, language=language)
        
        # Generate detailed running analysis PDF
        save_running_analysis_pdf(transformed_df, language=language, output_dir=output_dir)
        
        print(f"Generated {language} reports in {output_dir}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Analysis Tool")
    parser.add_argument(
        "file_path",
        nargs="?",
        default="activities.csv",
        help="Path to the CSV file with running activities data (default: activities.csv)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="charts",
        help="Directory to save all output files (default: charts)"
    )
    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        choices=["en", "ru"],
        default=["en"],
        help="Languages to generate reports in (default: en)"
    )
    
    args = parser.parse_args()
    
    success = main(
        file_path=args.file_path,
        output_dir=args.output_dir,
        languages=args.languages
    )
    
    if success:
        print(f"Analysis complete. Results saved to {args.output_dir}")
    else:
        print("Analysis failed. Please check the errors above.")