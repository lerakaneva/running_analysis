# Running Analysis Tool (for Garmin data)

A Python application for analyzing running workout data exported from Garmin devices. This tool generates comprehensive reports and visualizations to help track your training progress and identify trends in your running performance.

## Features

- Calculates weekly running metrics (distance, time, pace, heart rate)
- Creates visualizations for all key metrics
- Generates PDF reports with statistical analysis
- Identifies correlations between heart rate, training volume, and pace
- Supports both English and Russian reports

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Simply run the application with the default settings:

```
python main.py
```

This will analyze the default file `activities.csv` and save results to the `charts` directory in English.

### Advanced Usage

The tool accepts several command-line arguments:

```
python main.py [file_path] [--output-dir OUTPUT_DIR] [--languages LANGUAGES]
```

Arguments:
- `file_path`: Path to the CSV file with running activities data (default: activities.csv)
- `--output-dir`, `-o`: Directory to save all output files (default: charts)
- `--languages`, `-l`: Languages to generate reports in (choices: en, ru, default: en)

Example:
```
python main.py my_activities.csv --output-dir results --languages en ru
```

## Exporting Data from Garmin Connect

To export your running data from Garmin Connect:

1. Go to [Garmin Connect](https://connect.garmin.com/)
2. Click on Activities > All Activities
3. Use filters to show only running activities
4. Click the Export CSV button in the upper right corner

For detailed instructions, see [Garmin's official guide](https://support.garmin.com/en-US/?faq=W1TvTPW8JZ6LfJSfK512Q8).

## Input File Format

The CSV file should contain the following required columns:
- Date
- Distance (km)
- Time (in format HH:MM:SS)
- Avg HR (beats per minute)
- Avg Pace (in format MM:SS)

## Output

The tool generates:
- Weekly summary CSV
- PNG charts for each metric
- PDF with weekly metric charts
- Detailed running analysis PDF with statistical insights

## Acknowledgements

This code was developed with assistance from Claude AI (Anthropic).