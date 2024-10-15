import os
import json
import pandas as pd
from data_processing import prepare_data, establish_in_sample_period_unit, cointegration_testing
from backtester import Backtester, DataLoader, Portfolio, Strategy


def file_exists(file_name: str):
    """
    Check if a file exists in the 'data' directory

    Parameters:
        file_name (str): The name of the file to check for existence.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    file_path = os.path.join(os.getcwd(), 'data', file_name)
    return os.path.exists(file_path)


def load_config(file_name: str):
    """
    Load configuration from a JSON file.

    Parameters:
        file_name (str): The path to the JSON configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path, 'r') as config_file:
        config_x = json.load(config_file)
    return config_x


def run_full_process(raw_data_file: str, formatted_data_file: str, results_file: str, strategy_x: Strategy,
                     portfolio_x: Portfolio):
    """
Run the full data processing and backtesting pipeline.
    This function checks for existing formatted data and cointegration results.
    If found, it loads them; otherwise, it processes the raw data.
    It then runs the backtester using the formatted data and cointegration results.

    Parameters:
        raw_data_file (str): The name of the raw data file to process.
        formatted_data_file (str): The name of the formatted data file to load.
        results_file (str): The name of the cointegration results file to load.
        strategy_x (Strategy): An instance of the Strategy class defining trading strategy parameters.
        portfolio_x (Portfolio): An instance of the Portfolio class defining portfolio parameters.

    Returns:
        None
    """

    # Step 1: Check if formatted data and cointegration results exist
    if file_exists(formatted_data_file) and file_exists(results_file):

        # Establish in-sample period dates
        print("Formatted data and cointegration results found. Skipping data processing...")
        time_series_file = pd.read_csv(os.path.join(os.getcwd(), 'data', formatted_data_file),
                                       index_col='Timestamp', parse_dates=True)
        in_sample_dates = establish_in_sample_period_unit(time_series_file)

    # Step 2: If they don't exist, check if the raw data file is provided
    elif file_exists(raw_data_file):

        # Process the raw data to create a formatted dataset
        print("Raw data file found. Processing data...")
        testing_file = prepare_data(input_file_name=raw_data_file, output_file_name=formatted_data_file)

        # Establish in-sample period dates and perform cointegration testing
        in_sample_dates = establish_in_sample_period_unit(testing_file)
        cointegration_testing(testing_file, in_sample_dates, results_file_name=results_file)

    else:
        print("Required files not found. Ensure you have provided either raw data or preprocessed files.")
        return

    # Step 3: Run the backtester using the cointegration test results and formatted data file
    print("Running backtester...")
    backtester = Backtester(in_sample_dates, DataLoader(formatted_data_file), DataLoader(results_file),
                            strategy_x, portfolio_x)
    backtester.run()


# Entry point for running the full process
if __name__ == "__main__":
    config = load_config('config.json')

    # Initialise parameters from config
    raw_data = config['raw_data']
    formatted_data = config['formatted_data']
    cointegration_data = config['cointegration_data']

    strategy_config = config['strategy']
    strategy = Strategy(
        entry_threshold=strategy_config['entry_threshold'],
        exit_threshold=strategy_config['exit_threshold'],
        limit=strategy_config['limit'],
        size=strategy_config['size'],
        increment=strategy_config['increment']
    )

    portfolio_config = config['portfolio']
    portfolio = Portfolio(
        initial_cash=portfolio_config['initial_cash']
    )

    # Run the full process
    run_full_process(raw_data, formatted_data, cointegration_data, strategy, portfolio)
