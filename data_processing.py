import pandas as pd
import numpy as np
import os
from itertools import combinations, permutations
from statsmodels.tsa.stattools import coint
from joblib import Parallel, delayed
pd.set_option('display.max_rows', False)


def prepare_data(input_file_name: str, output_file_name: str):
    """
    Prepare and format the time series data from a CSV file.

    This function reads a CSV file, renames the first column to 'Timestamp',
    sets it as the index, converts the index to datetime format, and renames the
    last column to 'Benchmark'. It also exports the formatted data to a new CSV file
    while returning the DataFrame without the benchmark column.

    Parameters:
        input_file_name (str): The name of the input CSV file containing raw data.
        output_file_name (str): The name of the output CSV file for saving formatted data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the time series data without the benchmark column.
    """
    file_path = os.path.join(os.getcwd(), 'data')
    x = os.path.join(file_path, input_file_name)
    df = pd.read_csv(x)

    # Format and export time series data
    df.rename(columns={df.columns[0]: 'Timestamp'}, inplace=True)
    df.set_index('Timestamp', inplace=True)
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df.rename(columns={df.columns[-1]: 'Benchmark'}, inplace=True)

    x = os.path.join(file_path, output_file_name)
    df.to_csv(x, index=True)

    # Preparing df for statistical testing
    df_no_benchmark = df.drop(df.columns[-1], axis=1)
    return df_no_benchmark


def establish_in_sample_period_unit(df: pd.DataFrame):
    """
    Establish the in-sample period based on user-defined time units.

    The user is prompted to specify the unit of time (either 'yearly' or 'monthly').
    The function returns the starting dates of each period based on the input.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing time series data indexed by timestamp.

    Returns:
        list: A list of starting dates for the specified time units (yearly or monthly), or None if the input is invalid.
    """
    length = input("What unit of time is the in-sample period? (e.g., yearly, monthly). Answer: ")
    if length == "yearly":
        start_of_years = df.groupby(df.index.year).apply(lambda x: x.index.min())
        start_of_years = start_of_years.to_list()
        return start_of_years

    elif length == "monthly":
        start_of_months = df.groupby([df.index.year, df.index.month]).apply(lambda x: x.index.min())
        start_of_months = start_of_months.to_list()
        return start_of_months

    else:
        print("This input is not valid. Input must be either 'yearly' or 'monthly'")
        return


def cointegration_testing(df: pd.DataFrame, dates: list, results_file_name: str):
    """
    Perform cointegration testing on asset pairs within specified in-sample periods.

    This function generates all possible asset pairs from the DataFrame and tests
    for cointegration within the defined in-sample periods. Results are saved to a CSV file.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing time series data, with assets as columns
                           and timestamps as the index.
        dates (list): A list of date ranges defining the in-sample periods, where each
                      date range should be a tuple of (start_date, end_date).
        results_file_name (str): The name of the output CSV file for saving cointegration results.

    Returns:
        None
    """
    results = []
    in_sample_periods = list(combinations(dates, 2))

    for i, (start_month, end_month) in enumerate(in_sample_periods, start=1):
        print(f"{i}/{len(in_sample_periods)}")

        in_sample_period = df.loc[(df.index >= start_month) & (df.index <= end_month)]
        assets = df.columns
        asset_pairs = list(permutations(assets, 2))

        results_list = Parallel(n_jobs=-1)(
            delayed(statistical_testing)(in_sample_period, pair[0], pair[1]) for pair in asset_pairs
        )

        for y, x, p_value, correlation in results_list:
            if p_value < 0.05:
                x = {'start_month': start_month.strftime('%Y-%m-%d %H:%M:%S'),
                     'end_month': end_month.strftime('%Y-%m-%d %H:%M:%S'),
                     'y': y,
                     'x': x,
                     'p_value': p_value,
                     'correlation': correlation}
                results.append(x)

    results_df = pd.DataFrame(results)

    file_path = os.path.join(os.getcwd(), 'data')
    x = os.path.join(file_path, results_file_name)
    results_df.to_csv(x, index=False)


def statistical_testing(in_sample_period: pd.DataFrame, y: str, x: str):
    """
    Conduct statistical testing for cointegration between two asset time series.

    This function calculates the correlation and performs the cointegration test
    between two specified time series from the in-sample period.

    Parameters:
        in_sample_period (pd.DataFrame): A pandas DataFrame containing the in-sample data,
                                          where each column represents an asset and rows correspond to timestamps.
        y (str): The name of the first asset (dependent variable).
        x (str): The name of the second asset (independent variable).

    Returns:
        tuple: A tuple containing:
            - y (str): The name of the first asset.
            - x (str): The name of the second asset.
            - p_value (float): The p-value from the cointegration test.
            - corr (float): The correlation coefficient between the two assets.
    """
    corr = np.corrcoef(in_sample_period[y], in_sample_period[x])[0, 1]
    p = coint(in_sample_period[y], in_sample_period[x])
    if isinstance(p, tuple):
        return y, x, p[1], corr
