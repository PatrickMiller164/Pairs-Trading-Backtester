# Pairs Trading Backtester

## Overview

This repository provides a framework for backtesting statistical arbitrage strategies in financial markets. 
It utilizes cointegration and z-scores to identify trading opportunities between asset pairs. 
The framework is composed of three main modules:

1. **main.py**: Entry point for executing the backtesting process.
2. **data_processing.py**: Responsible for loading and processing the historical price data, 
and for carrying out cointegration testing.
3. **backtester.py**: Contains the core logic for executing trading strategies and managing trades.

## Features

- **Cointegration Analysis**: Identifies pairs of assets that exhibit a long-term statistical relationship.
- **Z-Score Calculation**: Uses z-scores to determine entry and exit points for trades.
- **Trade Execution**: Automates the execution of trades based on specified trading strategies.
- **Performance Metrics**: Tracks and displays performance metrics, including return, # of trades, beta and sharpe ratio

## Getting Started

### Requirements

- Python 3.7 or higher
- Pandas
- NumPy
- Statsmodels
- Joblib
- Itertools

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PatrickMiller164/Pairs-Trading-Backtester.git
   cd pairs-trading-backtester

2. Install the required libraries (if using a virtual environment):
	```bash
	pip install -r requirements.txt

### Usage

To run the backtesting framework, execute the following command:

	python main.py

The application will prompt you to enter the in-sample period unit (i.e., is the in-sample period to be 
measured in months or years. For large time series datasets, it's recommended you choose 'yearly' 
(choosing 'monthly' significantly increases the number of cointegration tests which need to be generated)

File Descriptions

1. main.py

- Purpose: Entry point for the application.

- Functions: Finds, and creates if necessary, the time series data file and the cointegration results file.
Runs the backtest on viable cointegrated pairs and displays the results.

2. data_processing.py

- Purpose: Loads and processes historical price data.
- Functions: Formats the historical price data in readiness for the backtester. Performs cointegration 
testing on all asset pairs from the dataset, on all possible in-sample period ranges 
(either in months or years, as specified by the user).


3. backtester.py

- Purpose: Contains logic for backtesting a pair of cointegrated assets.
- Functions: Contains the logic for executing the trading strategy. Iterates the strategy for all pairs which were shown
to be cointegrated for the in-sample period over the out-of-sample period specified.



Example Workflow

	1.	Load historical price data and cointegration results. If they don't exist, process the raw file first.
	2.	Set parameters for the trading strategy and the initial portfolio.
	3.	Define the in-sample and out-of-sample periods.
	4.	Execute the backtest.
	5.	Review the output metrics and performance summary.


License

This project is licensed under the MIT License - see the LICENSE file for details.