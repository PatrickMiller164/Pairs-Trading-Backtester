# Pairs Trading Backtester

## Overview

This repository provides a comprehensive framework for backtesting statistical arbitrage pairs trading strategies. 
Pairs trading relies on the notion that one can profit from the short-term mispricing of two highly correlated assets that are cointegrated, meaning a long-run relationship exists between them. 
The framework utilizes cointegration analysis to identify suitable pairs and z-scores to identify trading opportunities, while automating the execution of trades based on specified trading strategies, including risk management and position sizing. 
Additionally, it tracks and displays key performance metrics, such as total return, number of trades, beta, and Sharpe ratio, providing insights into the effectiveness of the trading strategy.

## How Pairs Trading Works
The primary objective of pairs trading is to exploit the cointegrated nature of an asset pair by executing trades when the short-term spread between the two assets diverges from the long-run average. The strategy involves going long on the undervalued asset while simultaneously shorting the overvalued asset. This approach aims to generate profit as the prices of the assets revert back to their long-run mean. Because the profits are derived from the relative performance of the two assets rather than the overall market direction, this strategy is considered market-neutral and theoretically has zero exposure to systematic risk.

Cointegration Testing: The cointegration relationship is tested during the in-sample period. If the p-value from the cointegration test is sufficiently low (e.g., less than 0.05), we conclude that the asset pair is cointegrated. In the out-of-sample period, we leverage the relationship derived from linear regression to predict the value of one asset based on the other. The predicted value relative to the actual value informs our trading decisions:

- When the predicted value exceeds the actual value, the asset is deemed undervalued. The strategy is to go long on this asset and short on the other.
- Conversely, when the predicted value is lower than the actual value, the strategy is to short this asset and go long on the other.

Trading Signals and Z-Scores: We derive trading signals from the z-scores of the mispricing between the actual and predicted values. A z-score indicates how many standard deviations the current price is from the mean:

- A positive z-score signals that the asset is overvalued.
- A negative z-score indicates the asset is undervalued.

In this strategy, we initiate a trade if the z-score is at least two standard deviations away from the mean (z-scores of 2 or -2). Additionally, we open further positions for every 0.05 standard deviation divergence beyond these thresholds.

Profit Taking and Loss Mitigation: The strategy takes profits when the z-score falls to 0.5 standard deviations from the mean. To mitigate losses, we exit positions when the z-score reaches 4 standard deviations from the mean, which indicates a low probability event and suggests the pair may no longer be cointegrated.

Rolling Window Testing: The strategy also conducts periodic cointegration tests using a rolling window format. As each additional day of data from the out-of-sample period is appended to the in-sample period, it replaces the oldest data in the in-sample period. If the p-value exceeds 0.05 during this rolling window, we conclude that the pair is no longer cointegrated. Consequently, all open positions are closed, and no new positions are initiated.

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

Before running the main.py file, ensure you have created a 'data' directory and an 'output' directory in 
the repository. Raw data, formatted data, and cointegration results data files will be stored in the 'data' directory, 
while the results from the backtest will be saved in the 'output' directory. 
When first running the program, ensure the raw data file can be found in the 'data' directory. 
In the config file, ensure you have inputted the correct name of the existing raw data file, as well as the name's of 
the formatted data file and cointegration results data file. 
When running the program the first time for a specific raw data file, the program will create the formatted data and 
cointegration results data files, saving them in the 'data' directory for future use. 
(Preloading cointegration results significantly reduces run time.)


To run the backtesting framework, execute the following command:

	python main.py

The application will prompt you to enter the in-sample period unit (is the in-sample period to be 
measured in months or years? For time series datasets spanning over several years, it's recommended you choose 'yearly' 
to reduce the number of in-sample periods for which cointegration tests will be run on.

Example Workflow

1. Set parameters for the file names, the trading strategy, and the portfolio in the config.json file.
2. Ensure at least the raw data file exists in the 'data' directory
3. Run the main.py file
4. Enter whether the in-sample period needs to be in months or years.
5. Load the formatted time series data and cointegration results files. If they don't exist, process the raw data file first.
6. Define the in-sample and out-of-sample periods from the list of dates provided.
7. Execute the backtest for asset pairs which are cointegrated during the in-sample period, onto the out-of-sample period.
8. Review the output metrics and performance summary in the results.csv file in the 'output' directory.


# License

This project is licensed under the MIT License - see the LICENSE file for details.