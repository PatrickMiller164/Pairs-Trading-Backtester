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

- Python 3
- Pandas
- NumPy
- Statsmodels
- Joblib

### Installation

1. Change directory to where you want to store the repository:
    ```bash
   cd /path/to/your/directory

2. Clone the repository:
   ```bash
   git clone https://github.com/PatrickMiller164/Pairs-Trading-Backtester.git
   
3. Change directory to the repository:
   ```bash
    cd Pairs-Trading-Backtester

4. Create a virtual environment:
    ```bash
   python3 -m venv venv

5. Activate virtual environment:
    ```bash
   source venv/bin/activate

6. Install the required libraries:
    ```bash
   pip3 install -r requirements.txt

7. Create _data_ and _output_ directories. Ensure raw data file is stored in _data_ directory. 
Ensure parameters are correct in _config.json_ file
    ```bash
   mkdir data
   mkdir output
   
8. Run the application:
    ```bash
    python3 main.py

## Usage

### Creating your raw data file
The raw data file must be in .csv format. 
The first column must contain the index of timestamps, with each subsequent column being a time series for an asset. 
The last column must be the benchmark time series. 
Here is an [example](https://docs.google.com/spreadsheets/d/1eKIyqQmjuK2n7H5-kF-pptQotW78esFFItPCo__xlIE/edit?usp=sharing) 
of the format which raw data files need to be in before they are exported as a .csv file and placed in the _data_ directory.

### Updating the configuration file (_config.json_)
Set data file names for the existing raw data file, and for formatted data and cointegration data files (regardless of whether they have been created yet).
- raw_data: your_raw_data_file.csv
- formatted_data: formatted_data.csv
- cointegration_data: cointegration_data.csv

Set the strategy parameters. Default parameters are as follows: 
- entry_threshold: 2 (z-score must be larger in magnitude than this value before a trade can open) 
- exit_threshold: 0.5 (profit-taking threshold, open position is closed when z-score falls below this level)
- limit: 4 (stop-loss threshold, open position is closed when z-score rises above this level)
- size: 125 (total amount of cash used to the long position and short position of a trade)
- increment: 0.05 (minimum z-score deviation needed from last opened trade before another trade can be placed)

Set the portfolio parameter.
- initial_cash: 1000 (initial amount of cash in portfolio, equal to initial portfolio value)

Set results file name.
- results_file: your_results.csv

### Running the application
The application will prompt you to enter the in-sample period unit. Is the in-sample period to be 
measured in months or years? 

For raw data files containing time series which span over several years, I recommended inputting _yearly_ as opposed to _monthly_. 
This is because the cointegration results file is generated by carrying out a cointegration test for every possible asset pair over every possible in-sample period. 
By selecting 'yearly' the number of cointegration tests needing to be computed will be significantly lower, significantly reducing run time.
For example, a raw data file comprising a three year time series comprises 3 yearly in-sample periods (Y1-Y2, Y2-Y3, and Y1-Y3) and 630 monthly in-sample periods (M1-M2, M1-M3 all the way until M35-M36). 
Obtaining cointegration results for each asset pair for 3 in-sample periods will be significantly faster than for 630 in-sample periods.

Following this, the application will look for existing formatted data and cointegration results files in the _data_ directory.
If they do not exist, the application will use the raw data file to create them, saving them there for future use.

Next, the application will ask you to specify the in-sample and out-of-sample periods from the list of valid dates provided above. 
In pairs trading, it is recommended to start the out-of-sample period right after the in-sample period ends.
The application will match the inputted in-sample period with a list of asset pairs found to be cointegrated (p-value < 0.05) in the same in-sample period in the cointegration results file.

Finally, the application will execute the trading strategy for each cointegrated pair over the out-of-sample period.
The results will be saved in the _output_ directory in the repository.

## License
This project is licensed under the MIT License - see the LICENSE file for details.