import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import warnings
import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure pandas display options for better readability of DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

# Define the columns for trades and daily logs
trade_columns = ['pair', 'entry_date', 'exit_date', 'entry_z_score', 'exit_z_score', 'trade_type', 'entry_price_y',
                 'exit_price_y', 'entry_price_x', 'exit_price_x', 'return', 'trade_size', 'value', 'is_open']

daily_columns = ['date', 'cash', 'invested assets', 'portfolio', 'benchmark', 'return', 'exit trading', 'z score']

# Define output columns for performance metrics
output_columns = ['pair', 'Return %', '# of Trades', 'Trade Profitability rate', 'Maximum drawdown',
                  'Average profit / average loss ratio', 'Beta', 'Sharpe ratio', 'Initial portfolio value',
                  'Final portfolio value', 'Z score out-of-sample average', 'Z score trade entry threshold',
                  'Z score take profit exit', 'Z score stop loss exit', 'Z score increment', 'Trade Size',
                  'In-sample period', 'Out-of-sample period', 'In-sample period cointegration p-value',
                  'Cointegration lost?', 'Correlation', 'y_weight', 'x_weight', 'slope', 'intercept']


class DataLoader:
    """
    DataLoader class responsible for loading and sorting data from CSV files.

        Attributes:
        file_name (str): The name of the CSV file to be loaded.
    """

    def __init__(self, file_name: str):
        """
        Initializes the DataLoader with the specified CSV file name.

        Parameters:
            file_name (str): The name of the CSV file containing the data.
        """
        self.file_name = file_name

    def sort_coint_data(self, start: datetime, end: datetime):
        """
        Load cointegration data from CSV and filter by start and end dates.

        Parameters:
        start (datetime): Start date for filtering.
        end (datetime): End date for filtering.

        Returns:
        DataFrame: Filtered DataFrame with cointegration data.
        """
        file_path = os.path.join(os.getcwd(), 'data', self.file_name)
        df = pd.read_csv(file_path)

        # Rename columns for clarity
        df.rename(columns={
            df.columns[0]: 'Start Date',
            df.columns[1]: 'End Date',
            df.columns[2]: 'y',
            df.columns[3]: 'x',
            df.columns[4]: 'pvalue',
            df.columns[5]: 'correlation',
        }, inplace=True)

        # Convert date columns to datetime
        df['Start Date'] = pd.to_datetime(df['Start Date'])
        df['End Date'] = pd.to_datetime(df['End Date'])

        # Filter the DataFrame by the provided start and end dates
        df = df[(df['Start Date'] == start) & (df['End Date'] == end)]
        df = df.reset_index(drop=True)
        return df

    def sort_time_series_data(self):
        """
        Load time series data from CSV and set the 'Timestamp' column as the index.

        Returns:
        DataFrame: DataFrame with the time series data indexed by timestamp.
        """
        file_path = os.path.join(os.getcwd(), 'data', self.file_name)
        df = pd.read_csv(file_path)

        # Set 'Timestamp' as the DataFrame index and convert to datetime
        df.set_index('Timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)
        return df


class Portfolio:
    """
    Portfolio class for managing cash and investments in a trading strategy.

    Attributes:
        initial_cash (float): The initial amount of cash available for trading.
        cash (float): The current amount of cash available.
        invested (float): The total amount currently invested in trades.
    """

    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = self.initial_cash
        self.invested = 0

    def reset(self):
        """
        Resets the portfolio to its initial state by restoring the cash to the initial amount
        and setting the invested amount to zero.
        """
        self.cash = self.initial_cash
        self.invested = 0

    def portfolio_value(self):
        """
        Calculates the total value of the portfolio by summing the available cash and the
        total invested amount.

        Returns:
            float: The total value of the portfolio (cash + invested).
        """
        return self.cash + self.invested


class Strategy:
    """
    Strategy class for defining trading parameters and thresholds for a trading strategy.

    Attributes:
        entry_threshold (float): The z-score level at which to enter a trade.
        exit_threshold (float): The z-score level at which to exit a trade for a profit.
        limit (float): The z-score level at which to exit a trade to limit losses.
        size (float): The size of each trade (in monetary units or shares).
        z_increment (float): The minimum change in z-score needed to open a new trade.
    """

    def __init__(self, entry_threshold: float, exit_threshold: float, limit: float, size: float, increment: float):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.limit = limit
        self.size = size
        self.z_increment = increment


class StatisticalModel:
    """
    The StatisticalModel class is designed for fitting a linear regression model to data
    and calculating z-scores for both in-sample and out-of-sample data.

    Attributes:
        y (str): The dependent variable's name in the dataset.
        x (str): The independent variable's name in the dataset.
        in_sample (DataFrame): The DataFrame containing the in-sample data.
        out_of_sample (DataFrame): The DataFrame containing the out-of-sample data.
        slope (float): The slope of the fitted regression line.
        intercept (float): The intercept of the fitted regression line.
    """

    def __init__(self, y1: str, x1: str, insample: pd.DataFrame, outofsample: pd.DataFrame):
        self.y = y1
        self.x = x1
        self.in_sample = insample
        self.out_of_sample = outofsample
        self.slope = None
        self.intercept = None

    def fit_model(self):
        """
        Fit a linear regression model to in-sample data.

        Returns:
        tuple: Slope and intercept of the fitted model.
        """
        # Add constant to the independent variable
        indep = sm.add_constant(self.in_sample[self.x])

        # Fit the Ordinary Least Squares (OLS) model
        model = sm.OLS(self.in_sample[self.y], indep)
        regress = model.fit()

        # Store the slope and intercept for later use
        self.slope = float(regress.params[self.x])
        self.intercept = float(regress.params['const'])
        return self.slope, self.intercept

    def calculate_z_scores(self):
        """
        Calculate z-scores for both in-sample and out-of-sample data.

        Returns:
        tuple: Whole sample z-scores, in-sample z-scores, and out-of-sample z-scores.
        """
        # Calculate differences for in-sample data
        in_sample_diff = self.in_sample[self.y] - (self.intercept + (self.slope*self.in_sample[self.x]))

        # Calculate mean and standard deviation for the in-sample differences
        mean, stdev = np.mean(in_sample_diff), np.std(in_sample_diff)

        # Calculate z-scores for in-sample data
        in_sample_z = (in_sample_diff - mean) / stdev

        # Predict out-of-sample values and calculate z-scores
        out_of_sample_yhat = self.intercept + (self.slope * self.out_of_sample[self.x])
        out_of_sample_diff = self.out_of_sample[self.y] - out_of_sample_yhat
        out_of_sample_z = (out_of_sample_diff - mean) / stdev

        # Combine in-sample and out-of-sample z-scores
        whole_sample_z = pd.concat([in_sample_z, out_of_sample_z], ignore_index=False, axis=0)
        return whole_sample_z, in_sample_z, out_of_sample_z


class Pair:
    """
    The Pair class represents a trading pair and manages its trading logic, including
    fitting a statistical model, executing trading strategies, and calculating trade performance metrics.

    Attributes:
        y (str): The name of the dependent variable (target) for the trading pair.
        x (str): The name of the independent variable (feature) for the trading pair.
        pvalue (float): The cointegration p-value for the trading pair.
        correlation (float): The correlation coefficient for the trading pair.
        in_sample (DataFrame): The DataFrame containing in-sample data for fitting the model.
        out_of_sample (DataFrame): The DataFrame containing out-of-sample data for prediction.
        stat_model (StatisticalModel): An instance of the StatisticalModel class for regression analysis.
        strategy (Strategy): The trading strategy to be applied to the pair.
        portfolio (Portfolio): The portfolio managing trades related to the pair.
        is_start (datetime): Start date of the in-sample period.
        is_end (datetime): End date of the in-sample period.
        oos_start (datetime): Start date of the out-of-sample period.
        oos_end (datetime): End date of the out-of-sample period.
        output (DataFrame): DataFrame to store performance metrics and logs.
        trades (DataFrame): DataFrame to log individual trade entries and exits.
        open_trades (DataFrame): DataFrame to manage currently open trades.
        daily (DataFrame): DataFrame to store daily performance metrics.
        out_of_sample_z (Series): Series of z-scores for out-of-sample data.
    """

    def __init__(self, y: str, x: str, pvalue1: float, correlation1: float, insample: pd.DataFrame,
                 outofsample: pd.DataFrame, strategy1: Strategy, portfolio1: Portfolio,
                 is_start: pd.Timestamp, is_end: pd.Timestamp, oos_start: pd.Timestamp, oos_end: pd.Timestamp,
                 output: pd.DataFrame):
        self.y = y
        self.x = x
        self.pvalue = pvalue1
        self.correlation = correlation1
        self.in_sample = insample
        self.out_of_sample = outofsample
        self.stat_model = StatisticalModel(y, x, insample, outofsample)
        self.strategy = strategy1
        self.portfolio = portfolio1
        self.is_start = is_start
        self.is_end = is_end
        self.oos_start = oos_start
        self.oos_end = oos_end
        self.output = output

        # Initialise DataFrames for trades and daily logs
        self.trades = pd.DataFrame(columns=trade_columns)
        self.open_trades = pd.DataFrame(columns=trade_columns)
        self.daily = pd.DataFrame(columns=daily_columns)

        self.out_of_sample_z = None

    def trade(self):
        """
        Execute the trading strategy based on calculated z-scores and model fit.

        This method fits the statistical model to the in-sample data, calculates z-scores,
        computes asset weights, and initiates a Trade object to execute the trading logic.
        """
        # Fit the model and calculate z-scores
        self.slope, self.intercept = self.stat_model.fit_model()
        whole_sample_z, in_sample_z, self.out_of_sample_z = self.stat_model.calculate_z_scores()

        # Calculate weights for the assets
        self.y_weight = self.calculate_weight('y')
        self.x_weight = self.calculate_weight('x')

        # Initialise Trade object and execute trading logic
        trade = Trade(self.y, self.x, self.pvalue, self.correlation, self.y_weight, self.x_weight,
                      self.in_sample, self.out_of_sample, self.strategy, self.portfolio,
                      self.trades, self.open_trades, self.daily, self.out_of_sample_z, in_sample_z)
        trade.execute()

    def calculate_weight(self, variable: str):
        """
        Calculate weights for y and x based on the slope.

        Parameters:
        variable (str): Variable name ('y' or 'x') to calculate weight for.

        Returns:
        float: Calculated weight.
        """
        if variable == 'y':
            return 1 / (1+self.slope)
        elif variable == 'x':
            return self.slope / (1+self.slope)
        return None

    def trade_performance(self, display_daily_summary: bool, display_trades: bool):
        """
        Calculate and display trade performance metrics.

        Parameters:
        display_daily_summary (bool): Whether to display daily summary.
        display_trades (bool): Whether to display individual trades.
        """
        if self.daily.empty:
            print("No data in daily")
            return

        # Display daily summary and trades if requested
        if display_daily_summary:
            print(self.daily)
        if display_trades:
            self.trades = self.trades.sort_values(by='entry_date')
            print(self.trades)

        # Calculate profit and performance metrics
        pnl = ((self.portfolio.portfolio_value() / self.portfolio.initial_cash) - 1) * 100

        trades_df = self.trades
        positive_trades = trades_df[trades_df['return'] > 0]
        negative_trades = trades_df[trades_df['return'] < 0]

        avg_profit = self.calculate_average(positive_trades, 'return', 0) * 100
        avg_loss = self.calculate_average(negative_trades, 'return', 0) * 100

        avg_pnl = avg_profit / abs(avg_loss) if avg_loss != 0 else 0

        success_rate = self.calculate_success_rate(trades_df, positive_trades)

        mpnlr = self.calculate_max_profit_loss_ratio(positive_trades, negative_trades)

        max_drawdown = self.calculate_max_drawdown()

        beta = self.calculate_beta()

        sharpe_ratio = self.calculate_sharpe_ratio(pnl)

        exit_trading = self.daily['exit trading'].iloc[-1]

        pair_performance = self.create_pair_performance_log(pnl, success_rate, avg_pnl,
                                                            max_drawdown, beta, sharpe_ratio, exit_trading)

        self.output.loc[len(self.output)] = pair_performance

    def calculate_average(self, trades: pd.DataFrame, column: str, default: float):
        """
        Calculate the average value for a specified column in the trades DataFrame.

        Parameters:
            trades (DataFrame): The DataFrame containing trade data.
            column (str): The name of the column for which to calculate the average.
            default (float): The default value to return if the trades DataFrame is empty.

        Returns:
            float: The average value of the specified column, or the default value.
        """
        if trades.empty:
            return default
        return np.average(trades[column]) if not trades.empty else default

    def calculate_success_rate(self, trades_df: pd.DataFrame, positive_trades: pd.DataFrame):
        """
        Calculate the success rate of trades as a percentage of positive trades.

        Parameters:
            trades_df (DataFrame): The DataFrame containing all trades.
            positive_trades (DataFrame): The DataFrame containing profitable trades.

        Returns:
            float: The success rate as a percentage.
        """
        try:
            return len(positive_trades) / len(trades_df) * 100
        except ZeroDivisionError:
            return 0

    def calculate_max_profit_loss_ratio(self, positive_trades: pd.DataFrame, negative_trades: pd.DataFrame):
        """
        Calculate the maximum profit/loss ratio from positive and negative trades.

        Parameters:
            positive_trades (DataFrame): The DataFrame containing profitable trades.
            negative_trades (DataFrame): The DataFrame containing losing trades.

        Returns:
            float: The maximum profit/loss ratio, or 0 if no trades exist.
        """
        if not positive_trades.empty and not negative_trades.empty:
            return np.max(positive_trades['return']) / abs(np.max(negative_trades['return']))
        return 0

    def calculate_max_drawdown(self):
        """
        Calculate the maximum drawdown of the portfolio.

        Returns:
            float: The maximum drawdown as a percentage of the portfolio value.
        """
        cum_returns = self.daily['portfolio'] / self.daily['portfolio'].iloc[0] - 1
        previous_peaks = cum_returns.cummax()
        drawdowns = cum_returns - previous_peaks
        return -drawdowns.min() * 100

    def calculate_beta(self):
        """
        Calculate the beta of the portfolio against a benchmark, indicating
        the portfolio's volatility relative to the market.

        Returns:
            float: The beta value, representing the sensitivity of the portfolio returns to benchmark returns.
                   Returns NaN if the lengths of portfolio and benchmark returns do not match.
        """
        portfolio_returns = self.daily['portfolio'].pct_change().dropna().reset_index(drop=True)
        benchmark_returns = self.daily['benchmark'].pct_change(fill_method=None).dropna().reset_index(drop=True)

        if len(portfolio_returns) == len(benchmark_returns):
            model = sm.OLS(portfolio_returns, sm.add_constant(benchmark_returns))
            beta_result = model.fit()
            return beta_result.params.iloc[1]
        return np.nan

    def calculate_sharpe_ratio(self, pnl: float):
        """
        Calculate the Sharpe ratio of the portfolio, which measures risk-adjusted return.

        Parameters:
            pnl (float): The profit and loss percentage from trading.

        Returns:
            float: The Sharpe ratio, indicating the excess return per unit of risk.
                   Returns 0 if standard deviation of portfolio value is zero.
        """
        pnl = pnl / 100
        risk_free_rate = 0.02
        annual_excess_return = ((1 + pnl) ** 12) - 1 - risk_free_rate
        daily_returns = self.daily['portfolio'].pct_change()
        std_portfolio_value = daily_returns.std() * (12 ** 0.5)
        return annual_excess_return / std_portfolio_value if std_portfolio_value != 0 else 0

    def create_pair_performance_log(self, pnl: float, success_rate: float, avg_pnl: float, max_drawdown: float,
                                    beta: float, sharpe_ratio: float, exit_trading: bool):
        """
        Create a log of performance metrics for the trading pair.

        Parameters:
            pnl (float): The total profit and loss percentage from trading.
            success_rate (float): The success rate of trades as a percentage.
            avg_pnl (float): The average profit to average loss ratio.
            max_drawdown (float): The maximum drawdown of the portfolio.
            beta (float): The beta of the portfolio against a benchmark.
            sharpe_ratio (float): The Sharpe ratio indicating risk-adjusted return.
            exit_trading (bool): Indicates whether trading should be exited.

        Returns:
            dict: A dictionary containing the performance metrics of the trading pair.
        """
        return {
            'pair': f"{self.y}-{self.x}",

            'Return %': pnl,
            '# of Trades': len(self.trades),
            'Trade Profitability rate': "{:.2f}%".format(success_rate),
            'Maximum drawdown': "{:.2f}%".format(max_drawdown),
            'Average profit / average loss ratio': "{:.0f}".format(avg_pnl),

            'Beta': "{:.2f}".format(beta),
            'Sharpe ratio': "{:.2f}".format(sharpe_ratio),

            'Initial portfolio value': self.portfolio.initial_cash,
            'Final portfolio value': "{:.2f}".format(self.portfolio.portfolio_value()),

            'Z score out-of-sample average': "{:.2f}".format(abs(np.average(self.out_of_sample_z))),
            'Z score trade entry threshold': self.strategy.entry_threshold,
            'Z score take profit exit': self.strategy.exit_threshold,
            'Z score stop loss exit': self.strategy.limit,
            'Z score increment': self.strategy.z_increment,
            'Trade Size': self.strategy.size,

            'In-sample period': f"{self.is_start}  -  {self.is_end}",
            'Out-of-sample period': f"{self.oos_start}  -  {self.oos_end}",

            'In-sample period cointegration p-value': "{:.4f}".format(self.pvalue),
            'Cointegration lost?': exit_trading,
            'Correlation': "{:.4f}".format(self.correlation),

            'y_weight': "{:.4f}".format(self.y_weight),
            'x_weight': "{:.4f}".format(self.x_weight),

            'slope': self.slope,
            'intercept': self.intercept,

        }


class Trade:
    """
    Trade class responsible for executing trades based on statistical arbitrage strategies.

    Attributes:
        y (str): The name of the first asset in the trading pair.
        x (str): The name of the second asset in the trading pair.
        pvalue (float): The p-value from the cointegration test.
        correlation (float): The correlation coefficient between the two assets.
        y_weight (float): The weight of the first asset in the trade.
        x_weight (float): The weight of the second asset in the trade.
        in_sample_z (pd.Series): The z-scores for the in-sample period.
        out_of_sample_z (pd.Series): The z-scores for the out-of-sample period.
        is_prices (pd.DataFrame): Price data for the in-sample period.
        oos_prices (pd.DataFrame): Price data for the out-of-sample period.
        strategy (object): The trading strategy being used.
        portfolio (object): The portfolio managing the cash and invested assets.
        trades (pd.DataFrame): DataFrame to log completed trades.
        open_trades (pd.DataFrame): DataFrame to log open trades.
        daily (pd.DataFrame): DataFrame to log daily portfolio performance.
    """

    def __init__(self, y1: str, x1: str, pvalue1: float, correlation1: float, y_weight1: float, x_weight1: float,
                 in_sample_price_data1: pd.DataFrame, out_of_sample_price_data1: pd.DataFrame, strategy: Strategy,
                 portfolio: Portfolio, trades: pd.DataFrame, open_trades: pd.DataFrame, daily1: pd.DataFrame,
                 out_of_sample_z1: pd.Series, in_sample_z1: pd.Series):
        self.y = y1
        self.x = x1
        self.pvalue = pvalue1
        self.correlation = correlation1
        self.y_weight = y_weight1
        self.x_weight = x_weight1
        self.in_sample_z = in_sample_z1
        self.out_of_sample_z = out_of_sample_z1
        self.is_prices = in_sample_price_data1
        self.oos_prices = out_of_sample_price_data1
        self.strategy = strategy
        self.portfolio = portfolio
        self.trades = trades
        self.open_trades = open_trades
        self.daily = daily1

    def execute(self):
        """
        Execute the trading strategy, managing entries and exits based on z-scores.

        This method loops through the out-of-sample z-scores to determine
        when to enter and exit trades, updating the portfolio and trade logs accordingly.
        """
        rolling = self.is_prices.copy()
        exit_trading = False
        print(f"{self.y}-{self.x}")

        for i, z_score in enumerate(self.out_of_sample_z):
            current_date = self.out_of_sample_z.index[i]
            check_frequency = int(np.floor(len(self.out_of_sample_z) / 10))
            if i % check_frequency == 0:
                if len(rolling) > check_frequency:
                    rolling = rolling.iloc[check_frequency:]
                else:
                    rolling = rolling.iloc[:0]
                latest_rows = self.oos_prices.iloc[i:i + check_frequency]
                rolling = pd.concat([rolling, latest_rows], ignore_index=False)
                rolling_coint = coint(rolling[self.y], rolling[self.x])
                if isinstance(rolling_coint, tuple):
                    pvalue = rolling_coint[1]
                    if pvalue > self.strategy.exit_threshold:
                        exit_trading = True

            self.exit_trades(i, z_score, current_date, exit_trading)
            self.enter_trades(i, z_score, current_date, exit_trading)
            self.log_daily(i, current_date, exit_trading, z_score)
            self.portfolio.invested = 0

    def exit_trades(self, i: int, z_score: float, current_date: datetime, exit_trading: bool):
        """
        Handle exiting trades based on the trading strategy and current conditions.

        Parameters:
            i (int): The index of the current iteration.
            z_score (float): The current z-score for the trading pair.
            current_date (datetime): The current date for the trading session.
            exit_trading (bool): Flag indicating whether to exit trading.
        """
        for index, trade in self.open_trades.iterrows():
            pnl_y, pnl_x = self.calculate_pnl(trade, i)
            pnl_total = (self.y_weight * pnl_y) + (self.x_weight * pnl_x)
            value_of_trade_change = self.strategy.size * pnl_total
            value_of_trade = self.strategy.size * (1 + pnl_total)
            self.open_trades.at[index, 'value'] = value_of_trade

            self.portfolio.invested += self.strategy.size
            self.portfolio.invested += value_of_trade_change
            if self.should_exit_trade(i, z_score, exit_trading):
                self.open_trades.at[index, 'exit_date'] = current_date
                self.open_trades.at[index, 'exit_price_y'] = self.oos_prices[self.y].iloc[i]
                self.open_trades.at[index, 'exit_price_x'] = self.oos_prices[self.x].iloc[i]
                self.open_trades.at[index, 'return'] = pnl_total
                self.open_trades.at[index, 'exit_z_score'] = z_score
                self.open_trades.at[index, 'is_open'] = False

                self.portfolio.cash += (self.strategy.size + value_of_trade_change)
                self.portfolio.invested -= (self.strategy.size + value_of_trade_change)

                self.trades.loc[len(self.trades)] = self.open_trades.loc[index]
                self.open_trades = self.open_trades[self.open_trades['is_open'] == True]

    def enter_trades(self, i: int, z_score: float, current_date: datetime, exit_trading: bool):
        """
        Handle entering new trades based on z-scores and strategy conditions.

        Parameters:
            i (int): The index of the current iteration.
            z_score (float): The current z-score for the trading pair.
            current_date (datetime): The current date for the trading session.
            exit_trading (bool): Flag indicating whether to exit trading.
        """
        abs_z = abs(z_score)
        if self.can_open_new_trade(i, abs_z, exit_trading):
            trade_type = f'long{self.y}_short{self.x}' if z_score < 0 else f'short{self.y}_long{self.x}'
            new_trade = {
                'pair': f"{self.y}-{self.x}",
                'entry_date': current_date,
                'entry_z_score': z_score,
                'trade_type': trade_type,
                'entry_price_y': self.oos_prices[self.y].iloc[i],
                'entry_price_x': self.oos_prices[self.x].iloc[i],
                'trade_size': self.strategy.size,
                'is_open': True
            }
            self.portfolio.cash -= self.strategy.size
            self.portfolio.invested += self.strategy.size
            self.open_trades = self.open_trades.dropna(how='all', axis=1)
            self.open_trades = pd.concat([self.open_trades, pd.DataFrame([new_trade])], ignore_index=True)

    def log_daily(self, i: int, current_date: datetime, exit_trading: bool, z_score: float):
        """
        Log daily portfolio performance metrics.

        Parameters:
            i (int): The index of the current iteration.
            current_date (datetime): The current date for the trading session.
            exit_trading (bool): Flag indicating whether to exit trading.
            z_score (float): The current z-score for the trading pair.
        """
        profit = ((self.portfolio.portfolio_value() / 1000) - 1) * 100
        day = {
            'date': current_date,
            'cash': self.portfolio.cash,
            'invested assets': self.portfolio.invested,
            'portfolio': self.portfolio.portfolio_value(),
            'benchmark': self.oos_prices['Benchmark'].iloc[i],
            'return': profit,
            'exit trading': exit_trading,
            'z score': "{:.2f}".format(z_score),
        }
        self.daily.loc[len(self.daily)] = day

    def calculate_pnl(self, trade: pd.Series, i: int):
        """
        Calculate the profit and loss for a given trade.

        Parameters:
            trade (pd.Series): A single row from the open trades DataFrame representing the trade.
            i (int): The index of the current iteration.

        Returns:
            tuple: A tuple containing the profit and loss for the first asset (y) and the second asset (x).
        """
        if trade['trade_type'] == f'long{self.y}_short{self.x}':
            # Long position in asset y and short position in asset x
            y = (self.oos_prices[self.y].iloc[i] - trade['entry_price_y']) / trade['entry_price_y']
            x = (trade['entry_price_x'] - self.oos_prices[self.x].iloc[i]) / trade['entry_price_x']
            return y, x
        else:
            # Short position in asset y and long position in asset x
            y = (trade['entry_price_y'] - self.oos_prices[self.y].iloc[i]) / trade['entry_price_y']
            x = (self.oos_prices[self.x].iloc[i] - trade['entry_price_x']) / trade['entry_price_x']
            return y, x

    def should_exit_trade(self, i: int, z_score: float, exit_trading: bool):
        """
        Determine whether to exit the current trade based on trading conditions.

        Parameters:
            i (int): The index of the current iteration.
            z_score (float): The current z-score for the trading pair.
            exit_trading (bool): Flag indicating whether to exit trading.

        Returns:
            bool: True if the trade should be exited; otherwise, False.
        """
        abs_z = abs(z_score)
        is_last_day = i == (len(self.out_of_sample_z) - 1)
        exit_conditions = (abs_z > self.strategy.limit or
                           abs_z < self.strategy.exit_threshold or
                           exit_trading or
                           is_last_day)

        return exit_conditions

    def can_open_new_trade(self, i: int, abs_z: float, exit_trading: bool):
        """
        Determine whether a new trade can be opened based on trading conditions.

        Parameters:
            i (int): The index of the current iteration.
            abs_z (float): The absolute value of the current z-score for the trading pair.
            exit_trading (bool): Flag indicating whether to exit trading.

        Returns:
            bool: True if a new trade can be opened; otherwise, False.
        """
        z_score_check = (self.strategy.entry_threshold < abs_z < self.strategy.limit)
        no_open_trades_or_increment_check = (self.open_trades.empty or
                                            abs((abs_z - abs(self.open_trades.iloc[-1]['entry_z_score']))) >=
                                             self.strategy.z_increment)
        enough_cash = (self.strategy.size < self.portfolio.cash)
        not_last_day = (i < len(self.out_of_sample_z) - 1)

        can_open_trade = (z_score_check and
                          no_open_trades_or_increment_check and
                          enough_cash and
                          not_last_day and
                          not exit_trading)
        return can_open_trade


class Backtester:
    """
    Backtester class for running trading strategy simulations and analyzing performance.

    Attributes:
        dates (pd.Index): A collection of dates for the backtesting period.
        price_data (DataFrame): Historical price data for the assets being traded.
        cointegration_data (DataFrame): Data containing cointegration information for asset pairs.
        strategy (Strategy): A Strategy object containing trading parameters and thresholds.
        portfolio (Portfolio): A Portfolio object to manage cash and investments during the backtest.
        is_start (datetime): The start date for the in-sample period.
        is_end (datetime): The end date for the in-sample period (start of out-of-sample).
        oos_start (datetime): The start date for the out-of-sample period.
        oos_end (datetime): The end date for the out-of-sample period.
        output (DataFrame): A DataFrame to store performance metrics of each pair tested.
        test_mode (bool): A flag indicating if the backtester is in test mode (using default values).
    """

    def __init__(self, dates_x: list, price_data_x: DataLoader, cointegration_data_x: DataLoader,
                 strategy_x: Strategy, portfolio_x: Portfolio, test_mode=False):
        self.dates = dates_x
        self.price_data = price_data_x
        self.cointegration_data = cointegration_data_x
        self.strategy = strategy_x
        self.portfolio = portfolio_x
        self.is_start = None
        self.is_end = None
        self.oos_start = None
        self.oos_end = None
        self.output = pd.DataFrame(columns=output_columns)
        self.test_mode = test_mode

    def get_input(self, prompt: str, default=None):
        """
        Gets user input with an optional default value.

        Parameters:
            prompt (str): The message to display for input.
            default (str, optional): The default value to return if test mode is enabled.

        Returns:
            str: The user input or the default value if in test mode.
        """
        if self.test_mode and default is not None:
            print(f"Using default value: {default}")
            return default
        return input(prompt)

    def run(self):
        """
        Runs the backtesting process, allowing the user to set in-sample and out-of-sample periods.

        The method processes price data, sets trading periods, and executes trades for each cointegrated pair.
        """
        price_data_df = self.price_data.sort_time_series_data()

        for date in self.dates:
            print(date)
        print("These are valid dates for setting the in-sample period")
        print("Enter using format YYYY-MM-DD")

        # Default dates for testing
        default_is_start = "2019-01-02"
        default_is_end = "2022-01-03"
        default_oos_start = "2022-01-03"
        default_oos_end = "2023-01-03"

        # Get user-defined dates for in-sample and out-of-sample periods
        self.is_start = pd.to_datetime(self.get_input("Input in-sample start date: ", default_is_start))
        self.is_end = pd.to_datetime(self.get_input("Input in-sample-period end date (out-of-sample start date): ", default_is_end))
        self.oos_start = pd.to_datetime(self.get_input("Input out-of-sample period start date: ", default_oos_start))
        self.oos_end = pd.to_datetime(self.get_input("Input out-of-sample period end date: ", default_oos_end))

        # Check if dates are valid within the price data
        if self.is_start not in price_data_df.index or self.is_end not in price_data_df.index:
            print("One or both of the in-sample dates are not available in the dataset.")
            return

        if self.oos_start not in price_data_df.index or self.oos_end not in price_data_df.index:
            print("One or both of the out-of-sample dates are not available in the dataset.")
            return

        # Select price data for the specified periods
        is_prices = price_data_df.loc[(price_data_df.index >= self.is_start) & (price_data_df.index <= self.is_end)]
        oos_prices = price_data_df.loc[(price_data_df.index >= self.oos_start) & (price_data_df.index <= self.oos_end)]
        coint_df = self.cointegration_data.sort_coint_data(self.is_start, self.is_end)

        # Execute trades for each pair in the cointegration dataset
        for i, row in coint_df.iterrows():
            self.portfolio.reset()
            pair = Pair(row['y'], row['x'], row['pvalue'], row['correlation'], is_prices, oos_prices,
                        self.strategy, self.portfolio, self.is_start, self.is_end, self.oos_start,
                        self.oos_end, self.output)
            pair.trade()
            pair.trade_performance(display_daily_summary=False, display_trades=False)

        # Sort and display the output results
        self.output = self.output.sort_values(by='Return %', ascending=False).reset_index(drop=True)
        file_path = os.path.join(os.getcwd(), 'output', 'results')
        self.output.to_csv(file_path, index=True)
