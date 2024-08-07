import numpy as np

from pandas import DataFrame
from typing import Tuple
from typing import List
from typing import Optional


from stock_frame import StockFrame
#from td.client import TDClient
from ibw.client import IBClient



class Portfolio():

    def __init__(self, account_number: Optional[str] = None) -> None:
        """Initalizes a new instance of the Portfolio object.

        Keyword Arguments:
        ----
        account_number {str} -- An account number to associate with the Portfolio. (default: {None})
        """

        self.positions = {}
        self.filled_orders = {}
        self.pending_orders = {}
        self.positions_count = 0

        self.profit_loss = 0.00
        self.market_value = 0.00
        self.risk_tolerance = 0.00
        self.account_number = account_number

        self._historical_prices = []

        self._ib_client: IBClient = None
        self._stock_frame: StockFrame = None
        self._stock_frame_daily: StockFrame = None

    def add_position(self, symbol: str, asset_type: str, purchase_date: Optional[str] = None,
                     quantity: int = 0, purchase_price: float = 0.0, description: str='', put_call_flag: str='',avg_mkt_price: float=0.0) -> dict:
        """Adds a single new position to the the portfolio.

        Arguments:
        ----
        symbol {str} -- The Symbol of the Financial Instrument. Example: 'AAPL' or '/ES'

        asset_type {str} -- The type of the financial instrument to be added. For example,
            'equity', 'forex', 'option', 'futures'

        Keyword Arguments:
        ----
        quantity {int} -- The number of shares or contracts you own. (default: {0})

        purchase_price {float} -- The price at which the position was purchased. (default: {0.00})

        purchase_date {str} -- The date which the asset was purchased. Must be ISO Format "YYYY-MM-DD"
            For example, "2020-04-01" (default: {None})

        Returns:
        ----
        {dict} -- A dictionary object that represents a position in the portfolio.

        Usage:
        ----
            >>> portfolio = Portfolio()
            >>> new_position = Portfolio.add_position(symbol='MSFT', 
                    asset_type='equity', 
                    quantity=2, 
                    purchase_price=4.00,
                    purchase_date="2020-01-31"
                )
            >>> new_position
            {
                'asset_type': 'equity', 
                'quantity': 2, 
                'purchase_price': 4.00,
                'symbol': 'MSFT',
                'purchase_date': '2020-01-31'
            }
        """

        self.positions[symbol] = {}
        self.positions[symbol]['symbol'] = symbol
        self.positions[symbol]['quantity'] = quantity
        self.positions[symbol]['purchase_price'] = purchase_price
        self.positions[symbol]['purchase_date'] = purchase_date
        self.positions[symbol]['asset_type'] = asset_type
        self.positions[symbol]['put_call_flag'] = put_call_flag
        self.positions[symbol]['description'] = description
        self.positions[symbol]['mktPrice'] = avg_mkt_price

        if purchase_date:
            self.positions[symbol]['ownership_status'] = True
        else:
            self.positions[symbol]['ownership_status'] = False

        return self.positions[symbol]

    def reduce_position(self, symbol: str, change_amount: int) -> Tuple[bool, str]:
        """updates a single position from the bportfolio.

        Arguments:
        ----
        symbol {str} -- The symbol of the instrument to be deleted. Example: 'AAPL' or '/ES'

        Returns:
        ----
        {Tuple[bool, str]} -- Returns `True` if successfully deleted, `False` otherwise
            along with a message.

        Usage:
        ----
            >>> portfolio = Portfolio()
        """

        if symbol in self.positions:
            if change_amount == self.positions[symbol]['quantity'] :
                self.remove_position(symbol)
                return True, "{symbol} position was successfully deleted".format(symbol=symbol)
            else:
                self.positions[symbol]['quantity'] = self.positions[symbol]['quantity'] - change_amount
                return True, "{symbol} position was successfully reduced".format(symbol=symbol)
        else:
            return False, "{symbol} did not exist in the porfolio.".format(symbol=symbol)

    def remove_position(self, symbol: str) -> Tuple[bool, str]:
        """Deletes a single position from the portfolio.

        Arguments:
        ----
        symbol {str} -- The symbol of the instrument to be deleted. Example: 'AAPL' or '/ES'

        Returns:
        ----
        {Tuple[bool, str]} -- Returns `True` if successfully deleted, `False` otherwise 
            along with a message.

        Usage:
        ----
            >>> portfolio = Portfolio()

            >>> new_position = Portfolio.add_position(
                    symbol='MSFT', 
                    asset_type='equity', 
                    quantity=2, 
                    purchase_price=4.00,
                    purchase_date="2020-01-31"
                )
            >>> delete_status = Portfolio.delete_position(symbol='MSFT')
            >>> delete_status
            (True, 'MSFT was successfully removed.')

            >>> delete_status = Portfolio.delete_position(symbol='AAPL')
            >>> delete_status
            (False, 'AAPL did not exist in the porfolio.')
        """

        if symbol in self.positions:
            del self.positions[symbol]
            return True, "{symbol} was successfully removed.".format(symbol=symbol)
        else:
            return False, "{symbol} did not exist in the porfolio.".format(symbol=symbol)

    def total_allocation(self) -> dict:
        """Returns a summary of the portfolio by asset allocation."""

        total_allocation = {
            'stocks': [],
            'fixed_income': [],
            'options': [],
            'futures': [],
            'furex': []
        }

        if len(self.positions.keys()) > 0:
            for symbol in self.positions:
                total_allocation[self.positions[symbol]['asset_type']].append(self.positions[symbol])

    def portfolio_variance(self, weights: dict, covariance_matrix: DataFrame) -> dict:

        sorted_keys = list(weights.keys())
        sorted_keys.sort()

        sorted_weights = np.array([weights[symbol] for symbol in sorted_keys])
        portfolio_variance = np.dot(
            sorted_weights.T,
            np.dot(covariance_matrix, sorted_weights)
        )

        return portfolio_variance

    def portfolio_metrics(self) -> dict:
        """Calculates different portfolio risk metrics using daily data.

        Overview:
        ----
        To build an effective summary of our portfolio we will need to
        calculate different metrics that help represent the risk of our
        portfolio and it's performance. The following metrics will be calculated
        in this method:

        1. Standard Deviation of Percent Returns.
        2. Covariance of Percent Returns.
        2. Variance of Percent Returns.
        3. Average Percent Return
        4. Weighted Average Percent Return.
        5. Portfolio Variance.

        Returns:
        ----
        dict -- [description]
        """

        if not self._stock_frame_daily:
            self._grab_daily_historical_prices()

        # Calculate the weights.
        porftolio_weights = self.portfolio_weights()

        # Calculate the Daily Returns (%)
        self._stock_frame_daily.frame['daily_returns_pct'] = self._stock_frame_daily.symbol_groups['close'].transform(
            lambda x: x.pct_change()
        )

        # Calculate the Daily Returns (Mean)
        self._stock_frame_daily.frame['daily_returns_avg'] = self._stock_frame_daily.symbol_groups['daily_returns_pct'].transform(
            lambda x: x.mean()
        )

        # Calculate the Daily Returns (Standard Deviation)
        self._stock_frame_daily.frame['daily_returns_std'] = self._stock_frame_daily.symbol_groups['daily_returns_pct'].transform(
            lambda x: x.std()
        )

        # Calculate the Covariance.
        returns_cov = self._stock_frame_daily.frame.unstack(
            level=0)['daily_returns_pct'].cov()

        # Take the other columns and get ready to add them to our dictionary.
        returns_avg = self._stock_frame_daily.symbol_groups['daily_returns_avg'].tail(
            n=1
        ).to_dict()

        returns_std = self._stock_frame_daily.symbol_groups['daily_returns_std'].tail(
            n=1
        ).to_dict()

        metrics_dict = {}

        portfolio_variance = self.portfolio_variance(
            weights=porftolio_weights,
            covariance_matrix=returns_cov
        )

        for index_tuple in returns_std:

            symbol = index_tuple[0]
            metrics_dict[symbol] = {}
            metrics_dict[symbol]['weight'] = porftolio_weights[symbol]
            metrics_dict[symbol]['average_returns'] = returns_avg[index_tuple]
            metrics_dict[symbol]['weighted_returns'] = returns_avg[index_tuple] * \
                metrics_dict[symbol]['weight']
            metrics_dict[symbol]['standard_deviation_of_returns'] = returns_std[index_tuple]
            metrics_dict[symbol]['variance_of_returns'] = returns_std[index_tuple] ** 2
            metrics_dict[symbol]['covariance_of_returns'] = returns_cov.loc[[
                symbol]].to_dict()

        metrics_dict['portfolio'] = {}
        metrics_dict['portfolio']['variance'] = portfolio_variance

        return metrics_dict

    def portfolio_weights(self) -> dict:
        """Calculate the weights for each position in the portfolio

        Returns:
        ----
        {dict} -- Each symbol with their designated weights.
        """

        weights = {}

        # First grab all the symbols.
        symbols = self.positions.keys()
        # Grab the quotes.
        quote_fields = [55, 7295, 86, 70, 71, 84, 31, 87]
        quotes = self._ib_client.market_data(symbols, since=None, fields=quote_fields)
        # Grab the projected market value.
        #################################   TO DO ###############################
        # Need to modify to accept quotes from IBK
        # J. Jones : Jan 4th, 2022
        projected_market_value_dict = self.projected_market_value(
            current_prices=quotes
        )

        # Loop through each symbol.
        for symbol in projected_market_value_dict:

            # Calculate the weights.
            if symbol != 'total':
                weights[symbol] = projected_market_value_dict[symbol]['total_market_value'] / \
                    projected_market_value_dict['total']['total_market_value']

        return weights

    def get_portfolio_symbols(self):
        # grab all the symbols.
        symbols = self.positions.keys()
        return symbols

    def portfolio_summary(self):
        """Generates a summary of our portfolio."""

        # First grab all the symbols.
        symbols = self.positions.keys()

        # Grab the quotes.
        quote_fields = [55, 7295, 86, 70, 71, 84, 31, 87]
        quotes = self._ib_client.market_data(symbols, since=None, fields=quote_fields)
        #################################   TO DO ###############################
        # Need to modify to accept quotes from IBK
        # J. Jones : Jan 4th, 2022
        #quotes = self._ib_client.get_quotes(instruments=list(symbols))

        portfolio_summary_dict = {}
        portfolio_summary_dict['projected_market_value'] = self.projected_market_value(
            current_prices=quotes
        )
        portfolio_summary_dict['portfolio_weights'] = self.portfolio_weights()
        portfolio_summary_dict['portfolio_risk'] = ""

        return portfolio_summary_dict

    def in_portfolio(self, symbol: str) -> bool:
        """checks if the symbol is in the portfolio.

        Arguments:
        ----
        symbol {str} -- The symbol of the instrument to be deleted. Example: 'AAPL' or '/ES'

        Returns:
        ----
        bool -- `True` if the position is in the portfolio, `False` otherwise.

        Usage:
        ----
            >>> portfolio = Portfolio()
            >>> new_position = Portfolio.add_position(
                symbol='MSFT', 
                asset_type='equity'
            )
            >>> in_position_flag = Portfolio.in_portfolio(symbol='MSFT')
            >>> in_position_flag
                True
        """

        if symbol in self.positions and self.positions[symbol]['quantity'] > 0:
            return True
        else:
            return False

    def get_ownership_status(self, symbol: str) -> bool:
        """Gets the ownership status for a position in the portfolio.

        Arguments:
        ----
        symbol {str} -- The symbol you want to grab the ownership status for.

        Returns:
        ----
        {bool} -- `True` if the we own the position, `False` if we do not own it.
        """

        if self.in_portfolio(symbol=symbol) and self.positions[symbol]['ownership_status']:
            return self.positions[symbol]['ownership_status']
        else:
            return False

    def set_ownership_status(self, symbol: str, ownership: bool) -> None:
        """Sets the ownership status for a position in the portfolio.

        Arguments:
        ----
        symbol {str} -- The symbol you want to change the ownership status for.

        ownership {bool} -- The ownership status you want the symbol to have. Can either
            be `True` or `False`.

        Raises:
        ----
        KeyError: If the symbol does not exist in the portfolio it will return an error.
        """

        if self.in_portfolio(symbol=symbol):
            self.positions[symbol]['ownership_status'] = ownership
        else:
            raise KeyError(
                "Can't set ownership status, as you do not have the symbol in your portfolio."
            )

    def is_profitable(self, symbol: str, current_price: float) -> bool:
        """Specifies whether a position is profitable.

        Arguments:
        ----
        symbol {str} -- The symbol of the instrument, to check profitability.

        current_price {float} -- The current trading price of the instrument.

        Returns:
        ----
        {bool} -- Specifies whether the position is profitable or flat `True` or not
            profitable `False`.

        Raises:
        ----
        KeyError: If the Symbol does not exist it will return a key error.

        Usage:
        ----
            >>> portfolio = Portfolio()
            >>> new_position = Portfolio.add_position(
                symbol='MSFT', 
                asset_type='equity',
                purchase_price=4.00,
                purchase_date="2020-01-31"
            )
            >>> is_profitable_flag = Portfolio.is_profitable(
                symbol='MSFT',
                current_price=7.00
            )
            >>> is_profitable_flag
            True
        """

        # Grab the purchase price, if it exists.
        if self.in_portfolio(symbol=symbol):
            purchase_price = self.positions[symbol]['purchase_price']
        else:
            raise KeyError("The Symbol you tried to request does not exist.")

        if purchase_price <= current_price:
            return True
        elif purchase_price > current_price:
            return False

    def projected_market_value(self, current_prices: dict) -> dict:
        """Returns the Projected market value for all the positions in the portfolio.

        Arguments:
        ----
        current_prices {dict} -- A dictionary of current quotes for each of the symbols
            in the portfolio.

        Returns:
        ----
        dict -- A summarized version of the portfolio with each position, purchase price, current price,
            and projected values.

        Usage:
        ----
            >>> portfolio = Portfolio()
            >>> new_position = portfolio.add_position(
                symbol='MSFT', 
                asset_type='equity',
                purchase_price=4.00,
                purchase_date="2020-01-31"
            )
            >>> portfolio_summary = portfolio.projected_market_value(current_prices={'MSFT':{'lastPrice': 8.00, 'openPrice': 7.50}})        
        """

        projected_value = {}
        total_value = 0.0
        total_invested_capital = 0.0
        total_profit_or_loss = 0.0

        position_count_profitable = 0
        position_count_not_profitable = 0
        position_count_break_even = 0

        for symbol in current_prices:

            if self.in_portfolio(symbol=symbol):

                projected_value[symbol] = {}
                current_quantity = self.positions[symbol]['quantity']
                purchase_price = self.positions[symbol]['purchase_price']
                current_price = current_prices[symbol]['lastPrice']
                is_profitable = self.is_profitable(
                    symbol=symbol, current_price=current_price)

                projected_value[symbol]['purchase_price'] = purchase_price
                projected_value[symbol]['current_price'] = current_prices[symbol]['lastPrice']
                projected_value[symbol]['quantity'] = current_quantity
                projected_value[symbol]['is_profitable'] = is_profitable

                # Calculate total market value.
                projected_value[symbol]['total_market_value'] = (
                    current_price * current_quantity
                )

                # Calculate total invested capital.
                projected_value[symbol]['total_invested_capital'] = (
                    current_quantity * purchase_price
                )

                projected_value[symbol]['total_loss_or_gain_$'] = ((current_price - purchase_price) * current_quantity) # this is projected to the quantity of stock owned rather than a forecast
                projected_value[symbol]['total_loss_or_gain_%'] = round(((current_price - purchase_price) / purchase_price), 4)

                total_value += projected_value[symbol]['total_market_value']
                total_profit_or_loss += projected_value[symbol]['total_loss_or_gain_$']
                total_invested_capital += projected_value[symbol]['total_invested_capital']

                if projected_value[symbol]['total_loss_or_gain_$'] > 0:
                    position_count_profitable += 1
                elif projected_value[symbol]['total_loss_or_gain_$'] < 0:
                    position_count_not_profitable += 1
                else:
                    position_count_break_even += 1

        projected_value['total'] = {}
        projected_value['total']['total_positions'] = len(self.positions)
        projected_value['total']['total_market_value'] = total_value
        projected_value['total']['total_invested_capital'] = total_invested_capital
        projected_value['total']['total_profit_or_loss'] = total_profit_or_loss
        projected_value['total']['number_of_profitable_positions'] = position_count_profitable
        projected_value['total']['number_of_non_profitable_positions'] = position_count_not_profitable
        projected_value['total']['number_of_breakeven_positions'] = position_count_break_even

        return projected_value

    @property
    def historical_prices(self) -> List[dict]:
        """Gets the historical prices for the Portfolio

        Returns:
        ----
        List[dict] -- A list of historical candle prices.
        """

        return self._historical_prices

    @historical_prices.setter
    def historical_prices(self, historical_prices: List[dict]) -> None:
        """Sets the historical prices for the Portfolio

        Arguments:
        ----
        historical_prices {List[dict]} -- A list of historical candle prices.
        """

        self._historical_prices = historical_prices

    @property
    def stock_frame(self) -> StockFrame:
        """Gets the StockFrame object for the Portfolio

        Returns:
        ----
        {StockFrame} -- A StockFrame object with symbol groups, and rolling windows.
        """

        return self._stock_frame

    @stock_frame.setter
    def stock_frame(self, stock_frame: StockFrame) -> None:
        """Sets the StockFrame object for the Portfolio

        Arguments:
        ----
        stock_frame {StockFrame} -- A StockFrame object with symbol groups, and rolling windows.
        """

        self._stock_frame = stock_frame

    @property
    def ib_client(self) -> IBClient:
        """Gets the IBClient object for the Portfolio

        Returns:
        ----
        {TDClient} -- An authenticated session with the TD API.
        """

        return self._ib_client

    @ib_client.setter
    def ib_client(self, ib_client: IBClient) -> None:
        """Sets the TDClient object for the Portfolio

        Arguments:
        ----
        IBC_client {TDClient} -- An authenticated session with the IBK API.
        """

        self._ib_client: IBClient = ib_client

# To Do ###################
# Haven't changed to IBK yet
# J. Jones : Jan 4, 2022
    def _grab_daily_historical_prices(self) -> StockFrame:
        """Grabs the daily historical prices for each position.

        Returns:
        ----
        {StockFrame} -- A StockFrame object with data organized, grouped, and sorted.
        """

        new_prices = []

        # Loop through each position.
        for symbol in self.positions:


            # Grab the historical prices.
            # To Do ###################
            # Haven't changed to IBK yet
            # J. Jones : Jan 4, 2022
            historical_prices_response = self._ib_client.get_price_history(
                symbol=symbol,
                period_type='year',
                period=1,
                frequency_type='daily',
                frequency=1,
                extended_hours=True
            )

            # Loop through the chandles.
            for candle in historical_prices_response['candles']:

                new_price_mini_dict = {}
                new_price_mini_dict['symbol'] = symbol
                new_price_mini_dict['open'] = candle['open']
                new_price_mini_dict['close'] = candle['close']
                new_price_mini_dict['high'] = candle['high']
                new_price_mini_dict['low'] = candle['low']
                new_price_mini_dict['volume'] = candle['volume']
                new_price_mini_dict['datetime'] = candle['datetime']
                new_prices.append(new_price_mini_dict)

        # Create and set the StockFrame
        self._stock_frame_daily = StockFrame(data=new_prices)
        self._stock_frame_daily.create_frame()

        return self._stock_frame_daily
