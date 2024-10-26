import time as time_true
import pathlib
import pandas as pd
import json
import logging

import datetime
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import pytz

import operator

from typing import List
from typing import Dict
from typing import Union

from portfolio import Portfolio
from stock_frame import StockFrame

from ibw.client import IBClient

class PyRobot():

    def __init__(self, client_id: str, \
                 redirect_uri: str, \
                 paper_trading: bool = True, \
                 credentials_path: str = None, \
                 json_path: str = None, \
                 trading_account: str = None, \
                 account_id: str = None, \
                 start_trading_time=None, \
                 end_trading_time=None, \
                 SECOND_START_TRADING_TIME=None, \
                 SECOND_END_TRADING_TIME=None,  \
                 liq_day_trades_time=None, \
                 default_order_type=None, \
                 no_loss_setting=None, \
                 default_buy_quantity=1, \
                 StopLoss=float(0.0), \
                 GainCap=float(0.0),\
                 lgfile = None
                 ) -> None:
        """Initalizes a new instance of the robot and logs into the API platform specified.

        Arguments:
        ----
        client_id {str} -- The Consumer ID assigned to you during the App registration.
            This can be found at the app registration portal.

        redirect_uri {str} -- This is the redirect URL that you specified when you created your
            TD Ameritrade Application.

        Keyword Arguments:
        ----
        credentials_path {str} -- The path to the session state file used to prevent a full
            OAuth workflow. (default: {None})

        trading_account {str} -- Your TD Ameritrade account number. (default: {None})

        """

        # Set the attributes
        self.trading_account = trading_account
        self.account_type = ''
        self.account_id = account_id
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        #self.credentials_path = credentials_path
        #self.json_path = json_path
        self.session: IBClient = self._create_session()
        self.trades = {}
        self.portfolio = {}
        self.bot_portfolio = {}
        self.bot_order_history = {}
        self.order_history = {}
        self.old_responses = []
        self.historical_prices = {}
        self.def_buy_quantity = default_buy_quantity
        self.stop_loss_perc = StopLoss
        self.gain_cap_perc = GainCap
        self.hit_stop_loss = False
        self.TradeOptions = True
        self.logfiler = lgfile
        self.session.logger_handle = self.logfiler

        # Trading stuff
        self.signals = []
        self.call_options = []
        self.put_options = []
        self.filled_orders = []

        est_tz = pytz.timezone('US/Eastern')
        nw = datetime.now(est_tz)
        self.earliest_order = datetime.combine(nw, datetime.strptime(start_trading_time, '%H:%M:%S EST').time())
        self.earliest_order = est_tz.localize(self.earliest_order)
        self.latest_order = datetime.combine(nw, datetime.strptime(end_trading_time, '%H:%M:%S EST').time())
        self.latest_order = est_tz.localize(self.latest_order)
        if SECOND_START_TRADING_TIME != '' :
            self.second_earliest_order = datetime.combine(nw, datetime.strptime(SECOND_START_TRADING_TIME, '%H:%M:%S EST').time())
            self.second_earliest_order = est_tz.localize(self.second_earliest_order)
            self.second_latest_order = datetime.combine(nw, datetime.strptime(SECOND_END_TRADING_TIME, '%H:%M:%S EST').time())
            self.second_latest_order = est_tz.localize(self.second_latest_order)
        else:
            self.second_earliest_order = datetime(2000,1,1,0,0)
            self.second_earliest_order = est_tz.localize(self.second_earliest_order)
            self.second_latest_order = datetime(2000,1,1,0,0)
            self.second_latest_order = est_tz.localize(self.second_latest_order)

        self.liquidate_all_positions_deadline = datetime.combine(nw, datetime.strptime(liq_day_trades_time, '%H:%M:%S EST').time())
        self.liquidate_all_positions_deadline = est_tz.localize(self.liquidate_all_positions_deadline)
        if no_loss_setting.upper() == 'TRUE' :
            self.no_trading_loss = True
        elif no_loss_setting.upper() == 'FALSE' :
            self.no_trading_loss = False
        else :
            self.no_trading_loss = False

        self.default_mkt_limit_order_type = default_order_type

        self.stock_frame: StockFrame = None
        self.paper_trading = paper_trading

        # J. Jones - check account type to ensure it is a cash account
        # Margin accounts are not allowed.
        # Must be called prior to using other account endpoints
        account_obj = self.session.portfolio_accounts()

#        for account_obj in acc_response :
        Sec_acct = account_obj[0]
        Acct_type = Sec_acct['type']
        TradingType = Sec_acct['tradingType']
        Acct_id = Sec_acct['accountId']

        if TradingType != 'STKCASH' and Acct_type != 'DEMO':
            logmsg = "Account " + Acct_id + " Is Type " + Acct_type
            self.logfiler.info(logmsg)
            self.logfiler.info("Only CASH Accounts Supported or Must be Running in Demo (Paper Trading) mode")
            self.logfiler.info("Bot session ending and Bot exiting")
            exit()

        self.logfiler.info("IBK Account : {acct}".format(acct=Acct_id))
        self.logfiler.info("IBK Display Name : {userdispname}".format(userdispname=Sec_acct['displayName']))
        self._bar_size = 1
        self._bar_type = 'minute'

        if Acct_id != trading_account :
            self.logfiler.info("IBK Account : {acct} does not match trading account in config.ini {tr_acct}".format(acct=Acct_id, tr_acct=self.trading_account))

    def _create_session(self) -> IBClient:
        """Start a new session.

        Creates a new session with the Interactive Brokers API and logs the user into
        the new session.

        Returns:
        ----
        IBClient -- An IBClient object with an authenticated sessions.

        """

        # Create a new instance of the client
        ib_client = IBClient(
            username=self.client_id,
            account=self.account_id,
            is_server_running=True
#            client_id=self.client_id,
#            account_number=self.trading_account,
#            redirect_uri=self.redirect_uri,
#            credentials_path=self.credentials_path
        )

        # log the client into the new session
        #td_client.login()

        return ib_client


    def milliseconds_since_epoch(self, dt_object: datetime) -> int:
        """converts a datetime object to milliseconds since 1970, as an integer

        Arguments:
        ----------
        dt_object {datetime.datetime} -- Python datetime object.

        Returns:
        --------
        [int] -- The timestamp in milliseconds since epoch.
        """

        return int(dt_object.timestamp() * 1000)

    def datetime_from_milliseconds_since_epoch(self, ms_since_epoch: int, timezone: timezone = None) -> datetime :
        """Converts milliseconds since epoch to a datetime object.

        Arguments:
        ----------
        ms_since_epoch {int} -- Number of milliseconds since epoch.

        Keyword Arguments:
        --------
        timezone {datetime.timezone} -- The timezone of the new datetime object. (default: {None})

        Returns:
        --------
        datetime.datetime -- A python datetime object.
        """

        return datetime.fromtimestamp((ms_since_epoch / 1000), tz=timezone)


    def within_trading_hours(self) -> bool :
        """Checks if within allowable trading window for trading strategy to operate successfully
        uses the variables set in the config.ini file

        Uses the datetime module to operate in EST hours

        Returns:
        ----
        bool -- True if current time is within allowable start and end trading time, False otherwise.

        """

        est_tz = pytz.timezone('US/Eastern')
        timenow = datetime.now(est_tz)
        # J. Jones : Don't submit any trades outside of trading hours defined in config.ini
        if (timenow >= self.earliest_order) and (timenow <= self.latest_order) :
            return True
        elif (timenow >= self.second_earliest_order) and (timenow <= self.second_latest_order) :
            return True
        else :
            return False

    def before_trading_hours(self) -> bool :
        """Checks if within allowable trading window for trading strategy to operate successfully
        uses the variables set in the config.ini file

        Uses the datetime module to operate in EST hours

        Returns:
        ----
        bool -- True if current time is within allowable start and end trading time, False otherwise.

        """

        est_tz = pytz.timezone('US/Eastern')
        timenow = datetime.now(est_tz)
        # J. Jones : Don't submit any trades outside of trading hours defined in config.ini
        if (timenow <= self.earliest_order) :
            return True
        else:
            return False

    def past_liquidation_time(self) -> bool :
        """Checks if current time is past the set liquidation time
        for the trading day.

        Uses the variables set in the config.ini file

        Uses the datetime module to operate in EST hours

        Returns:
        ----
        bool -- True if past the liquidation time, False otherwise.

        """

        est_tz = pytz.timezone('US/Eastern')
        timenow = datetime.now(est_tz)
        # J. Jones : Don't submit any trades outside of trading hours defined in config.ini
        if (timenow >= self.liquidate_all_positions_deadline) :
            return True
        else:
            return False

    @property
    def pre_market_open(self) -> bool:
        """Checks if pre-market is open.

        Uses the datetime module to create US Pre-Market Equity hours in
        UTC time.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> pre_market_open_flag = trading_robot.pre_market_open
            >>> pre_market_open_flag
            True

        Returns:
        ----
        bool -- True if pre-market is open, False otherwise.

        """

        pre_market_start_time = datetime.utcnow().replace(
            hour=8,
            minute=00,
            second=00
        ).timestamp()

        market_start_time = datetime.utcnow().replace(
            hour=13,
            minute=30,
            second=00
        ).timestamp()

        right_now = datetime.utcnow().timestamp()

        if market_start_time >= right_now >= pre_market_start_time:
            return True
        else:
            return False

    @property
    def post_market_open(self):
        """Checks if post-market is open.

        Uses the datetime module to create US Post-Market Equity hours in
        UTC time.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> post_market_open_flag = trading_robot.post_market_open
            >>> post_market_open_flag
            True

        Returns:
        ----
        bool -- True if post-market is open, False otherwise.

        """

        post_market_end_time = datetime.utcnow().replace(
            hour=00,
            minute=00,
            second=00
        ).timestamp()

        market_end_time = datetime.utcnow().replace(
            hour=20,
            minute=00,
            second=00
        ).timestamp()

        right_now = datetime.utcnow().timestamp()

        if post_market_end_time >= right_now >= market_end_time:
            return True
        else:
            return False

    @property
    def regular_market_open(self):
        """Checks if regular market is open.

        Uses the datetime module to create US Regular Market Equity hours in
        UTC time.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> market_open_flag = trading_robot.market_open
            >>> market_open_flag
            True

        Returns:
        ----
        bool -- True if post-market is open, False otherwise.

        """

        market_start_time = datetime.utcnow().replace(
            hour=13,
            minute=30,
            second=00
        ).timestamp()

        market_end_time = datetime.utcnow().replace(
            hour=20,
            minute=00,
            second=00
        ).timestamp()

        right_now = datetime.utcnow().timestamp()

        if market_end_time >= right_now >= market_start_time:
            return True
        else:
            return False

    def create_portfolio(self) -> Portfolio:
        """Create a new portfolio.

        Creates a Portfolio Object to help store and organize positions
        as they are added and removed during trading.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> portfolio = trading_robot.create_portfolio()
            >>> portfolio
            <pyrobot.portfolio.Portfolio object at 0x0392BF88>

        Returns:
        ----
        Portfolio -- A pyrobot.Portfolio object with no positions.
        """

        # Initalize the portfolio.
        self.portfolio = Portfolio(account_number=self.trading_account)

        # Assign the Client
        ###########################   TO DO ############################################

        self.portfolio.td_client = self.session

        # This is the synthetic portofolio that the bot is tracking
        self.bot_portfolio = Portfolio(account_number=self.trading_account)

        # Assume zero positions upon startup for bot synthetic portfolio.
        ###########################   TO DO ############################################

        symbols = self.bot_portfolio.positions.keys()
        for sym in symbols:
            self.bot_portfolio.remove_position(sym)

        self.bot_portfolio._ib_client = self.session

        return self.portfolio, self.bot_portfolio



    def grab_current_quotes(self) -> dict:
        """Grabs the current quotes for all positions in the portfolio.

        Makes a call to the TD Ameritrade Get Quotes endpoint with all
        the positions in the portfolio. If only one position exist it will
        return a single dicitionary, otherwise a nested dictionary.

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_portfolio.add_position(
            symbol='MSFT',
            asset_type='equity'
            )
            >>> current_quote = trading_robot.grab_current_quotes()
            >>> current_quote
            {
                "MSFT": {
                    "assetType": "EQUITY",
                    "assetMainType": "EQUITY",
                    "cusip": "594918104",
                    ...
                    "regularMarketPercentChangeInDouble": 0,
                    "delayed": true
                }
            }

            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_portfolio.add_position(
            symbol='MSFT',
            asset_type='equity'
            )
            >>> trading_robot_portfolio.add_position(
            symbol='AAPL',
            asset_type='equity'
            )
            >>> current_quote = trading_robot.grab_current_quotes()
            >>> current_quote

            {
                "MSFT": {
                    "assetType": "EQUITY",
                    "assetMainType": "EQUITY",
                    "cusip": "594918104",
                    ...
                    "regularMarketPercentChangeInDouble": 0,
                    "delayed": False
                },
                "AAPL": {
                    "assetType": "EQUITY",
                    "assetMainType": "EQUITY",
                    "cusip": "037833100",
                    ...
                    "regularMarketPercentChangeInDouble": 0,
                    "delayed": False
                }
            }

        Returns:
        ----
        dict -- A dictionary containing all the quotes for each position.

        """

        # First grab all the symbols.
        symbols = self.portfolio.positions.keys()
        id_list = ''
        for sym in symbols:
            id_list = id_list + str(sym) + ','
        if len(id_list) > 1:
            # remove trailing comma
            id_request_list = id_list[:-1]
        else :
            quotes = {}
            return quotes

        # Grab the quotes.
        quote_fields = [55, 7295, 86, 70, 71, 84, 31, 87]
        quotes = self.session.market_data(id_list, since=None, fields=quote_fields)
        quote_record = quotes[0]
        if '86' not in quote_record.keys(): # ask a second time if data isn't returned
            quotes = self.session.market_data(id_list, since=None, fields=quote_fields)
        return quotes

    def grab_historical_prices(self, bar_size: int = 1,
                               bar_type: str = 'minute', symbols: List[str] = None) -> List[dict]:
        """Grabs the historical prices for all the postions in a portfolio.

        Overview:
        ----
        Any of the historical price data returned will include extended hours
        price data by default.

        Arguments:
        ----
        start {datetime} -- Defines the start date for the historical prices.

        end {datetime} -- Defines the end date for the historical prices.

        Keyword Arguments:
        ----
        bar_size {int} -- Defines the size of each bar. (default: {1})

        bar_type {str} -- Defines the bar type, can be one of the following:
            `['minute', 'week', 'month', 'year']` (default: {'minute'})

        symbols {List[str]} -- A list of ticker symbols to pull. (default: None)

        Returns:
        ----
        {List[Dict]} -- The historical price candles.

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
                )
            >>> start_date = datetime.today()
            >>> end_date = start_date - timedelta(days=30)
            >>> historical_prices = trading_robot.grab_historical_prices(
                    start=end_date,
                    end=start_date,
                    bar_size=1,
                    bar_type='minute'
                )
        """

        self._bar_size = bar_size
        self._bar_type = bar_type

        new_prices = []

        if self._bar_type != 'minute' :
            self.logfiler.info('Only retrieving minute candles')
            exit(-1)

        if not symbols:
            symbols = self.bot_portfolio.positions.keys()

        for symbol in symbols:

            # Get contract ID of the symbol
            Sym_Details = self.session.symbol_search(symbol)
            Sym_conid = Sym_Details[0]['conid']
            self.conid = Sym_conid


            historical_prices_response = self.session.market_data_history(conid=Sym_conid,period='8h',bar='5min', outsideRth=True)


            self.historical_prices[symbol] = {}
            self.historical_prices[symbol]['candles'] = historical_prices_response['data']

            for candle in historical_prices_response['data']:

                new_price_mini_dict = {}
                new_price_mini_dict['symbol'] = symbol
                new_price_mini_dict['open'] = candle['o']
                new_price_mini_dict['close'] = candle['c']
                new_price_mini_dict['high'] = candle['h']
                new_price_mini_dict['low'] = candle['l']
                new_price_mini_dict['volume'] = candle['v']
                new_price_mini_dict['datetime'] = candle['t']
                new_prices.append(new_price_mini_dict)

        self.historical_prices['aggregated'] = new_prices

        return self.historical_prices

    def get_latest_bar(self, TDSession, symbol, lastbartime) -> List[dict]:
        """Returns the latest bar for each symbol in the portfolio.

        Returns:
        ---
        {List[dict]} -- A simplified quote list.

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> latest_bars = trading_robot.get_latest_bar()
            >>> latest_bars
        """

        # Grab the info from the last quest.
        bar_size = self._bar_size
        bar_type = self._bar_type


        # Define the start and end date.
        end_date = datetime.today()
#        print("Time Of End Date For Historical Data Extract :", end_date.hour,":",end_date.minute)
        # Incoming datetime is a UTC time
        newtimezone = pytz.timezone("US/Eastern")
        ny_lastbartime = lastbartime.astimezone(newtimezone)
#        print("Time of last bar :", ny_lastbartime.hour,":",ny_lastbartime.minute)
#        end_date = end_date - timedelta(minutes=1)
#        end_date = end_date.replace(second=59, microsecond=999998)    # J. Jones - removed seconds to always get a full minute of volume and price action
#        start_date = end_date - timedelta(days=1)
        start_date = end_date - timedelta(minutes=20)           # J. Jones - only taking 20 minutes before instead of a day
        start = str(self.milliseconds_since_epoch(dt_object=start_date))
        end = str(self.milliseconds_since_epoch(dt_object=end_date))

        latest_prices = []
        symbols = [symbol]

        # Loop through each symbol.
        # previously looping through self.portfolio.positions
        for symbol in symbols:

            try:
                # Reuse conid from original market data historical request
                Sym_conid = self.conid
                # request all the historical data again.
                # Using 5min candles 1/31/24
                historical_prices_response = self.session.market_data_history(conid=Sym_conid, period='8h', bar='5min',
                                                                              outsideRth=True)
                self.logfiler.info('Retrieving Prices For %s', symbol)

            except:
                time_true.sleep(2)
                self.logfiler.info('Exception On market_data_history')
                # Reuse conid from original market data historical request
                Sym_conid = self.conid
                # request all the historical data again.
                historical_prices_response = self.session.market_data_history(conid=Sym_conid, period='8h', bar='5min',
                                                                              outsideRth=False)
            # parse the candles.
            # J. Jones - Changed to take the next to last candle, since the last candle is only a partial
            # candle of the last time period.  This partial candle does not provide a full reflection of the
            # volume for that last time period.  Since several of the indicators are dependent upon a full volume
            # for a candle, we have to take the next to last candle which has the volume for that entire minute.
            #
            # Also need to ensure that all the candles starting from the last row of the dataframe
            # are added to the end of the frame to avoid missing any candles (due to TDAmeritrade sometimes not returning
            # the same rows
            #
            backcnter=-5
            if 'data' in historical_prices_response.keys() :
                for candle in historical_prices_response['data'][-5:]:
                    candle_datetime = datetime.fromtimestamp(candle['t'] / 1000)
                    candle_datetime = candle_datetime.astimezone(newtimezone)
                    if candle_datetime <= ny_lastbartime or backcnter >= 0:
                        backcnter += 1
                        continue
                    else:
                        new_price_mini_dict = {}
                        new_price_mini_dict['symbol'] = symbol
                        new_price_mini_dict['open'] = candle['o']
                        new_price_mini_dict['close'] = candle['c']
                        new_price_mini_dict['high'] = candle['h']
                        new_price_mini_dict['low'] = candle['l']
                        new_price_mini_dict['volume'] = candle['v']
                        new_price_mini_dict['datetime'] = candle['t']
                    #print("Date Time Of Candle Just Added :",
                      #candle_datetime.strftime("%Y-%m-%d %H:%M:%S"))
                        latest_prices.append(new_price_mini_dict)
                        backcnter += 1


        return latest_prices

    def round_to_next_5_min_mark(self, dt):
        mins_to_add = (-(dt.minute % 5) + 5) % 5
        if mins_to_add == 0:
            mins_to_add = 5
        dt += timedelta(minutes=mins_to_add)
        dt = dt.replace(second=0, microsecond=0)
        return dt

    def wait_till_next_bar(self, last_bar_timestamp: pd.DatetimeIndex) -> None:
        """Waits the number of seconds till the next bar is released.

        Arguments:
        ----
        last_bar_timestamp {pd.DatetimeIndex} -- The last bar's timestamp.
        """
        curr_bar_time = datetime.now(tz=timezone.utc)
        last_bar_time = last_bar_timestamp.replace(tzinfo=timezone.utc)


        next_bar_time = self.round_to_next_5_min_mark(curr_bar_time)

        last_bar_timestamp = int(last_bar_time.timestamp())
        next_bar_timestamp = int(next_bar_time.timestamp())
        curr_bar_timestamp = int(curr_bar_time.timestamp())

        time_to_wait_now = next_bar_timestamp - curr_bar_timestamp

        # J. Jones - added 6 seconds as IBK isn't passing the most current candle if requesting right at
        # the top of the minute.
        if time_to_wait_now < 0:
            time_to_wait_now = 0
        else:
            time_to_wait_now += 6

        logmsg = "=" * 80
        self.logfiler.info(logmsg)
        self.logfiler.info("Pausing for the next bar")
        logmsg = "-" * 80
        self.logfiler.info(logmsg)
        self.logfiler.info("Curr Time: %s", curr_bar_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.logfiler.info("Next Time: %s", next_bar_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.logfiler.info("Sleep Time: {seconds}".format(seconds=time_to_wait_now))
        self.logfiler.info(logmsg)

        time_true.sleep(time_to_wait_now)

    def create_stock_frame(self, data: List[dict]) -> StockFrame:
        """Generates a new StockFrame Object.

        Arguments:
        ----
        data {List[dict]} -- The data to add to the StockFrame object.

        Returns:
        ----
        StockFrame -- A multi-index pandas data frame built for trading.
        """

        # Create the Frame.
        self.stock_frame = StockFrame(data=data)

        return self.stock_frame

    def execute_signals(self, signals: List[pd.Series], trades_to_execute: dict) -> List[dict]:
        """Executes the specified trades for each signal.

        Arguments:
        ----
        signals {list} -- A pandas.Series object representing the buy signals and sell signals.
            Will check if series is empty before making any trades.

        Trades:
        ----
        trades_to_execute {dict} -- the trades you want to execute if signals are found.

        Returns:
        ----
        {List[dict]} -- Returns all order responses.

        Usage:
        ----
            >>> trades_dict = {
                    'MSFT': {
                        'trade_func': trading_robot.trades['long_msft'],
                        'trade_id': trading_robot.trades['long_msft'].trade_id
                    }
                }
            >>> signals = indicator_client.check_signals()
            >>> trading_robot.execute_signals(
                    signals=signals,
                    trades_to_execute=trades_dict
                )
        """

        # Define the Buy and sells.
        buys: pd.Series = signals['buys']
        sells: pd.Series = signals['sells']

        order_responses = []

        # If we have buys or sells continue.
        if not buys.empty:

            # Grab the buy Symbols.
            symbols_list = buys.index.get_level_values(0).to_list()

            # Loop through each symbol.
            for symbol in symbols_list:

                # Check to see if there is a Trade object.
                if symbol in trades_to_execute:

                    if self.portfolio.in_portfolio(symbol=symbol):
                        self.portfolio.set_ownership_status(
                            symbol=symbol,
                            ownership=True
                        )

                    # Set the Execution Flag.
                    trades_to_execute[symbol]['has_executed'] = True
                    trade_obj: Trade = trades_to_execute[symbol]['buy']['trade_func']

                    if not self.paper_trading:

                        # Execute the order.
                        order_response = self.execute_orders(
                            trade_obj=trade_obj
                        )

                        order_response = {
                            'order_id': order_response['order_id'],
                            'request_body': order_response['request_body'],
                            'timestamp': datetime.now().isoformat()
                        }

                        order_responses.append(order_response)

                    else:

                        order_response = {
                            'order_id': trade_obj._generate_order_id(),
                            'request_body': trade_obj.order,
                            'timestamp': datetime.now().isoformat()
                        }

                        order_responses.append(order_response)

        elif not sells.empty:

            # Grab the buy Symbols.
            symbols_list = sells.index.get_level_values(0).to_list()

            # Loop through each symbol.
            for symbol in symbols_list:

                # Check to see if there is a Trade object.
                if symbol in trades_to_execute:

                    # Set the Execution Flag.
                    trades_to_execute[symbol]['has_executed'] = True

                    if self.portfolio.in_portfolio(symbol=symbol):
                        self.portfolio.set_ownership_status(
                            symbol=symbol,
                            ownership=False
                        )

                    trade_obj: Trade = trades_to_execute[symbol]['sell']['trade_func']

                    if not self.paper_trading:

                        # Execute the order.
                        order_response = self.execute_orders(
                            trade_obj=trade_obj
                        )

                        order_response = {
                            'order_id': order_response['order_id'],
                            'request_body': order_response['request_body'],
                            'timestamp': datetime.now().isoformat()
                        }

                        order_responses.append(order_response)

                    else:

                        order_response = {
                            'order_id': trade_obj._generate_order_id(),
                            'request_body': trade_obj.order,
                            'timestamp': datetime.now().isoformat()
                        }

                        order_responses.append(order_response)

        # Save the response.
        self.save_orders(order_response_dict=order_responses)

        return order_responses



    def find_bot_owned_option(self, put_call_flag) :
        owned_option = 'Could not find position to sell'
        for position in self.bot_portfolio.positions :
            if put_call_flag == self.bot_portfolio.positions[position]['put_call_flag']  :
                if self.bot_portfolio.positions[position]['quantity'] > 0 :
                    owned_option = position
                    return owned_option

        if put_call_flag == 'C' :
            full_word = 'Call'
        elif put_call_flag == 'P' :
            full_word = 'Put'
        elif self.TradeOptions :
            full_word = 'Unknown Option'
        else :
            full_word = 'Stock'

        self.logfiler.info("Could not find a {putcall} to sell in Bot positions".format(putcall=full_word))
        return owned_option

    def process_orders(self, First_Loop, symbol, orders ):
        """Executes a orders from the prototype Bot.

        Overview:
        ----
        The `` method will execute orders according to signals in the 'buy_condition' column that
        was created by the prototype bot. The 'buy_condition' column exists in the self.stock_frame dataframe.

        Returns:
        ----
        {dict} -- An order response dictionary.

        Portfolio position format
            {
                'asset_type': 'equity',
                'quantity': 2,
                'purchase_price': 4.00,
                'symbol': 'MSFT',
                'purchase_date': '2020-01-31'
            }
        {order_list}
        order_list = {'buy_calls_count': buy_calls_count, 'buy_puts_count': buy_puts_count,
            'no_action_calls_count': no_action_calls_count, 'no_action_puts_count': no_action_puts_count}
        """

        # Required variables
        buy_and_sell_count = 0
        buy_calls_count = orders['buy_calls_count']
        buy_puts_count = orders['buy_puts_count']
        stock_data = self.stock_frame
        order = {}
        order_response = {}
        purchprice = 0.0

        # Query portfolio for existing orders.
        # J. Jones : NOte, this only returns positions that are copied from IBKR
        # The Bot positions are always refreshed from IBKR.
        filled_orders, calls_quantity, puts_quantity, remaining_quantity, days_transactions = self.query_orders(
            symbol, First_Loop)

        # Check for Stop Loss
        # There is an option in the config.ini file to set the stop loss to a max percentage loss
        # If that is exceeded, that position will be sold.
        #
        current_signal = self.signals[-1]
        previous_signal = self.signals[-2]
        # allow bot to reverse positions if signals change quickly
        if (current_signal.startswith("Buy Calls") and previous_signal.startswith("Buy Puts")) or (current_signal.startswith("Buy Puts") and previous_signal.startswith("Buy Calls")):
            signal = self.signals[-1]
            self.logfiler.info("Reversing Positions!")
        elif self.Stop_Loss_Exceeded() or self.Gain_Limit_Exceeded() :
            self.signals[-1] = "StopGainLoss"
            self.stock_frame.at[self.stock_frame.index[-1],'buy_condition'] = "StopGainLoss"
            signal = "StopGainLoss"
            self.hit_stop_loss = True
        else :
            signal = self.signals[-1]

        # last element in the calls_option column
        if self.TradeOptions :
            call_symbol = self.call_options[-1]
            put_symbol = self.put_options[-1]
        else :
            call_symbol = self.conid # Just buy or sell the stock
            put_symbol = self.conid # Just buy or sell the stock

        if buy_calls_count == 0 and buy_puts_count == 0:
            buy_n = 0
        elif buy_puts_count == 0:
            buy_n = buy_calls_count
        else:
            buy_n = buy_puts_count


        self.logfiler.info("Signal {sig}, Close {cls}, VWAP {vp} Vol Chg % {vchg}".format(sig=signal, cls=str(self.stock_frame.at[self.stock_frame.index[-1],'close']), vp=str(self.stock_frame.at[self.stock_frame.index[-1],'vwap']), vchg=str(self.stock_frame.at[self.stock_frame.index[-1],'per_chg_volume'])))

        if (self.past_liquidation_time()):
            self.logfiler.info("Past Liquidation Deadline {closeitup}, Closing Open Option Positions".format(
                closeitup=self.liquidate_all_positions_deadline.strftime("%H:%M:%S")))
            signal = 'No action'   # Will trigger selling logic
            if calls_quantity <= 0 and puts_quantity <= 0 :
                self.logfiler.info("No Open Bot Positions, Bot Exiting")
                exit(0)

        # J. Jones : Don't submit any trades outside of hours defined in config.ini
        # We do need to sell though, if past liquidation time or between trading periods
        if self.within_trading_hours() or self.past_liquidation_time() :
            if signal.startswith("Buy Calls")  :
                # buy condition met and no position held in Interactive Brokers
                # For VWAP Strategy, changed to buy when one buy signal occurrs
                # 1/24/2022 added the condition to avoid buying calls when holding puts to avoid going vertical unintentially
                instruction = "BUY"
                if buy_n >= 1 and calls_quantity < 1 :
                    # if we don't own any of the current symbol, otherwise, don't buy
                    if (not self.bot_portfolio.in_portfolio(call_symbol)) :
                        if self.TradeOptions :
                            self.logfiler.info("Buying CALL option {call_sym} for {sym} at time: {tm}".format(call_sym=call_symbol, sym=symbol, tm=datetime.now().strftime("%H:%M:%S")))
                        else :
                            self.logfiler.info("Buying {id} {sym} at time: {tm}".format(id=self.conid, sym=symbol, tm=datetime.now().strftime("%H:%M:%S")))

                        # BUY
                        if self.def_buy_quantity != 0 :
                            order, order_response = self.buy_stock(symbol=symbol,
                                                               option_symbol_str=call_symbol,
                                                               instruction=instruction)
                    elif self.bot_portfolio.in_portfolio(call_symbol):
                        if self.TradeOptions :
                            self.logfiler.info("Already have option {csym}.".format(csym=call_symbol))
                        if not self.TradeOptions :
                            self.logfiler.info("Already have Stock Position {csym}.".format(csym=call_symbol))
                    else:
                        self.logfiler.info("Something went wrong.")

                    buy_and_sell_count = 1
                    buy_calls_count += 1
                    self.logfiler.info("Buy and sell count: %d", buy_and_sell_count)

            # Sell CALLS logic
            elif signal.startswith("Buy Puts") or signal.startswith("StopGainLoss"):
                self.logfiler.info("Sell CALLS (or stock) if we have a position.")
                # sell condition met and we have CALLS in the portfolio
                if self.TradeOptions :
                    call_symbol_owned = self.find_bot_owned_option('C')
                else :
                    call_symbol_owned = self.find_bot_owned_option('')

                if (calls_quantity >= 1) :
                    # J. Jones - if we are past the liquidation time, we sell regardless of the sell conditions
                    instruction = "SELL"
                    if (self.past_liquidation_time()) :
                         # if we own a call, sell it, otherwise, don't sell
                        if self.TradeOptions :
                             call_symbol_owned = self.find_bot_owned_option('C')
                        else :
                             call_symbol_owned = self.find_bot_owned_option('')

                        if call_symbol_owned  == '' :
                            if self.TradeOptions :
                                self.logfiler.info("Do not own any Calls")
                                purchprice = 999.0
                            else :
                                self.logfiler.info("Do not own any Stock {sym}".format(sym=self.conid))
                        else:
                            purchprice = float(self.bot_portfolio.positions[call_symbol_owned]['purchase_price'])


                    if (self.bot_portfolio.in_portfolio(call_symbol_owned)) :
                        purchprice = float(self.bot_portfolio.positions[call_symbol_owned]['purchase_price'])
                        if self.TradeOptions :
                            self.logfiler.info("Selling CALL option {c_sym}".format(c_sym=call_symbol_owned))

                        else :
                            self.logfiler.info("Selling stock {c_sym}".format(c_sym=call_symbol_owned))

                        # SELL THE Position
                        if self.def_buy_quantity != 0 :
                            order, order_response = self.sell_stock(symbol=symbol,
                                                        option_symbol=call_symbol_owned,
                                                        instruction=instruction,
                                                        purchase_price = purchprice)

                    else:
                        if self.TradeOptions :
                            self.logfiler.info("Do not have option %s.",call_symbol_owned)
                            self.logfiler.info("Calls held, but did not meet Call selling conditions")
                        else :
                            self.logfiler.info("Do not have Stock %s.", call_symbol_owned)

                    buy_and_sell_count = 0

                else :
                    if self.TradeOptions :
                        self.logfiler.info("Do not own any Calls")
                    else :
                        self.logfiler.info("Do not own any Stock {sym}".format(sym=self.conid))


            # Buys and sells PUTS options ===============================================================
            # Buy PUTS logic
            if signal.startswith("Buy Puts") :
                # Buy Puts is the equivalent of not being long.  So, if not trading options, we would sell stock position
                # buy puts condition met and no position held
                # J. Jones - changed to buy position when buy_n 1 or greater for vwap strategy
                if buy_n >= 1 and puts_quantity < 1 :
                #if buy_n >= 1 and puts_quantity < 1 and calls_quantity < 1:
                    instruction = "BUY"
                    # if we don't own any of the current option symbol, otherwise, don't buy
                    if not self.bot_portfolio.in_portfolio(put_symbol) and self.TradeOptions :
                        self.logfiler.info(
                            "Buying PUT option {p_sym} for {sm} at time: {tm}".format(p_sym=put_symbol, sm=symbol, tm=datetime.now().time()))
                        # BUY THE PUTS
                        if self.def_buy_quantity != 0 :
                            order, order_response = self.buy_stock(symbol=symbol,
                                                               option_symbol_str=put_symbol,
                                                               instruction=instruction)
                    elif self.bot_portfolio.in_portfolio(put_symbol) and self.TradeOptions :
                        self.logfiler.info("Already have option %s", put_symbol)
                    elif not self.bot_portfolio.in_portfolio(put_symbol) :
                        self.logfiler.info("No stock to sell")
                    elif self.bot_portfolio.in_portfolio(put_symbol) :
                        self.logfiler.info(
                        "Selling Stock {id} {sm} at time: {tm}".format(id=put_symbol, sm=symbol,
                                                                                  tm=datetime.now().time()))
                        instruction = "SELL"
                        if self.def_buy_quantity != 0 :
                            order, order_response = self.sell_stock(symbol=symbol,
                                                               option_symbol_str=put_symbol,
                                                               instruction=instruction, purchase_price=float(999.0) )
                else:
                    self.logfiler.info("Something went wrong.")

                    buy_and_sell_count = 1
                    buy_puts_count += 1
                    self.logfiler.info("Buy and sell count: %d", buy_and_sell_count)

            elif signal.startswith("Buy Calls") or signal.startswith("StopGainLoss"):
                if self.TradeOptions :
                    self.logfiler.info("Sell PUTS if we have them.")
                else :
                    self.logfiler.info("Sell Stock Due To Gain/Loss Limit")
                # Potential optimization to Sell Logic : add 'or' statement to sell puts if 3 MA Slope turns positive
                # Sell PUTS logic
                if (puts_quantity >= 1) and self.TradeOptions :
                    put_symbol_owned = self.find_bot_owned_option('P')
                    instruction = "SELL"
                    purchprice = 0.0
                    if (self.past_liquidation_time()) :
                        # if we own a call, sell it, otherwise, don't sell
                        if put_symbol_owned  == '' :
                            self.logfiler.info("Do not own any Puts")
                            purchprice = 999.0
                        else:
                            purchprice = float(self.bot_portfolio.positions[put_symbol_owned]['purchase_price'])

                    if (self.bot_portfolio.in_portfolio(put_symbol_owned)) and purchprice != 999.0 :
                        purchprice = float(self.bot_portfolio.positions[put_symbol_owned]['purchase_price'])
                        self.logfiler.info("Selling PUT option {p_sym}".format(p_sym=put_symbol_owned))
                        # SELL THE CALLS
                        if self.def_buy_quantity != 0 :
                            order, order_response = self.sell_stock(symbol=symbol,
                                                                option_symbol=put_symbol_owned,
                                                                instruction=instruction,
                                                                purchase_price = purchprice)
                    else:
                        self.logfiler.info("Do not have option %s", put_symbol_owned)
                        self.logfiler.info("Puts held, but do not meet sell indicator conditions")

                    stock_data["buy_count"] = -1
                    buy_and_sell_count = 0
                elif not self.TradeOptions and signal.startswith("StopGainLoss"):
                    purchprice = 0.0
                    put_symbol_owned = self.find_bot_owned_option('')
                    if (self.past_liquidation_time()) :
                        # if we own stock, sell it
                        if put_symbol_owned == '' :
                            self.logfiler.info("Do not own any Stock")
                            purchprice = 999.0
                        else:
                            purchprice = float(self.bot_portfolio.positions[put_symbol_owned]['purchase_price'])

                    if (self.bot_portfolio.in_portfolio(put_symbol_owned)) and purchprice != 999.0 :
                        purchprice = float(self.bot_portfolio.positions[put_symbol_owned]['purchase_price'])
                        self.logfiler.info("Selling Stock {p_sym}".format(p_sym=put_symbol_owned))
                        # SELL THE CALLS
                        if self.def_buy_quantity != 0 :
                            order, order_response = self.sell_stock(symbol=symbol,
                                                                option_symbol=put_symbol_owned,
                                                                instruction=instruction,
                                                                purchase_price = purchprice)

                else:
                    self.logfiler.info("Do not own any Puts")

        else :
            if not self.within_trading_hours() :
                self.logfiler.info("Outside of allowable trading times {startt} to {endt} EST".format(startt=self.earliest_order.strftime("%H:%M:%S"), endt=self.latest_order.strftime("%H:%M:%S")))
                self.logfiler.info("Or secondary trading times {startt} to {endt} EST".format(startt=self.second_earliest_order.strftime("%H:%M:%S"), endt=self.second_latest_order.strftime("%H:%M:%S")))
                if self.TradeOptions :
                    self.logfiler.info("Sell CALLS if we have them.")
                else :
                    self.logfiler.info("Sell stock if we hold any")

                if (calls_quantity >= 1):
                    # J. Jones - if we are not within trading hours, we sell regardless of the sell conditions
                    instruction = "SELL"
                    # if we own a call, sell it, otherwise, don't sell
                    if self.TradeOptions :
                        call_symbol_owned = self.find_bot_owned_option('C')
                    else :
                        call_symbol_owned = self.find_bot_owned_option('')
                    purchprice = 0.0
                    if call_symbol_owned == '':
                        if self.TradeOptions :
                            self.logfiler.info("Do not own any Calls")
                        else :
                            self.logfiler.info("Do not own any Stock")
                        purchprice = 999.0
                    else:
                        purchprice = float(self.bot_portfolio.positions[call_symbol_owned]['purchase_price'])

                    if (self.bot_portfolio.in_portfolio(call_symbol_owned)) and purchprice != 999.0 :
                        purchprice = float(self.bot_portfolio.positions[call_symbol_owned]['purchase_price'])
                        if self.TradeOptions :
                            self.logfiler.info("Selling CALL option {c_sym}".format(c_sym=call_symbol_owned))
                        else :
                            self.logfiler.info("Selling Stock {c_sym}".format(c_sym=call_symbol_owned))

                        # SELL THE CALLS
                        if self.def_buy_quantity != 0 :
                            order, order_response = self.sell_stock(symbol=symbol,
                                                                option_symbol=call_symbol_owned,
                                                                instruction=instruction,
                                                                purchase_price=purchprice)
                else :
                    self.logfiler.info("Do not own any Calls")

                if self.TradeOptions :
                    self.logfiler.info("Sell PUTS if we have them.")
                # Sell PUTS logic.  We should never have a puts_quantity over zero if we are trading stock only
                if (puts_quantity >= 1) :
                    if self.TradeOptions :
                        put_symbol_owned = self.find_bot_owned_option('P')
                    else:
                        put_symbol_owned = self.find_bot_owned_option('')
                    instruction = "SELL"
                    purchprice = 0.0
                    # if we own a put, sell it, otherwise, don't sell
                    if put_symbol_owned  == '' and self.TradeOptions:
                        self.logfiler.info("Do not own any Puts")
                        purchprice = 999.0
                    else:
                        purchprice = float(self.bot_portfolio.positions[put_symbol_owned]['purchase_price'])

                    if (self.bot_portfolio.in_portfolio(put_symbol_owned)) and purchprice != 999.0 :
                        purchprice = float(self.bot_portfolio.positions[put_symbol_owned]['purchase_price'])
                        self.logfiler.info("Selling PUT option {p_sym}".format(p_sym=put_symbol_owned))
                        # SELL THE CALLS
                        if self.def_buy_quantity != 0 :
                            order, order_response = self.sell_stock(symbol=symbol,
                                                                option_symbol=put_symbol_owned,
                                                                instruction=instruction,
                                                                purchase_price = purchprice)
                else :
                    self.logfiler.info("Do not own any Puts")

        return order, order_response, days_transactions

    def Stop_Loss_Exceeded(self):
        if self.stop_loss_perc != 0.0 :
            for position_record in self.bot_portfolio.positions:
                if self.bot_portfolio.positions[position_record]['quantity'] > 0 :
                    quantity = self.bot_portfolio.positions[position_record]['quantity']
                    average_price = float(self.bot_portfolio.positions[position_record]['purchase_price'])
                    position_desc = self.bot_portfolio.positions[position_record]['description']
                    Mkt_Price_Position = float(self.bot_portfolio.positions[position_record]['mktPrice'])
                    if Mkt_Price_Position < (average_price*(1.0-self.stop_loss_perc)) :
                        self.logfiler.info("Hit Stop Loss of {per}% On {descr}, Purch Price {avg_price}, Mkt Price {mkt_price}".format(\
                            per=str((self.stop_loss_perc*100)), \
                            descr=position_desc, \
                            avg_price=str(average_price), \
                            mkt_price=str(Mkt_Price_Position)))
                        return True
                    else :
                        self.logfiler.info(
                            "Did not Hit Stop Loss of {per}% On {descr}, Purch Price {avg_price}, Mkt Price {mkt_price}".format( \
                                per=str((self.stop_loss_perc * 100)), \
                                descr=position_desc, \
                                avg_price=str(average_price), \
                                mkt_price=str(Mkt_Price_Position)))

        return False

    def Gain_Limit_Exceeded(self):
        if self.gain_cap_perc != 0.0 :
            for position_record in self.bot_portfolio.positions:
                if self.bot_portfolio.positions[position_record]['quantity'] > 0 :
                    quantity = self.bot_portfolio.positions[position_record]['quantity']
                    average_price = float(self.bot_portfolio.positions[position_record]['purchase_price'])
                    position_desc = self.bot_portfolio.positions[position_record]['description']
                    Mkt_Price_Position = float(self.bot_portfolio.positions[position_record]['mktPrice'])
                    if Mkt_Price_Position > (average_price*(1.0+self.gain_cap_perc)) :
                        self.logfiler.info("Hit Gain Limit of {per}% On {descr}, Purch Price {avg_price}, Mkt Price {mkt_price}".format(\
                            per=str((self.gain_cap_perc*100)), \
                            descr=position_desc, \
                            avg_price=str(average_price), \
                            mkt_price=str(Mkt_Price_Position)))
                        return True
                    else :
                        self.logfiler.info(
                            "Did not Hit Gain Limit of {per}% On {descr}, Purch Price {avg_price}, Mkt Price {mkt_price}".format( \
                                per=str((self.gain_cap_perc * 100)), \
                                descr=position_desc, \
                                avg_price=str(average_price), \
                                mkt_price=str(Mkt_Price_Position)))
        return False

    def buy_stock(self, symbol, option_symbol_str, instruction):

        # Define the Order.
        order_response = {}
        order_template = {}
        # Bot will not buy more than 10 contracts at a time.
        if self.def_buy_quantity <= 10 :
            default_quantity = self.def_buy_quantity
        else :
            default_quantity = 1
            self.logfiler.info("Default Buy Quantity Too High {dbq}, buy quant set to 1".format(dbq=self.def_buy_quantity))

        if option_symbol_str.isnumeric() :
            option_symbol = int(option_symbol_str)
        else :
            option_symbol = 0

        if option_symbol == 0  :
            self.logfiler.info(
            "Invalid Option Symbol IBKR Number {osym}".format(osym=option_symbol))
            return order_template, order_response


        orderType = self.default_mkt_limit_order_type

        if orderType == 'MKT':
            if self.TradeOptions :
                order_template = {
                    'acctid': self.account_id,
                    'conid': option_symbol,
                    'ticker': symbol,
                    'secType': str(option_symbol) + ':' + 'OPT',
                    'orderType': orderType,
                    'quantity': default_quantity,
                    'side': 'BUY',
                    'tif': 'DAY'
                }
            else :
                order_template = {
                    'acctid': self.account_id,
                    'conid': int(self.conid),
                    'ticker': symbol,
                    'secType': 'STK',
                    'orderType': orderType,
                    'quantity': default_quantity,
                    'side': 'BUY',
                    'tif': 'DAY'
                }
        elif orderType == 'LMT' :
            # Retrieve latest price for option to set limit to
            quote_fields = [55, 7295, 86, 70, 71, 84, 31, 87]
            quote_snapshot = self.session.market_data([str(option_symbol)], since=None, fields=quote_fields)
            quote_record=quote_snapshot[0]
            if '86' in quote_record.keys():
                ask_price = float(quote_record['86'])
                bid_price = float(quote_record['84'])
            else: # Second attempt always works for some reason
                quote_snapshot = self.session.market_data([str(option_symbol)], since=None, fields=quote_fields)
                quote_record = quote_snapshot[0]
                if '86' in quote_record.keys():
                    ask_price = float(quote_record['86'])
                    bid_price = float(quote_record['84'])
                else:
                    self.logfiler.info(
                        "Error trying obtain quote for {sy} ".format(sy=option_symbol))
                    return order_template, order_response
            LimitPrice = (bid_price + ask_price) / 2.0
            LimitPrice = round(LimitPrice,2)

            if self.TradeOptions :
                order_template = {
                'acctid': self.account_id,
                'conid': int(option_symbol),
                'ticker': symbol,
                'secType': str(option_symbol) + ':' + 'OPT',
                'orderType': orderType,
                'quantity': default_quantity,
                'side': 'BUY',
                'price': LimitPrice,
                'tif': 'DAY'
                }
            else :
                order_template = {
                'acctid': self.account_id,
                'conid': int(option_symbol),
                'ticker': symbol,
                'secType': 'STK',
                'orderType': orderType,
                'quantity': default_quantity,
                'side': 'BUY',
                'price': LimitPrice,
                'tif': 'DAY'
                }
        else :
            self.logfiler.info("Invalid Order Type %s", orderType)
            order_template = {}
            return order_template, order_response

        # Place the Order.
        try:
            # J. Jones - added dump of order template
            self.logfiler.info("Order Template {ord}".format(ord=json.dumps(order_template, indent=4)))

            order_response = self.session.place_order(
                account_id=self.account_id,
                order=order_template
            )

            if 'order_id' in order_response[0].keys():
                self.logfiler.info("Order Submitted, OrderID = {id}".format(id=order_response[0]['order_id']))
                return order_template, order_response
            else:  # We have messages to respond to, we by default just respond 'true' to all of them
                if 'messageIds' in order_response[0].keys():
                    self.logfiler.info("Message received on order : {msg}".format(msg=order_response[0]['message']))
                    order_response_question = self.session.place_order_reply(reply_id=order_response[0]['id'])
                    if 'messageIds' in order_response_question[0].keys():
                        self.logfiler.info(
                            "Message received on order : {msg}".format(msg=order_response_question[0]['message']))
                        order_response_question2 = self.session.place_order_reply(reply_id=order_response_question[0]['id'])
                    elif 'order_id' in order_response_question[0].keys():
                        self.logfiler.info("Order Submitted, OrderID = {id}".format(id=order_response_question[0]['order_id']))
                        return order_template, order_response_question
                    if 'order_id' in order_response_question2[0].keys():
                        self.logfiler.info(
                            "Order Submitted, OrderID = {id}".format(id=order_response_question2[0]['order_id']))
                        return order_template, order_response_question2
                else :
                    self.logfiler.info("{inst} order unsuccessfully placed for {sy}".format(inst=instruction, sy=option_symbol))
                    return order_template, order_response

            self.logfiler.info("{inst} order unsuccessfully placed for {sy}".format(inst=instruction, sy=option_symbol))

            return order_template, order_response

        except Exception as e:
            self.logfiler.info("Error trying to place {inst} for {sy} with error {err}".format(inst=instruction, sy=symbol, err=str(e)))
            return order_template, order_response

    def sell_stock(self, symbol, option_symbol, instruction: str, purchase_price: float):

        bidPrice = 0.0
        
        order_response = {}

        # Bot will not buy more than 10 contracts at a time.
        if self.def_buy_quantity <= 10:
            default_quantity = self.def_buy_quantity
        else:
            default_quantity = 1
            self.logfiler.info(
                "Default Sell Quantity Too High {dbq}, buy quant set to 1".format(dbq=self.def_buy_quantity))

        # Check if we have position in our portfolio
        quantity = 0
        if self.TradeOptions :
            if self.bot_portfolio.in_portfolio(option_symbol):
                quantity = self.bot_portfolio.positions[option_symbol]['quantity']
            else:
                self.logfiler.info("Do not have any {sm}".format(sm=option_symbol))
                return
        else :
            if self.bot_portfolio.in_portfolio(option_symbol):
                quantity = self.bot_portfolio.positions[option_symbol]['quantity']
            else:
                self.logfiler.info("Do not have any {sm}".format(sm=option_symbol))
                return

        if (quantity != default_quantity) and (quantity != 0) :
            default_quantity = quantity

        #orderType = self.default_mkt_limit_order_type
        orderType = 'MKT'      #changed to avoid filling orders on the sell side during fast moving market
        if orderType == 'MKT':
            if self.TradeOptions :
                order_template = {
                    'acctid': self.account_id,
                    'conid': int(option_symbol),
                    'ticker': symbol,
                    'secType': str(option_symbol) + ':' + 'OPT',
                    'orderType': orderType,
                    'quantity': default_quantity,
                    'side': 'SELL',
                    'tif': 'DAY'
                }
            else :
                order_template = {
                    'acctid': self.account_id,
                    'conid': int(option_symbol),
                    'ticker': symbol,
                    'secType': 'STK',
                    'orderType': orderType,
                    'quantity': default_quantity,
                    'side': 'SELL',
                    'tif': 'DAY'
                }
        elif orderType == 'LMT' :
            # Retrieve latest bid price for option to set limit to
            quote_fields = [55, 7295, 86, 70, 71, 84, 31, 87]
            quote_snapshot = self.session.market_data([str(option_symbol)], since=None, fields=quote_fields)
            quote_record=quote_snapshot[0]
            if '84' in quote_record.keys():
                bid_price = float(quote_record['84'])
                ask_price = float(quote_record['86'])
            else: # Second attempt always works for some reason
                quote_snapshot = self.session.market_data([str(option_symbol)], since=None, fields=quote_fields)
                quote_record = quote_snapshot[0]
                if '84' in quote_record.keys():
                    bid_price = float(quote_record['84'])
                    ask_price = float(quote_record['86'])
                else:
                    self.logfiler.info(
                        "Error trying obtain quote for {sy} ".format(sy=option_symbol))
                    order_template = {}
                    return order_template, order_response

            LimitPrice = (bid_price + ask_price) / 2
            LimitPrice = round(LimitPrice,2)

            if self.TradeOptions :
                order_template = {
                    'acctid': self.account_id,
                    'conid': int(option_symbol),
                    'ticker': symbol,
                    'secType': str(option_symbol) + ':' + 'OPT',
                    'orderType': orderType,
                    'quantity': default_quantity,
                    'side': 'SELL',
                    'price': LimitPrice,
                    'tif': 'DAY'
                }
            else:
                order_template = {
                    'acctid': self.account_id,
                    'conid': int(option_symbol),
                    'ticker': symbol,
                    'secType': 'STK',
                    'orderType': orderType,
                    'quantity': default_quantity,
                    'side': 'SELL',
                    'price': LimitPrice,
                    'tif': 'DAY'
                }

        else :
            self.logfiler.info("Invalid Order Type %s", orderType)
            order_template = {}
            return order_template, order_response

        try:
            # Added .02 to purchase price to cover commissions
            # Can't implement this with Market orders (VWAP Strategy)
            # Added 30% stop loss
            if self.no_trading_loss : # and (((purchase_price + .02) < LimitPrice) or (LimitPrice < (.85 * purchase_price))) :
                self.logfiler.info(
                    "Can't do a no loss condition for {opt_sym}".format(opt_sym=option_symbol))
                    #"Selling {opt_sym}, Selling price {sp} purchase price {purchp}".format(opt_sym=option_symbol, \
                    #               sp=str(LimitPrice), purchp=str(purchase_price)))
                self.logfiler.info("Order Template {ord}".format(ord=json.dumps(order_template, indent=4)))

                order_response = self.session.place_order(
                    account_id=self.account_id,
                    order=order_template)

                if 'order_id' in order_response[0].keys():
                    self.logfiler.info("Order Submitted, OrderID = {id}".format(id=order_response[0]['order_id']))
                    return order_template, order_response
                else:  # We have messages to respond to, we by default just respond 'true' to all of them
                    if 'messageIds' in order_response[0].keys():
                        self.logfiler.info("Message received on order : {msg}".format(msg=order_response[0]['message']))
                        order_response_question = self.session.place_order_reply(reply_id=order_response[0]['id'])
                        if 'messageIds' in order_response_question[0].keys():
                            self.logfiler.info(
                                "Message received on order : {msg}".format(msg=order_response_question[0]['message']))
                            order_response_question2 = self.session.place_order_reply(reply_id=order_response_question[0]['id'])
                        elif 'order_id' in order_response_question[0].keys():
                            self.logfiler.info(
                                "Order Submitted, OrderID = {id}".format(id=order_response_question[0]['order_id']))
                            return order_template, order_response_question
                        if 'order_id' in order_response_question2[0].keys():
                            self.logfiler.info(
                                "Order Submitted, OrderID = {id}".format(id=order_response_question2[0]['order_id']))
                            return order_template, order_response_question2
                    else:
                        self.logfiler.info(
                            "{inst} order unsuccessfully placed for {sy}".format(inst=instruction, sy=option_symbol))
                        return order_template, order_response
            elif not self.no_trading_loss :
                self.logfiler.info(
                    "Selling {opt_sym}, purchase price {purchp}".format(opt_sym=option_symbol, \
                                   purchp=str(purchase_price)))
                self.logfiler.info("Order Template {ord}".format(ord=json.dumps(order_template, indent=4)))

                order_response = self.session.place_order(
                    account_id=self.account_id,
                    order=order_template)

                if 'order_id' in order_response[0].keys():
                    self.logfiler.info("Order Submitted, OrderID = {id}".format(id=order_response[0]['order_id']))
                    return order_template, order_response
                else:  # We have messages to respond to, we by default just respond 'true' to all of them
                    if 'messageIds' in order_response[0].keys():
                        self.logfiler.info("Message received on order : {msg}".format(msg=order_response[0]['message']))
                        order_response_question = self.session.place_order_reply(reply_id=order_response[0]['id'])
                        if 'messageIds' in order_response_question[0].keys():
                            self.logfiler.info(
                                "Message received on order : {msg}".format(msg=order_response_question[0]['message']))
                            order_response_question2 = self.session.place_order_reply(reply_id=order_response_question[0]['id'])
                        elif 'order_id' in order_response_question[0].keys():
                            self.logfiler.info(
                                "Order Submitted, OrderID = {id}".format(id=order_response_question[0]['order_id']))
                            return order_template, order_response_question
                        if 'order_id' in order_response_question2[0].keys():
                            self.logfiler.info(
                                "Order Submitted, OrderID = {id}".format(id=order_response_question2[0]['order_id']))
                            return order_template, order_response_question2
                    else:
                        self.logfiler.info(
                            "{inst} order unsuccessfully placed for {sy}".format(inst=instruction, sy=option_symbol))
                        return order_template, order_response

            else :
                self.logfiler.info("Not selling {opt_sym}, Purchase price {purchp} + .02 Commission greater than selling price".format(opt_sym=option_symbol, \
                            purchp=str(purchase_price)))
                order_template = {}
                return order_template, order_response

            return order_template, order_response

        except Exception as e:
                self.logfiler.info("Error trying to place {inst} for {sy} with error code {er}".format(inst=instruction, sy=option_symbol, er=str(e)))

                return order_template, order_response


    def get_positions_for_symbol(self, underlying_symbol) -> dict:

        # Need to remove any positions which might be in the data structure
        # as we are repopulating from IBK
        self.portfolio.positions = {}
        positions_response = self.session.portfolio_account_positions(account_id=self.account_id, page_id=0)

        for position in positions_response:
            if 'acctId' not in position.keys():
                self.logfiler.info("Received Error While Requesting Positions : {pos}".format(pos=position))
                return self.portfolio.positions
            else :
                account_id = position['acctId']
            if account_id == self.account_id:
                if 'conid' in position.keys():
                    position_symbol = position['conid']
                    quantity = position['position']
                    average_price = position['avgPrice']
                    position_desc = position['contractDesc']
                    asset_type = position['assetClass']
                    if self.TradeOptions :
                        PutCall = position['putOrCall']
                    else :
                        PutCall = ''
                    avg_mkt_price = float(position['mktPrice'])

                    if position_desc.startswith(underlying_symbol):
                        if quantity != 0 :
                            if self.TradeOptions and PutCall not in ('P', 'C'):
                                # The PutCall indicator is not populated in Positions records from IBKR sometimes
                                self.logfiler.info(
                                    "Contract {pos_desc} has no Put/Call indicator while copying IBKR Positions".format(
                                        pos_desc=position_desc))
                                # We will look at the decription and find the C or P right before the [ character
                                char_position = position_desc.find('[')
                                if char_position != -1:
                                    PutCall = position_desc[(char_position - 2):(char_position - 1)]
                                    self.logfiler.info(
                                    "Extracted P/C For {pos_desc} is {PorC}".format(
                                        pos_desc=position_desc,PorC=PutCall))
                                else:
                                    self.logfiler.info(
                                        "Contract {pos_desc} Did Not Have a [ in it to extract PutCall".format(
                                            pos_desc=position_desc))

                            new_position = self.portfolio.add_position(symbol=position_symbol,
                                                                       asset_type=asset_type,
                                                                       quantity=quantity,
                                                                       purchase_price=average_price,
                                                                       description = position_desc,
                                                                       put_call_flag = PutCall,
                                                                       avg_mkt_price = avg_mkt_price)
                    else :
                        self.logfiler.info(
                            "Ignoring Position Not Associate With Bot : {pos_desc}".format(pos_desc=position_desc))
                else :
                    self.logfiler.info("Position with no conid : {pos}".format(pos=position))
            else :
                self.logfiler.info("Positions returned for {pos_acct} not associated with trading account : {acct}".format(pos_acct=account_id,acct=self.account_id))

        return self.portfolio.positions

    def Bot_Filled_Order(self, orderId) :
        for order in self.filled_orders :
            if int(orderId) == int(order['orderId']) :
                return True
        return False

    # This method assumes that the bot_portfolio has already been loaded into the bot
    #
    # We have to test if the spreadsheet reconciles with the bot_positions
    # and we have to test if the bot_positions reconcile with the spreadsheet
    # We also need to insure that the TDA positions are not zero when the
    # bot is assuming it has > 0 positions to manage (or sell).
    # If IBK has zero and bot thinks it has more, then the bot must exit
    #
    def verify_bot_positions(self, First_Initial_Test, symbol) :
        position_records = self.get_positions_for_symbol(symbol)
        self.logfiler.info('Positions At Interactive Brokers *****')
        ibk_pos_counter = 0
        pos_counter = 0
        call_counter = 0
        put_counter = 0
        if len(position_records) > 0 :
            for position_record in position_records.keys() :
                if position_records[position_record]['quantity'] > 0 :
                    ibk_pos_counter += 1
                    self.logfiler.info('    Symbol {sym}, Description {desc}, quantity {qu}'.format(sym=position_record, \
                                                                            desc=position_records[position_record]['description'],\
                                                                            qu=position_records[position_record]['quantity']))
            if ibk_pos_counter == 0 :
                self.logfiler.info('    No IBK Positions for Symbol {sym}'.format(sym=symbol))
        else :
            self.logfiler.info('    No IBK Positions for Symbol {sym}'.format(sym=symbol))

        # The boolean First_Initial_Test is set to True when the bot first starts
        # when the bot positions are assumed to be zero.
#         if not First_Initial_Test or ibk_pos_counter >= 0 :
#             # Accumulate position records for bot from the spreadsheet
#             # check quantities are equal to the bot_portfolio
#             position_tracker = {}
#             Bot_Positions_Exist = False
#             for frame_cnter in range(len(self.stock_frame)):
#                 row_id = self.stock_frame.index[frame_cnter]
#                 if (float(self.stock_frame.loc[row_id, 'order_id']) > float(0)):
#                     tick = self.stock_frame.loc[row_id, 'Ticker']
#                     quant = self.stock_frame.loc[row_id, 'quantity']
#                     # Check if orderid in our filled orders for the bot
#                     #
#                     if self.Bot_Filled_Order(self.stock_frame.loc[row_id, 'order_id']) :
#                         Bot_Positions_Exist = True
#                         if tick in position_tracker.keys():
#                             if self.stock_frame.loc[row_id, 'instruction'] == 'BUY':
#                                 position_tracker[tick] += quant
#                             else:
#                                 position_tracker[tick] -= quant
#                         else:
#                             if self.stock_frame.loc[row_id, 'instruction'] == 'BUY':
#                                 position_template = {tick: quant}
#                             else:
#                                 position_template = {tick: (quant * -1)}
#                             position_tracker.update(position_template)
#
#             if len(self.bot_portfolio.positions) > 0 :
#                 for position_record in self.bot_portfolio.positions :
#                     # Make sure the bot synthetic position is equal to
#                     # the quantity expected from reconciling with the orders on the
#                     # spreadsheet
#                     pos_symbol = self.bot_portfolio.positions[position_record]['symbol']
#                     bot_quantity = self.bot_portfolio.positions[position_record]['quantity']
#                     if (self.bot_portfolio.positions[position_record]['put_call_flag']  == 'P') :
#                         if bot_quantity > 0 :
#                             put_counter += bot_quantity
#                     elif (self.bot_portfolio.positions[position_record]['put_call_flag']  == 'C') :
#                         if bot_quantity > 0 :
#                             call_counter += bot_quantity
#                     else :
#                         self.logfiler.info('Bot has non Put/Call positions for %s', str(pos_symbol))
#                 Bot_Positions_Exist = True
#             else :
#                 Bot_Positions_Exist = False
#             # Check to make sure that IBK doesn't think Bot positions are zero when
#             # bot does not.
#             if Bot_Positions_Exist :
#                 for pos_record in self.bot_portfolio.positions.keys() :
#                     if not self.portfolio.in_portfolio(pos_record) :
#                         self.logfiler.info('Interactive Brokers has no position for %s', str(pos_record))
#                         self.logfiler.info('Bot position is %d', self.bot_portfolio.positions[position_record]['quantity'])
# #                        self.logfiler.info('Bot is exiting')
# #                        exit()
#
#                 self.logfiler.info('Positions Within Bot *****')
#                 pos_counter=0
#                 for pos_record in self.bot_portfolio.positions.keys():
#                     pos_counter += 1
#                     self.logfiler.info(
#                     '    Symbol {sym} quantity {qu}'.format(sym=pos_record, qu=self.bot_portfolio.positions[pos_record]['quantity']))
#                 if pos_counter == 0 :
#                     self.logfiler.info('    No Bot Positions')
#             else:
#                 self.logfiler.info('Positions Within Bot *****')
#                 self.logfiler.info('    No Bot Positions')

        call_counter = 0
        put_counter = 0
        over_ride_flag = 1
        #if ibk_pos_counter != pos_counter :
            #self.logfiler.info('Positions Within Bot Do Not Match IBKR Positions *****')
        if (True) :
            # self.logfiler.info('Overriding Positions Within Bot To Match IBKR Positions *****')
            self.bot_portfolio.positions = {}
            for position_record in self.portfolio.positions:
                position_symbol = self.portfolio.positions[position_record]['symbol']
                quantity = self.portfolio.positions[position_record]['quantity']
                average_price = self.portfolio.positions[position_record]['purchase_price']
                position_desc = self.portfolio.positions[position_record]['description']
                asset_type = self.portfolio.positions[position_record]['asset_type']
                Mkt_Price_Position = float(self.portfolio.positions[position_record]['mktPrice'])

                if self.TradeOptions :
                    PutCall = self.portfolio.positions[position_record]['put_call_flag']
                else :
                    PutCall = ''

                if self.TradeOptions and PutCall not in ('P', 'C'):
                    self.logfiler.info(
                    "Contract {pos_desc} has no Put/Call indicator while copying positions".format(pos_desc=position_desc))

                # There is some delay in the market prices in the portfolio positions.  So, updating the market price
                # while copying to the bot.
                #quote_fields = [55, 7295, 86, 70, 71, 84, 31, 87]
                #quote_snapshot = self.session.market_data([str(position_symbol)], since=None, fields=quote_fields)
                #quote_record = quote_snapshot[0]
                #MktData_Value = 0
                #if '31' in quote_record.keys():
                #    MktData_Value = float(quote_record['31'])
                #    self.logfiler.info("Position Market Price Positions {Pos_Mkt}, versus MktData {MD_Mkt}".format(
                #        Pos_Mkt=str(Mkt_Price_Position), MD_Mkt=quote_record['31']))
                #else :
                #    self.logfiler.info("Position Market Price Positions {Pos_Mkt}, Last Price Not Available".format(
                #        Pos_Mkt=str(Mkt_Price_Position)))
                #    MktData_Value = Mkt_Price_Position

                new_position = self.bot_portfolio.add_position(symbol=position_symbol,
                                                           asset_type=asset_type,
                                                           quantity=quantity,
                                                           purchase_price=average_price,
                                                           description=position_desc,
                                                           put_call_flag=PutCall,
                                                            avg_mkt_price = Mkt_Price_Position)
            over_ride_flag = 1
            if len(self.bot_portfolio.positions) > 0 :
                for position_record in self.bot_portfolio.positions :
                    # Make sure the bot synthetic position is equal to
                    # the quantity expected from reconciling with the orders on the
                    # spreadsheet
                    pos_symbol = self.bot_portfolio.positions[position_record]
                    bot_quantity = self.bot_portfolio.positions[position_record]['quantity']
                    if (self.bot_portfolio.positions[position_record]['put_call_flag']  == 'P') :
                        if bot_quantity > 0 :
                            put_counter += bot_quantity
                    elif (self.bot_portfolio.positions[position_record]['put_call_flag']  == 'C') :
                        if bot_quantity > 0 :
                            call_counter += bot_quantity
                    elif (not self.TradeOptions) and (self.bot_portfolio.positions[position_record]['put_call_flag'] == '') :
                        if bot_quantity > 0 :
                            call_counter += bot_quantity
                    else :
                        self.logfiler.info('Bot/IBKR has non Put/Call positions for %s', str(pos_symbol['symbol']))


        return over_ride_flag, put_counter, call_counter

    def save_orders(self, order_response_dict: dict) -> bool:
        """Saves the order to a JSON file for further review.

        Arguments:
        ----
        order_response {dict} -- A single order response.

        Returns:
        ----
        {bool} -- `True` if the orders were successfully saved.
        """

        def default(obj):

            if isinstance(obj, bytes):
                return str(obj)

        # Define the folder.
        folder: pathlib.PurePath = pathlib.Path(
            __file__
        ).parents[1].joinpath("data")

        # See if it exist, if not create it.
        if not folder.exists():
            folder.mkdir()

        # Define the file path.
        file_path = folder.joinpath('orders.json')

        # First check if the file alread exists.
        if file_path.exists():
            with open('data/orders.json', 'r') as order_json:
                orders_list = json.load(order_json)
        else:
            orders_list = []

        # Combine both lists.
        orders_list = orders_list + order_response_dict

        # Write the new data back.
        with open(file='data/orders.json', mode='w+') as order_json:
            json.dump(obj=orders_list, fp=order_json, indent=4, default=default)

        return True

    def query_orders(self, symbol, First_Loop):
        """Returns order confirmed, quantity filled, and quantity remaining"""

        order_response = self.session.get_live_orders()
        if 'orders' in order_response.keys() :
            transactions_info = order_response['orders']
        else :
            self.logfiler.info("No orders in order response {ord}".format(ord=order_response))

        # Re-establishing bot positions
        self.bot_portfolio.positions = {}

        # search for FILLED transactions
        filled_orders = []
        cumulative_calls_quantity = 0
        cumulative_puts_quantity = 0
        remaining_quantity = 0
        # Need to track options which have open positions
        orderID_column = self.stock_frame['order_id']
        est_tz = pytz.timezone('US/Eastern')
        utc_tz = pytz.UTC
        if len(transactions_info) > 0 :
            # We have to create a sorted list to ensure the last position has the latest
            # price we paid
            sort_transactions_info = sorted(transactions_info, key=operator.itemgetter('lastExecutionTime_r'))
            for order in sort_transactions_info:
                order_time_string = order['lastExecutionTime']
                # Convert to est
                date_obj= pd.to_datetime(order_time_string, format='%y%m%d%H%M%S')
                date_obj.tz_localize(tz=utc_tz)
                today_obj = datetime.now(tz=est_tz)
                date_obj.tz_localize(est_tz)  # converts to local timezone
                order_time = date_obj.strftime("%m/%d/%Y %H:%M:%S")
                #Only process active/filled trades from today with an underlying
                # that is equal to our underlying symbol
                if (order['status'] != 'Inactive') and (today_obj.date() == date_obj.date()) and (order['ticker'] == symbol):
                    historical_order_id = order['orderId']
                    OrderFound = False
                    # Loop through the DF matching up order ID's
                    for frame_cnter in range(len(orderID_column)):
                        #row_id = self.stock_frame.index[frame_cnter]
                        if float(orderID_column[frame_cnter]) == float(historical_order_id) :
                            row_id = self.stock_frame.index[frame_cnter]
                            OrderFound = True
                            contract_symbol = order['conid']
                            asset_type = order['secType']
                            if self.TradeOptions :
                                desc = order['description1'] + " " + order['description2']
                            else :
                                desc = order['description1']
                            quantity = order['remainingQuantity'] + order['filledQuantity']
                            self.stock_frame.loc[row_id, 'order_time'] = order_time

                            if order['filledQuantity'] > 0 and order['status'] == 'Filled':
                                filled_orders.append(order)
                                fill_time = order_time
                                hist_execution_price = order['avgPrice']
                                #print(contract_symbol, " Execution Price ***************", hist_execution_price)
                                self.stock_frame.loc[row_id, 'fill_time'] = fill_time
                                self.stock_frame.loc[row_id, 'price'] = hist_execution_price
                                self.stock_frame.loc[row_id, 'order_status'] = order['status']


                                # update bot synthetic positions while tracking buys and sells
                                if order['side'] == 'BUY' :
                                    putcall_flag = ''
                                    if desc.endswith('Call'):
                                        putcall_flag = 'C'
                                        cumulative_calls_quantity += quantity
                                    elif desc.endswith('Put'):
                                        putcall_flag = 'P'
                                        cumulative_puts_quantity += quantity
                                    elif not self.TradeOptions :
                                        cumulative_calls_quantity += quantity

                                    if not self.bot_portfolio.in_portfolio(contract_symbol):
                                        self.bot_portfolio.add_position(symbol=contract_symbol,
                                                                  asset_type=asset_type,
                                                                  quantity=order['filledQuantity'],
                                                                  purchase_price=hist_execution_price,
                                                                  put_call_flag=putcall_flag)
                                    else:
                                        # If already exists in portfolio, add to the quantity
                                        existing_quantity = self.bot_portfolio.positions[contract_symbol]['quantity']
                                        self.bot_portfolio.positions[contract_symbol]['quantity'] = existing_quantity + order['filledQuantity']
                                        self.bot_portfolio.positions[contract_symbol]['purchase_price'] = hist_execution_price

                                    if self.bot_portfolio.positions[contract_symbol]['quantity'] == 0 :
                                        self.bot_portfolio.remove_position(contract_symbol)

                                    remaining_quantity += order['remainingQuantity']

                                        # update sell
                                if order['side'] == 'SELL' :
                                    putcall_flag = ''
                                    if desc.endswith('Call'):
                                        putcall_flag = 'C'
                                        cumulative_calls_quantity -= order['filledQuantity']
                                    elif desc.endswith('Put'):
                                        putcall_flag = 'P'
                                        cumulative_puts_quantity -= order['filledQuantity']
                                    elif not self.TradeOptions :
                                        cumulative_calls_quantity += order['filledQuantity']

                                    # J. Jones : We might have Sell orders presented in the list before buys.
                                    # In that case, we have to add a position with a negative quantity.
                                    if not self.bot_portfolio.in_portfolio(contract_symbol):
                                        self.bot_portfolio.add_position(symbol=contract_symbol,
                                                                    asset_type=asset_type,
                                                                    quantity=order['filledQuantity'] * -1,
                                                                    purchase_price=hist_execution_price,
                                                                    put_call_flag=putcall_flag)
                                    else:
                                        # If already exists in portfolio, update the quantity
                                        self.bot_portfolio.reduce_position(contract_symbol,order['filledQuantity'] )

                                    remaining_quantity -= order['remainingQuantity']

                            else : # Order not filled, but we need to update counters to avoid adding more orders if some are pending
                                if order['side'] == 'SELL' :
                                    if desc.endswith('Call'):
                                        cumulative_calls_quantity -= 1
                                    elif desc.endswith('Put'):
                                        cumulative_puts_quantity -= 1
                                    elif not self.TradeOptions :
                                        cumulative_calls_quantity -= 1

                                if order['side'] == 'BUY' :
                                    if desc.endswith('Call'):
                                        cumulative_calls_quantity += 1
                                    elif desc.endswith('Put'):
                                        cumulative_puts_quantity += 1
                                    elif not self.TradeOptions :
                                        cumulative_calls_quantity += 1

                    if not OrderFound :
                        self.logfiler.info("Ignoring Order / Position For Order ID %d Not Originated By Bot", historical_order_id)

        self.filled_orders = filled_orders

        # If bot positions are not matching IBKR positions, we force the bot to accept the IBKR positions
        override_positions, verify_put_count, verify_call_count = self.verify_bot_positions(First_Loop, symbol)
        if override_positions == 1 :
            cumulative_puts_quantity = verify_put_count
            cumulative_calls_quantity = verify_call_count

        return filled_orders, cumulative_calls_quantity, cumulative_puts_quantity, remaining_quantity, transactions_info

    def get_accounts(self, account_number: str = None, all_accounts: bool = False) -> dict:
        """Returns all the account balances for a specified account.

        Keyword Arguments:
        ----
        account_number {str} -- The account number you want to query. (default: {None})

        all_accounts {bool} -- Specifies whether you want to grab all accounts `True` or not
            `False`. (default: {False})

        Returns:
        ----
        Dict -- A dictionary containing all the information in your account.

        Usage:
        ----

            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_accounts = trading_robot.session.get_accounts(
                account_number="<YOUR ACCOUNT NUMBER>"
            )
            >>> trading_robot_accounts
            [
                {
                    'account_number': 'ACCOUNT_ID',
                    'account_type': 'CASH',
                    'available_funds': 0.0,
                    'buying_power': 0.0,
                    'cash_available_for_trading': 0.0,
                    'cash_available_for_withdrawl': 0.0,
                    'cash_balance': 0.0,
                    'day_trading_buying_power': 0.0,
                    'long_market_value': 0.0,
                    'maintenance_call': 0.0,
                    'maintenance_requirement': 0.0,
                    'short_balance': 0.0,
                    'short_margin_value': 0.0,
                    'short_market_value': 0.0
                }
            ]
        """

        # Depending on how the client was initalized, either use the state account
        # or the one passed through the function.
        if all_accounts:
            account = 'all'
        elif self.trading_account:
            account = self.trading_account
        else:
            account = account_number

        # Grab the accounts.
        accounts = self.session.get_accounts(
            account=account
        )

        # Parse the account info.
        accounts_parsed = self._parse_account_balances(
            accounts_response=accounts
        )

        return accounts_parsed

    def _parse_account_balances(self, accounts_response: Union[Dict, List]) -> List[Dict]:
        """Parses an Account response into a more simplified dictionary.

        Arguments:
        ----
        accounts_response {Union[Dict, List]} -- A response from the `get_accounts` call.

        Returns:
        ----
        List[Dict] -- A list of simplified account dictionaries.
        """

        account_lists = []

        if isinstance(accounts_response, dict):

            account_dict = {}

            for account_type_key in accounts_response:

                account_info = accounts_response[account_type_key]

                account_id = account_info['accountId']
                account_type = account_info['type']
                account_current_balances = account_info['currentBalances']
                # account_inital_balances = account_info['initialBalances']

                account_dict['account_number'] = account_id
                account_dict['account_type'] = account_type
                account_dict['cash_balance'] = account_current_balances['cashBalance']
                account_dict['long_market_value'] = account_current_balances['longMarketValue']

                account_dict['cash_available_for_trading'] = account_current_balances.get(
                    'cashAvailableForTrading', 0.0
                )
                account_dict['cash_available_for_withdrawl'] = account_current_balances.get(
                    'cashAvailableForWithDrawal', 0.0
                )
                account_dict['available_funds'] = account_current_balances.get(
                    'availableFunds', 0.0
                )
                account_dict['buying_power'] = account_current_balances.get(
                    'buyingPower', 0.0
                )
                account_dict['day_trading_buying_power'] = account_current_balances.get(
                    'dayTradingBuyingPower', 0.0
                )
                account_dict['maintenance_call'] = account_current_balances.get(
                    'maintenanceCall', 0.0
                )
                account_dict['maintenance_requirement'] = account_current_balances.get(
                    'maintenanceRequirement', 0.0
                )

                account_dict['short_balance'] = account_current_balances.get(
                    'shortBalance', 0.0
                )
                account_dict['short_market_value'] = account_current_balances.get(
                    'shortMarketValue', 0.0
                )
                account_dict['short_margin_value'] = account_current_balances.get(
                    'shortMarginValue', 0.0
                )

                account_lists.append(account_dict)

        elif isinstance(accounts_response, list):

            for account in accounts_response:

                account_dict = {}

                for account_type_key in account:

                    account_info = account[account_type_key]

                    account_id = account_info['accountId']
                    account_type = account_info['type']
                    account_current_balances = account_info['currentBalances']
                    # account_inital_balances = account_info['initialBalances']

                    account_dict['account_number'] = account_id
                    account_dict['account_type'] = account_type
                    account_dict['cash_balance'] = account_current_balances['cashBalance']
                    account_dict['long_market_value'] = account_current_balances['longMarketValue']

                    account_dict['cash_available_for_trading'] = account_current_balances.get(
                        'cashAvailableForTrading', 0.0
                    )
                    account_dict['cash_available_for_withdrawl'] = account_current_balances.get(
                        'cashAvailableForWithDrawal', 0.0
                    )
                    account_dict['available_funds'] = account_current_balances.get(
                        'availableFunds', 0.0
                    )
                    account_dict['buying_power'] = account_current_balances.get(
                        'buyingPower', 0.0
                    )
                    account_dict['day_trading_buying_power'] = account_current_balances.get(
                        'dayTradingBuyingPower', 0.0
                    )
                    account_dict['maintenance_call'] = account_current_balances.get(
                        'maintenanceCall', 0.0
                    )
                    account_dict['maintenance_requirement'] = account_current_balances.get(
                        'maintenanceRequirement', 0.0
                    )
                    account_dict['short_balance'] = account_current_balances.get(
                        'shortBalance', 0.0
                    )
                    account_dict['short_market_value'] = account_current_balances.get(
                        'shortMarketValue', 0.0
                    )
                    account_dict['short_margin_value'] = account_current_balances.get(
                        'shortMarginValue', 0.0
                    )

                    account_lists.append(account_dict)

        return account_lists

    def get_positions(self, account_number: str = None, all_accounts: bool = False) -> List[Dict]:
        """Gets all the positions for a specified account number.

        Arguments:
        ----
        account_number (str, optional): The account number of the account you want
            to pull positions for. Defaults to None.

        all_accounts (bool, optional): If you want to return all the positions for every
            account then set to `True`. Defaults to False.

        Returns:
        ----
        List[Dict]: A list of Position objects.

        Usage:
        ----

            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_positions = trading_robot.session.get_positions(
                account_number="<YOUR ACCOUNT NUMBER>"
            )
            >>> trading_robot_positions
            [
                {
                    'account_number': '111111111',
                    'asset_type': 'EQUITY',
                    'average_price': 0.00,
                    'current_day_profit_loss': -0.96,
                    'current_day_profit_loss_percentage': -5.64,
                    'cusip': '565849106',
                    'description': '',
                    'long_quantity': 3.0,
                    'market_value': 16.05,
                    'settled_long_quantity': 3.0,
                    'settled_short_quantity': 0.0,
                    'short_quantity': 0.0,
                    'sub_asset_type': '',
                    'symbol': 'MRO',
                    'type': ''
                },
                {
                    'account_number': '111111111',
                    'asset_type': 'EQUITY',
                    'average_price': 5.60667,
                    'current_day_profit_loss': -0.96,
                    'current_day_profit_loss_percentage': -5.64,
                    'cusip': '565849106',
                    'description': '',
                    'long_quantity': 3.0,
                    'market_value': 16.05,
                    'settled_long_quantity': 3.0,
                    'settled_short_quantity': 0.0,
                    'short_quantity': 0.0,
                    'sub_asset_type': '',
                    'symbol': 'MRO',
                    'type': ''
                }
            ]
        """

        if all_accounts:
            account = 'all'
        elif self.trading_account and account_number is None:
            account = self.trading_account
        else:
            account = account_number

        # Grab the positions.
        positions = self.session.get_accounts(
            account=account,
            fields=['positions']
        )

        # Parse the positions.
        positions_parsed = self._parse_account_positions(
            positions_response=positions
        )

        return positions_parsed

    def _parse_account_positions(self, positions_response: Union[List, Dict]) -> List[Dict]:
        """Parses the response from the `get_positions` into a more simplified list.

        Arguments:
        ----
        positions_response {Union[List, Dict]} -- Either a list or a dictionary that represents a position.

        Returns:
        ----
        List[Dict] -- A more simplified list of positions.
        """

        positions_lists = []

        if isinstance(positions_response, dict):

            for account_type_key in positions_response:

                account_info = positions_response[account_type_key]

                account_id = account_info['accountId']
                positions = account_info['positions']

                for position in positions:
                    position_dict = {}
                    position_dict['account_number'] = account_id
                    position_dict['average_price'] = position['averagePrice']
                    position_dict['market_value'] = position['marketValue']
                    position_dict['current_day_profit_loss_percentage'] = position['currentDayProfitLossPercentage']
                    position_dict['current_day_profit_loss'] = position['currentDayProfitLoss']
                    position_dict['long_quantity'] = position['longQuantity']
                    position_dict['short_quantity'] = position['shortQuantity']
                    position_dict['settled_long_quantity'] = position['settledLongQuantity']
                    position_dict['settled_short_quantity'] = position['settledShortQuantity']

                    position_dict['symbol'] = position['instrument']['symbol']
                    position_dict['cusip'] = position['instrument']['cusip']
                    position_dict['asset_type'] = position['instrument']['assetType']
                    position_dict['sub_asset_type'] = position['instrument'].get(
                        'subAssetType', ""
                    )
                    position_dict['description'] = position['instrument'].get(
                        'description', ""
                    )
                    position_dict['type'] = position['instrument'].get(
                        'type', ""
                    )

                    positions_lists.append(position_dict)

        elif isinstance(positions_response, list):

            for account in positions_response:

                for account_type_key in account:

                    account_info = account[account_type_key]

                    account_id = account_info['accountId']
                    positions = account_info['positions']

                    for position in positions:
                        position_dict = {}
                        position_dict['account_number'] = account_id
                        position_dict['average_price'] = position['averagePrice']
                        position_dict['market_value'] = position['marketValue']
                        position_dict['current_day_profit_loss_percentage'] = position['currentDayProfitLossPercentage']
                        position_dict['current_day_profit_loss'] = position['currentDayProfitLoss']
                        position_dict['long_quantity'] = position['longQuantity']
                        position_dict['short_quantity'] = position['shortQuantity']
                        position_dict['settled_long_quantity'] = position['settledLongQuantity']
                        position_dict['settled_short_quantity'] = position['settledShortQuantity']

                        position_dict['symbol'] = position['instrument']['symbol']
                        position_dict['cusip'] = position['instrument']['cusip']
                        position_dict['asset_type'] = position['instrument']['assetType']
                        position_dict['sub_asset_type'] = position['instrument'].get(
                            'subAssetType', ""
                        )
                        position_dict['description'] = position['instrument'].get(
                            'description', ""
                        )
                        position_dict['type'] = position['instrument'].get(
                            'type', ""
                        )

                        positions_lists.append(position_dict)

        return positions_lists




    
