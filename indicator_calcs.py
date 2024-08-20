import numpy as np
import pandas as pd
import operator
import math
import re
import json


from typing import Any
from typing import Dict
from typing import Union
from typing import Optional
from typing import Tuple
from stock_frame import StockFrame
from portfolio import Portfolio

import datetime
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import pytz


pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)



class Indicators():
    """
    Represents an Indicator Object which can be used
    to easily add technical indicators to a StockFrame.
    """

    def __init__(self, price_data_frame: StockFrame, lgfile=None) -> None:
        """Initalizes the Indicator Client.

        Arguments:
        ----
        price_data_frame {pyrobot.StockFrame} -- The price data frame which is used to add indicators to.
            At a minimum this data frame must have the following columns: `['timestamp','close','open','high','low']`.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.price_data_frame
        """

        self._stock_frame: StockFrame = price_data_frame
        # change this price_groups = self._stock_frame.symbol_groups
        self._price_groups = price_data_frame.symbol_groups
        # J. Jones - removed the _current_indicators object variable to
        # disable the use of local symbol table and other related manipulations
        #l self._current_indicators = {}
        self._indicator_signals = {}
        self.option_data = []
        self.stock_data = None
        self.indicator_signal_list = []
        self.calls_options = []
        self.puts_options = []
        self.fixed_call_option_strike = ''
        self.fixed_put_option_strike = ''
        self.order_list = {'buy_calls_count': 0, 'buy_puts_count': 0, 'no_action_calls_count': 0, 'no_action_puts_count' : 0}

        # TODO: use Alex's add_rows() function instead of updating whole dataframe
        self._frame = self._stock_frame.frame

        # add by nikhil for storing buy count
        self.buy_count = 0

        self._indicators_comp_key = []
        self._indicators_key = []
        self.logfiler = lgfile

        if self.is_multi_index:
            True

    def get_indicator_signal(self, indicator: str = None) -> Dict:
        """Return the raw Pandas Dataframe Object.

        Arguments:
        ----
        indicator {Optional[str]} -- The indicator key, for example `ema` or `sma`.

        Returns:
        ----
        {dict} -- Either all of the indicators or the specified indicator.
        """

        if indicator and indicator in self._indicator_signals:
            return self._indicator_signals[indicator]
        else:
            return self._indicator_signals

    def set_indicator_signal(self, indicator: str, buy: float, sell: float, condition_buy: Any, condition_sell: Any,
                             buy_max: float = None, sell_max: float = None, condition_buy_max: Any = None,
                             condition_sell_max: Any = None) -> None:
        """Used to set an indicator where one indicator crosses above or below a certain numerical threshold.

        Arguments:
        ----
        indicator {str} -- The indicator key, for example `ema` or `sma`.

        buy {float} -- The buy signal threshold for the indicator.

        sell {float} -- The sell signal threshold for the indicator.

        condition_buy {str} -- The operator which is used to evaluate the `buy` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.

        condition_sell {str} -- The operator which is used to evaluate the `sell` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.

        buy_max {float} -- If the buy threshold has a maximum value that needs to be set, then set the `buy_max` threshold.
            This means if the signal exceeds this amount it WILL NOT PURCHASE THE INSTRUMENT. (defaults to None).

        sell_max {float} -- If the sell threshold has a maximum value that needs to be set, then set the `buy_max` threshold.
            This means if the signal exceeds this amount it WILL NOT SELL THE INSTRUMENT. (defaults to None).

        condition_buy_max {str} -- The operator which is used to evaluate the `buy_max` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`. (defaults to None).

        condition_sell_max {str} -- The operator which is used to evaluate the `sell_max` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`. (defaults to None).
        """

        # Add the key if it doesn't exist.
        if indicator not in self._indicator_signals:
            self._indicator_signals[indicator] = {}
            self._indicators_key.append(indicator)

            # Add the signals.
        self._indicator_signals[indicator]['buy'] = buy
        self._indicator_signals[indicator]['sell'] = sell
        self._indicator_signals[indicator]['buy_operator'] = condition_buy
        self._indicator_signals[indicator]['sell_operator'] = condition_sell

        # Add the max signals
        self._indicator_signals[indicator]['buy_max'] = buy_max
        self._indicator_signals[indicator]['sell_max'] = sell_max
        self._indicator_signals[indicator]['buy_operator_max'] = condition_buy_max
        self._indicator_signals[indicator]['sell_operator_max'] = condition_sell_max

    def set_indicator_signal_compare(self, indicator_1: str, indicator_2: str, condition_buy: Any,
                                     condition_sell: Any) -> None:
        """Used to set an indicator where one indicator is compared to another indicator.

        Overview:
        ----
        Some trading strategies depend on comparing one indicator to another indicator.
        For example, the Simple Moving Average crossing above or below the Exponential
        Moving Average. This will be used to help build those strategies that depend
        on this type of structure.

        Arguments:
        ----
        indicator_1 {str} -- The first indicator key, for example `ema` or `sma`.

        indicator_2 {str} -- The second indicator key, this is the indicator we will compare to. For example,
            is the `sma` greater than the `ema`.

        condition_buy {str} -- The operator which is used to evaluate the `buy` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.

        condition_sell {str} -- The operator which is used to evaluate the `sell` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.
        """

        # Define the key.
        key = "{ind_1}_comp_{ind_2}".format(
            ind_1=indicator_1,
            ind_2=indicator_2
        )

        # Add the key if it doesn't exist.
        if key not in self._indicator_signals:
            self._indicator_signals[key] = {}
            self._indicators_comp_key.append(key)

            # Grab the dictionary.
        indicator_dict = self._indicator_signals[key]

        # Add the signals.
        indicator_dict['type'] = 'comparison'
        indicator_dict['indicator_1'] = indicator_1
        indicator_dict['indicator_2'] = indicator_2
        indicator_dict['buy_operator'] = condition_buy
        indicator_dict['sell_operator'] = condition_sell

    @property
    def price_data_frame(self) -> pd.DataFrame:
        """Return the raw Pandas Dataframe Object.

        Returns:
        ----
        {pd.DataFrame} -- A multi-index data frame.
        """

        return self._frame

    @price_data_frame.setter
    def price_data_frame(self, price_data_frame: pd.DataFrame) -> None:
        """Sets the price data frame.

        Arguments:
        ----
        price_data_frame {pd.DataFrame} -- A multi-index data frame.
        """

        self._frame = price_data_frame

    @property
    def is_multi_index(self) -> bool:
        """Specifies whether the data frame is a multi-index dataframe.

        Returns:
        ----
        {bool} -- `True` if the data frame is a `pd.MultiIndex` object. `False` otherwise.
        """

        if isinstance(self._frame.index, pd.MultiIndex):
            return True
        else:
            return False

    def change_in_price(self, column_name: str = 'change_in_price') -> pd.DataFrame:
        """Calculates the Change in Price.

        Returns:
        ----
        {pd.DataFrame} -- A data frame with the Change in Price included.
        """

        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.diff()
        )

        return self._frame

    def rsi(self, period: int, method: str = 'wilders', column_name: str = 'rsi') -> pd.DataFrame:
        """Calculates the Relative Strength Index (RSI).

        Arguments:
        ----
        period {int} -- The number of periods to use to calculate the RSI.

        Keyword Arguments:
        ----
        method {str} -- The calculation methodology. (default: {'wilders'})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the RSI indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.rsi(period=14)
            >>> price_data_frame = inidcator_client.price_data_frame
        """

        # First calculate the Change in Price.
        if 'per_of_change' not in self._frame.columns:
            self.change_in_price()

        # Define the up days.
        self._frame['up_day'] = self._price_groups['per_of_change'].transform(
            lambda x: np.where(x >= 0, x, 0)
        )

        # Define the down days.
        self._frame['down_day'] = self._price_groups['per_of_change'].transform(
            lambda x: np.where(x < 0, x.abs(), 0)
        )

        # Calculate the EWMA for the Up days.
        self._frame['ewma_up'] = self._price_groups['up_day'].transform(
            lambda x: x.ewm(span=period).mean()
        )

        # Calculate the EWMA for the Down days.
        self._frame['ewma_down'] = self._price_groups['down_day'].transform(
            lambda x: x.ewm(span=period).mean()
        )

        # Calculate the Relative Strength
        relative_strength = self._frame['ewma_up'] / self._frame['ewma_down']

        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

        # Add the info to the data frame.
        self._frame['rsi'] = np.where(relative_strength_index == 0, 100, 100 - (100 / (1 + relative_strength_index)))

        # Clean up before sending back.
        self._frame.drop(
            labels=['ewma_up', 'ewma_down', 'down_day', 'up_day'],
            axis=1,
            inplace=True
        )

        return self._frame

    def abs_3_minus_50_direction(self, column_name: str = 'abs_3_minus_50_direction'):

        # grab sma_9 and sma_50
        sma_3 = self._frame["sma_3"]
        sma_50 = self._frame["sma_50"]

        # calculate abs_diff_slope
        temp_sma_3_minus_50 = []
        for val_3, val_50 in zip(sma_3, sma_50):
            if math.isnan(val_3) or math.isnan(val_50):
                temp_sma_3_minus_50.append(0)
            else:
                temp_sma_3_minus_50.append(round(abs(val_3 - val_50) * 0.0174533 * 1000, 4))

        self._frame[column_name] = pd.Series(temp_sma_3_minus_50).values

        return self._frame

    def abs_9_minus_50_direction(self, column_name: str = 'abs_9_minus_50_direction') -> pd.DataFrame:

        # grab sma_9 and sma_50
        sma_9 = self._frame["sma_9"]
        sma_50 = self._frame["sma_50"]

        # calculate abs_diff_slope
        temp_sma_9_minus_50 = []
        for val_9, val_50 in zip(sma_9, sma_50):
            if math.isnan(val_9) or math.isnan(val_50):
                temp_sma_9_minus_50.append(0)
            else:
                temp_sma_9_minus_50.append(round(abs(val_9 - val_50) * 0.0174533 * 1000, 4))

        self._frame[column_name] = pd.Series(temp_sma_9_minus_50).values

        return self._frame

    def sma(self, period: int, column_name: str = 'sma') -> pd.DataFrame:
        """Calculates the Simple Moving Average (SMA).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating the SMA.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the SMA indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.sma(period=100)
        """

        # print(self.price_data_frame)
        # Add the SMA
        self._frame[column_name + '_' + str(period)] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # nikhil modified
        # adding logic for sma 9 slope
        index_count = 1
        prev = 0
        next = 0
        temp_list = []
        for index, row in self._frame.iterrows():

            if index_count == 1:
                prev = next = row[column_name + '_' + str(period)]
                temp_list.append(0)

            else:
                prev = next
                next = row[column_name + '_' + str(period)]

                if math.isnan(prev) or math.isnan(next):
                    temp_list.append(0)
                else:
                    temp_list.append(round(((next - prev) * 0.0174533) * 10000, 4))

            # print(prev, next)
            index_count += 1
        self._frame[column_name + '_' + str(period) + '_slope'] = pd.Series(temp_list).values
        # print(row[column_name])

        # print(self._frame)
        return self._frame

    def sma_volume(self, period: int, column_name: str = 'sma_volume') -> pd.DataFrame:
        """Calculates the Simple Moving Average (SMA).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating the SMA.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the SMA indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.sma(period=100)
        """

        # print(self.price_data_frame)
        # Add the SMA
        self._frame[column_name + '_' + str(period)] = self._price_groups['volume'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # nikhil modified
        # adding logic for sma 9 slope
        index_count = 1
        prev = 0
        next = 0
        temp_list = []
        for index, row in self._frame.iterrows():

            if index_count == 1:
                prev = next = row[column_name + '_' + str(period)]
                temp_list.append(0)

            else:
                prev = next
                next = row[column_name + '_' + str(period)]

                if math.isnan(prev) or math.isnan(next):
                    temp_list.append(0)
                else:
                    temp_list.append(round(((next - prev) * 0.00174533), 4))

            # print(prev, next)
            index_count += 1

        self._frame[column_name + '_' + str(period) + '_slope'] = pd.Series(temp_list).values
        # print(row[column_name])

        # print(self._frame)
        return self._frame

    def per_of_change(self) -> pd.DataFrame:

        # calculate per of change
        per_of_change = []
        count = 0

        while count < len(self._frame):
            if count == 0:
                per_of_change.append(0)
                prev = next = self._frame["close"][count]
            else:
                prev = next
                next = self._frame["close"][count]
                per_of_change.append(round(((next - prev) / prev) * 100, 3))
            count += 1

        self._frame['per_of_change'] = pd.Series(per_of_change).values

        return self._frame

    def vwap(self, start_time_vwap, column_name: str = 'vwap') -> pd.DataFrame:
        # """Calculates the Volume Weighted Adjusted Price (VWAP).
        #
        # Returns:
        # ----
        # {pd.DataFrame} -- A Pandas data frame with the vwap indicator included.
        #
        # Usage:
        # ----
        #     >>> historical_prices_df = trading_robot.grab_historical_prices(
        #         start=start_date,
        #         end=end_date,
        #         bar_size=1,
        #         bar_type='minute'
        #     )
        #     >>> price_data_frame = pd.DataFrame(data=historical_prices)
        #     >>> indicator_client = Indicators(price_data_frame=price_data_frame)
        #     >>> indicator_client.vwap()
        # """
        est_tz = pytz.timezone('US/Eastern')


        self._frame['typical_mult_volume'] = ((self._frame["low"] + self._frame["close"] + self._frame["high"]) / 3) * \
                                             self._frame["volume"]

        if 'Cumulative_TPV' not in self._frame.columns:
            self._frame['Cumulative_TPV'] = 0.0

        if 'Cumulative_Vol' not in self._frame.columns:
            self._frame['Cumulative_Vol'] = 0.0

        for i in range(len(self._frame)):
            string_datetimevalue = self._frame['new york time'].values[i]
            candle_datetime = datetime.strptime(string_datetimevalue, "%Y-%m-%d %H:%M:%S")
            candle_datetime = est_tz.localize(candle_datetime)
            if candle_datetime < start_time_vwap or i == 0 :
                self._frame["Cumulative_TPV"].values[i] = 0.0
                self._frame["Cumulative_Vol"].values[i] = 0.0
            else :
                self._frame["Cumulative_TPV"].values[i] = self._frame["typical_mult_volume"].values[i] + self._frame["Cumulative_TPV"].values[i-1]
                self._frame["Cumulative_Vol"].values[i] = self._frame["volume"].values[i] + self._frame["Cumulative_Vol"].values[i-1]


        # print(self._frame)
        temp_list = []
        # print(self._frame["typical_mult_volume"][mantain_index:mantain_index+2].values.mean())
        # print(self._frame["volume"][mantain_index:mantain_index+2].values.mean())
        for i in range(len(self._frame)):
            # print(self._frame["typical_mult_volume"][mantain_index:mantain_index + 2].values)
            # print(self._frame["volume"][mantain_index:mantain_index + 2].values)
            if self._frame["Cumulative_Vol"].values[i] != 0 :
                temp_list.append(self._frame["Cumulative_TPV"].values[i] / self._frame["Cumulative_Vol"].values[i])
            else :
                temp_list.append(0.0)

        # print(len(temp_list), len(self._frame))
        self._frame[column_name] = pd.Series(temp_list).values

        # The VWAP should only be intra day, so is only calculated at start of the trading day
        #
        #for i in range(len(self._frame)):
        #    string_datetimevalue = self._frame['new york time'].values[i]
        #    candle_datetime = datetime.datetime.strptime(string_datetimevalue, "%Y-%m-%d %H:%M:%S")
        #    candle_datetime = est_tz.localize(candle_datetime)
        #    if candle_datetime < earliest_order  :
        #        self._frame[column_name].values[i] = math.nan

        # 12/10/2023 We do not need slope for the VWAP Strategy
        #index_count = 1
        #prev = 0
        #next = 0
        #temp_list = []
        #for index, row in self._frame.iterrows():

        #    if index_count == 1:
        #        prev = next = row[column_name]
        #        temp_list.append(0)

        #    else:
        #        prev = next
        #        next = row[column_name]

        #        if math.isnan(prev) or math.isnan(next):
        #            temp_list.append(0)
        #        else:
        #            temp_list.append(round(((next - prev) * 0.174533) * 10000, 4))

            # print(prev, next)
        #    index_count += 1
        #self._frame[column_name + '_slope'] = pd.Series(temp_list).values

        return self._frame

    def reformat_symbol(self, symbol):
        """Returns the contract identifier from the symbol"""
        symbol_new_format_list = symbol.split('_')
        symbol_new_format = '.' + symbol_new_format_list[0] + symbol_new_format_list[1][4:6] + \
                            symbol_new_format_list[1][0:2] + symbol_new_format_list[1][2:4] + \
                            symbol_new_format_list[1][6:]
        return symbol_new_format

    def filter_non_numeric(self, input_str=str) -> str :
        if input_str == '' :
            numeric_str = '0.0'
        else :
            numeric_str = re.sub("[^\d\.]", "", input_str)
        return numeric_str

    def max_option_chain(self, IBSession, symbol, conid):

        AddOneRow = False

        params = {
            "symbol" : symbol,
            "conid": conid,
            "range": "NTM",
            "strikeCount" : 6
        }

        # Acquiring all the strikes and options is very time consuming, so only doing at the beginning of the session
        if len(self.option_data) == 0 :
            option_chain = IBSession.get_options_chain(conid=conid, option_chain_params=params)
            options_list_calls = option_chain[conid]['call_options']
            options_list_puts = option_chain[conid]['put_options']
            self.option_data.append(option_chain)
        else :
            option_chain = self.option_data[0]
            options_list_calls = option_chain[conid]['call_options']
            options_list_puts = option_chain[conid]['put_options']

        #55=underlying symbol(str), 7295=open_price(float), 86=ask_price(float), 70=high(float),
        # 71=low(float), 84=bid_price(float), 31=last_price(float), 87=volume(int)
        quote_fields = [55, 7295, 86, 70, 71, 84, 31, 87]
        # Gather the ID's to request the market data fields for each option
        id_list=''
        for option_record in options_list_calls :
            id_list = id_list + str(option_record['conid']) + ','

        if len(id_list) > 1 :
            #remove trailing comma
            id_request_list = id_list[:-1]
        else :
            self.logfiler.info("No call option ID list to obtain market prices")
            return self._frame

        Calls_snapshot = IBSession.market_data(id_request_list, since=None, fields=quote_fields)

        id_list = ''
        for option_record in options_list_puts:
            id_list = id_list + str(option_record['conid']) + ','

        if len(id_list) > 1 :
            #remove trailing comma
            id_request_list = id_list[:-1]
        else :
            self.logfiler.info("No put option ID list to obtain market prices")
            return self._frame
        # remove trailing comma
        id_request_list = id_list[:-1]

        Puts_snapshot = IBSession.market_data(id_request_list, since=None, fields=quote_fields)

        # Data for the options are in the following data objects
        #         options_list_calls: Contains the month, strikes, descriptions and conid's of the NTM calls
        #         options_list_puts: Contains the month, strikes, descriptions and conid's of the NTM puts
        #         Calls_snapshot : Contains the market data quotes for the NTM calls
        #         Puts_snapshot : Contains the market dat quotes for the NTM puts
        # Insert the bid, ask, last in the options_list_calls
        # Create a dataframe with the options information
        call_df = pd.DataFrame(options_list_calls)
        put_df = pd.DataFrame(options_list_puts)
        # add the market data columns to the dataframes
        Market_Data_Column_Names = [
            'option_id',
            'right',
            'total_volume_option',
            'bid',
            'ask',
            'last',
            'quoteTime',
            'daysToExpiration'
        ]
        # add the column names
        for i in range(len(Market_Data_Column_Names)):
            call_df[Market_Data_Column_Names[i]] = float('NaN')
            put_df[Market_Data_Column_Names[i]] = float('NaN')
        # add the market data into each dataframe
        # if we don't get market data back for a row (ie. if it has a value of 'NaN', we delete the row
        call_df.set_index('conid', inplace=True)
        put_df.set_index('conid', inplace=True)
        for call_record in Calls_snapshot :
            conid_int = int(call_record['conid'])
            if conid_int in call_df.index :
                call_df.loc[conid_int, 'right'] = 'C'
                if '87_raw' in call_record.keys() :
                    # 87 is the volume but in a nn.nk format where k is the thousands, 87_raw is the actual float number
                    volume = float(call_record['87_raw'])
                    call_df.loc[conid_int,'total_volume'] = volume
                    call_df.loc[conid_int,'option_id'] = conid_int
                    if '31' in call_record.keys():
                        call_df.loc[conid_int,'last'] = float(self.filter_non_numeric(call_record['31']))
                    if '84' in call_record.keys():
                        call_df.loc[conid_int,'bid'] = float(self.filter_non_numeric(call_record['84']))
                    if '86' in call_record.keys():
                        call_df.loc[conid_int,'ask'] = float(self.filter_non_numeric(call_record['86']))
                    if '_updated' in call_record.keys():
                        call_df.loc[conid_int,'quoteTime'] = float(call_record['_updated'])

                    date_value_str = str(call_df.loc[conid_int,'maturityDate'])
                    datetime_obj = pd.to_datetime(date_value_str, format='%Y%m%d')
                    est_tz = pytz.timezone('US/Eastern')
                    datetime_obj.tz_localize(est_tz)   # converts to local timezone
                    timestampObj_est = datetime_obj
                    timestampObj_est.replace(hour=0, minute=0, second=0, microsecond=0)
                    daystoexpiration = (timestampObj_est - datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)).days
                    call_df.loc[conid_int,'daysToExpiration'] = daystoexpiration
                else :
                    call_df.loc[conid_int, 'right'] = 'C'
                    call_df.loc[conid_int,'total_volume'] = 0
                    call_df.loc[conid_int,'daysToExpiration'] = 0
                    self.logfiler.info("No volume returned for call ID %d", conid_int)
            else :
                self.logfiler.info("Invalid ID in call quotes %d", conid_int)
                return self._frame

        for put_record in Puts_snapshot :
            conid_int = int(put_record['conid'])
            if conid_int in put_df.index :
                put_df.loc[conid_int, 'right'] = 'P'
                if '87_raw' in put_record.keys() :
                    # 87 is the volume but in a nn.nk format where k is the thousands, 87_raw is the actual float number
                    volume = float(put_record['87_raw'])
                    put_df.loc[conid_int,'total_volume'] = volume
                    put_df.loc[conid_int,'option_id'] = conid_int
                    if '31' in put_record.keys():
                        put_df.loc[conid_int,'last'] = float(self.filter_non_numeric(put_record['31']))
                    if '84' in put_record.keys():
                        put_df.loc[conid_int,'bid'] = float(self.filter_non_numeric(put_record['84']))
                    if '86' in put_record.keys():
                        put_df.loc[conid_int,'ask'] = float(self.filter_non_numeric(put_record['86']))
                    if '_updated' in put_record.keys():
                        put_df.loc[conid_int,'quoteTime'] = float(put_record['_updated'])

                    date_value_str = str(put_df.loc[conid_int,'maturityDate'])
                    datetime_obj = pd.to_datetime(date_value_str, format='%Y%m%d')
                    est_tz = pytz.timezone('US/Eastern')
                    datetime_obj.tz_localize(est_tz)  # converts to local timezone
                    timestampObj_est = datetime_obj
                    timestampObj_est.replace(hour=0, minute=0, second=0, microsecond=0)
                    daystoexpiration = (timestampObj_est - datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)).days
                    put_df.loc[conid_int,'daysToExpiration'] = daystoexpiration
                else :
                    put_df.loc[conid_int, 'right'] = 'P'
                    put_df.loc[conid_int,'total_volume'] = 0
                    put_df.loc[conid_int,'daysToExpiration'] = 0
                    self.logfiler.info("No volume returned for put ID %d", conid_int)
            else :
                self.logfiler.info("Invalid ID in put quotes %d", conid_int)
                return self._frame
        call_df.drop(columns=['listingExchange', 'cusip', 'coupon'], inplace=True)
        call_df.dropna(inplace=True)   # if we don't get market data back for certain rows, we delete them
        if len(call_df) == 0 :
            self.logfiler.info("No Call Market Data, proceeding without Option information")
            put_df.drop(columns=['listingExchange', 'cusip', 'coupon'], inplace=True)
            put_df.dropna(inplace=True)  # if we don't get market data back for certain rows, we delete them
            self.calls_options.append(str(int(0)))
            self.puts_options.append(str(int(0)))
            return self._frame
        put_df.drop(columns=['listingExchange', 'cusip', 'coupon'], inplace=True)
        put_df.dropna(inplace=True)   # if we don't get market data back for certain rows, we delete them
        if len(put_df) == 0 :
            self.logfiler.info("No Put Market Data, proceeding without Option information")
            self.calls_options.append(str(int(0)))
            self.puts_options.append(str(int(0)))
            return self._frame


        # Get the calls with max volume
        df_result = call_df.loc[(call_df['daysToExpiration'] > 5) and (call_df['right'] == 'C')]
        df2_result = df_result.loc[df_result['total_volume'] == df_result['total_volume'].max()]
        if len(df2_result) == 0 :
            self.logfiler.info("No Call Market Data With Max Volume, proceeding without Option information")
            self.calls_options.append(str(int(0)))
            self.puts_options.append(str(int(0)))
            return self._frame
        max_vol_record = df2_result.iloc[0]
        max_volume_calls_index = int(max_vol_record['option_id'])
        if self.fixed_call_option_strike == '' :
            self.calls_options.append(str(int(max_volume_calls_index)))
            self.fixed_call_option_strike = str(int(max_volume_calls_index))
        else :
            self.calls_options.append(str(int(max_volume_calls_index)))

        # Use the quoteTime to determine where in the dataframe to insert the data
        # so it matches up against the underlying candles from the previous minute
        # Loop through the DF
        timestampvalue = call_df.loc[max_volume_calls_index,'quoteTime'] / 1000
        timestampObj = datetime.fromtimestamp(timestampvalue)  # convert to est
        timestampObj.astimezone()    # assigns local timezone to the object
        timestampObj_local = timestampObj
        est_tz = pytz.timezone('US/Eastern')
        utc_tz = pytz.UTC
        #row_id = self._frame.index[0]
        timestampObj_utc =  timestampObj.astimezone(utc_tz)
        timestampObj_utc = timestampObj_utc.replace(second=0, microsecond=0)
        timestampObj_est = timestampObj_local.astimezone(est_tz)
        timestampstring = timestampObj_est.strftime("%Y-%m-%d %H:%M:%S")
        #utc_timestampstring = timestampObj_utc.strftime("%Y-%m-%d %H:%M:%S")
        Call_Description = call_df.loc[max_volume_calls_index,'symbol'] + \
                                str(call_df.loc[max_volume_calls_index, 'maturityDate']) + \
                                str(call_df.loc[max_volume_calls_index, 'strike']) + \
                                call_df.loc[max_volume_calls_index, 'right'] + " LastPrice=" + \
                                str(call_df.loc[max_volume_calls_index,'last'])
        self.logfiler.info("EST Time {est_time} For {Option_str} ".format(est_time=timestampstring, Option_str=Call_Description))
        Option_Prices_Current = False
        for frame_cnter in range(len(self._frame)):
            row_id = self._frame.index[frame_cnter]
            dt_rowdate = row_id[1]   # need to make this a UTC aware time
            dt_rowdate = utc_tz.localize(dt_rowdate)
            dt_rowdate = dt_rowdate.replace(second=0, microsecond=0)
            # if quote time in between row_date (time) and row_date (time) + 1 minutes
            #  add the order info to the dataframe
            #if (timestampObj_utc >= dt_rowdate) and timestampObj_utc <= (dt_rowdate + timedelta(minutes=1)):
            if (timestampObj_utc == dt_rowdate) :
                Option_Prices_Current = True
                self._frame.loc[row_id,'quoteTime'] = timestampstring
                self._frame.loc[row_id,'option_id'] = max_volume_calls_index
                self._frame.loc[row_id, 'right'] = call_df.loc[max_volume_calls_index, 'right']
                self._frame.loc[row_id, 'strike'] = call_df.loc[max_volume_calls_index, 'strike']
                self._frame.loc[row_id, 'expiration'] = call_df.loc[max_volume_calls_index, 'maturityDate']
                self._frame.loc[row_id,'description'] = Call_Description
                self._frame.loc[row_id,'bid'] = call_df.loc[max_volume_calls_index,'bid']
                self._frame.loc[row_id,'ask'] = call_df.loc[max_volume_calls_index,'ask']
                self._frame.loc[row_id,'last'] = call_df.loc[max_volume_calls_index,'last']
                self._frame.loc[row_id,'total_volume'] = call_df.loc[max_volume_calls_index,'total_volume']
                self._frame.loc[row_id,'daysToExpiration'] = call_df.loc[max_volume_calls_index,'daysToExpiration']
                break      #J. Jones - if we found the row to insert, we move on to the puts

        #
        # make put columns
        #
        # J. JOnes - use the same row_id for puts as we did for calls to keep all data together for a candle.

        # Get the put with max volume
        df_result = put_df.loc[(put_df['daysToExpiration'] > 5)]
        df2_result = df_result.loc[df_result['total_volume'] == df_result['total_volume'].max()]
        if len(df2_result) == 0 :
            self.logfiler.info("No Put Market Data With Max Volume, proceeding without Option information")
            self.calls_options.append(str(int(0)))
            self.puts_options.append(str(int(0)))
            return self._frame
        max_vol_record = df2_result.iloc[0]
        max_volume_puts_index = int(max_vol_record['option_id'])
        if self.fixed_put_option_strike == '' :
            self.puts_options.append(str(int(max_volume_puts_index)))
            self.fixed_put_option_strike = str(int(max_volume_puts_index))
        else :
            self.puts_options.append(str(int(max_volume_puts_index)))

        if not Option_Prices_Current :
            self.logfiler.info("Option Quote Times {tm} Outside of Historical Price Data".format(tm=timestampstring))
            return self._frame
        Put_Description = put_df.loc[max_volume_puts_index, 'symbol'] + \
                                str(put_df.loc[max_volume_puts_index, 'maturityDate']) + \
                                str(put_df.loc[max_volume_puts_index, 'strike']) + \
                                put_df.loc[max_volume_puts_index, 'right'] + " LastPrice=" + \
                                str(put_df.loc[max_volume_puts_index, 'last'])
        self.logfiler.info("EST Time {est_time} For {Option_str} ".format(est_time=timestampstring, Option_str=Put_Description))

        self._frame.loc[row_id, 'quoteTime_put'] = timestampstring
        self._frame.loc[row_id, 'right'] = put_df.loc[max_volume_puts_index, 'right']
        self._frame.loc[row_id, 'option_id'] = max_volume_puts_index
        self._frame.loc[row_id, 'strike'] = put_df.loc[max_volume_puts_index, 'strike']
        self._frame.loc[row_id, 'expiration'] = put_df.loc[max_volume_puts_index, 'maturityDate']
        self._frame.loc[row_id, 'description'] = Put_Description
        self._frame.loc[row_id, 'bid'] = put_df.loc[max_volume_puts_index, 'bid']
        self._frame.loc[row_id, 'ask'] = put_df.loc[max_volume_puts_index, 'ask']
        self._frame.loc[row_id, 'last'] = put_df.loc[max_volume_puts_index, 'last']
        self._frame.loc[row_id, 'total_volume'] = put_df.loc[max_volume_puts_index, 'total_volume']

        self._frame.loc[row_id, 'daysToExpiration'] = put_df.loc[max_volume_puts_index, 'daysToExpiration']

        return self._frame

    def populate_order_data_2(self, order, underlying_symbol, order_response, hist_orders):
        """Populates order data columns in dataframe"""

        est_tz = pytz.timezone('US/Eastern')
        utc_tz = pytz.UTC

        if order:
            order_type = order['orderType']
            instruction = order['side']
            quantity = order['quantity']
            symbol = order['conid']
            asset_type = order['secType']
        else:
            order_type = ''
            session = ''
            duration = ''
            instruction = ''
            quantity = ''
            symbol = ''
            asset_type = ''

        order_response_id = 0
        if order_response :    # If we receive an order response, try to extract the Order ID
            if 'order_id' in order_response[0].keys() :
                order_response_id = order_response[0]['order_id']
                self.logfiler.info("Order Response {ord}".format(ord=order_response[0]))


        # we have to be careful not to overwrite a previous candle which might have order
        # information already populated.
        if self._frame.iloc[-1, self._frame.columns.get_loc('order_type')] == '' or (isinstance(self._frame.iloc[-1, self._frame.columns.get_loc('order_type')], float) and math.isnan(self._frame.iloc[-1, self._frame.columns.get_loc('order_type')])) :
            self._frame.iloc[-1, self._frame.columns.get_loc('order_type')] = order_type
            self._frame.iloc[-1, self._frame.columns.get_loc('instruction')] = instruction
            self._frame.iloc[-1, self._frame.columns.get_loc('quantity')] = quantity
            self._frame.iloc[-1, self._frame.columns.get_loc('Ticker')] = symbol
            self._frame.iloc[-1, self._frame.columns.get_loc('asset_type')] = asset_type
            self._frame.iloc[-1, self._frame.columns.get_loc('order_id')] = order_response_id

        # Update spreadsheet to ensure we capture any subsequent executions / fills
        # from the all_orders object
        # Loop through the DF matching up order ID's
        if len(hist_orders) == 0 :
            return self._frame

        orderID_column = self._frame['order_id']

        for order in hist_orders:

            order_time_string = order['lastExecutionTime']
            # Convert to est
            date_obj = pd.to_datetime(order_time_string, format='%y%m%d%H%M%S')
            date_obj.tz_localize(tz=utc_tz)
            today_obj = datetime.now(tz=est_tz)
            date_obj.tz_localize(est_tz)  # converts to local timezone
            order_time = date_obj.strftime("%m/%d/%Y %H:%M:%S")
            # Only process active/filled trades from today with an underlying
            # that is equal to our underlying symbol
            if (order['status'] != 'Inactive') and (today_obj.date() == date_obj.date()) and (
                    order['ticker'] == underlying_symbol):
                historical_order_id = order['orderId']
                OrderFound = False
                # Loop through the DF matching up order ID's
                for col_cnter in range(len(orderID_column)):
                    # row_id = self.stock_frame.index[frame_cnter]
                    if float(orderID_column[col_cnter]) == float(historical_order_id):
                        row_id = self._frame.index[col_cnter]
                        OrderFound = True
                        contract_symbol = order['conid']
                        asset_type = order['secType']
                        desc = order['description1'] + " " + order['description2']
                        quantity = order['remainingQuantity'] + order['filledQuantity']
                        self._frame.loc[row_id, 'order_time'] = order_time

                        if order['filledQuantity'] > 0 and order['status'] == 'Filled':
                            fill_time = order_time
                            hist_execution_price = order['avgPrice']
                            # print(contract_symbol, " Execution Price ***************", hist_execution_price)
                            self._frame.loc[row_id, 'fill_time'] = fill_time
                            self._frame.loc[row_id, 'price'] = hist_execution_price
                            self._frame.loc[row_id, 'order_status'] = order['status']
                            self._frame.loc[row_id, 'price'] = hist_execution_price
                            self._frame.loc[row_id, 'order_status'] = 'Filled'
                        else :
                            self._frame.loc[row_id, 'order_status'] = 'Not Filled'

                        self._frame.loc[row_id, 'order_type'] = order['orderType']
                        self._frame.loc[row_id, 'instruction'] = order['side']
                        self._frame.loc[row_id, 'order_time'] = order_time
                        self._frame.loc[row_id, 'quantity'] = quantity
                        self._frame.loc[row_id, 'Ticker'] = contract_symbol
                        self._frame.loc[row_id, 'asset_type'] = asset_type
                        break

        return self._frame

    # J. Jones - Implemented this method to allow the day's previous orders
    # to appear in the spreadsheet .
    #
    # J. JOnes - 11/9/2021 changed logic to ensure we are only managing / populating orders that
    # were made previously by the bot, if the bot was restarted during the defined
    # trading window.
    #

    def populate_historical_orders(self, underlying_symbol, trading_robot):
        """
        Takes historical orders and populates the dataframe order columns.
        These are all assumed to be manual orders so they are going into the manual orders
        columns
        """
        df_order_columns = [
            'order_type',
            'instruction',
            'order_time',
            'fill_time',
            'order_id',
            'quantity',
            'asset_type',
            'order_status',
            'Ticker',
            'price',
            'Manual_order_type',
            'Manual_instruction',
            'Manual_order_time',
            'Manual_fill_time',
            'Manual_order_id',
            'Manual_quantity',
            'Manual_asset_type',
            'Manual_order_status',
            'Manual_Ticker',
            'Manual_price'
        ]

        #if not prev_df.empty and trading_robot.within_trading_hours()  :
        #    for i in range(len(df_order_columns)) :
        #        # We are copying order data from previous cycle's dataframe to preserve
        #        # orders made by the bot during this trading day.
        #        self._frame[df_order_columns[i]] = prev_df[df_order_columns[i]]
        #else:
        for i in range(len(df_order_columns)):
            self._frame[df_order_columns[i]] = float('NaN')

        historical_orders_obj = trading_robot.order_history     # incorporate historical orders into spreadsheet
        if len(historical_orders_obj) > 0 :
            historical_orders = historical_orders_obj['orders']
        else :
            return self._frame

        est_tz = pytz.timezone('US/Eastern')
        utc_tz = pytz.UTC
        #
        # If we are starting at beginning of day or during the day, then we are adding historical orders to the spreadsheet
        # regardless of whether it exists in the spreadsheet already or not.
        #
        if trading_robot.before_trading_hours() or trading_robot.within_trading_hours() :

            for order in historical_orders:

                order_time_string = order['lastExecutionTime']
                # Convert to est
                date_obj = pd.to_datetime(order_time_string, format='%y%m%d%H%M%S')
                date_obj.tz_localize(tz=utc_tz)
                order_date_utc = date_obj
                today_obj = datetime.now(tz=est_tz)
                date_obj.tz_localize(est_tz)  # converts to local timezone
                order_time = date_obj.strftime("%m/%d/%Y %H:%M:%S")
                # Only process active/filled trades from today with an underlying
                # that is equal to our underlying symbol
                if (order['status'] != 'Inactive' or order['status'] != 'Cancelled') and (today_obj.date() == date_obj.date()) and (
                        order['ticker'] == underlying_symbol):
                    dt_ordertime = order_date_utc
                    Order_Recorded = False
                    hist_order_ID = order['orderId']
                    for frame_cnter in range(len(self._frame)):
                        row_id = self._frame.index[frame_cnter]
                        dt_rowdate = row_id[1]
                        if (dt_ordertime >= dt_rowdate) and dt_ordertime <= (dt_rowdate + timedelta(minutes=1)):
                            #################Need to finish this  ######################
                            self.add_manual_order_to_frame(row_id, order, order_time)
                            Order_Recorded = True
                    if not Order_Recorded:
                        self.logfiler.info(
                            "Order Not Posted {orderId} On Startup Into Spreadsheet ".format(orderId=hist_order_ID))

            self.stock_data = self._frame
            return self._frame

    def add_manual_order_to_frame(self, row_id, order, order_time):
        hist_symbol = order['conid']
        hist_asset_type = order['secType']
        hist_instrument = order['description1'] + " " + order['description2']
        hist_execution_quantity = order['remainingQuantity'] + order['filledQuantity']
        hist_order_ID = order['orderId']
        hist_order_type = order['orderType']
        hist_instruction = order['side']
        if 'avgPrice' in order.keys() :
            hist_execution_price = order['avgPrice']
        else :
            hist_execution_price = 0
        fill_time = order_time
        hist_execution_quantity = order['filledQuantity']
        self._frame.loc[row_id, 'Manual_order_id'] = hist_order_ID
        self._frame.loc[row_id, 'Manual_order_type'] = hist_order_type
        self._frame.loc[row_id, 'Manual_instruction'] = hist_instruction
        self._frame.loc[row_id, 'Manual_order_time'] = order_time
        self._frame.loc[row_id, 'Manual_Ticker'] = str(hist_symbol) + " " + hist_instrument
        self._frame.loc[row_id, 'Manual_asset_type'] = hist_asset_type
        self._frame.loc[row_id, 'Manual_order_status'] = 'Not Filled'
        self._frame.loc[row_id, 'Manual_fill_time'] = fill_time
        self._frame.loc[row_id, 'Manual_quantity'] = hist_execution_quantity
        self._frame.loc[row_id, 'Manual_price'] = hist_execution_price
        self._frame.loc[row_id, 'Manual_order_status'] = 'Filled'

        return


    def buy_condition(self, earliest_order, symbol, Signal_Volume_Threshold):

        signal_list = []
        #print("In Buy Condition****************************************")

        #global buy_and_sell_count  J. Jones commented this line out
        buy_calls_count = 0
        sell_calls_count = 0
        buy_puts_count = 0
        sell_puts_count = 0
        no_action_calls_count = 0
        no_action_puts_count = 0
        est_tz = pytz.timezone('US/Eastern')

        # Generates signals column called buy_condition
        for i in range(len(self._frame)):

            string_datetimevalue = self._frame['new york time'].values[i]
            candle_datetime = datetime.strptime(string_datetimevalue, "%Y-%m-%d %H:%M:%S")
            candle_datetime = est_tz.localize(candle_datetime)

            if candle_datetime >= earliest_order and self._frame['vwap'][i] != 0.0 :
                # Testing if signal validity via Volume test
                # Repeat last signal unless significant volume is present
                if self._frame['volume'][i] > Signal_Volume_Threshold:  # Threshold for accepting a candle as a valid signal
                    # Buy CALLS condition (VWAP < Close) AND (Previous VWAP > Previous Close)
                    # We only go long when the Close crosses over the VWAP.
                    if (self._frame['vwap'][i] < self._frame['close'][i]) and (self._frame['vwap'][i-1] > self._frame['close'][i-1]) :
                        no_action_calls_count = 0
                        buy_calls_count += 1
                        buy_puts_count = 0
                        signal_list.append('Buy Calls ' + str(buy_calls_count) + ' ' + symbol)
                    elif (self._frame['vwap'][i] > self._frame['close'][i]) and ((self._frame['vwap'][i-1] < self._frame['close'][i-1])) :
                    # Buy PUTS condition (VWAP > Close) AND (Previous VWAP < Previous Close)
                    # We only go short when the Close crosses below the VWAP.
                        no_action_puts_count = 0
                        buy_puts_count += 1
                        buy_calls_count = 0
                        signal_list.append('Buy Puts ' + str(buy_puts_count) + ' ' + symbol)
                    # NO ACTION condition
                    else:
                        buy_calls_count = 0
                        buy_puts_count = 0
                        no_action_calls_count += 1
                        no_action_puts_count += 1
                        signal_list.append('No action')
                else :
                    # repeat last signal if not adequate volume
                    last_signal_item = signal_list[-1]
                    signal_list.append(last_signal_item)
            else :
                buy_calls_count = 0
                buy_puts_count = 0
                no_action_calls_count += 1
                no_action_puts_count += 1
                signal_list.append('No action')

        # Sets a column in the dataframe containing the  signals: ['Buy Calls 1 PETS', 'Buy Puts 1 PETS', 'No action1']
        #order_list = {'buy_calls_count': buy_calls_count, 'buy_puts_count': buy_puts_count, 'no_action_calls_count': no_action_calls_count, 'no_action_puts_count': no_action_puts_count}
        # J. Jones - changed order_list to be an object level variable
        self.order_list['buy_calls_count'] = buy_calls_count
        self.order_list['buy_puts_count'] = buy_puts_count
        self.order_list['no_action_calls_count'] = no_action_calls_count
        self.order_list['no_action_puts_count'] = no_action_puts_count

        self._frame["buy_condition"] = pd.Series(signal_list).values
        self.stock_data = self._frame
        self.indicator_signal_list = signal_list
        return self._frame, signal_list, self.order_list

    def sma9_crossed_sma50(self):

        # def calculate(row):
        #     if row["sma_9"].item() == 0 or row["sma_50"].item() == 0:
        #         val = 0
        #     elif self._frame["sma_9"].item() > self._frame["sma_50"].item():
        #         value = '9maAbove50ma'
        #     else:
        #         value = '9maBelow50ma'
        #     return  value

        temp_list = []
        crossed_above = False
        crossed_below = False
        for i in range(len(self._frame)):
            if self._frame["sma_3"][i] == 0 or self._frame["sma_50"][i] == 0 or self._frame["sma_50"][i] == \
                    self._frame["sma_3"][i]:
                temp_list.append(0)
            elif self._frame["sma_3"][i] > self._frame["sma_50"][i]:
                if crossed_above == False:
                    crossed_above = True
                    crossed_below = False
                    temp_list.append('3maCrossedAbove50ma')
                else:
                    temp_list.append('3maAbove50ma')
            else:
                if crossed_below == False:
                    crossed_below = True
                    crossed_above = False
                    temp_list.append('3maCrossedBelow50ma')
                else:
                    temp_list.append('3maBelow50ma')

        self._frame["sma3_crossed_sma50"] = pd.Series(temp_list).values

        # calculate colum for 9 mov cross to 50 mov

        # self._frame['sma9_crossed_sma50'] = np.where(self._frame["sma_9"] > self._frame["sma_50"], '9maAbove50ma', "9maBelow50ma")

        return self._frame

    def ema(self, period: int, alpha: float = 0.0, column_name='ema') -> pd.DataFrame:
        """Calculates the Exponential Moving Average (EMA).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating the EMA.

        alpha {float} -- The alpha weight used in the calculation. (default: {0.0})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the EMA indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.ema(period=50, alpha=1/50)
        """

        # Add the EMA
        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.ewm(span=period).mean()
        )

        return self._frame

    def rate_of_change(self, period: int = 1, column_name: str = 'rate_of_change') -> pd.DataFrame:
        """Calculates the Rate of Change (ROC).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the ROC. (default: {1})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the ROC indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.rate_of_change()
        """

        # Add the Momentum Price indicator.
        # print(self._price_groups)
        # for key, item in self._price_groups:
        #     print(self._price_groups.get_group(key), "\n\n")

        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.pct_change(periods=period)
        )

        return self._frame

    def option_volume_for_candle(self, period: int = 1, column_name: str = 'call or put total volume') -> pd.DataFrame:

        return self._frame

    def rate_of_change_volume(self, period: int = 1, column_name: str = 'rate_of_change_volume') -> pd.DataFrame:
        """Calculates the Rate of Change (ROC) based on valume.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the ROC. (default: {1})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the ROC indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.rate_of_change_volume()
        """

        # Add the Momentum Volume indicator.
        # print(self._price_groups)
        # for key, item in self._price_groups:
        #     print(self._price_groups.get_group(key), "\n\n")

        self._frame[column_name] = self._price_groups['volume'].transform(
            lambda x: x.pct_change(periods=period)
        )

        return self._frame

    def bollinger_bands(self, period: int = 20, column_name: str = 'bollinger_bands') -> pd.DataFrame:
        """Calculates the Bollinger Bands.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Bollinger Bands. (default: {20})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Lower and Upper band
            indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.bollinger_bands()
        """

        # Define the Moving Avg.
        self._frame['moving_avg'] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Define Moving Std.
        self._frame['moving_std'] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).std()
        )

        # Define the Upper Band.
        self._frame['band_upper'] = 4 * (self._frame['moving_std'] / self._frame['moving_avg'])

        # Define the lower band
        self._frame['band_lower'] = (
                (self._frame['close'] - self._frame['moving_avg']) +
                (2 * self._frame['moving_std']) /
                (4 * self._frame['moving_std'])
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['moving_avg', 'moving_std'],
            axis=1,
            inplace=True
        )

        return self._frame

    def average_true_range(self, period: int = 14, column_name: str = 'average_true_range') -> pd.DataFrame:
        """Calculates the Average True Range (ATR).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the ATR. (default: {14})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the ATR included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.average_true_range()
        """

        # Calculate the different parts of True Range.
        self._frame['true_range_0'] = abs(self._frame['high'] - self._frame['low'])
        self._frame['true_range_1'] = abs(self._frame['high'] - self._frame['close'].shift())
        self._frame['true_range_2'] = abs(self._frame['low'] - self._frame['close'].shift())

        # Grab the Max.
        self._frame['true_range'] = self._frame[['true_range_0', 'true_range_1', 'true_range_2']].max(axis=1)

        # Calculate the Average True Range.
        self._frame['average_true_range'] = self._frame['true_range'].transform(
            lambda x: x.ewm(span=period, min_periods=period).mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['true_range_0', 'true_range_1', 'true_range_2', 'true_range'],
            axis=1,
            inplace=True
        )

        return self._frame

    def stochastic_oscillator(self, column_name: str = 'stochastic_oscillator') -> pd.DataFrame:
        """Calculates the Stochastic Oscillator.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Stochastic Oscillator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.stochastic_oscillator()
        """

        # Calculate the stochastic_oscillator.
        self._frame['stochastic_oscillator'] = (
                self._frame['close'] - self._frame['low'] /
                self._frame['high'] - self._frame['low']
        )

        return self._frame

    def macd(self, fast_period: int = 12, slow_period: int = 26, column_name: str = 'macd') -> pd.DataFrame:
        """Calculates the Moving Average Convergence Divergence (MACD).

        Arguments:
        ----
        fast_period {int} -- The number of periods to use when calculating
            the fast moving MACD. (default: {12})

        slow_period {int} -- The number of periods to use when calculating
            the slow moving MACD. (default: {26})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the MACD included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.macd(fast_period=12, slow_period=26)
        """

        # Calculate the Fast Moving MACD.
        self._frame['macd_fast'] = self._frame['close'].transform(
            lambda x: x.ewm(span=fast_period, min_periods=fast_period).mean()
        )

        # Calculate the Slow Moving MACD.
        self._frame['macd_slow'] = self._frame['close'].transform(
            lambda x: x.ewm(span=slow_period, min_periods=slow_period).mean()
        )

        # Calculate the difference between the fast and the slow.
        self._frame['macd_diff'] = self._frame['macd_fast'] - self._frame['macd_slow']

        # Calculate the Exponential moving average of the fast.
        self._frame['macd'] = self._frame['macd_diff'].transform(
            lambda x: x.ewm(span=9, min_periods=8).mean()
        )

        return self._frame

    def mass_index(self, period: int = 9, column_name: str = 'mass_index') -> pd.DataFrame:
        """Calculates the Mass Index indicator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the mass index. (default: {9})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Mass Index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.mass_index(period=9)
        """

        # Calculate the Diff.
        self._frame['diff'] = self._frame['high'] - self._frame['low']

        # Calculate Mass Index 1
        self._frame['mass_index_1'] = self._frame['diff'].transform(
            lambda x: x.ewm(span=period, min_periods=period - 1).mean()
        )

        # Calculate Mass Index 2
        self._frame['mass_index_2'] = self._frame['mass_index_1'].transform(
            lambda x: x.ewm(span=period, min_periods=period - 1).mean()
        )

        # Grab the raw index.
        self._frame['mass_index_raw'] = self._frame['mass_index_1'] / self._frame['mass_index_2']

        # Calculate the Mass Index.
        self._frame['mass_index'] = self._frame['mass_index_raw'].transform(
            lambda x: x.rolling(window=25).sum()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['diff', 'mass_index_1', 'mass_index_2', 'mass_index_raw'],
            axis=1,
            inplace=True
        )

        return self._frame

    def force_index(self, period: int, column_name: str = 'force_index') -> pd.DataFrame:
        """Calculates the Force Index.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the force index.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the force index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.force_index(period=9)
        """

        # Calculate the Force Index.
        self._frame[column_name] = self._frame['close'].diff(period) * self._frame['volume'].diff(period)

        return self._frame

    def ease_of_movement(self, period: int, column_name: str = 'ease_of_movement') -> pd.DataFrame:
        """Calculates the Ease of Movement.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Ease of Movement.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Ease of Movement included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.ease_of_movement(period=9)
        """

        # Calculate the ease of movement.
        high_plus_low = (self._frame['high'].diff(1) + self._frame['low'].diff(1))
        diff_divi_vol = (self._frame['high'] - self._frame['low']) / (2 * self._frame['volume'])
        self._frame['ease_of_movement_raw'] = high_plus_low * diff_divi_vol

        # Calculate the Rolling Average of the Ease of Movement.
        self._frame['ease_of_movement'] = self._frame['ease_of_movement_raw'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['ease_of_movement_raw'],
            axis=1,
            inplace=True
        )

        return self._frame

    def commodity_channel_index(self, period: int, column_name: str = 'commodity_channel_index') -> pd.DataFrame:
        """Calculates the Commodity Channel Index.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Commodity Channel Index.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Commodity Channel Index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.commodity_channel_index(period=9)
        """

        # Calculate the Typical Price.
        self._frame['typical_price'] = (self._frame['high'] + self._frame['low'] + self._frame['close']) / 3

        # Calculate the Rolling Average of the Typical Price.
        self._frame['typical_price_mean'] = self._frame['pp'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Calculate the Rolling Standard Deviation of the Typical Price.
        self._frame['typical_price_std'] = self._frame['pp'].transform(
            lambda x: x.rolling(window=period).std()
        )

        # Calculate the Commodity Channel Index.
        self._frame[column_name] = self._frame['typical_price_mean'] / self._frame['typical_price_std']

        # Clean up before sending back.
        self._frame.drop(
            labels=['typical_price', 'typical_price_mean', 'typical_price_std'],
            axis=1,
            inplace=True
        )

        return self._frame

    def standard_deviation(self, period: int, column_name: str = 'standard_deviation') -> pd.DataFrame:
        """Calculates the Standard Deviation.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the standard deviation.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Standard Deviation included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.standard_deviation(period=9)
        """

        # Calculate the Standard Deviation.
        self._frame[column_name] = self._frame['close'].transform(
            lambda x: x.ewm(span=period).std()
        )

        return self._frame

    def chaikin_oscillator(self, period: int, column_name: str = 'chaikin_oscillator') -> pd.DataFrame:
        """Calculates the Chaikin Oscillator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Chaikin Oscillator.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Chaikin Oscillator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.chaikin_oscillator(period=9)
        """

        # Calculate the Money Flow Multiplier.
        money_flow_multiplier_top = 2 * (self._frame['close'] - self._frame['high'] - self._frame['low'])
        money_flow_multiplier_bot = (self._frame['high'] - self._frame['low'])

        # Calculate Money Flow Volume
        self._frame['money_flow_volume'] = (money_flow_multiplier_top / money_flow_multiplier_bot) * self._frame[
            'volume']

        # Calculate the 3-Day moving average of the Money Flow Volume.
        self._frame['money_flow_volume_3'] = self._frame['money_flow_volume'].transform(
            lambda x: x.ewm(span=3, min_periods=2).mean()
        )

        # Calculate the 10-Day moving average of the Money Flow Volume.
        self._frame['money_flow_volume_10'] = self._frame['money_flow_volume'].transform(
            lambda x: x.ewm(span=10, min_periods=9).mean()
        )

        # Calculate the Chaikin Oscillator.
        self._frame[column_name] = self._frame['money_flow_volume_3'] - self._frame['money_flow_volume_10']

        # Clean up before sending back.
        self._frame.drop(
            labels=['money_flow_volume_3', 'money_flow_volume_10', 'money_flow_volume'],
            axis=1,
            inplace=True
        )

        return self._frame

    def kst_oscillator(self, r1: int, r2: int, r3: int, r4: int, n1: int, n2: int, n3: int, n4: int,
                       column_name: str = 'kst_oscillator') -> pd.DataFrame:
        """Calculates the Mass Index indicator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the mass index. (default: {9})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Mass Index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.mass_index(period=9)
        """

        # Calculate the ROC 1.
        self._frame['roc_1'] = self._frame['close'].diff(r1 - 1) / self._frame['close'].shift(r1 - 1)

        # Calculate the ROC 2.
        self._frame['roc_2'] = self._frame['close'].diff(r2 - 1) / self._frame['close'].shift(r2 - 1)

        # Calculate the ROC 3.
        self._frame['roc_3'] = self._frame['close'].diff(r3 - 1) / self._frame['close'].shift(r3 - 1)

        # Calculate the ROC 4.
        self._frame['roc_4'] = self._frame['close'].diff(r4 - 1) / self._frame['close'].shift(r4 - 1)

        # Calculate the Mass Index.
        self._frame['roc_1_n'] = self._frame['roc_1'].transform(
            lambda x: x.rolling(window=n1).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_2_n'] = self._frame['roc_2'].transform(
            lambda x: x.rolling(window=n2).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_3_n'] = self._frame['roc_3'].transform(
            lambda x: x.rolling(window=n3).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_4_n'] = self._frame['roc_4'].transform(
            lambda x: x.rolling(window=n4).sum()
        )

        self._frame[column_name] = 100 * (
                    self._frame['roc_1_n'] + 2 * self._frame['roc_2_n'] + 3 * self._frame['roc_3_n'] + 4 * self._frame[
                'roc_4_n'])
        self._frame[column_name + "_signal"] = self._frame['column_name'].transform(
            lambda x: x.rolling().mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['roc_1', 'roc_2', 'roc_3', 'roc_4', 'roc_1_n', 'roc_2_n', 'roc_3_n', 'roc_4_n'],
            axis=1,
            inplace=True
        )

        return self._frame


    def refresh(self, earliest_order, IBSession, symbol, conid, Signal_Volume_Threshold):
        """Updates the Indicator columns after adding the new rows."""

        # First update the groups since, we have new rows.
        self.logfiler.info("Starting Refresh Of Indicators")


        self._price_groups = self._stock_frame.symbol_groups

        # Only calculate vwap intra day, from 9:30am EST through 4pm EST
        est_tz = pytz.timezone('US/Eastern')
        nw = datetime.now(est_tz)
        vwap_start_time = datetime(nw.year, nw.month, nw.day, 9, 30, second=0, microsecond=0)
        vwap_start_time = est_tz.localize(vwap_start_time)

        self.per_of_change()
        #self.rsi(period=20)
        #self.sma(period=3)
        #self.sma(period=9)
        #self.sma(period=50)
        #self.sma9_crossed_sma50()
        #self.sma(period=200)
        #self.sma_volume(period=9)
        #self.sma_volume(period=50)
        #self.sma_volume(period=200)
        #self.abs_3_minus_50_direction()
        #self.abs_9_minus_50_direction()
        #self.ease_of_movement(period=9)
        # Need to preserve order and option data
        # from previous frame
        self.max_option_chain(IBSession, symbol, conid)
        self.vwap(vwap_start_time, column_name='vwap')
        self.buy_condition(earliest_order, symbol, Signal_Volume_Threshold)
        self.logfiler.info("Ending Refresh Of Indicators")

        return

    def check_signals(self) -> Union[pd.DataFrame, None]:
        """Checks to see if any signals have been generated.

        Returns:
        ----
        {Union[pd.DataFrame, None]} -- If signals are generated then a pandas.DataFrame
            is returned otherwise nothing is returned.
        """

        signals_df = self._stock_frame._check_signals(
            indicators=self._indicator_signals,
            indciators_comp_key=self._indicators_comp_key,
            indicators_key=self._indicators_key
        )

        return signals_df
