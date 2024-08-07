import sys, getopt, os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import pytz
import logging
import m_logger
import pandas as pd

from functions import setup_func
from indicator_calcs import Indicators


def Run_Bot(inputsymbol, default_buy_quantity, logger, now, Signal_Volume_Threshold) :

    symbol = inputsymbol
    trading_options = True

    # Sets up the robot class, robot's portfolio, and the TDSession object
    trading_robot, IBClient = setup_func(logger)

    historical_prices = trading_robot.grab_historical_prices(
        bar_size=5,
        bar_type='minute',
        symbols=[symbol]
    )


    # Get current date and create the excel sheet name
    # J. Jones - changed default timezone to EST
    # added seconds to the file name
    filename = "{}_run_{}".format(symbol, now)
    json_path = './config'
    full_path = json_path + r"/" + filename + ".csv"


    # Convert data to a Data StockFrame.
    stock_frame = trading_robot.create_stock_frame(data=historical_prices['aggregated'])

    First_Loop = True

    # If there are options positions for the underlying symbol for this session, the
    # bot will manage those (ie. sell them if signals indicate so).

    # Create an indicator Object.
    indicator_client = Indicators(price_data_frame=stock_frame, lgfile=logger)

    # Add required indicators
    indicator_client.per_of_change()
    indicator_client.max_option_chain(IBClient, symbol, trading_robot.conid)

    # Only calculate vwap intra day, from 9:30am EST through 4pm EST
    est_tz = pytz.timezone('US/Eastern')
    nw = datetime.today()
    nw = est_tz.localize(nw)
    vwap_start_time = datetime(nw.year, nw.month, nw.day,9,30, second=0, microsecond=0)
    vwap_start_time = est_tz.localize(vwap_start_time)

    indicator_client.vwap(vwap_start_time, column_name='vwap')
    stock_info_df, signal_list, init_order_list = indicator_client.buy_condition(trading_robot.earliest_order,symbol, Signal_Volume_Threshold)

    # J. Jones
    # A user may create orders in their account outside of the bot for other symbols.  Those
    # positions will be ignored though.  So, you can have multiple bot running all for different symbols.
    #
    indicator_client.populate_historical_orders(symbol, trading_robot)

    # Define initial refresh time so we know when to refresh the IBClient
    # J. Jones : Changed default timezone to EST
    # Note : The bot has only been tested in the EST timzone
    est_tz = pytz.timezone('US/Eastern')
    refresh_time = datetime.now(est_tz) + timedelta(minutes=3)

    # J. Jones
    # If user sets the -q command line option to set the default_buy_quantity,
    # this will override what is set in the config.ini file
    if default_buy_quantity != 0 :
        if default_buy_quantity <= 10 :
            logger.info('Overriding Buy Quantity in config.ini from {ini_dfb} to {dfbq}'.format(ini_dfb=str(trading_robot.def_buy_quantity),dfbq=str(default_buy_quantity)))
            trading_robot.def_buy_quantity = default_buy_quantity
        else:
            logger.info('Default Buy Quantity Too High, Setting to 1')
            trading_robot.def_buy_quantity = 1

    while True :
        if First_Loop :
             First_Loop = False
        else:
             # Grab the latest bar.
             last_bar_timestamp = trading_robot.stock_frame.tail(n=1).index.get_level_values(1)
             last_bar_time = last_bar_timestamp.to_pydatetime()[0].replace(tzinfo=timezone.utc) # Note - in UTC timezone
             latest_bars = trading_robot.get_latest_bar(TDSession=IBClient, symbol=symbol, lastbartime=last_bar_time)

             # Add to the Stock Frame.
             stock_frame.add_rows(data=latest_bars)

             # Update the stock frame in the robot and indicator client again
             trading_robot.stock_frame = stock_frame.frame
             indicator_client.stock_data = stock_frame.frame

             # Refresh the Indicators.  Note : Sending previous cycle's dataframe to preserve
             # order data and option prices
             indicator_client.refresh(trading_robot.earliest_order, IBClient, symbol, trading_robot.conid, Signal_Volume_Threshold)

        # Get the stock DF from indicators
        stock_df = indicator_client.stock_data
    #    print(stock_df.tail())

        # Send signals, puts, and calls to the bot client
        trading_robot.signals = indicator_client.indicator_signal_list
        trading_robot.call_options = indicator_client.calls_options
        trading_robot.put_options = indicator_client.puts_options

        # Set the StockFrame in the print robot to the same as the indicator one
        trading_robot.stock_frame = stock_df

        # If we hit a stoploss with the previous candle,
        # Then we don't enter into another position unless volumne in the last candle is over
        # 4000 for 5 min candles.
        # So, we set the last candle to StopLoss to keep from buying again until we see adequate volume again

        if trading_robot.hit_stop_loss :
            if trading_robot.stock_frame["volume"][-1] < Signal_Volume_Threshold :
                trading_robot.stock_frame.at[trading_robot.stock_frame.index[-1], 'buy_condition'] = "StopLoss"
                trading_robot.signals[-1] = "StopLoss"
            else:
                trading_robot.hit_stop_loss = False

        # logger.info('Last Buy Condition Value {dfbq}'.format(dfbq=trading_robot.stock_frame.at[trading_robot.stock_frame.index[-1], 'buy_condition']))

        order, order_response, all_orders = trading_robot.process_orders(First_Loop, symbol=symbol, orders=indicator_client.order_list)


        # Add order info to the dataframe

        stock_info_df = indicator_client.populate_order_data_2(order=order, underlying_symbol=symbol, order_response=order_response, hist_orders=all_orders)

        # Save an excel sheet with the data
        # The Excel sheet is actually saved in a csv file for efficiency
        stock_info_df.to_csv(full_path)

        # Grab the last bar.
        last_bar_timestamp = trading_robot.stock_frame.tail(n=1).index.get_level_values(1)

        # J. Jones - The stock_frame has the next to the last minute candle, to ensure a fully populated volume
        # We are adding a minute to the last_bar_timestamp to ensure we wait the appropriate time period to ensure
        # the continuous cycle of time based on the current time (not a historical time)
        last_bar_time = last_bar_timestamp.to_pydatetime()[0].replace(tzinfo=timezone.utc)
        curr_bar_time = datetime.now(tz=timezone.utc)
        if (last_bar_time + timedelta(minutes=1)) < curr_bar_time :
            last_historical_date = last_bar_time + timedelta(minutes=1)
        elif last_bar_time < curr_bar_time :
            last_historical_date = last_bar_time
        else :
            last_historical_date = curr_bar_time

        # Wait till the next bar.
        trading_robot.wait_till_next_bar(last_bar_timestamp=last_historical_date)

# Check to see if this file is being executed as the "Main" python
# script instead of being used as a module by some other python script
# This allows us to use the module which ever way we want.
def main(argv):
    inputsymbol = ''
    defBuyQuant = 0
    Signal_Volume_Threshold = -1.0    # If Volume isn't above this threshold, the previous candle's signal is used.

    try:
       opts, args = getopt.getopt(argv,"hs:v:q:")
    except getopt.GetoptError:
       print('TradeBot -s <ticker_symbol> -v <volume threshold in candle> -q <Optional default buy quantity>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
           print('TradeBot -s <ticker_symbol> -v <volume threshold in candle> -q <Optional default buy quantity>')
           sys.exit()
       elif opt in ("-s", "-S"):
           inputsymbol = arg
       elif opt in ('-q', '-Q') :
           defBuyQuant = int(arg)
       elif opt in ('-v', '-V') :
           Signal_Volume_Threshold = float(arg)

    if inputsymbol == '' :
        inputsymbol = "No_Sym_Defined"
    # J. Jones
    # Setting up a logging service for the bot to be able to retrieve
    # runtime messages from a log file
    est_tz = pytz.timezone('US/Eastern')
    now = datetime.now(est_tz).strftime("%Y_%m_%d-%H%M%S")
    logfilename = "{}_logfile_{}".format(inputsymbol, now)
    logfilename = logfilename + ".txt"
    logger = m_logger.getlogger(logfilename)

    if inputsymbol == "No_Sym_Defined" :
       logger.info("No Input Symbol Provided")
       logger.info("Please start with a Symbol using -s command line argument")
       logger.info("TradeBot -s <ticker_symbol> -v <volume threshold in candle> -q <Optional default buy quantity>")
       exit()
    else :
       logger.info('Running With Ticker Symbol : {sym}'.format(sym=inputsymbol))

    if Signal_Volume_Threshold == -1.0 or Signal_Volume_Threshold <= 0 :
        logger.info("Must provide a threshold for volume that qualifies for a valid signal")
        logger.info("Please start with a Number using -v command line argument")
        logger.info("TradeBot -s <ticker_symbol> -v <volume threshold in candle> -q <Optional default buy quantity>")
        exit()
    else :
        logger.info('Signal Volume Threshold : {sigvol}'.format(sigvol=str(Signal_Volume_Threshold)))

    Run_Bot(inputsymbol, defBuyQuant, logger, now, Signal_Volume_Threshold)

    return True

if __name__ == "__main__":
   main(sys.argv[1:])
