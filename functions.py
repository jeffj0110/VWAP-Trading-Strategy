import platform
import logging
import glob
import os
import pandas as pd
from os.path import exists

from datetime import datetime
import pytz
from pytz import timezone
from configparser import ConfigParser
from robot import PyRobot

def setup_func(logger_hndl=None):
    # Get credentials

    CLIENT_ID, REDIRECT_URI, ACCOUNT_NUMBER, ACCOUNT_ID, \
    START_TRADING_TIME, END_TRADING_TIME, SECOND_START_TRADING_TIME, SECOND_END_TRADING_TIME, LIQUIDATE_DAY_TRADES_TIME, \
    DEFAULT_ORDER_TYPE, NO_TRADING_LOSS, DEFAULT_BUY_QUANTITY, Stop_Loss_Perc  = import_credentials(log_hndl=logger_hndl)

    # Initalize the robot with my credentials.
    trading_robot = PyRobot(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        trading_account=ACCOUNT_NUMBER,
        account_id=ACCOUNT_ID,
        start_trading_time = START_TRADING_TIME,
        end_trading_time = END_TRADING_TIME,
        SECOND_START_TRADING_TIME = SECOND_START_TRADING_TIME,
        SECOND_END_TRADING_TIME = SECOND_END_TRADING_TIME,
        liq_day_trades_time = LIQUIDATE_DAY_TRADES_TIME,
        default_order_type = DEFAULT_ORDER_TYPE,
        no_loss_setting=NO_TRADING_LOSS,
        default_buy_quantity=int(DEFAULT_BUY_QUANTITY),
        StopLoss=float(float(Stop_Loss_Perc) / 100.0),
        lgfile = logger_hndl
    )

    # J. Jones - setting bots default timezone to EST.
    est_tz = pytz.timezone('US/Eastern')
    dt = datetime.now(est_tz).strftime("%Y_%m_%d-%I%M%S_%p")
    logmsg = "Bot created at " + dt + " EST"
    logger_hndl.info(logmsg)

    # Create IBSession
    # td_client = trading_robot._create_session()
    ib_client = trading_robot.session
    logger_hndl.info("Session created.")

    # Create a Portfolio
    ib_portfolio, trading_robot_portfolio = trading_robot.create_portfolio()
    logger_hndl.info("Portfolio created.")
    logger_hndl.info("Trading with account: %s",trading_robot.account_id)

    # Needs to be called before any other operations associated with the account
    response = trading_robot.session.portfolio_accounts()

    # Grab any orders made previously during the day prior to the Bot starting
    logger_hndl.info("Retrieving Previous Orders For Today For Account : %s",trading_robot.account_id)
    trading_robot.order_history = ib_client.get_live_orders()
    logmsg = '='*80
    logger_hndl.info(logmsg)

    return trading_robot, ib_client


def import_credentials(log_hndl=None):
    system = platform.system()
    config = ConfigParser()
    currWorkingDirectory = os.getcwd()
    log_hndl.info("Working from default directory %s ", currWorkingDirectory)
    if exists('./config/config.ini') :
        config_str = config.read(r'./config/config.ini')
        log_hndl.info("Reading ./config/config.ini file ")
    else :
        log_hndl.info("No ./config/config.ini file found in %s", currWorkingDirectory)
        exit(-1)

    if config.has_option('main', 'CLIENT_ID') :
        CLIENT_ID = config.get('main', 'CLIENT_ID')
    else:
        log_hndl.info("No CLIENT_ID Found in config.ini file")
        exit(-1)
    REDIRECT_URI = config.get('main', 'REDIRECT_URI')
    ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')
    ACCOUNT_ID = config.get('main', 'ACCOUNT_ID')
    #CREDENTIALS_PATH = config.get('main', 'CREDENTIALS_PATH')
    START_TRADING_TIME = config.get('main', 'START_TRADING_TIME')
    END_TRADING_TIME = config.get('main', 'END_TRADING_TIME')
    if config.has_option('main', '2ND_START_TRADING_TIME') :
        SECOND_START_TRADING_TIME = config.get('main', '2ND_START_TRADING_TIME')
    else:
        SECOND_START_TRADING_TIME = ''
    if config.has_option('main', '2ND_START_TRADING_TIME'):
        SECOND_END_TRADING_TIME = config.get('main', '2ND_END_TRADING_TIME')
    else:
        SECOND_END_TRADING_TIME = ''
    LIQUIDATE_DAY_TRADES_TIME = config.get('main', 'LIQUIDATE_ALL_POSITIONS_TIME')
    DEFAULT_ORDER_TYPE = config.get('main', 'DEFAULT_ORDER_TYPE')
    No_Trading_Loss = "FALSE"

    if config.has_option('main', 'STOP_LOSS_PERCENTAGE') :
        Stop_Loss_Perc = float(config.get('main', 'STOP_LOSS_PERCENTAGE'))
    else:
        Stop_Loss_Perc = float(0.0)

    DEFAULT_BUY_QUANTITY = config.get('main', 'DEFAULT_BUY_QUANTITY')

    return CLIENT_ID, REDIRECT_URI, ACCOUNT_NUMBER, ACCOUNT_ID, \
           START_TRADING_TIME, END_TRADING_TIME, SECOND_START_TRADING_TIME, SECOND_END_TRADING_TIME,\
           LIQUIDATE_DAY_TRADES_TIME, \
           DEFAULT_ORDER_TYPE, No_Trading_Loss, DEFAULT_BUY_QUANTITY, Stop_Loss_Perc
