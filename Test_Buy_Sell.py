
import sys, getopt, os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import pytz
import logging
import m_logger
import pandas as pd

from functions import setup_func

import time as time_true
import pathlib
import pandas as pd
import json
import logging
import m_logger


from datetime import datetime
from datetime import timedelta
from datetime import timezone
import pytz

from portfolio import Portfolio
from stock_frame import StockFrame
from ibw.client import IBClient





def _create_session() -> IBClient:
        """Start a new session.

        Creates a new session with the Interactive Brokers API and logs the user into
        the new session.

        Returns:
        ----
        IBClient -- An IBClient object with an authenticated sessions.

        """

        # Create a new instance of the client
        ib_client = IBClient(
            username="jeffjones4",
            account="U7949765",
            is_server_running=True
#            client_id=self.client_id,
#            account_number=self.trading_account,
#            redirect_uri=self.redirect_uri,
#            credentials_path=self.credentials_path
        )

        # log the client into the new session
        #td_client.login()

        return ib_client

def sell_stock(logfiler, session, symbol, option_symbol, instruction: str, quant : int):
    """
    Portfolio format
        {
            'asset_type': 'equity',
            'quantity': 2,
            'purchase_price': 4.00,
            'symbol': 'MSFT',
            'purchase_date': '2020-01-31'
        }
    """

    bidPrice = 0.0
    account = "U7949765"


    order_response = {}
    if quant == 0 :
        default_quantity = 1
    else :
        default_quantity = quant

    orderType = 'MKT'
    if orderType == 'MKT':
        order_template = {
            'acctid': account,
            'conid': int(option_symbol),
            'ticker' : symbol,
            'secType': str(option_symbol) + ':' + 'OPT',
            'orderType': orderType,
            'quantity': default_quantity,
            'side': 'SELL',
            'tif': 'DAY'
            }
    else:
        logfiler.info("Invalid Order Type %s", orderType)
        order_template = {}
        return order_template, order_response

    try:
        # J. Jones - added dump of order template
        logfiler.info("Order Template {ord}".format(ord=json.dumps(order_template, indent=4)))

        order_response = session.place_order(
            account_id=account,
            order=order_template
        )

        if 'order_id' in order_response[0].keys() :
            return order_template, order_response
        else : # We have messages to respond to, we by default just respond 'true' to all of them
            if 'messageIds' in order_response[0].keys() :
                logfiler.info("Message received on order : {msg}".format(msg=order_response[0]['message']))
                order_response_question = session.place_order_reply(reply_id = order_response[0]['id'], reply_resp=True)
                if 'messageIds' in order_response_question[0].keys() :
                    logfiler.info("Message received on order : {msg}".format(msg=order_response_question[0]['message']))
                    order_response_question2 = session.place_order_reply(reply_id = order_response_question[0]['id'], reply_resp=True)
                elif 'order_id' in order_response_question[0].keys() :
                    return order_template, order_response_question
                if 'order_id' in order_response_question2[0].keys() :
                    return order_template, order_response_question2
            elif 'order_id' in order_response[0].keys() :
                return order_template, order_response

        logfiler.info("{inst} order unsuccessfully placed for {sy}".format(inst=instruction, sy=option_symbol))

        logfiler.info("Order response {ord}".format(ord=json.dumps(order_response_question, indent=4)))

        return order_template, order_response

    except Exception as e:
        logfiler.info(
            "Error trying to place {inst} for {sy} with error code {er}".format(inst=instruction, sy=option_symbol,
                                                                                er=str(e)))

        return order_template, order_response



# Check to see if this file is being executed as the "Main" python
# script instead of being used as a module by some other python script
# This allows us to use the module which ever way we want.
def main(argv):
    underlying_symbol = ''
    defQuant = 1
    try:
       opts, args = getopt.getopt(argv,"hs:o:q:")
    except getopt.GetoptError:
       print('Sell -s underlying_ticker -o <ticker_IBK_Number> -q <Optional quantity>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('Sell -s underlying_ticker -o <ticker_IBK_Number> -q <Optional quantity>')
          sys.exit()
       elif opt in ("-s", "-S"):
          underlying_symbol = arg
       elif opt in ('-q', '-Q') :
          defQuant = int(arg)
       elif opt in ('-o', '-O') :
           option_symbol_number = int(arg)

    if underlying_symbol == '' :
        underlying_symbol = "No_Sym_Defined"
    # J. Jones
    # Setting up a logging service for the bot to be able to retrieve
    # runtime messages from a log file
    est_tz = pytz.timezone('US/Eastern')
    now = datetime.now(est_tz).strftime("%Y_%m_%d-%H%M%S")
    logfilename = "{}_logfile_{}".format(underlying_symbol, now)
    logfilename = logfilename + ".txt"
    logger = m_logger.getlogger(logfilename)

    if underlying_symbol == "No_Sym_Defined" :
       logger.info("No Input Symbol Provided")
       logger.info("Please start with a Symbol using -s command line argument")
       logger.info("Sell -s underlying_ticker -o <ticker_IBK_Number> -q <Optional quantity>")
       exit()
    else :
       logger.info('Running With Ticker Symbol : {sym}'.format(sym=underlying_symbol))

    IBC_Session = _create_session()

    positions_response = IBC_Session.portfolio_account_positions(account_id="U7949765", page_id=0)

    logger.info("Positions response {ord}".format(ord=json.dumps(positions_response, indent=4)))


    order_template, order_response = sell_stock(logger, IBC_Session, underlying_symbol, option_symbol_number, "SELL", defQuant)


    logger.info("Order response {ord}".format(ord=json.dumps(order_response, indent=4)))

    positions_response = IBC_Session.portfolio_account_positions(account_id="U7949765", page_id=0)

    logger.info("Positions response {ord}".format(ord=json.dumps(positions_response, indent=4)))

    return True

if __name__ == "__main__":
   main(sys.argv[1:])
