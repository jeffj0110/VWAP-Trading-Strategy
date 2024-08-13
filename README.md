# VWAP Trading Strategy
 Implements VWAP Trading Strategy Using IBKR Rest API.  Instead of going long or short on the stock, calls/puts are used to reduce capital requirements.

For the Client Portal API, please refer to the official documentation https://www.interactivebrokers.com/campus/ibkr-api-page/webapi-doc/

This implementation of the Client Portal API is using an API developed by Alex Reed (areed1192) called ibw.

Setup Requirements The following requirements must be met to use this API:

A Interactive Broker account, you'll need your account password and account number to use the API.
Java 8 update 192 or higher installed (gateway is compatible with higher Java versions including OpenJDK 11).
Download the Client Portal Gateway
Setup Client Portal
Once you've downloaded the latest client portal or if you chose to use the one provided by the repo. You need to unzip the folder and place it in the repo where this code is stored.
Setup API Key and Credentials The API does not require any API keys to use it, all of the authentication is handled by the Client Portal Gateway. Everytime a user starts a new session with the API they will need to proivde their login credentials for the account they wish to use. The Interactive Broker Web API does offer the ability to use the API using a paper account.

Important: Your account number and account password should be kept secret.

Setup Installation pip install interactive-broker-python-web-api Setup Writing Account Information The Client needs specific account information to create a and validate a new session. 

Write a Config File: It's common in Python to have a config file that contains information you need to use during the setup of a script. Additionally, you can make this file in a standard way so that way it's easy to read everytime. In Python, there is a module called configparser which can be used to create config files that mimic that of Windows INI files.

 Below is an example of a config.ini file for this implementation : 

[main]
client_id = IBKR-UserID
redirect_uri = https://localhost
json_path = ./config
account_number = IBKRAccount
account_id = IBKRAccount
start_trading_time = 10:00:00 EST
end_trading_time = 15:45:00 EST
liquidate_all_positions_time = 15:48:00 EST
default_order_type = MKT
no_trading_loss = FALSE
stop_loss_percentage = 10
DEFAULT_BUY_QUANTITY = 4

Usage :

The config.ini file should reside in a 'config' directory that you are running the data capture program from.
Ensure the IBKR Gateway is up and you have successfully logged into that.

From a command line with the default directory having the IB_TradeBot executable - 

.\TradeBot -s <ticker_symbol> -v <volume threshold in candle> -q <Optional default buy quantity>

The ticker_symbol should be the ticker of the equity you wish to trade the VWAP Strategy on.
