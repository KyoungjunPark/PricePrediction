import json
import time
import sys
from datetime import datetime
import yfinance as yf
import logging
from yahoo_fin.stock_info import tickers_nasdaq, tickers_sp500, tickers_dow

from pgportfolio.marketdata.info_source import InfoSource

if sys.version_info[0] == 3:
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode
else:
    from urllib2 import Request, urlopen
    from urllib import urlencode

minute = 60
hour = minute*60
day = hour*24
week = day*7
month = day*30
year = day*365

# Possible Commands
PUBLIC_COMMANDS = ['returnTicker', 'return24hVolume', 'returnOrderBook', 'returnTradeHistory', 'returnChartData', 'returnCurrencies', 'returnLoanOrders']


class Yahoo(InfoSource):
    def __init__(self, ticker=None):
        self.ticker = []
        self.history = {}

    def market_chart(self, target, start=time.time() - (week * 1), period=day, end=time.time()):
        print(self.history[target])
        exit(1)

    #####################
    # Main Api Function #
    #####################
    def api(self, command, args=None):
        """
        returns 'False' if invalid command or if no APIKey or Secret is specified (if command is "private")
        returns {"error":"<error message>"} if API error
        """
        if args is None:
            args = {}
        if command in PUBLIC_COMMANDS:
            url = 'https://poloniex.com/public?'
            args['command'] = command
            ret = urlopen(Request(url + urlencode(args)))
            return json.loads(ret.read().decode(encoding='UTF-8'))
        else:
            return False

    def add_market_nasdaq_ticker(self):
        ticker_list = tickers_nasdaq()
        self.ticker.append(ticker_list)
        index = 0
        for ticker in ticker_list:
            yf_obj = yf.Ticker(ticker)
            if index % 100 == 0:
                logging.info("%d/%d" % (index, len(ticker_list)))
            index += 1
            self.history[ticker] = yf_obj.history()
        return ticker_list

    def add_market_sp500_ticker(self):
        ticker_list = tickers_sp500()
        self.ticker.append(ticker_list)
        index = 0
        for ticker in ticker_list:
            yf_obj = yf.Ticker(ticker)
            if index % 100 == 0:
                logging.info("%d/%d" % (index, len(ticker_list)))
            index += 1
            self.history[ticker] = yf_obj.history()
        return ticker_list

    def add_market_dow_ticker(self):
        ticker_list = tickers_dow()
        self.ticker.append(ticker_list)
        index = 0
        for ticker in ticker_list:
            yf_obj = yf.Ticker(ticker)
            if index % 100 == 0:
                logging.info("%d/%d" % (index, len(ticker_list)))
            index += 1
            self.history[ticker] = yf_obj.history()
        return ticker_list

    def get_recent_volume(self, ticker):
        return self.history[ticker].tail(1)['Volume'].iloc[0]

    def get_recent_prices(self, ticker):
        return self.history[ticker].tail(1)['Close'].iloc[0]


