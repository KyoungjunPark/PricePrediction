import json
import time
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf
import logging
from yahoo_fin.stock_info import tickers_nasdaq, tickers_sp500, tickers_dow

from pgportfolio.constants import FIVE_MINUTES, FIVE_MINUTES_STR, DAY_STR, DAY
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
    def __init__(self):
        self.history = {}

    def market_chart(self, target, start=time.time() - (week * 1), period=DAY, end=time.time()):
        if period == DAY:
            # print("history")
            # print(self.history[target])
            # self.history[target]['Date'] = pd.to_datetime(self.history[target]['Date'])
            start = datetime.fromtimestamp(int(start)).strftime('%Y-%m-%d')
            end = datetime.fromtimestamp(int(end)).strftime('%Y-%m-%d')
            if target not in self.history.keys():
                logging.info("%s does not exist in DB" % target)
                exit(1)
            else:
                data_set = self.history[target]
                data_set['Date'] = data_set.index
                return json.loads(data_set[start:end].to_json(orient="records", date_unit="s"))
        else:
            logging.info("cannot happen")
            print(period)
            exit(1)

    def add_market_nasdaq_ticker(self, start, end):
        ticker_list = tickers_nasdaq()
        self.add_market_ticker(ticker_list, start, end)
        return True

    def add_market_sp500_ticker(self, start, end):
        ticker_list = tickers_sp500()
        self.add_market_ticker(ticker_list, start, end)
        return True

    def add_market_dow_ticker(self, start, end):
        ticker_list = tickers_dow()
        self.add_market_ticker(ticker_list, start, end)
        return True

    def add_market_ticker(self, stock_list, start, end):
        start = datetime.fromtimestamp(int(start)).strftime('%Y-%m-%d')
        end = datetime.fromtimestamp(int(end)).strftime('%Y-%m-%d')
        index = 0
        tmp_start = time.time()
        for ticker in stock_list:
            yf_obj = yf.Ticker(ticker)
            if index % 100 == 0:
                logging.info("%d/%d (elapsed time: %s)" % (index, len(stock_list), time.time() - tmp_start))
                tmp_start = time.time()
            index += 1
            stock_df = yf_obj.history(start=start, end=end, interval=DAY_STR)
            if not stock_df.empty:
                self.history[ticker] = stock_df
        return True

    def add_all_possible_ticker(self, start, end):
        self.add_market_nasdaq_ticker(start, end)
        self.add_market_sp500_ticker(start, end)
        self.add_market_dow_ticker(start, end)
        return self.history.keys()

    def get_recent_volume(self, ticker):
        return self.history[ticker].tail(1)['Volume'].iloc[0]

    def get_recent_prices(self, ticker):
        return self.history[ticker].tail(1)['Close'].iloc[0]

    def get_tickers(self):
        return self.history.keys()


