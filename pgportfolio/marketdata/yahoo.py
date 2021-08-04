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
        self.ticker = []
        self.history = {}

    def market_chart(self, target, start=time.time() - (week * 1), period=DAY, end=time.time()):
        if period == DAY:
            # print("history")
            # print(self.history[target])
            # self.history[target]['Date'] = pd.to_datetime(self.history[target]['Date'])
            start = datetime.fromtimestamp(int(start)).strftime('%Y-%m-%d')
            end = datetime.fromtimestamp(int(end)).strftime('%Y-%m-%d')

            if target not in self.history.keys():
                print(target)
                print(self.history.keys())
                print("Cannot happen!!")
                exit(1)
            else:
                data_set = self.history[target]
                data_set['Date'] = data_set.index
                return json.loads(data_set[start:end].to_json(orient="records"))
        else:
            logging.info("cannot happen")
            print(period)
            exit(1)

    def add_market_nasdaq_ticker(self, start, end):
        ticker_list = tickers_nasdaq()
        self.ticker.append(ticker_list)
        index = 0
        for ticker in ticker_list:
            yf_obj = yf.Ticker(ticker)
            if index % 100 == 0:
                logging.info("%d/%d" % (index, len(ticker_list)))
            index += 1
            self.history[ticker] = yf_obj.history(start=start, end=end, interval=DAY_STR)
        return ticker_list

    def add_market_sp500_ticker(self, start, end):
        ticker_list = tickers_sp500()
        self.ticker.append(ticker_list)

        start = datetime.fromtimestamp(int(start)).strftime('%Y-%m-%d')
        end = datetime.fromtimestamp(int(end)).strftime('%Y-%m-%d')
        index = 0
        for ticker in ticker_list:
            yf_obj = yf.Ticker(ticker)
            if index % 100 == 0:
                logging.info("%d/%d" % (index, len(ticker_list)))
            index += 1

            self.history[ticker] = yf_obj.history(start=start, end=end, interval=DAY_STR)
        return ticker_list

    def add_market_dow_ticker(self, start, end):
        ticker_list = tickers_dow()
        self.ticker.append(ticker_list)
        index = 0

        start = datetime.fromtimestamp(int(start)).strftime('%Y-%m-%d')
        end = datetime.fromtimestamp(int(end)).strftime('%Y-%m-%d')
        for ticker in ticker_list:
            yf_obj = yf.Ticker(ticker)
            if index % 100 == 0:
                logging.info("%d/%d" % (index, len(ticker_list)))
            index += 1
            self.history[ticker] = yf_obj.history(start=start, end=end, interval=DAY_STR)
        return ticker_list

    def get_recent_volume(self, ticker):
        return self.history[ticker].tail(1)['Volume'].iloc[0]

    def get_recent_prices(self, ticker):
        return self.history[ticker].tail(1)['Close'].iloc[0]

    def add_market_ticker(self, stock_list, start, end):
        self.ticker.append(stock_list)

        start = datetime.fromtimestamp(int(start)).strftime('%Y-%m-%d')
        end = datetime.fromtimestamp(int(end)).strftime('%Y-%m-%d')
        index = 0
        for ticker in stock_list:
            yf_obj = yf.Ticker(ticker)
            if index % 100 == 0:
                logging.info("%d/%d" % (index, len(stock_list)))
            index += 1
            self.history[ticker] = yf_obj.history(start=start, end=end, interval=DAY_STR)
        return stock_list


