from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from pgportfolio.marketdata.yahoo import Yahoo
from pgportfolio.tools.data import get_chart_until_success
import pandas as pd
from datetime import datetime
import logging
from pgportfolio.constants import *


class StockList(object):
    def __init__(self, stock_list, start, end, volume_average_days=1, volume_forward=0):
        self._yahoo = Yahoo()
        # connect the internet to accees volumes
        # vol = self._yahoo.marketVolume()
        try:
            # self._yahoo.add_market_dow_ticker(start, end)
            self._yahoo.add_market_ticker(stock_list, start, end)
        except Exception:
            logging.info("Error occurred while loading yahoo finance info.")
            pass
        # ticker = self._yahoo.add_market_ticker(stock_list, start, end)
        pairs = []
        stocks = []
        volumes = []
        prices = []

        ticker = self._yahoo.get_tickers()

        logging.info("select stock online from %s to %s" % (datetime.fromtimestamp(start).
                                                           strftime('%Y-%m-%d %H:%M'),
                                                           datetime.fromtimestamp(end).
                                                           strftime('%Y-%m-%d %H:%M')))
        for tick in ticker:
            volume = self._yahoo.get_recent_volume(tick)
            price = self._yahoo.get_recent_prices(tick)
            if volume is None or price is None:
                logging.info("cannot load %s (%s, %s)" % (tick, volume, price))
                pass
            else:
                pairs.append(tick)
                stocks.append(tick)
                volumes.append(volume)
                prices.append(price)
        self._df = pd.DataFrame({'stock': stocks, 'pair': pairs, 'volume': volumes, 'price': prices})
        self._df = self._df.set_index('stock')

    @property
    def all_active_stocks(self):
        return self._df

    @property
    def all_stocks(self):
        return self._yahoo.marketStatus().keys()

    @property
    def polo(self):
        return self._yahoo

    def get_chart_until_success(self, stock, period, start, end):
        return get_chart_until_success(self._yahoo, stock, period, start, end)

    # get several days volume
    def __get_total_volume(self, pair, global_end, days, forward):
        start = global_end-(DAY*days)-forward
        end = global_end-forward
        chart = self.get_chart_until_success(stock=pair, period=DAY, start=start, end=end)
        result = 0
        for one_day in chart:
            if pair.startswith("BTC_"):
                result += one_day['volume']
            else:
                result += one_day["quoteVolume"]
        return result

    def topNVolume(self, n=5, order=True, minVolume=0):
        if minVolume == 0:
            r = self._df.loc[self._df['price'] > 2e-6]
            r = r.sort_values(by='volume', ascending=False)[:n]
            print(r)
            if order:
                return r
            else:
                return r.sort_index()
        else:
            return self._df[self._df.volume >= minVolume]
