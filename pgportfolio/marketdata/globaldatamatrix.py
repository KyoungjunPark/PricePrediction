from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from pgportfolio.marketdata.coinlist import CoinList
import numpy as np
import pandas as pd

from pgportfolio.marketdata.stocklist import StockList
from pgportfolio.tools.data import panel_fillna, multiIndex_fillna
from pgportfolio.constants import *
import sqlite3
from datetime import datetime
import logging


class HistoryManagerCoin:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(self, coin_number, end, volume_average_days=1, volume_forward=0, online=True):
        self.initialize_db()
        self.__storage_period = FIVE_MINUTES  # keep this as 300
        self._coin_number = coin_number
        self._online = online
        if self._online:
            self._coin_list = CoinList(end, volume_average_days, volume_forward)
        self.__volume_forward = volume_forward
        self.__volume_average_days = volume_average_days
        self.__coins = None

    @property
    def coins(self):
        return self.__coins

    def initialize_db(self):
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS History_Coin (date INTEGER,'
                           ' coin varchar(20), high FLOAT, low FLOAT,'
                           ' open FLOAT, close FLOAT, volume FLOAT, '
                           ' quoteVolume FLOAT, weightedAverage FLOAT,'
                           'PRIMARY KEY (date, coin));')
            connection.commit()

    def get_global_data_matrix(self, start, end, period=300, features=('close',)):
        """
        :return a numpy ndarray whose axis is [feature, coin, time]
        """
        return self.get_global_data(start, end, period, features).values

    def get_global_data(self, start, end, period=300, features=('close',)):
        """
        :param start/end: linux timestamp in seconds
        :param period: time interval of each data access point
        :param features: tuple or list of the feature names
        :return a panel, [feature, coin, time]
        """
        start = int(start - (start % period))
        end = int(end - (end % period))
        coins = self.select_coins(start=end - self.__volume_forward - self.__volume_average_days * DAY,
                                  end=end - self.__volume_forward)
        self.__coins = coins
        for coin in coins:
            self.update_data(start, end, coin)

        if len(coins) != self._coin_number:
            raise ValueError("the length of selected coins %d is not equal to expected %d"
                             % (len(coins), self._coin_number))

        logging.info("feature type list is %s" % str(features))
        self.__checkperiod(period)

        time_index = pd.to_datetime(list(range(start, end + 1, period)), unit='s')
        # panel = pd.Panel(items=features, major_axis=coins, minor_axis=time_index, dtype=np.float32)
        df = pd.DataFrame(index=pd.MultiIndex.from_product([features, coins], names=['feature', 'coin']),
                          columns=time_index, dtype=np.float64)

        connection = sqlite3.connect(DATABASE_DIR)
        try:
            for row_number, coin in enumerate(coins):
                for feature in features:
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = ("SELECT date+300 AS date_norm, close FROM History_Coin WHERE"
                               " date_norm>={start} and date_norm<={end}"
                               " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                            start=start, end=end, period=period, coin=coin))
                    elif feature == "open":
                        sql = ("SELECT date+{period} AS date_norm, open FROM History_Coin WHERE"
                               " date_norm>={start} and date_norm<={end}"
                               " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                            start=start, end=end, period=period, coin=coin))
                    elif feature == "volume":
                        sql = ("SELECT date_norm, SUM(volume)" +
                               " FROM (SELECT date+{period}-(date%{period}) "
                               "AS date_norm, volume, coin FROM History_Coin)"
                               " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                               " GROUP BY date_norm".format(
                                   period=period, start=start, end=end, coin=coin))
                    elif feature == "high":
                        sql = ("SELECT date_norm, MAX(high)" +
                               " FROM (SELECT date+{period}-(date%{period})"
                               " AS date_norm, high, coin FROM History_Coin)"
                               " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                               " GROUP BY date_norm".format(
                                   period=period, start=start, end=end, coin=coin))
                    elif feature == "low":
                        sql = ("SELECT date_norm, MIN(low)" +
                               " FROM (SELECT date+{period}-(date%{period})"
                               " AS date_norm, low, coin FROM History_Coin)"
                               " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                               " GROUP BY date_norm".format(
                                   period=period, start=start, end=end, coin=coin))
                    else:
                        msg = ("The feature %s is not supported" % feature)
                        logging.error(msg)
                        raise ValueError(msg)
                    serial_data = pd.read_sql_query(sql, con=connection,
                                                    parse_dates=["date_norm"],
                                                    index_col="date_norm")
                    idx = pd.IndexSlice
                    df.loc[idx[feature, coin], serial_data.index] = serial_data.squeeze()
                    df.loc[idx[feature, coin], :] = df.loc[idx[feature, coin], :].fillna(method="bfill").fillna(
                        method="ffill")
        finally:
            connection.commit()
            connection.close()
        return df

    # select top coin_number of coins by volume from start to end
    def select_coins(self, start, end):
        if not self._online:
            logging.info(
                "select coins offline from %s to %s" % (datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                                                        datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')))
            connection = sqlite3.connect(DATABASE_DIR)
            try:
                cursor = connection.cursor()
                cursor.execute('SELECT coin,SUM(volume) AS total_volume FROM History_Coin WHERE'
                               ' date>=? and date<=? GROUP BY coin'
                               ' ORDER BY total_volume DESC LIMIT ?;',
                               (int(start), int(end), self._coin_number))
                coins_tuples = cursor.fetchall()

                if len(coins_tuples) != self._coin_number:
                    logging.error("the sqlite error happened")
            finally:
                connection.commit()
                connection.close()
            coins = []
            for tmp_tuple in coins_tuples:
                coins.append(tmp_tuple[0])
        else:
            coins = list(self._coin_list.topNVolume(n=self._coin_number).index)
        logging.debug("Selected coins are: " + str(coins))
        return coins

    def __checkperiod(self, period):
        if period == FIVE_MINUTES:
            return
        elif period == FIFTEEN_MINUTES:
            return
        elif period == HALF_HOUR:
            return
        elif period == TWO_HOUR:
            return
        elif period == FOUR_HOUR:
            return
        elif period == DAY:
            return
        else:
            raise ValueError('peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day')

    # add new history data into the database
    def update_data(self, start, end, coin):
        connection = sqlite3.connect(DATABASE_DIR)
        try:
            cursor = connection.cursor()
            min_date = cursor.execute('SELECT MIN(date) FROM History_Coin WHERE coin=?;', (coin,)).fetchall()[0][0]
            max_date = cursor.execute('SELECT MAX(date) FROM History_Coin WHERE coin=?;', (coin,)).fetchall()[0][0]

            if min_date is None or max_date is None:
                self.__fill_data(start, end, coin, cursor)
            else:
                if max_date + 10 * self.__storage_period < end:
                    if not self._online:
                        raise Exception("Have to be online")
                    self.__fill_data(max_date + self.__storage_period, end, coin, cursor)
                if min_date > start and self._online:
                    self.__fill_data(start, min_date - self.__storage_period - 1, coin, cursor)

            # if there is no data
        finally:
            connection.commit()
            connection.close()

    def __fill_data(self, start, end, coin, cursor):
        duration = 7819200  # three months
        bk_start = start
        for bk_end in range(start + duration - 1, end, duration):
            self.__fill_part_data(bk_start, bk_end, coin, cursor)
            bk_start += duration
        if bk_start < end:
            self.__fill_part_data(bk_start, end, coin, cursor)

    def __fill_part_data(self, start, end, coin, cursor):
        chart = self._coin_list.get_chart_until_success(
            pair=self._coin_list.allActiveCoins.at[coin, 'pair'],
            start=start,
            end=end,
            period=self.__storage_period)
        logging.info("fill %s data from %s to %s" % (coin, datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                                                     datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')))
        for c in chart:
            if c["date"] > 0:
                if c['weightedAverage'] == 0:
                    weightedAverage = c['close']
                else:
                    weightedAverage = c['weightedAverage']

                # NOTE here the USDT is in reversed order
                if 'reversed_' in coin:
                    cursor.execute('INSERT INTO History_Coin VALUES (?,?,?,?,?,?,?,?,?)',
                                   (c['date'], coin, 1.0 / c['low'], 1.0 / c['high'], 1.0 / c['open'],
                                    1.0 / c['close'], c['quoteVolume'], c['volume'],
                                    1.0 / weightedAverage))
                else:
                    cursor.execute('INSERT INTO History_Coin VALUES (?,?,?,?,?,?,?,?,?)',
                                   (c['date'], coin, c['high'], c['low'], c['open'],
                                    c['close'], c['volume'], c['quoteVolume'],
                                    weightedAverage))


class HistoryManagerStock:
    # if offline ,the stock_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(self, start, end, stock_list, volume_average_days=1, volume_forward=0, online=True):
        self.initialize_db()
        self.__storage_period = DAY  # keep this as DAY
        self._online = online
        if self._online:
            self._stock_list = StockList(stock_list, start, end, volume_average_days, volume_forward)
        else:
            self._stock_list = StockList(stock_list, start, end, volume_average_days, volume_forward)
        self.__volume_forward = volume_forward
        self.__volume_average_days = volume_average_days
        self.__stocks = stock_list

    @property
    def stocks(self):
        return self.__stocks

    def initialize_db(self):
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS History_Stock (date INTEGER,'
                           ' stock varchar(20), high FLOAT, low FLOAT,'
                           ' open FLOAT, close FLOAT, volume FLOAT, '
                           ' dividends FLOAT, splits FLOAT,'
                           'PRIMARY KEY (date, stock));')
            connection.commit()

    def get_global_data_matrix(self, start, end, period=300, features=('close',)):
        """
        :return a numpy ndarray whose axis is [feature, stock, time]
        """
        return self.get_global_data(start, end, period, features).values

    def get_global_data(self, start, end, period=300, features=('close',)):
        """
        :param start: linux timestamp in seconds
        :param end: linux timestamp in seconds
        :param period: time interval of each data access point
        :param features: tuple or list of the feature names
        :return a panel, [feature, stock, time]
        """
        # print(datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'))
        # print(datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M'))
        start = start + HOUR * 9  # Stock Opening at 9 am
        end = end + HOUR * 9
        # end = int(end + (end % period))
        for stock in self.stocks:
            self.update_data(start, end, stock)

        logging.info("feature type list is %s" % str(features))
        self.__checkperiod(period)

        time_index = pd.to_datetime(list(range(start, end, period)), unit='s').strftime('%Y-%m-%d')
        # panel = pd.Panel(items=features, major_axis=stocks, minor_axis=time_index, dtype=np.float32)
        df = pd.DataFrame(index=pd.MultiIndex.from_product([features, self.stocks], names=['feature', 'stock']),
                          columns=time_index, dtype=np.float64)

        connection = sqlite3.connect(DATABASE_DIR)
        try:
            for row_number, stock in enumerate(self.stocks):
                for feature in features:
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = ("SELECT date AS date_norm, close FROM History_Stock WHERE"
                               " date_norm>={start} and date_norm<={end}"
                               " and date_norm%{period}=0 and stock=\"{stock}\""
                               .format(start=start, end=end, period=period, stock=stock))
                    elif feature == "open":
                        sql = ("SELECT date AS date_norm, open FROM History_Stock WHERE"
                               " date_norm>={start} and date_norm<={end}"
                               " and date_norm%{period}=0 and stock=\"{stock}\""
                               .format(start=start, end=end, period=period, stock=stock))
                    elif feature == "volume":
                        sql = ("SELECT date AS date_norm, volume FROM History_Stock WHERE"
                               " date_norm>={start} and date_norm<={end} and stock=\"{stock}\""
                               .format(period=period, start=start, end=end, stock=stock))
                    elif feature == "high":
                        sql = ("SELECT date AS date_norm, high FROM History_Stock WHERE"
                               " date_norm>={start} and date_norm<={end} and stock=\"{stock}\""
                               .format(period=period, start=start, end=end, stock=stock))
                    elif feature == "low":
                        sql = ("SELECT date AS date_norm, low FROM History_Stock WHERE"
                               " date_norm>={start} and date_norm<={end} and stock=\"{stock}\""
                               .format(period=period, start=start, end=end, stock=stock))
                    else:
                        msg = ("The feature %s is not supported" % feature)
                        logging.error(msg)
                        raise ValueError(msg)

                    serial_data = pd.read_sql_query(sql, con=connection,
                                                    parse_dates=["date_norm"],
                                                    index_col="date_norm")
                    idx = pd.IndexSlice
                    df.loc[idx[feature, stock], serial_data.index] = serial_data.squeeze()
                    df.loc[idx[feature, stock], :] = df.loc[idx[feature, stock], :].fillna(method="bfill").fillna(
                        method="ffill")

        finally:
            connection.commit()
            connection.close()
        return df

    def __checkperiod(self, period):
        if period == FIVE_MINUTES:
            return
        elif period == FIFTEEN_MINUTES:
            return
        elif period == HALF_HOUR:
            return
        elif period == TWO_HOUR:
            return
        elif period == FOUR_HOUR:
            return
        elif period == DAY:
            return
        else:
            raise ValueError('peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day')

    # add new history data into the database
    def update_data(self, start, end, stock):
        connection = sqlite3.connect(DATABASE_DIR)
        try:
            cursor = connection.cursor()
            min_date = cursor.execute('SELECT MIN(date) FROM History_Stock WHERE stock=?;', (stock,)).fetchall()[0][0]
            max_date = cursor.execute('SELECT MAX(date) FROM History_Stock WHERE stock=?;', (stock,)).fetchall()[0][0]

            if min_date is None or max_date is None:
                self.__fill_data(start, end, stock, cursor)
            else:
                logging.info("Data from %s to %s already exists about %s"
                             % (datetime.fromtimestamp(min_date).strftime('%Y-%m-%d'),
                                datetime.fromtimestamp(max_date).strftime('%Y-%m-%d'), stock))
                pass

            # if there is no data
        finally:
            connection.commit()
            connection.close()

    def __fill_data(self, start, end, stock, cursor):
        duration = 2592000 * 12  # 1 year
        bk_start = start
        for bk_end in range(start + duration - DAY, end, duration):
            self.__fill_part_data(bk_start, bk_end, stock, cursor)
            bk_start += duration
        if bk_start < end:
            self.__fill_part_data(bk_start, end, stock, cursor)

    def __fill_part_data(self, start, end, stock, cursor):
        chart = self._stock_list.get_chart_until_success(
            stock=stock,
            start=start,
            end=end,
            period=self.__storage_period)
        logging.info("fill %s data from %s to %s" % (stock, datetime.fromtimestamp(start).strftime('%Y-%m-%d'),
                                                     datetime.fromtimestamp(end).strftime('%Y-%m-%d')))

        for c in chart:
            if c["Date"] > 0:
                cursor.execute('INSERT INTO History_Stock VALUES (?,?,?,?,?,?,?,?,?)',
                               (c['Date'], stock, c['High'], c['Low'], c['Open'],
                                c['Close'], c['Volume'], c['Dividends'], c['Stock Splits']))
