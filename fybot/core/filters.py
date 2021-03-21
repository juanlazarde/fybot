import asyncio
import pandas as pd
import logging

import requests
import talib
from lxml import html
from time import time as t

from core.patterns import candlestick_patterns
from core.settings import S
from core.snapshot import Snapshot
from core.database import Database

log = logging.getLogger(__name__)


class Filter:
    def __init__(self, filters=None, source_file=False):
        """Apply all filter and signal functions here.

        This class will return a signals table and save it in a signals
        database

        :param dict or None filters: dictionary of settings
        :param bool source_file: True will read Price data from file"""

        s = t()
        self.filters = filters if filters is not None else S.DEFAULT_FILTERS

        # read price data and format
        if source_file:
            price_data = pd.read_pickle(S.PRICE_FILE)

            log.info("Filter is reading price data from the file")
        else:
            # TODO: GET SPECIFIC SYMBOLS
            # get price data from ALL the symbols
            price_data = Snapshot.GetPrice().data

            log.info("Filter is reading price data from the database")

        # cleanup and format price_history
        price_data = price_data.drop(['adj_close'], axis=1)
        price_data = (price_data.set_index(['date', 'symbol'])
                      .unstack().swaplevel(axis=1)
                      )

        # extract symbols from dataframe
        self.symbols = list(price_data.columns.levels[0])
        self.symbols.sort()

        # define filters to use and create table with signals
        columns = [k for k in self.filters if self.filters[k]['go']]
        self.signal = pd.DataFrame(index=self.symbols, columns=columns)

        # process filters to populate signal table
        asyncio.run(self.process_filters(columns=columns,
                                         price_data=price_data)
                    )

        # Save the signal table
        self.save()

        log.info(f"{self.__class__.__name__} took {t() - s} seconds")

    async def process_filters(self, columns, price_data):
        """Fill out signal table with different filters and populate singl
        table"""

        # for every symbol in a row
        for symbol in self.symbols:
            # add a column with a filter name
            for column in columns:
                # locate the cell and add the returned value
                self.signal.loc[symbol, column] = \
                    getattr(self, column)(df=price_data[symbol], symbol=symbol)

            # Remove latest symbol to optimize for speed
            price_data.drop(symbol, axis=1, level=0, inplace=True)

        log.info("Signal table finished")

    def save(self):
        """save the signal dataframe in a file and the database"""

        self.save_to_file()
        asyncio.run(self.save_to_database())

    def save_to_file(self):
        """Save signal table to pickle file"""

        self.signal.to_pickle(S.SIGNALS_FILE)

        log.info("Saved signal table to pickle file")

    async def save_to_database(self):
        """Save signal table to database"""

        # long format table
        signal_df = (self.signal.reset_index()
                     .melt(id_vars=['index'])
                     .fillna('')
                     )

        with Database() as db:
            # erase previous table contents
            sql = "TRUNCATE TABLE signals"
            db.execute(sql)
            db.commit()

            # save data into signals table database
            # pd.values deals with an boolean conflict between pandas and SQL
            values = [tuple(x) for x in signal_df.values]
            sql = """INSERT INTO signals (symbol_id, study, value)
                     VALUES ((SELECT id FROM symbols WHERE symbol=%s), %s, %s)
                     ON CONFLICT (symbol_id, study) DO UPDATE
                     SET symbol_id = excluded.symbol_id,
                         study = excluded.study,
                         value = excluded.value;"""
            db.executemany(sql, values)

            db.timestamp('signals')
            db.commit()

        log.info("Saved signal table to database")

    @staticmethod
    def read_signals():
        """Read the signals table from the database"""

        query = """SELECT symbols.symbol, signals.study, signals.value
                   FROM signals
                   INNER JOIN symbols
                   ON signals.symbol_id = symbols.id;"""
        with Database() as db:
            df = pd.read_sql_query(query, db.connection)

        # wide format table
        df = df.pivot(index='symbol', columns='study', values='value')
        df = df.replace({'true': True, 'false': False}).fillna('')

        return df

    def consolidating(self, **kwargs):
        df = kwargs['df']
        pct = kwargs['pct'] if 'pct' in kwargs.keys() else 0
        pct = pct if pct > 0 else self.filters['consolidating']['pct']
        recent_candlesticks = df[-15:]
        max_close = recent_candlesticks['close'].max()
        min_close = recent_candlesticks['close'].min()
        threshold = 1 - (pct / 100)
        if min_close > (max_close * threshold):
            return True
        return False

    def breakout(self, **kwargs):
        df = kwargs['df']
        pct = self.filters['breakout']['pct']
        last_close = df[-1:]['close'].values[0]
        if self.consolidating(df=df[:-1], pct=pct):
            recent_closes = df[-16:-1]
            if last_close > recent_closes['close'].max():
                return True
        return False

    @staticmethod
    def ttm_squeeze(**kwargs):
        df = kwargs['df']
        sq = pd.DataFrame()
        sq['20sma'] = df['close'].rolling(window=20).mean()
        sq['stddev'] = df['close'].rolling(window=20).std()
        sq['lower_band'] = sq['20sma'] - (2 * sq['stddev'])
        sq['upper_band'] = sq['20sma'] + (2 * sq['stddev'])
        sq['TR'] = abs(df['high'] - df['low'])
        sq['ATR'] = sq['TR'].rolling(window=20).mean()
        sq['lower_keltner'] = sq['20sma'] - (sq['ATR'] * 1.5)
        sq['upper_keltner'] = sq['20sma'] + (sq['ATR'] * 1.5)

        def in_squeeze(_):
            return _['lower_band'] > _['lower_keltner'] and \
                   _['upper_band'] < _['upper_keltner']

        sq['squeeze_on'] = sq.apply(in_squeeze, axis=1)

        if 'in_the_squeeze' in kwargs.keys():
            if kwargs['in_the_squeeze']:
                return sq

        if sq.iloc[-3]['squeeze_on'] and not sq.iloc[-1]['squeeze_on']:
            # is coming out the squeeze
            return True
        return False

    def in_the_squeeze(self, **kwargs):
        # is in the squeeze
        sq = self.ttm_squeeze(df=kwargs['df'], in_the_squeeze=True)
        result = True if sq.iloc[-1]['squeeze_on'] else False
        return result

    @staticmethod
    def ema_stacked(**kwargs):
        df = kwargs['df']
        ema = pd.DataFrame()
        ema_list = [8, 21, 34, 55, 89]
        for period in ema_list:
            col = '{}{}'.format('ema', period)
            ema[col] = talib.EMA(df['close'], timeperiod=period)

        if (ema['ema8'].iloc[-1] > ema['ema21'].iloc[-1]) and \
                (ema['ema21'].iloc[-1] > ema['ema34'].iloc[-1]) and \
                (ema['ema34'].iloc[-1] > ema['ema55'].iloc[-1]) and \
                (ema['ema55'].iloc[-1] > ema['ema89'].iloc[-1]) and \
                (df['close'].iloc[-1] > ema['ema21'].iloc[-1]):
            return True
        return False

    def candlestick(self, **kwargs):
        """Based on hackthemarket"""
        df = kwargs['df']
        symbol = kwargs['symbol']
        bul, ber = [], []

        for pattern, description in candlestick_patterns.items():
            pattern_function = getattr(talib, pattern)
            results = pattern_function(
                df['open'], df['high'], df['low'], df['close'])
            last = int(results.tail(1).values[0])
            if last > 0:
                bul.append(description)
            elif last < 0:
                ber.append(description)

        # create column in signal table with summary of bears & bull
        status = False
        if len(bul) != 0:
            self.signal.loc[symbol, 'cdl_sum_bul'] = ', '.join(bul)
            status = True
        if len(ber) != 0:
            self.signal.loc[symbol, 'cdl_sum_ber'] = ', '.join(ber)
            status = True
        return status

    def sma_filter(self, **kwargs):
        df = kwargs['df']
        sma_fast = talib.SMA(df['close'],
                             timeperiod=self.filters['sma_filter']['fast'])
        sma_slow = talib.SMA(df['close'],
                             timeperiod=self.filters['sma_filter']['slow'])

        result = ((sma_fast[-1] > sma_slow[-1]) and
                  (sma_fast[-2] < sma_slow[-2]))
        return True if result else False

    def investor_reco(self, **kwargs):
        # scrape finviz for latest 3 investor status
        symbol = kwargs['symbol']

        # reco = yf.Ticker(symbol).recommendations['To Grade'][-3:]
        # reco = ", ".join(reco)

        site = 'https://www.finviz.com/quote.ashx?t={}'.format(symbol)
        headers = {'User-Agent': 'Mozilla/5.0'}
        search = '//table[@class="fullview-ratings-outer"]//tr[1]//text()'
        screen = requests.get(site, headers=headers)
        tree = html.fromstring(screen.content).xpath(search)
        bull_words = ['Buy', 'Outperform', 'Overweight', 'Upgrade']
        bear_words = ['Sell', 'Underperform', 'Underweight', 'Downgrade']
        bull = sum([tree.count(x) for x in bull_words])
        bear = sum([tree.count(x) for x in bear_words])
        reco = "Bull {}, Bear {}".format(bull, bear)

        self.signal.loc[symbol, 'investor_sum'] = reco
        return True
