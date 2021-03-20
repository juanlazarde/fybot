"""Stock scanner

Downloads list of stocks, filters and analyzes, and provides results on HTML
via Flask,

To run flask, open the python terminal and run: flask run


-- Template format
http://semantic-ui.com


"""
import datetime
# noinspection PyUnresolvedReferences
import pickle
import sys
# noinspection PyUnresolvedReferences
from time import time as t
import json
import os
from ftplib import FTP

import pandas as pd
import plotly as pt
import plotly.graph_objs as go
import psycopg2
import psycopg2.extras
import pytz
import requests
# noinspection PyPackageRequirements
import talib
import yfinance as yf
import yahooquery as yq
from flask import Flask, render_template, request, flash, url_for, redirect
from lxml import html

from fybot.common.patterns import candlestick_patterns
from psaw import PushshiftAPI

import aiohttp
import asyncpg
import asyncio
from fybot.common import config
import logging
import time
from functools import wraps
from memory_profiler import memory_usage
from tda.auth import easy_client as tda_connection
from cryptography.fernet import Fernet
import subprocess
from ast import literal_eval

from fybot.common.encryption import Encryption

log = logging.getLogger(name=__name__)


# noinspection DuplicatedCode
class S:
    # define files
    DATASET_DIR = '../datasets'
    SYMBOLS_FILE = 'symbols.csv'
    PRICE_FILE = 'price.pkl'
    FUNDAMENTALS_FILE = 'fundamentals.pkl'
    SIGNALS_FILE = 'signals.pkl'

    # other constants
    DAYS = 150

    # FOR DEBUG ONLY
    DEBUG = True
    MAX_SYMBOLS = 10 if DEBUG else None

    # database settings
    DB_HOST = 'localhost'
    DB_USER = 'postgres'
    DB_PASSWORD = 'diamondhands'
    DB_NAME = 'source'

    # flask settings
    FLASK_SECRET_KEY = "jPRgt7XGwp5fuvVh846TAZDbAbTKTF8hpaHSacAqyEuTgkE6Kkuv" \
                       "yW2KDzQhhdVu2xKR7UExyfuQ8Nd4nxjezbHKJUwJ8QjFrsfRGVcH" \
                       "AHUzyvdAjCWZDLr7zFENhRA"

    # Encryption
    SECRET_KEY_FILE = config.SECRET_KEY

    # TDA Ameritrade Settings
    TDA_API_KEY = config.TDA_API_KEY
    TDA_REDIRECT_URI = config.TDA_REDIRECT_URI
    TDA_TOKEN = config.TDA_TOKEN
    TDA_ACCOUNT_E = config.TDA_ACCOUNT

    # Alpaca Settings
    # you need to create a config.py to house the credentials
    ALPACA_KEY = config.ALPACA_API_KEY
    ALPACA_SECRET = config.ALPACA_SECRET_KEY
    ALPACA_ENDPOINT = config.ALPACA_ENDPOINT

    SYMBOLS_FILE = os.path.join(DATASET_DIR, SYMBOLS_FILE)
    PRICE_FILE = os.path.join(DATASET_DIR, PRICE_FILE)
    FUNDAMENTALS_FILE = os.path.join(DATASET_DIR, FUNDAMENTALS_FILE)
    SIGNALS_FILE = os.path.join(DATASET_DIR, SIGNALS_FILE)

    # create dataset directory if not there
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    DEFAULT_FILTERS = {'consolidating': {'go': False, 'pct': 6},
                       'breakout': {'go': False, 'pct': 2.5},
                       'ttm_squeeze': {'go': False},
                       'in_the_squeeze': {'go': True},
                       'candlestick': {'go': True},
                       'sma_filter': {'go': False, 'fast': 25, 'slow': 100},
                       'ema_stacked': {'go': True},
                       'investor_reco': {'go': False}
                       }


class Encryption:
    cipher_suite = None

    def __init__(self, password=''):
        """Encryption process using 'cryptography' package.

         Use:
             crypto = Encryption(password='tothemoon')
             secret = crypto.encrypt('secret message')
             revealed = crypto.decrypt('mm432kl4m32')
             If no password is used, then it reads the key file.
             If no key file is present, it creates one.

         :param str password: Password, otherwise it will read it from key file
         """

        # name of file with key
        if password.strip() != '':
            key = password.strip()
        else:
            key = self.create_save_key_file(S.SECRET_KEY_FILE)
        self.cipher_suite = Fernet(key)

    @staticmethod
    def create_save_key_file(filename):
        """Generate or load Key and save it in file.

        :param str filename: path and file name to key file
        :return key: encypting key
        """

        if os.path.isfile(filename):
            with open(filename, "rb") as key_file:
                key = key_file.read()
        else:
            key = Fernet.generate_key()
            with open(filename, "wb") as key_file:
                key_file.write(key)
        return key

    def encrypt(self, text: str):
        """Encode text using master password."""

        _byte = text.encode('ascii')
        return self.cipher_suite.encrypt(_byte)

    def decrypt(self, text: str):
        """Decode text using master password."""

        _byte = text.encode('ascii')
        decoded_text = self.cipher_suite.decrypt(_byte)
        return decoded_text.decode('ascii')

    def save_encrypted_to_text_file(self, message: str):
        """Writes an 'encrypted.txt' file to the root folder with the
        message, then opens it for you to copy/paste.

        Use:
            Encryption(password='1234').save_encrypted_to_text_file('message')
        * Password (optional) if empty/None, it reads the key file

        :param str message: Plain text to be encrypted"""

        secret = self.encrypt(message).decode('ascii')
        assert secret == self.decrypt(message)

        # write file
        filename = "encrypted.txt"
        with open(filename, "w") as f:
            f.write(secret)

        # open the file
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])


class Logger:
    """Logger to bertter diagnose and get relevant messages"""

    # logging.DEBUG, INFO, WARNING, ERROR, CRITICAL
    LEVEL = logging.DEBUG if S.DEBUG else logging.INFO
    FORMAT = "%(asctime)s, %(name)s, %(module)s, %(funcName)s, %(lineno)s, " \
             "%(levelname)s:  %(message)s "
    logging.basicConfig(filename="log.log",
                        filemode='w',
                        format=FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=LEVEL
                        )

    console = logging.StreamHandler()
    console.setLevel(LEVEL)
    console.setFormatter(logging.Formatter(FORMAT))
    logging.getLogger().addHandler(console)


class TDA:
    client = None

    def __init__(self):
        """Connects with TDA database using TDA-API."""

        API_KEY = S.TDA_API_KEY
        REDIRECT_URL = S.TDA_REDIRECT_URI
        TOKEN = S.TDA_TOKEN

        try:
            self.client = tda_connection(api_key=API_KEY,
                                         redirect_uri=REDIRECT_URL,
                                         token_path=TOKEN,
                                         webdriver_func=self.webdriver,
                                         asyncio=False)
        except Exception as e:
            print("\nTDA authorization ERROR.\n" + str(e) +
                  "\nEstablishing authorization via Chrome")

            log.info("\nTDA error. Establishing authorization via Chrome")

            try:
                cont = input("Delete token file? Y/N: ")

                if cont[0].strip().upper() == "Y":
                    if os.path.isfile(TOKEN):
                        os.remove(TOKEN)
                    self.client = tda_connection(api_key=API_KEY,
                                                 redirect_uri=REDIRECT_URL,
                                                 token_path=TOKEN,
                                                 webdriver_func=self.webdriver,
                                                 asyncio=False)
                else:
                    raise

            except Exception as e:
                log.critical(f"\nTDA authorization ERROR.\n{e}")
                sys.exit(e)

        log.info("Established TDA Ameritrade connection")

    @staticmethod
    def webdriver():
        """Establishes oAuth with API via Chrome.

        Creates Token file, once is created and valid, webdriver is not used.

        ChromeDriver is a dependency used to authorize TDA portal download
        driver from https://chromedriver.chromium.org/downloads
        Save ChromeDriver in same location as this Script

        Returns:
            driver (object): API authorization.
        """
        # Import selenium here because it's slow to import
        from selenium import webdriver
        import atexit
        import webbrowser

        try:
            driver = webdriver.Chrome()
            atexit.register(lambda: driver.quit())
            log.info("ChromeDriver created successfully")
            return driver
        except Exception as e:
            str_url = "https://chromedriver.chromium.org/downloads"
            print("\n**There was an error: {}"
                  "\nDOWNLOAD ChromeDriver from {} & INSTALL it in the same "
                  "folder as the script.".format(str(e), str_url))
            webbrowser.open(str_url, new=1)
            log.error("Error loading ChromeDriver.")
            raise


def performance(fn):
    """Measure of performance for Time and Memory. NOTE: It runs the
    function twice, once for time, once for memory.

    To measure time for each method we use the built-in *time* module.
    *Perf_counter* provides the clock with the highest available resolution
    To measure *memory* consumption we use the package memory-profiler.

    Use:
        @performance
        def some_function():
            x += 1
            print(x)

    """

    @wraps(fn)
    def inner(*args, **kwargs):
        fn_kwargs_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        print(f'\n{fn.__name__}({fn_kwargs_str})')

        # Measure time
        perf_time = time.perf_counter()
        # retval = fn(*args, **kwargs)
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - perf_time
        print(f'Time   {elapsed:0.4}')

        # Measure memory
        mem, retval = memory_usage((fn, args, kwargs),
                                   retval=True,
                                   timeout=200,
                                   interval=1e-7)

        print(f'Memory {max(mem) - min(mem)}')
        return retval

    return inner


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


class Chart:
    """Chart library"""

    @staticmethod
    def style_candlestick(df):
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])
        fig_json = json.dumps(fig, cls=pt.utils.PlotlyJSONEncoder)
        return fig_json

    @staticmethod
    def finviz():
        pass

    @staticmethod
    def chart(df):
        candlestick = go.Candlestick(x=df['Date'], open=df['Open'],
                                     high=df['High'], low=df['Low'],
                                     close=df['Close'])
        upper_band = go.Scatter(x=df['Date'], y=df['upper_band'],
                                name='Upper Bollinger Band',
                                line={'color': 'red'})
        lower_band = go.Scatter(x=df['Date'], y=df['lower_band'],
                                name='Lower Bollinger Band',
                                line={'color': 'red'})

        upper_keltner = go.Scatter(x=df['Date'], y=df['upper_keltner'],
                                   name='Upper Keltner Channel',
                                   line={'color': 'blue'})
        lower_keltner = go.Scatter(x=df['Date'], y=df['lower_keltner'],
                                   name='Lower Keltner Channel',
                                   line={'color': 'blue'})

        fig = go.Figure(
            data=[candlestick, upper_band, lower_band, upper_keltner,
                  lower_keltner])
        fig.layout.xaxis.type = 'category'
        fig.layout.xaxis.rangeslider.visible = False
        fig.show()


class Database:
    """Connect with PostgresSQL. host, db, usr, pwd defined elsewhere"""

    def __init__(self):
        try:
            self._conn = psycopg2.connect(host=S.DB_HOST,
                                          database=S.DB_NAME,
                                          user=S.DB_USER,
                                          password=S.DB_PASSWORD)
            self._cursor = self.connection.cursor(
                cursor_factory=psycopg2.extras.DictCursor)
        except Exception as e:
            print(f"Unable to connect to database!\n{e}")
            sys.exit(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.close()

    @property
    def connection(self):
        return self._conn

    @property
    def cursor(self):
        return self._cursor

    def commit(self):
        self.connection.commit()

    def close(self, commit=True):
        if commit:
            self.commit()
        self.connection.close()

    def execute(self, sql, params=None):
        self.cursor.execute(sql, params or ())

    def executemany(self, sql, params=None):
        self.cursor.executemany(sql, params)

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()

    def query(self, sql, params=None):
        self.cursor.execute(sql, params or ())
        response = self.fetchall()
        return response[0] if len(response) == 1 else response

    @staticmethod
    def parse_sql(sql_file_path):
        """Read *.sql file, parse lines"""
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            data = f.read().splitlines()
        stmt = ''
        stmts = []
        for line in data:
            if line:
                if line.startswith('--'):
                    continue
                stmt += line.strip() + ' '
                if ';' in stmt:
                    stmts.append(stmt.strip())
                    stmt = ''
        return stmts

    def create_table(self):
        """create table in database"""

        sql = self.parse_sql("resources/create_tables.sql")
        for i in sql:
            self.execute(i)

        self.commit()
        self.close()
        print("Done creating tables for database")

    def last_update(self, table, close=False) -> bool or None:
        """Gets timestamp from last_update table.

        :param str table: string with table name
        :param close: True to close the Database connection
        :return: True if last update within 24 hours. None if error"""
        try:
            query = f"SELECT date FROM last_update WHERE tbl = '{table}';"
            timestamp = self.query(query)[0]
            if close:
                self.close()
            now_utc = pytz.utc.localize(datetime.datetime.utcnow())
            within24 = None if timestamp is None else \
                (now_utc - timestamp).seconds <= 24 * 60 * 60
        except Exception as e:
            log.debug(e)
            within24 = None
        return within24

    def timestamp(self, table, close=False):
        query = f"""INSERT INTO last_update (tbl)
                    VALUES ('{table}')
                    ON CONFLICT (tbl)
                    DO UPDATE SET date = NOW();"""
        self.execute(query)
        if close:
            self.commit()
            self.close()

    def reset_tables(self):
        table = "symbols, last_update"
        query = f"TRUNCATE {table} CASCADE;"
        self.execute(query)
        self.close(commit=True)

    def update_rejected_symbols(self, symbols, current, close=False):
        """Updates rejected_symbols table with symbols that have problems.

               :param list symbols: list of symbols
               :param list current: list of rejected symbols
               :param close: True to close database connection"""

        values = list(set(symbols).difference(current))
        if len(values) != 0:
            values = [(value,) for value in values]
            sql = """INSERT INTO rejected_symbols (symbol_id)
                     VALUES ((SELECT id FROM symbols WHERE symbol = %s))
                     ON CONFLICT (symbol_id)
                     DO UPDATE SET date = NOW();"""
            self.executemany(sql, values)
            self.commit()
            if close:
                self.close()


class Portfolio:
    def __init__(self):
        self.get_automatic_portfolio()
        # self.create_custom_portfolio()

    def get_automatic_portfolio(self):
        self.ARK()

    class ARK:
        def __init__(self):
            url_base = \
                "https://ark-funds.com/wp-content/fundsiteliterature/csv/"
            files = [
                "ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
                "ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv",
                "ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv",
                "ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS.csv",
                "ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv",
                "THE_3D_PRINTING_ETF_PRNT_HOLDINGS.csv",
                "ARK_ISRAEL_INNOVATIVE_TECHNOLOGY_ETF_IZRL_HOLDINGS.csv"
            ]
            # funds = [self.get_ark_funds(url_base + file) for file in files]
            # funds = pd.concat(funds, ignore_index=True)
            urls = [url_base + file for file in files]
            funds = asyncio.run(self.get_ark_funds(urls))
            self.save_ark(funds)

        @staticmethod
        async def download(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url=url) as response:
                    return await response.text()

        async def get_ark_funds(self, url):
            dataset = await asyncio.gather(*[self.download(u) for u in url])
            # data = requests.get(url).text
            funds = []
            for data in dataset:
                data = data.replace('"', '')
                data = data.replace('(%)', '').replace('($)', '')
                data = data.splitlines()[:-2]
                data = [i.split(",") for i in data]
                data = pd.DataFrame(data=data[1:], columns=data[0])
                data['date'] = pd.to_datetime(data['date'])
                data = data[data['fund'].str.strip().astype(bool)]
                funds.append(data)
            return pd.concat(funds, ignore_index=True)

        @staticmethod
        def save_ark(data):
            with Database() as db:
                etfs = data['fund'].unique()
                etfs = ", ".join("'{0}'".format(x) for x in etfs)
                query = f"""SELECT id, symbol
                            FROM symbols 
                            WHERE symbol IN ({etfs})"""
                etfs = db.query(query)
                etfs = {i['symbol']: i['id'] for i in etfs}
                for row in data.itertuples():
                    if not row.ticker:
                        continue
                    query = "SELECT id FROM symbols WHERE symbol = %s"
                    holding = db.query(query, (row.ticker,))
                    if not holding:
                        continue
                    query = """
                     INSERT INTO portfolio 
                                 (id, holding_id, date, shares, weight)
                     VALUES (%s, %s, %s, %s, %s)
                     ON CONFLICT DO NOTHING;"""
                    db.execute(query, (etfs[row.fund], holding['id'], row.date,
                                       row.shares, row.weight))
                db.commit()


class News:
    """Get news from different sources"""

    @staticmethod
    def capture_reddit():
        """Get news from Reddit"""
        # API: reddit.com/dev/api
        # Other API: pushshift.io
        # Pushshift wrapper: psaw.readthedocs.io
        sub = 'wallstreetbets'
        filters = ['url', 'title', 'subreddit']
        limit = 1000
        start_date = '2/19/2021'

        dt = datetime.datetime
        start_date = dt.strptime(start_date, "%m/%d/%Y")
        start_date = int(start_date.timestamp())
        api = PushshiftAPI()
        submitted = list(api.search_submissions(after=start_date,
                                                subreddit=sub,
                                                filter=filters,
                                                limit=limit))
        with Database() as db:
            query = "SELECT id, symbol FROM symbols"
            rows = db.query(query)
        symbols = {f"${row['symbol']}": row['id'] for row in rows}

        args = []
        for submit in submitted:
            words = submit.title.split()

            def extract(word):
                return word.lower().startswith('$')

            cashtags = list(set(filter(extract, words)))
            if len(cashtags) == 0:
                continue
            for cashtag in cashtags:
                if cashtag not in symbols:
                    continue
                submit_time = dt.fromtimestamp(submit.created_utc).isoformat()
                arg = (submit_time, symbols[cashtag], submit.title,
                       sub, submit.url)
                args.append(arg)

        query = """
            INSERT INTO mention (date, symbol_id, mentions, source, url)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING"""
        if len(args) > 0:
            db.cursor.executemany(query, args)
            db.connection.commit()

    @staticmethod
    def show_reddit():
        analysis = {}
        with Database as db:
            query = "SELECT COUNT(*) FROM mention"
            analysis['count'] = db.query(query)

            query = """
                    SELECT COUNT(*) AS num_mentions, symbol_id, symbol
                    FROM mention JOIN symbols ON symbols.id = mention.symbol_id
                    GROUP BY symbol_id, symbol
                    HAVING COUNT(*) > 2
                    ORDER BY num_mentions DESC;"""
            analysis['mentions'] = db.query(query)

        for k, v in analysis.items():
            print(k, v)


class Snapshot:
    def __init__(self, forced=False):
        """Get symbols, pricing, fundamentals from internet, database or file.
        Then save them to a file and database. Sources: NASDAQ, Wikipedia,
        Alpaca

        :param forced: True to force data refresh"""

        s = t()

        assets = self.GetAssets(forced)
        self.GetFundamental(assets.symbols, forced)
        self.GetPrice(assets.symbols, forced)

        log.info(f"{self.__class__.__name__} took {t() - s} seconds")

    class GetAssets:
        def __init__(self, forced=False):
            """Load symbols from database or file, when older than 24 hours it
            downloads. If DEBUG then symbols list is limited.

            :param forced: True to force symbols download
            :return: list of symbols, dataframe with symbol/security names"""

            self.data = self.load(forced)
            self.unique_id()
            self.save()
            self.data, wihtin24 = self.load_database()
            self.symbols = self.data['symbol'].to_list()
            self.symbols = self.symbols[:S.MAX_SYMBOLS]

        def load(self, forced):
            """Gets the assets from database, file or internet

            :param bool forced: True to force symbols download
            :return: list of symbols, dataframe with symbol/security names"""

            # get symbols from database or file
            try:
                df, within24 = self.load_database()
            except Exception as e:
                log.info(e)
                try:
                    df, within24 = self.load_file()
                except Exception as e:
                    log.info(e)
                    df, within24 = None, False

            # download the symbols if data it's older than 24 hours or forced
            if not within24 or forced:
                df = self.download()

            # format symbols for YFinance and create symbols list
            df.replace({'symbol': r'\.'}, '-', regex=True, inplace=True)

            return df

        @staticmethod
        def load_database():
            """Load symbols & security names from database, excludes rejects"""

            with Database() as db:
                query = """SELECT symbol, security FROM symbols
                           WHERE symbols.id
                           NOT IN (SELECT symbol_id FROM rejected_symbols)
                           ORDER BY symbol;"""
                df = pd.read_sql_query(query, con=db.connection)
                within24 = db.last_update(table='symbols')

            if df.empty:
                raise Exception("Empty symbols database")

            log.info("Symbols loaded from database")

            return df, within24

        @staticmethod
        def load_file():
            """Loads symbols from pickle file in the datasets directory"""

            if os.path.isfile(S.SYMBOLS_FILE):
                df = pd.read_csv(S.SYMBOLS_FILE)
                if df.empty:
                    raise Exception("Empty symbols file")
                df = df.rename(columns=str.lower)
                df = df[['symbol', 'security']]

                timestamp = os.stat(S.SYMBOLS_FILE).st_mtime
                timestamp = datetime.datetime.fromtimestamp(
                    timestamp, tz=datetime.timezone.utc)
                now_utc = pytz.utc.localize(datetime.datetime.utcnow())
                within24 = (now_utc - timestamp).seconds <= 24 * 60 * 60

                log.info('Symbols loaded from CSV file')

                return df, within24
            else:
                log.info("Symbols file not found")
                raise Exception("Empty symbols file")

        def download(self):
            """Downloads symbols from the internet, cascade through sources"""

            # download from Nasdaq
            try:
                df = self.download_from_nasdaq()
                log.info("Symbols downloaded from Nasdaq")
                return df
            except Exception as e:
                log.warning(f"Nasdaq download failed. {e}")

            # or download from Alpaca
            try:
                df = self.download_alpaca()
                log.info("Symbols downloaded from Alpaca")
                return df
            except Exception as e:
                log.warning(f"Alpaca download failed. {e}")

            # or download S&P 500 from Wikipedia
            try:
                df = self.download_sp500()
                log.info("Symbols downloaded from Wikipedia S&P 500")
                return df
            except Exception as e:
                log.warning(f"Wikipedia S&P500 download failed. {e}")

            # unable to download symbols
            log.critical('Could not download the Symbols. Check code/sources')
            sys.exit(1)

        @staticmethod
        def download_from_nasdaq():
            """Downloads NASDAQ-traded stock/ETF Symbols, from NASDAQ FTP.

            Filtering: NASDAQ-traded, no test issues, round Lots,
            good financial status, no NextShares, Symbols with fewer than 5
            digits, and other keywords (no funds, notes, depositary, bills,
            warrants, unit, security)

            Info is saved into 'nasdaqtraded.txt' file, available via FTP,
            updated nightly. Logs into ftp.nasdaqtrader.com, anonymously, and
            browses for SymbolDirectory.
            ftp://ftp.nasdaqtrader.com/symboldirectory
            http://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs

            :return: Symbols and Security names in DataFrame object"""

            lines = []
            with FTP(host='ftp.nasdaqtrader.com') as ftp:
                ftp.login()
                ftp.cwd('symboldirectory')
                ftp.retrlines("RETR nasdaqtraded.txt", lines.append)
                ftp.quit()
            lines = [lines[x].split('|') for x in range(len(lines) - 1)]
            df = pd.DataFrame(data=lines[1:], columns=lines[0])
            cols = {'NASDAQ Symbol': 'symbol', 'Security Name': 'security'}
            drop = " warrant|depositary|- unit| notes |interest|%| rate "
            df = df.loc[(df['Nasdaq Traded'] == 'Y') &
                        (df['Listing Exchange'] != 'V') &
                        (df['Test Issue'] == 'N') &
                        (df['Round Lot Size'] == '100') &
                        (df['NextShares'] == 'N') &
                        (df['Symbol'].str.len() <= 5) &
                        (df['Financial ''Status'].isin(['', 'N'])) &
                        (~df['Security Name'].str.contains(drop, case=False)),
                        list(cols.keys())]
            df.rename(columns=cols, inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df

        @staticmethod
        def download_alpaca():
            """Downloads assets from Alpaca and retrieves symbols. Assets
            from Alpaca contain a lot more than symbols and names.

            :return: Symbols and Security names in DataFrame object"""

            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(key_id=S.ALPACA_KEY,
                                secret_key=S.ALPACA_SECRET,
                                base_url=S.ALPACA_ENDPOINT)
            active_assets = api.list_assets(status='active')
            tradable_assets = [(asset.symbol, asset.name) for asset in
                               active_assets if asset.tradable]
            result = [dict(zip(['symbol', 'security'], asset)) for asset in
                      tradable_assets]
            df = pd.DataFrame(result, columns=['symbol', 'security'])

            return df

        @staticmethod
        def download_sp500():
            """Scrape wikipedia for S&P500 list

            :return: Symbols and Security names in DataFrame object"""

            table = pd.read_html(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            df = table[0]
            df = df[['Symbol', 'Security']]
            df = df.rename(columns=str.lower)

            return df

        def unique_id(self):
            """Create and embeds a Unique ID for every symbol.

            symbol_id = symbol filled with 0 up to 6 digits + first 6 chars
            of name, all alphanumeric. Example: 0000PGPROCTE"""
            self.data['symbol_id'] = (
                    self.data['symbol']
                    .str.replace('[^a-zA-Z0-9]', '', regex=True)
                    .str[:6]
                    .str.zfill(6) +
                    self.data['security']
                    .str.replace('[^a-zA-Z0-9]', '', regex=True)
                    .str[:6]
                    .str.upper()
                    .str.zfill(6)
            )

            log.info("Unique ID created")

        def save(self):
            """Saves the symbol data in a pickle file and the database"""

            self.save_to_file()
            asyncio.run(self.save_to_database())

        def save_to_file(self):
            """Save symbols to file"""

            self.data.to_csv(S.SYMBOLS_FILE, index=False)

            log.info(f"Symbols saved to CSV file: {S.SYMBOLS_FILE}")

        async def save_to_database(self):
            """ASYNC function to save database"""

            values = self.data.to_records(index=False)
            sql = """INSERT INTO symbols (symbol, security, id)
                     VALUES (%s, %s, %s)
                     ON CONFLICT (id) DO UPDATE
                     SET security=excluded.security;"""
            with Database() as db:
                db.executemany(sql, values)
                db.timestamp('symbols')
                db.commit()

            log.info("Symbols saved to database")

    class GetFundamental:
        def __init__(self, symbols, forced=False):
            """Load fundamentals from datbase or file, when older than 24 hours
            it downloads.

            :param list symbols: list of symbols
            :param bool forced: True bypasses within 24 hour requirement
            :return: dataframe with fundamental info."""

            self.data = None
            within24 = Database().last_update('fundamentals', close=True)

            # download or load from database/file. If from file, save database
            if not within24 or forced:
                try:
                    self.download(symbols)
                    self.save()
                except Exception as e:
                    log.info(f"Error downloading or saving Fundamentals {e}")
                    sys.exit(e)
            else:
                try:
                    self.load_database()
                    self.process_loaded_fundamentals()
                except Exception as e:
                    log.info(e)
                    try:
                        self.load_file()
                        asyncio.run(self.save_to_database())
                    except Exception as e:
                        log.critical(f"Could not load fundamental data. {e}")
                        sys.exit(1)

        def download(self, symbols):
            """Download fundamentals from different sources and returns a
            single dataframe, four columns: stock, source, variable & value.

            List of symbols is chunkable by changing the chunk_size:
                * chunk_size is optional. Empty downloads all.
                * it can be a constant, i.e. 250.
                * or divide all symbols into equal chunks

            :param list symbols: list of symbols to download fundamentals"""

            chunk_size = 5
            # n = 3  # 3 for divide symbols into thirds
            # chunk_size = (len(symbols) + n) // n  # into n-ths

            df_tda = self.download_from_tda(symbols, chunk_size)
            df_yahoo = self.download_from_yahoo(symbols, chunk_size)

            # concatenate tables using 'symbol' as index
            self.data = pd.concat([df_tda, df_yahoo],
                                  keys=['tda', 'yahoo'],
                                  axis=1)
            self.data.rename_axis('symbol', inplace=True)

            # update symbol rejects database
            Database().update_rejected_symbols(
                symbols=symbols,
                current=list(self.data.index),
                close=True)

        @staticmethod
        def unpack_dictionary_fundamental(df, packages: list):
            """Unpack dictionaries and concatenate to table"""
            # df = pd.concat([pd.DataFrame(None, index=['package']), df]) # DBG
            for package in packages:
                on_hand = df.loc[df[package].notnull(), package]
                unpack_df = pd.json_normalize(on_hand).set_index(on_hand.index)
                # unpack_df.loc['package'] = package[1:]  # DBG
                df.drop(package, axis=1, inplace=True)
                df = pd.concat([df, unpack_df], axis=1)
            df.rename_axis('symbol', inplace=True)
            return df

        def download_from_tda(self, symbols: list, chunk_size: int = None):
            """Get TD Ameritrade fundamental data in batches, if necessary."""
            # Create connection, return empty df if there is no connection
            try:
                con = TDA()
            except Exception as e:
                log.info(f"TDA Connection failed. {e}")
                return pd.DataFrame()

            # download stocks data in chunks and concatenate to large table
            chunk_size = len(symbols) if chunk_size is None else chunk_size

            df = pd.DataFrame()
            for i in range(0, len(symbols), chunk_size):
                symbol_chunk = symbols[i:i + chunk_size]
                tickers = con.client.search_instruments(
                    symbols=symbol_chunk,
                    projection=con.client.Instrument.Projection.FUNDAMENTAL)
                if tickers.status_code == 200:
                    chunk_df = pd.DataFrame(tickers.json()).T
                    df = pd.concat([df, chunk_df])
                else:
                    raise Exception("TDA data download, unsuccesful")

            # unpack the dictionaries
            """['fundamental']"""
            packages = ['fundamental']
            # packages = df.columns  # for unpacking all

            df = self.unpack_dictionary_fundamental(df, packages)
            log.info("TDA fundamentals downloaded")
            return df

        def download_from_yahoo(self, symbols: list, chunk_size: int = None):
            """Get Yahoo fundamental data in batches, if necessary."""
            # download stocks data in chunks and concatenate to large table
            chunk_size = len(symbols) if chunk_size is None else chunk_size

            df = pd.DataFrame()
            for i in range(0, len(symbols), chunk_size):
                symbol_chunk = symbols[i:i + chunk_size]
                tickers = yq.Ticker(symbols=symbol_chunk,
                                    asynchronous=True).all_modules
                chunk_df = pd.DataFrame(tickers).T
                df = pd.concat([df, chunk_df])

            # unpack the dictionaries
            """['assetProfile', 'recommendationTrend', 'industryTrend',
             'cashflowStatementHistory', 'indexTrend', 
             'defaultKeyStatistics', 'quoteType', 'fundOwnership',
             'incomeStatementHistory', 'summaryDetail', 'insiderHolders', 
             'calendarEvents', 'upgradeDowngradeHistory', 'price',
             'balanceSheetHistory', 'earningsTrend', 'secFilings',
             'institutionOwnership', 'majorHoldersBreakdown',
             'balanceSheetHistoryQuarterly', 'earningsHistory', 
             'esgScores', 'summaryProfile', 'netSharePurchaseActivity', 
             'insiderTransactions',  'sectorTrend', 'fundPerformance', 
             'incomeStatementHistoryQuarterly', 'financialData', 
             'cashflowStatementHistoryQuarterly', 'earnings', 'pageViews',
             'fundProfile', 'topHoldings']"""
            packages = ['summaryProfile', 'summaryDetail', 'quoteType']
            # packages = df.columns  # for unpacking all

            # adding prefix to avoid duplicated label errors unpacking
            df = df.add_prefix("_")
            packages = ["_" + i for i in packages]

            # unpack dictionaries and concatenate to table
            df = self.unpack_dictionary_fundamental(df, packages)

            # deal with duplicates
            erase = ['maxAge', 'symbol']
            try:
                df.drop(erase, axis='columns', inplace=True, errors='ignore')
            except AttributeError as e:
                log.info(f"Dataframe may be empty. {e}")
                raise

            log.info("Yahoo fundamentals downloaded")
            return df

        def load_database(self):
            """Retrieves fundamental data from database, filtered of rejects"""
            query = """SELECT symbols.symbol,
                              fundamentals.source,
                              fundamentals.var,
                              fundamentals.val 
                       FROM fundamentals
                       INNER JOIN symbols
                       ON fundamentals.symbol_id = symbols.id
                       WHERE symbols.id
                       NOT IN (SELECT symbol_id FROM rejected_symbols)
                       ORDER BY symbols.symbol;"""
            with Database() as db:
                self.data = pd.read_sql_query(query, con=db.connection)

            log.info("Loaded fundamentals from database")

        def load_file(self):
            """Gets fundamentals data from pickle file"""
            self.data = pd.read_pickle(S.FUNDAMENTALS_FILE)

            log.info("Loaded fundamentals from file")

        def process_loaded_fundamentals(self):
            """Transform database format to user-friendly format"""
            def literal_return(val):
                try:
                    return literal_eval(val)
                except (ValueError, SyntaxError):
                    return val

            df = self.data
            df['val'] = df['val'].apply(lambda x: literal_return(x))
            self.data = df.pivot(index=['symbol'],
                                 columns=['source', 'var'],
                                 values=['val'])['val']

        def save(self):
            """Saves the fundamental data in a pickle file and the database"""

            self.save_to_file()
            asyncio.run(self.save_to_database())

        def save_to_file(self):
            """Save fundamental data in a pickle file"""

            self.data.to_pickle(S.FUNDAMENTALS_FILE)
            self.data.T.to_csv(S.FUNDAMENTALS_FILE + ".csv")
            log.info(f"Fundamentals saved to pickle file: "
                     f"{S.FUNDAMENTALS_FILE}")

        async def save_to_database(self):
            """Save fundamental data in a database"""
            df = self.data
            df = df.unstack().reset_index()
            df.columns = ['source', 'variable', 'symbol', 'value']
            df = df[['symbol', 'source', 'variable', 'value']]
            df['value'] = df['value'].astype(str)

            values = df.to_records(index=False)
            query = """INSERT INTO fundamentals (symbol_id, source, var, val)
                       VALUES ((SELECT id FROM symbols WHERE symbol=$1),
                               $2, $3, $4)
                       ON CONFLICT (symbol_id, source, var)
                       DO UPDATE
                       SET symbol_id=excluded.symbol_id, 
                           source=excluded.source, 
                           var=excluded.var, 
                           val=excluded.val;"""

            # using asynciopf as the Database connection manager for Async
            creds = {'host': S.DB_HOST, 'database': S.DB_NAME,
                     'user': S.DB_USER, 'password': S.DB_PASSWORD}
            async with asyncpg.create_pool(**creds) as pool:
                async with pool.acquire() as db:
                    async with db.transaction():
                        await db.executemany(query, values)

            Database().timestamp('fundamentals', close=True)

        # list of columns in YFinance
        """['zip', 'sector', 'fullTimeEmployees']
        ['longBusinessSummary', 'city', 'phone']
        ['state', 'country', 'companyOfficers']
        ['website', 'maxAge', 'address1']
        ['industry', 'previousClose', 'regularMarketOpen']
        ['twoHundredDayAverage', 'trailingAnnualDividendYield', 'payoutRatio']
        ['volume24Hr', 'regularMarketDayHigh', 'navPrice', , 'totalAssets']
        ['averageDailyVolume10Day', 'regularMarketPreviousClose']
        ['fiftyDayAverage', 'trailingAnnualDividendRate', 'open']
        ['toCurrency', 'averageVolume10days', 'expireDate']
        ['yield', 'algorithm', 'dividendRate']
        ['exDividendDate', 'beta', 'circulatingSupply']
        ['startDate', 'regularMarketDayLow', 'priceHint']
        ['currency', 'trailingPE', 'regularMarketVolume']
        ['lastMarket', 'maxSupply', 'openInterest']
        ['marketCap', 'volumeAllCurrencies', 'strikePrice']
        ['averageVolume', 'priceToSalesTrailing12Months', 'dayLow']
        ['ask', 'ytdReturn', 'askSize']
        ['volume', 'fiftyTwoWeekHigh', 'forwardPE']
        ['fromCurrency', 'fiveYearAvgDividendYield', 'fiftyTwoWeekLow']
        ['bid', 'tradeable', 'dividendYield']
        ['bidSize', 'dayHigh', 'exchange']
        ['shortName', 'longName', 'exchangeTimezoneName', 'isEsgPopulated']
        ['exchangeTimezoneShortName', 'gmtOffSetMilliseconds']
        ['quoteType', 'symbol', 'messageBoardId']
        ['market', 'annualHoldingsTurnover', 'enterpriseToRevenue']
        ['beta3Year', 'profitMargins', 'enterpriseToEbitda']
        ['52WeekChange', 'morningStarRiskRating', 'forwardEps']
        ['revenueQuarterlyGrowth', 'sharesOutstanding', 'fundInceptionDate']
        ['annualReportExpenseRatio', 'bookValue', 'sharesShort']
        ['sharesPercentSharesOut', 'fundFamily', 'lastFiscalYearEnd']
        ['heldPercentInstitutions', 'netIncomeToCommon', 'trailingEps']
        ['lastDividendValue', 'SandP52WeekChange', 'priceToBook']
        ['heldPercentInsiders', 'nextFiscalYearEnd', 'mostRecentQuarter']
        ['shortRatio', 'sharesShortPreviousMonthDate', 'floatShares']
        ['enterpriseValue', 'threeYearAverageReturn', 'lastSplitDate']
        ['lastSplitFactor', 'legalType', 'lastDividendDate']
        ['morningStarOverallRating', 'earningsQuarterlyGrowth']
        ['dateShortInterest', 'pegRatio', 'lastCapGain', 'shortPercentOfFloat']
        ['sharesShortPriorMonth', 'impliedSharesOutstanding', 'category']
        ['fiveYearAverageReturn', 'regularMarketPrice', 'logo_url']
        """

    class GetPrice:
        def __init__(self, symbols=None, forced=False):
            """Gets price history data from download if not within 24 hours,
               then from data, finally from file.

               :param symbols: list of symbols
               :param forced: True bypassess 24 hour download requirement"""

            self.data = None
            within24 = Database().last_update('price_history', close=True)

            # download or load from database or file. If from file, save db
            if not within24 or forced:
                asyncio.run(self.download(symbols))
                self.save()
            else:
                try:
                    self.data = self.load_database()
                except Exception as e:
                    log.info(e)
                    try:
                        self.load_file()
                        asyncio.run(self.save_to_database())
                    except Exception as e:
                        log.critical(f"Could not load price data. {e}")
                        sys.exit(1)

        async def download(self, symbols):
            """Download price data.

            List of symbols is chunkable by changing the chunk_size.

            :param list symbols: list of symbols to download fundamentals"""

            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=S.DAYS)

            collector = pd.DataFrame()

            # chunk_size = 2
            # chunk_size = len(symbols)  # use actual chunk size i.e. 200
            n = 3
            chunk_size = (len(symbols) + n) // n  # into n-ths

            for i in range(0, len(symbols), chunk_size):
                symbol_chunk = symbols[i:i + chunk_size]

                # download chunk of fundamentals from YFinance
                df_chunk = await self.yfinance_price_history(symbol_chunk,
                                                             start_date,
                                                             end_date)

                # YFinance won't return symbol label if it's only one symbol
                if len(symbol_chunk) == 1:
                    df_chunk.columns = pd.MultiIndex.from_product(
                        [df_chunk.columns, symbol_chunk]).swaplevel(0, 1)

                collector = pd.concat([collector, df_chunk], axis=1)

            # cleanup any NaN values in the dataframe
            collector.dropna(how='all', inplace=True)
            collector.dropna(how='all', axis=1, inplace=True)

            self.data = collector.stack(level=0).reset_index()
            self.data.rename(columns={'level_1': 'symbol'}, inplace=True)

            # update symbol rejects database
            Database().update_rejected_symbols(
                symbols=symbols,
                current=self.data['symbol'].unique(),
                close=True)

            log.info("Price history downloaded")

        # TODO: JUST GET LATEST PRICE DATA, MAY REQUIRE SINGLE TICKER DOWNLOAD
        # @staticmethod
        # def date_range(self):
        #     """Dates to download history.
        #     Total days comes from Settings, whether you want 150 days back.
        #     The dat of the last update is the latest date in the symbols
        #     price history date field.
        #     """
        #     return start_date, end_date

        @staticmethod
        async def yfinance_price_history(symbol_chunk, start_date, end_date):
            """ASYNC function to download price history data from YFinance"""

            return yf.download(symbol_chunk,  # list of tickers
                               start=start_date,  # start date
                               end=end_date,  # end date
                               period='max',  # change instead of start/end
                               group_by='ticker',  # or 'column'
                               progress=False,  # show progress bar
                               rounding=False,  # round 2 descimals
                               actions=False,  # include dividend/earnings
                               threads=True,  # multi threads bool or int
                               auto_adjust=False,  # adjust OHLC with adj_close
                               back_adjust=False,
                               prepost=False,  # include before/after market
                               )

        @staticmethod
        def load_database():
            """Retrieves the price data from database, filtered of rejects"""

            query = """ SELECT price_history.date,
                            symbols.symbol,
                            price_history.open,
                            price_history.high,
                            price_history.low,
                            price_history.close,
                            price_history.adj_close,
                            price_history.volume
                        FROM price_history
                        INNER JOIN symbols
                        ON price_history.symbol_id = symbols.id
                        WHERE symbols.id
                        NOT IN (SELECT symbol_id FROM rejected_symbols)
                        ORDER BY symbols.symbol, price_history.date;"""

            with Database() as db:
                df = pd.read_sql_query(query, con=db.connection)

            # df.drop(['symbol_id'], inplace=True, axis=1)

            log.info("Loaded price history from database")

            return df

        def load_file(self):
            self.data = pd.read_pickle(S.PRICE_FILE)

        def save(self):
            """Saves the pricing data in a pickle file and the database"""

            self.save_to_file()
            asyncio.run(self.save_to_database())

        def save_to_file(self):
            """Save pricing data in a pickle file"""

            self.data.to_pickle(S.PRICE_FILE)

            log.info(f"Price data saved to pickle file: {S.PRICE_FILE}")

        async def save_to_database(self):
            """Save price data in a database"""

            rows = self.data.itertuples(index=False)
            values = [list(row) for row in rows]
            query = """INSERT INTO price_history (date, symbol_id, open,
                                                  high, low, close, 
                                                  adj_close, volume)
                       VALUES ($1, (SELECT id FROM symbols WHERE symbol=$2),
                                $3, $4, $5, $6, $7, $8)
                       ON CONFLICT (symbol_id, date)
                       DO UPDATE
                       SET  symbol_id=excluded.symbol_id,
                            date=excluded.date, open=excluded.open,
                            high=excluded.high, low=excluded.low,
                            close=excluded.close,
                            adj_close=excluded.adj_close,
                            volume=excluded.volume;"""

            # using asynciopg as the Database connection manager for Async
            creds = {'host': S.DB_HOST, 'database': S.DB_NAME,
                     'user': S.DB_USER, 'password': S.DB_PASSWORD}
            async with asyncpg.create_pool(**creds) as pool:
                async with pool.acquire() as db:
                    async with db.transaction():
                        await db.executemany(query, values)

            Database().timestamp('price_history', close=True)


class Index:
    signals = None
    stocks = None

    def __init__(self, active_filter):
        self.settings = active_filter
        self.load_data()
        self.apply_filters()
        self.prep_arguments()

    def load_data(self):
        # load symbols
        try:
            symbols, within24 = Snapshot.GetAssets.load_database()
            if symbols.empty:
                raise Exception("Blank database")
        except Exception as e:
            print(e)
            symbols = pd.read_csv(
                S.SYMBOLS_FILE, index_col='symbol', usecols=[0, 1])
        # self.stocks = symbols.T.to_dict()
        self.stocks = symbols.set_index('symbol').T.to_dict('dict')

        # load signal table
        try:
            self.signals = Filter.read_signals()
            if self.signals.empty:
                raise Exception("Blank database")
        except Exception as e:
            print(e)
            self.signals = pd.read_pickle(S.SIGNALS_FILE)

    def apply_filters(self):
        """Filter signal table with selections"""
        signals = self.signals

        # only show the filters wanted
        for k in self.settings.keys():
            if k in signals.columns:
                signals = signals[signals[k] == self.settings[k]['go']]

        # remove the nan
        signals = signals.fillna('')

        self.signals = signals

    def prep_arguments(self):
        """Prepare argument to be sent to Flask/HTML"""
        stocks = self.stocks

        # only show symbols that exist in signals
        stocks = {k: v
                  for k, v in stocks.items()
                  if k in list(self.signals.index)}

        # send candle info to server
        if self.settings['candlestick']['go']:
            for symbol, row in self.signals.iterrows():
                stocks[symbol]['cdl_sum_ber'] = row.cdl_sum_ber
                stocks[symbol]['cdl_sum_bul'] = row.cdl_sum_bul

        if self.settings['investor_reco']['go']:
            for symbol, row in self.signals.iterrows():
                stocks[symbol]['investor_sum'] = row.investor_sum

        self.stocks = stocks

    @staticmethod
    def last_update(table, source_file=False):
        """last update date"""
        now_utc = pytz.utc.localize(datetime.datetime.utcnow())
        timestamp = now_utc - datetime.timedelta(hours=25)
        if source_file:
            if os.path.isfile(S.PRICE_FILE):
                timestamp = os.stat(S.PRICE_FILE).st_mtime
        else:
            query = f"SELECT date FROM last_update WHERE tbl = '{table}';"
            with Database() as db:
                timestamp_db = db.query(query)[0]
            timestamp = timestamp if timestamp_db is None else timestamp_db
        within_24 = (now_utc - timestamp).seconds <= 24 * 60 * 60
        timestamp_est = timestamp.astimezone(pytz.timezone('US/Eastern'))
        return timestamp_est.strftime('%m-%d-%Y %I:%M %p %Z'), within_24


# run in terminal 'flask run', auto-update with 'export FLASK_ENV=development'
app = Flask(__name__)
app.config['SECRET_KEY'] = S.FLASK_SECRET_KEY


@app.route("/stock/<symbol>")
def stock_detail(symbol):
    try:
        symbols, within24 = Snapshot.GetAssets.load_database()
        if symbols.empty:
            raise Exception("Blank database")
    except Exception as e:
        print(e)
        symbols = pd.read_csv(
            S.SYMBOLS_FILE, index_col='symbol', usecols=[0, 1])
    # self.stocks = symbols.T.to_dict()
    stocks = symbols.set_index('symbol').T.to_dict('dict')

    # df = pd.read_csv(S.SYMBOLS_FILE, index_col='symbol', usecols=[0, 1])
    # stocks = df.T.to_dict()
    security = stocks[symbol]['security']

    try:
        data = Snapshot.GetPrice.load_database()
        if data.empty:
            raise Exception("Blank database")
    except Exception as e:
        print(e)
        data = pd.read_pickle(S.PRICE_FILE)
    bars_df = data.loc[(data['symbol'] == symbol)]
    bars_df = bars_df.drop(['symbol'], axis=1)
    # df.reset_index(drop=True, inplace=True)
    #    bars_df = data[symbol]
    # bars_df = bars_df.reset_index()
    # data = data.set_index('symbol').T.to_dict('dict')
    bars_df = bars_df.sort_values(by=['date'], ascending=False)
    bars = bars_df.to_dict('records')
    stock = {'symbol': symbol,
             'security': security,
             'bars': bars}

    return render_template("stock_detail.html",
                           stock=stock)


@app.route("/snapshot")
def snapshot_wrapper():
    """Downloads all data from Snapshot class"""

    Snapshot(forced=True)

    return redirect(url_for('index'))


@app.route("/", methods=['GET', 'POST'])
def index():
    """Renders index.html page"""

    # Defaults and variable initialization
    active_filters = S.DEFAULT_FILTERS
    data = {}
    settings = ''

    # refresh data by pressing 'Refresh data' button
    if request.method == 'GET':
        # TODO: IN HTML PROVIDE OPTIONS OF WHAT SYMBOLS TO DOWNLOAD
        Snapshot(forced=False)

    # apply filters by pressing 'Submit' button
    if request.method == 'POST':
        # read current options from index form under name='filter'
        filters_selected = request.form.getlist('filter')
        if len(filters_selected) == 0:
            flash("Please make a selection")
        else:
            s = t()

            # use only the filters available as options
            def fltr_me(x):
                return x in filters_selected

            frm_get = request.form.get
            active_filters = {
                'consolidating':
                    {'go': fltr_me('consolidating'),
                     'pct': float(frm_get('consolidating_pct'))},

                'breakout':
                    {'go': fltr_me('breakout'),
                     'pct': float(frm_get('breakout_pct'))},

                'ttm_squeeze':
                    {'go': fltr_me('ttm_squeeze')},

                'in_the_squeeze':
                    {'go': fltr_me('in_the_squeeze')},

                'candlestick':
                    {'go': fltr_me('candlestick')},

                'sma_filter':
                    {'go': fltr_me('sma_filter'),
                     'fast': float(frm_get('sma_fast')),
                     'slow': float(frm_get('sma_slow'))},

                'ema_stacked':
                    {'go': fltr_me('ema_stacked')},

                'investor_reco':
                    {'go': fltr_me('investor_reco')}
            }

            # run filter or signal function and save result to database
            Filter(active_filters)

            # prepare information to be presented to html
            data = Index(active_filters).stocks

            # message to front
            flash(f"Done thinking. It took me {t() - s:0.2} seconds.")

    # get date of last price_history from file or database
    price_update_dt, within24 = Index.last_update('price_history')

    return render_template("index.html",
                           stocks=data,
                           data_last_modified=price_update_dt,
                           active_filters=active_filters,
                           settings=settings)


if __name__ == "__main__":
    Logger()

    try:
        if sys.argv[1] in ["create_tables", "create_table", "create",
                           "table", "reset", "nukeit", "createtable",
                           "start", "go"]:
            Database().create_table()
            sys.exit(0)
        else:
            pass
    except Exception as exception_e:
        pass

    app.run(debug=True)

    # globals()[sys.argv[1]]()

# stocks = {}
#     if pattern:
#         pattern_function = getattr(talib, pattern)
#         with open("datasets/symbols.csv") as f:
#             stocks = {row[0]: {"company": row[1]} for row in csv.reader(f)}
#         data = pd.read_pickle("datasets/data.pkl")
#         signals = pd.read_pickle("datasets/signals.pkl")
#         for k in settings.keys():
#             signals = signals[signals[k] == settings[k]['go']]
#
#         for symbol, fltr in signals.iterrows():  # stocks.keys():
#             df = data[symbol]
#             try:
#                 results = pattern_function(
#                     df["Open"], df["High"], df["Low"], df["Close"]
#                 )
#                 last = results.tail(1).values[0]
#                 stocks[symbol]['chart'] = Chart.style_candlestick(df)
#                 if last > 0:
#                     stocks[symbol][pattern] = "Bullish"
#                 elif last < 0:
#                     stocks[symbol][pattern] = "Bearish"
#                 else:
#                     stocks[symbol][pattern] = None
#             except Exception as e:
#                 print("error", str(e))

# if os.path.isfile(self.symbols):
#     with open(self.symbols, mode='r') as f:
#         companies = {rows[0]: rows[1] for rows in csv.reader(f)}
# else:
#     companies = self.download_sp500()

# symbols = [k for k in companies.keys()]
# symbols = list(set(symbols))
# symbols.sort()

# Update de-listed symbols to original symbol file

# matched = {k: v for k, v in companies.items() if k in data.columns}
# sorted(matched)
# with open("datasets/symbols.csv", "w", newline="") as f:
#     w = csv.writer(f)
#     for k, v in matched.items():
#         w.writerow([k, v])
#
#
# # df = df.sort_values(by='Security', key=lambda col: col.str.lower())
# df.to_csv(symbols_file, index=False)
# # df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])


# return render_template(
#      "index.html",
#      candlestick_patterns=candlestick_patterns,
#      stocks=stocks,
#      pattern=pattern,
#      # tables=[signals.to_html(classes='data')],
#      # titles=signals.columns.values
#  )

#
# df = pd.read_csv(Const.symbols_file)
# df = df[['Symbol', 'Security']]
# df = df.T
# df.columns = df.iloc[0]
# df = df.drop(df.index[0])
# self.stocks = df.to_dict()

# #        # replace '.' with '-' a Yahoo Finance issue
#         df = df.assign(
#             Symbol=lambda x: x['Symbol'].str.replace('.', '-', regex=True))

# with open(Const.symbols_file) as f:
#     self.stocks = {row[0]: {"company": row[1]}
#                    for row in csv.reader(f)}

# # replace '.' with '-' a Yahoo Finance issue
# symbols = [x.replace('.', '-') for x in symbols]

# symbol = kwargs['symbol']
#         site = 'https://www.finviz.com/quote.ashx?t={}'.format(symbol)
#         headers = {'User-Agent': 'Mozilla/5.0'}
#         search = '//table[@class="fullview-ratings-outer"]//tr[1]//text()'
#
#         screen = requests.get(site, headers=headers)
#         tree = html.fromstring(screen.content).xpath(search)
#
#         bull_words = ['Buy', 'Outperform', 'Overweight', 'Upgrade']
#         bear_words = ['Sell', 'Underperform', 'Underweight', 'Downgrade']
#         bull = sum([tree.count(x) for x in bull_words])
#         bear = sum([tree.count(x) for x in bear_words])
#
#         self.signal.loc[symbol, 'investor_sum'] = \
#             "Bull {}, Bear {}".format(bull, bear)
#         return True

# !/usr/bin/env/python

# import psycopg2
# import os
# from io import StringIO
# import pandas as pd
#
# # Get a database connection
# dsn = os.environ.get('DB_DSN')  # Use ENV vars: keep it secret, keep it safe
# conn = psycopg2.connect(dsn)
#
# # Do something to create your dataframe here...
# df = pd.read_csv("file.csv")
#
# # Initialize a string buffer sio = StringIO() sio.write(df.to_csv(
# index=None, header=None))  # Write the Pandas DataFrame as a csv to the
# buffer sio.seek(0)  # Be sure to reset the position to the start of the
# stream
#
# # Copy the string buffer to the database, as if it were an actual file
# with conn.cursor() as c:
#     c.copy_from(sio, "schema.table", columns=df.columns, sep=',')
#     conn.commit()


# vectorizing pandas: https://gdcoder.com/speed-up-pandas-apply-function
# -using-dask-or-swifter-tutorial/ vectorizing:
# https://stackoverflow.com/questions/52673285/performance-of-pandas-apply
# -vs-np-vectorize-to-create-new-column-from-existing-c

# df.replace({'Symbol': '\.'}, '-', regex=True, inplace=True)
# def repl(x):
#     return x['symbol'].str.replace('.', '-', regex=True)
#
#
# df = df.assign(symbol=repl(df))

# to show localized time to user:
# now_eastern = now_utc.astimezone(pytz.timezone('US/Eastern'))
# https://vinta.ws/code/timezone-in-python-offset-naive-and-offset-aware-datetimes.html

# import tempfile
# def faster_read(query, db_connection):
#     # uses less RAM as it keeps the panda in a tmpfile
#     with tempfile.TemporaryFile() as tmpfile:
#         copy_sql = f"COPY ({query}) TO STDOUT WITH CSV HEADER"
#         db.cursor.copy_expert(copy_sql, tmpfile)
#         tmpfile.seek(0)
#         result_df = pd.read_csv(tmpfile)
#         return result_df

#
# for i in range(len(data.symbols)):
#   temp = pd.DataFrame.from_dict(data.tickers[i].info, orient="index")
#   temp.reset_index(inplace=True)
#   temp.columns = ['Attribute', 'Recent']
#   ticker_data[data.tickers[i].ticker] = temp


# measure performance
# from https://hakibenita.com/fast-load-data-python-postgresql
# Time
# >>> import time
# >>> start = time.perf_counter()
# >>> time.sleep(1) # do work
# >>> elapsed = time.perf_counter() - start
# >>> print(f'Time {elapsed:0.4}')
# memory usage
# python -m memory_profiler example.py
# from memory_profiler import memory_usage
# mem, retval = memory_usage((fn, args, kwargs), retval=True, interval=1e-7)
#
# All together
# import time
# from functools import wraps
# from memory_profiler import memory_usage
#
# def profile(fn):
#     @wraps(fn)
#     def inner(*args, **kwargs):
#         fn_kwargs_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
#         print(f'\n{fn.__name__}({fn_kwargs_str})')
#
#         # Measure time
#         t = time.perf_counter()
#         retval = fn(*args, **kwargs)
#         elapsed = time.perf_counter() - t
#         print(f'Time   {elapsed:0.4}')
#
# # Measure memory mem, retval = memory_usage((fn, args, kwargs),
# retval=True, timeout=200, interval=1e-7)
#
#         print(f'Memory {max(mem) - min(mem)}')
#         return retval
#
# return inner use as decorator @profile # FAST ONE
# def insert_execute_values_iterator(connection, beers: Iterator[Dict[str,
# Any]]) -> None: with connection.cursor() as cursor: create_staging_table(
# cursor) psycopg2.extras.execute_values(cursor, """ INSERT INTO
# staging_beers VALUES %s; """, (( beer['id'], beer['name'],
# beer['tagline'], parse_first_brewed(beer['first_brewed']),
# beer['description'], beer['image_url'], beer['abv'], beer['ibu'],
# beer['target_fg'], beer['target_og'], beer['ebc'], beer['srm'],
# beer['ph'], beer['attenuation_level'], beer['brewers_tips'],
# beer['contributed_by'], beer['volume']['value'], ) for beer in beers))


# https://www.pythonsheets.com/notes/python-asyncio.html

# Tables to html
# https://stackoverflow.com/questions/52644035/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table

# TA-Lib reference
# https://mrjbq7.github.io/ta-lib/

# Sraping xpath cheatsheet
# https://gist.github.com/LeCoupa/8c305ec8c713aad07b14

# multiprocessing and scraping https://likegeeks.com/downloading-files-using
# -python/

# class TurboSave:
#      def __init__(self, assets, price_history, fundamental):
#          self.assets = assets
#          self.fundamental = fundamental
#          self.price = price_history
#          asyncio.run(self.to_file())
#          asyncio.run(self.to_database())
#
#      async def to_file(self):
#          # self.assets.data.to_csv(S.SYMBOLS_FILE, index=False)
#          # self.fundamental.data.to_pickle(S.FUNDAMENTALS_FILE)
#          self.price.data.to_pickle(S.PRICE_FILE)
#
#      async def to_database(self):
#          def timestamp(table):
#              return f"""INSERT INTO last_update (tbl)
#                         VALUES ('{table}')
#                         ON CONFLICT (tbl)
#                         DO UPDATE SET date = NOW();"""
#
#          # separating each table ingestion into nested functions.
#          async def symbols(assets):
#              values = assets.to_records(index=False)
#              query = """INSERT INTO symbols (symbol, security, id)
#                         VALUES ($1, $2, $3)
#                         ON CONFLICT (symbol) DO UPDATE
#                         SET security=excluded.security;"""
#              async with pool.acquire() as db:
#                  async with db.transaction():
#                      await db.executemany(query, values)
#                      await db.execute(timestamp('symbols'))
#
#          async def fundamental(fndmntl):
#              def jsonize(x): return json.dumps(x)
#
#              fndmntl['yfinance'] = list(map(jsonize, fndmntl['yfinance']))
#              values = fndmntl.to_records(index=False)
#              query = """INSERT INTO fundamentals (id, yfinance)
#                         VALUES ((SELECT id FROM symbols WHERE symbol=$1),
#                                 $2)
#                         ON CONFLICT (id)
#                         DO UPDATE
#                         SET yfinance=excluded.yfinance;"""
#              async with pool.acquire() as db:
#                  async with db.transaction():
#                      await db.executemany(query, values)
#                      await db.execute(timestamp('fundamentals'))
#
#          async def price_history(price):
#              rows = price.itertuples(index=False)
#              values = [list(row) for row in rows]
#              query = """INSERT INTO price_history (date, symbol_id, open,
#                                                    high, low, close,
#                                                    adj_close, volume)
#                         VALUES ($1,
#                                 (SELECT id
#                                  FROM symbols
#                                  WHERE symbol=$2),
#                                  $3, $4, $5, $6, $7, $8)
#                         ON CONFLICT (symbol_id, date)
#                         DO UPDATE
#                         SET  symbol_id=excluded.symbol_id,
#                              date=excluded.date, open=excluded.open,
#                              high=excluded.high, low=excluded.low,
#                              close=excluded.close,
#                              adj_close=excluded.adj_close,
#                              volume=excluded.volume;"""
#              async with pool.acquire() as db:
#                  async with db.transaction():
#                      await db.executemany(query, values)
#                      await db.execute(timestamp('price_history'))
#
# # main database saving function try: async with asyncpg.create_pool(
# host=S.DB_HOST, database=S.DB_NAME, user=S.DB_USER,
# password=S.DB_PASSWORD) as pool: await symbols(self.assets.data) await
# asyncio.gather( fundamental(self.fundamental.data), price_history(
# self.price.data) ) except Exception as e: print(e) sys.exit()


# # function to collect data in a json format
# def jsonize(x): return json.dumps(x)
#
# # save fundamental data in a json in the column of the source
# self.data['yfinance'] = list(map(jsonize, self.data['yfinance']))


# values = df.to_records(index=False)
# values = [tuple(x) for x in signal_df.to_records(index=False)]

# df = df2 = pd.DataFrame(j)
# # df.loc['fundamental', 'ACN']['beta']
# df_unpack = pd.json_normalize(df.loc['fundamental'])
# df = df.T.join(df_unpack.set_index('symbol'))
#
# df_unpack = pd.json_normalize(df2.loc['fundamental'])
# df2 = df2.T.join(df_unpack.set_index('symbol'))
#
# # df3 = pd.concat([df, df2], keys=['TDA', 'Yahoo'], axis=1)
#
# df3 = df.merge(df2, right_index=True, left_index=True, how='outer',
# suffixes=('_TDA', '_YAHOO')) df3.columns[df3.columns.str.contains('_TDA')]
#
# https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#select-modifiers

# https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html
