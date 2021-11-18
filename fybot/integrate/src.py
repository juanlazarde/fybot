"""Serve integrated data to UX server"""

import datetime
import os

import pandas as pd

import pytz

import logging

from core.database import Database
from core.settings import S
from core.snp.assets import GetAssets
from core.filters import Signals

log = logging.getLogger()


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
            symbols = GetAssets().symbols
            if symbols.empty:
                raise Exception("Blank database")
        except Exception as e:
            print(e)
            symbols = pd.read_csv(
                S.SYMBOLS_FILE, index_col='symbol', usecols=[0, 1])
        # _wtc.stocks = symbols.T.to_dict()
        self.stocks = symbols.set_index('symbol').T.to_dict('dict')

        # load signal table
        try:
            self.signals = Signals.load_database()
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
# app = Flask(__name__)
# app.config['SECRET_KEY'] = S.FLASK_SECRET_KEY
#
#
# @app.route("/stock/<symbol>")
# def stock_detail(symbol):
#     try:
#         symbols, within24 = Snapshot.GetAssets.load_database()
#         if symbols.empty:
#             raise Exception("Blank database")
#     except Exception as e:
#         print(e)
#         symbols = pd.read_csv(
#             S.SYMBOLS_FILE, index_col='symbol', usecols=[0, 1])
#     # _wtc.stocks = symbols.T.to_dict()
#     stocks = symbols.set_index('symbol').T.to_dict('dict')
#
#     # df = pd.read_csv(S.SYMBOLS_FILE, index_col='symbol', usecols=[0, 1])
#     # stocks = df.T.to_dict()
#     security = stocks[symbol]['security']
#
#     try:
#         data = Snapshot.GetPrice.load_database()
#         if data.empty:
#             raise Exception("Blank database")
#     except Exception as e:
#         print(e)
#         data = pd.read_pickle(S.PRICE_FILE)
#     bars_df = data.loc[(data['symbol'] == symbol)]
#     bars_df = bars_df.drop(['symbol'], axis=1)
#     # df.reset_index(drop=True, inplace=True)
#     #    bars_df = data[symbol]
#     # bars_df = bars_df.reset_index()
#     # data = data.set_index('symbol').T.to_dict('dict')
#     bars_df = bars_df.sort_values(by=['date'], ascending=False)
#     bars = bars_df.to_dict('records')
#     stock = {'symbol': symbol,
#              'security': security,
#              'bars': bars}
#
#     return render_template("stock_detail.html",
#                            stock=stock)
#
#
# def index():
#     """Renders index.html page"""
#
#     # Defaults and variable initialization
#     active_filters = S.DEFAULT_FILTERS
#     data = {}
#     settings = ''
##
#     # apply filters by pressing 'Submit' button
#     if request.method == 'POST':
#         # read current options from index form under name='filter'
#         filters_selected = request.form.getlist('filter')
#         if len(filters_selected) == 0:
#             flash("Please make a selection")
#         else:
#             s = t()
#
#             # use only the filters available as options
#             def fltr_me(x):
#                 return x in filters_selected
#
#             frm_get = request.form.get
#             active_filters = {
#                 'consolidating':
#                     {'go': fltr_me('consolidating'),
#                      'pct': float(frm_get('consolidating_pct'))},
#
#                 'breakout':
#                     {'go': fltr_me('breakout'),
#                      'pct': float(frm_get('breakout_pct'))},
#
#                 'ttm_squeeze':
#                     {'go': fltr_me('ttm_squeeze')},
#
#                 'in_the_squeeze':
#                     {'go': fltr_me('in_the_squeeze')},
#
#                 'candlestick':
#                     {'go': fltr_me('candlestick')},
#
#                 'sma_filter':
#                     {'go': fltr_me('sma_filter'),
#                      'fast': float(frm_get('sma_fast')),
#                      'slow': float(frm_get('sma_slow'))},
#
#                 'ema_stacked':
#                     {'go': fltr_me('ema_stacked')},
#
#                 'investor_reco':
#                     {'go': fltr_me('investor_reco')}
#             }
#
#             # run filter or signal function and save result to database
#             Filter(active_filters)
#
#             # prepare information to be presented to html
#             data = Index(active_filters).stocks
#
#             # message to front
#             flash(f"Done thinking. It took me {t() - s:0.2} seconds.")
#
#     # get date of last price_history from file or database
#     price_update_dt, within24 = Index.last_update('price_history')
#
#     return render_template("index.html",
#                            stocks=data,
#                            data_last_modified=price_update_dt,
#                            active_filters=active_filters,
#                            settings=settings)


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

# if os.path.isfile(_wtc.symbols):
#     with open(_wtc.symbols, mode='r') as f:
#         companies = {rows[0]: rows[1] for rows in csv.reader(f)}
# else:
#     companies = _wtc.download_sp500()

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
# _wtc.stocks = df.to_dict()

# #        # replace '.' with '-' a Yahoo Finance issue
#         df = df.assign(
#             Symbol=lambda x: x['Symbol'].str.replace('.', '-', regex=True))

# with open(Const.symbols_file) as f:
#     _wtc.stocks = {row[0]: {"company": row[1]}
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
#         _wtc.signal.loc[symbol, 'investor_sum'] = \
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
#      def __init__(_wtc, assets, price_history, fundamental):
#          _wtc.assets = assets
#          _wtc.fundamental = fundamental
#          _wtc.price = price_history
#          asyncio.run(_wtc.to_file())
#          asyncio.run(_wtc.to_database())
#
#      async def to_file(_wtc):
#          # _wtc.assets.data.to_csv(S.SYMBOLS_FILE, index=False)
#          # _wtc.fundamental.data.to_pickle(S.FUNDAMENTALS_FILE)
#          _wtc.price.data.to_pickle(S.PRICE_FILE)
#
#      async def to_database(_wtc):
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
# password=S.DB_PASSWORD) as pool: await symbols(_wtc.assets.data) await
# asyncio.gather( fundamental(_wtc.fundamental.data), price_history(
# _wtc.price.data) ) except Exception as e: print(e) sys.exit()


# # function to collect data in a json format
# def jsonize(x): return json.dumps(x)
#
# # save fundamental data in a json in the column of the source
# _wtc.data['yfinance'] = list(map(jsonize, _wtc.data['yfinance']))


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
