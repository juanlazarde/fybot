"""Stock scanner

Downloads list of stocks, filters and analyzes, and provides results on HTML
via Flask,

To run flask, open the python terminal and run: flask run"""

import datetime
import json
import os

import requests
from lxml import html

import pandas as pd
import plotly as pt
import plotly.graph_objs as go
import talib
import yfinance as yf
from flask import Flask, render_template, request

from patterns import candlestick_patterns


class S:
    # define files
    dataset_dir = 'datasets'
    symbols_file = 'symbols.csv'
    data_file = 'data.pkl'
    signals_file = 'signals.pkl'

    symbols_file = os.path.join(dataset_dir, symbols_file)
    data_file = os.path.join(dataset_dir, data_file)
    signals_file = os.path.join(dataset_dir, signals_file)

    # other constants
    download = 180


class Filter:
    def __init__(self, active_filter):
        self.settings = active_filter
        # Read data and optimize
        df = pd.read_pickle(S.data_file)
        df = df.drop(['Volume', 'Adj Close'], axis=1, level=1)

        # Extract symbols in data
        self.symbols = list(df.columns.levels[0])
        self.symbols.sort()

        # Define filters to use and create table with signals
        columns = [k for k in self.settings if self.settings[k]['go']]
        self.signal = pd.DataFrame(index=self.symbols, columns=columns)

        # Fill out signal table with different filters
        for symbol in self.symbols:
            for column in columns:
                self.signal.loc[symbol, column] = \
                    getattr(self, column)(df=df[symbol], symbol=symbol)
            # Remove latest symbol to optimize for speed
            df = df.drop(symbol, axis=1, level=0)

        # Save the signal table
        self.signal.to_pickle(S.signals_file)

    def consolidating(self, **kwargs):
        df = kwargs['df']
        pct = kwargs['pct'] if 'pct' in kwargs.keys() else 0
        pct = pct if pct > 0 else self.settings['consolidating']['pct']
        recent_candlesticks = df[-15:]
        max_close = recent_candlesticks['Close'].max()
        min_close = recent_candlesticks['Close'].min()
        threshold = 1 - (pct / 100)
        if min_close > (max_close * threshold):
            return True
        return False

    def breakout(self, **kwargs):
        df = kwargs['df']
        pct = self.settings['breakout']['pct']
        last_close = df[-1:]['Close'].values[0]
        if self.consolidating(df=df[:-1], pct=pct):
            recent_closes = df[-16:-1]
            if last_close > recent_closes['Close'].max():
                return True
        return False

    @staticmethod
    def ttm_squeeze(**kwargs):
        df = kwargs['df']
        sq = pd.DataFrame()
        sq['20sma'] = df['Close'].rolling(window=20).mean()
        sq['stddev'] = df['Close'].rolling(window=20).std()
        sq['lower_band'] = sq['20sma'] - (2 * sq['stddev'])
        sq['upper_band'] = sq['20sma'] + (2 * sq['stddev'])
        sq['TR'] = abs(df['High'] - df['Low'])
        sq['ATR'] = sq['TR'].rolling(window=20).mean()
        sq['lower_keltner'] = sq['20sma'] - (sq['ATR'] * 1.5)
        sq['upper_keltner'] = sq['20sma'] + (sq['ATR'] * 1.5)

        def in_squeeze(_):
            return _['lower_band'] > _['lower_keltner'] and \
                   _['upper_band'] < _['upper_keltner']

        sq['squeeze_on'] = sq.apply(in_squeeze, axis=1)

        if sq.iloc[-3]['squeeze_on'] and not sq.iloc[-1]['squeeze_on']:
            # is coming out the squeeze
            return True
        return False
    
    @staticmethod
    def in_the_squeeze(**kwargs):
        df = kwargs['df']
        sq = pd.DataFrame()
        sq['20sma'] = df['Close'].rolling(window=20).mean()
        sq['stddev'] = df['Close'].rolling(window=20).std()
        sq['lower_band'] = sq['20sma'] - (2 * sq['stddev'])
        sq['upper_band'] = sq['20sma'] + (2 * sq['stddev'])
        sq['TR'] = abs(df['High'] - df['Low'])
        sq['ATR'] = sq['TR'].rolling(window=20).mean()
        sq['lower_keltner'] = sq['20sma'] - (sq['ATR'] * 1.5)
        sq['upper_keltner'] = sq['20sma'] + (sq['ATR'] * 1.5)

        def in_squeeze(_):
            return _['lower_band'] > _['lower_keltner'] and \
                   _['upper_band'] < _['upper_keltner']

        sq['squeeze_on'] = sq.apply(in_squeeze, axis=1)

        if sq.iloc[-1]['squeeze_on']:
            # is coming out the squeeze
            return True
        return False

    def ema_stacked(self, **kwargs):
        df = kwargs['df']
        ema = pd.DataFrame()
        ema_list = [8,21,34,55,89]
        for period in ema_list:
            col = '{}{}'.format('ema', period)
            ema[col] = talib.EMA(df['Close'], timeperiod=period)

        if  (ema['ema8'].iloc[-1] > ema['ema21'].iloc[-1]) and \
            (ema['ema21'].iloc[-1] > ema['ema34'].iloc[-1]) and \
            (ema['ema34'].iloc[-1] > ema['ema55'].iloc[-1]) and \
            (ema['ema55'].iloc[-1] > ema['ema89'].iloc[-1]) and \
            (df['Close'].iloc[-1] > ema['ema21'].iloc[-1]):
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
                df['Open'], df['High'], df['Low'], df['Close'])
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
        sma_fast = talib.SMA(df['Close'],
                             timeperiod=self.settings['sma_filter']['fast'])
        sma_slow = talib.SMA(df['Close'],
                             timeperiod=self.settings['sma_filter']['slow'])

        return True if sma_fast[-1] > sma_slow[-1] else False

    def investor_reco(self, **kwargs):
        # scrape finviz for latest 3 investor status
        symbol = kwargs['symbol']
        site = 'https://www.finviz.com/quote.ashx?t={}'.format(symbol)
        headers = {'User-Agent': 'Mozilla/5.0'}
        search = '//table[@class="fullview-ratings-outer"]//tr[1]//text()'

        screen = requests.get(site, headers=headers)
        tree = html.fromstring(screen.content).xpath(search)

        bull_words = ['Buy', 'Outperform', 'Overweight', 'Upgrade']
        bear_words = ['Sell', 'Underperform', 'Underweight', 'Downgrade']
        bull = sum([tree.count(x) for x in bull_words])
        bear = sum([tree.count(x) for x in bear_words])

        self.signal.loc[symbol, 'investor_sum'] = \
            "Bull {}, Bear {}".format(bull, bear)
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


class Snapshot:
    def __init__(self):
        """Save table with symbol financial data from Yahoo Finance"""

        # create dataset directory if not there
        if not os.path.exists(S.dataset_dir):
            os.makedirs(S.dataset_dir)

        # load symbols, if not there download s&p500
        if os.path.isfile(S.symbols_file):
            df = pd.read_csv(S.symbols_file)
        else:
            df = self.download_sp500(S.symbols_file)

        # download and save financial data for each symbol
        symbols = list(df['Symbol'])
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=S.download)
        data = yf.download(symbols,
                           start=start_date,
                           end=end_date,
                           group_by="ticker")
        data = data.dropna(how='all')
        data = data.dropna(how='all', axis=1)
        data.to_pickle(S.data_file)

    @staticmethod
    def download_sp500(symbols_file):
        # scrape wikipedia for S&P500 list
        table = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        df = df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
        # replace '.' with '-' a Yahoo Finance issue
        df = df.assign(
            Symbol=lambda x: x['Symbol'].str.replace('.', '-', regex=True))
        df.to_csv(symbols_file, index=False)
        return df


class Index:
    signals = None
    stocks = None

    def __init__(self, active_filter):
        self.settings = active_filter
        self.load_data()
        self.apply_filters()
        self.prep_arguments()

    def load_data(self):
        # load company names and signal table
        self.signals = pd.read_pickle(S.signals_file)

        df = pd.read_csv(S.symbols_file, index_col='Symbol', usecols=[0, 1])
        self.stocks = df.T.to_dict()

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


# run in terminal 'flask run', auto-update with 'export FLASK_ENV=development'
app = Flask(__name__)


@app.route("/snapshot")
def snapshot_wrapper():
    Snapshot()
    return "Success downloading data. " \
           "<a href='http://localhost:5000'>Go Back</a>"


@app.route("/", methods=['GET', 'POST'])
def index():
    stocks = {}
    settings = ''

    # DEFAULTS
    active_filters = {'consolidating':
                          {'go': True,
                           'pct': 7},

                      'breakout':
                          {'go': False,
                           'pct': 2.5},

                      'ttm_squeeze':
                          {'go': False},

                      'in_the_squeeze':
                          {'go': False},

                      'candlestick':
                          {'go': True},

                      'sma_filter':
                          {'go': False,
                           'fast': 25,
                           'slow': 50},

                      'ema_stacked':
                          {'go': False},

                      'investor_reco':
                          {'go': False}
                      }

    modified_time = os.stat(S.data_file).st_mtime
    last_modified = datetime.datetime.fromtimestamp(modified_time)
    last_modified_fmt = last_modified.strftime('%d-%m-%Y %H:%M %p')

    if request.method == 'GET':
        if os.path.isfile(S.data_file):
            now = datetime.datetime.now()
            last24 = datetime.timedelta(hours=24)
            if now - last_modified > last24:
                Snapshot()

    if request.method == 'POST':
        # read arguments from form
        filters_selected = request.form.getlist('filter')
        if len(filters_selected) == 0:
            return "Please go back and make a selection. " \
                   "<a href='http://localhost:5000'>Go Back</a>"

        # shorten functions
        def fltr_me(x): return x in filters_selected
        frm_get = request.form.get

        active_filters = {'consolidating':
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

        Filter(active_filters)
        stocks = Index(active_filters).stocks

    return render_template("index.html",
                           stocks=stocks,
                           data_last_modified=last_modified_fmt,
                           active_filters=active_filters,
                           settings=settings)


if __name__ == "__main__":
    app.run(debug=False)

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


# Tables to html
# https://stackoverflow.com/questions/52644035/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table

# TA-Lib reference
# https://mrjbq7.github.io/ta-lib/

# Sraping xpath cheatsheet
# https://gist.github.com/LeCoupa/8c305ec8c713aad07b14
