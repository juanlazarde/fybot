"""Stock scanner

Downloads list of stocks, filters and analyzes, and provides results on HTML
via Flask,

To run flask, open the python terminal and run: flask run"""

import csv
import datetime
import json
import os

import pandas as pd
import plotly as pt
import plotly.graph_objs as go
import talib
import yfinance as yf
from flask import Flask, request, render_template

from patterns import candlestick_patterns

# default settings, superseded from html form
settings = {'consolidating': {'go':False, 'pct': 5},
            'breakout': {'go': False, 'pct': 5},
            'ttm_squeeze': {'go': False},
            'candlestick': {'go': True},
            'sma_filter': {'go': False, 'fast': 25, 'slow': 50}}


class Const:
    # define files
    dataset_dir = 'datasets'
    symbols_file = 'symbols.csv'
    data_file = 'data.pkl'
    signals_file = 'signals.pkl'

    symbols_file = os.path.join(dataset_dir, symbols_file)
    data_file = os.path.join(dataset_dir, data_file)
    signals_file = os.path.join(dataset_dir, signals_file)

    # other constants
    download = 100


class Filter:
    def __init__(self):
        # Read data and optimize
        df = pd.read_pickle(Const.data_file)
        remove_cols = ['Volume', 'Adj Close']
        df = df.drop(remove_cols, axis=1, level=1)

        # Extract symbols in data
        self.symbols = list(df.columns.levels[0])
        self.symbols.sort()

        # Define filters to use and create table with of signals
        columns = [k for k in settings if settings[k]['go']]
        self.signal = pd.DataFrame(index=self.symbols, columns=columns)

        # Fill out signal table with different filters
        i = 0
        for symbol in self.symbols:
            # try:
            for column in columns:
                self.signal.loc[symbol, column] = \
                    getattr(self, column)(df=df[symbol], symbol=symbol)
            # Remove latest symbol to optimize for speed
            df = df.drop(symbol, axis=1, level=0)
            # except Exception as e:
            #    i += 1
            #    print('{} errors. {} symbol. {}'.format(i, symbol, e))

        # Save the signal table
        self.signal.to_pickle(Const.signals_file)

    @staticmethod
    def consolidating(df, pct=0, symbol='THE_DUMMY'):
        pct = pct if pct > 0 else settings['consolidating']['pct']
        recent_candlesticks = df[-15:]
        max_close = recent_candlesticks['Close'].max()
        min_close = recent_candlesticks['Close'].min()
        threshold = 1 - (pct / 100)
        if min_close > (max_close * threshold):
            return True
        return False

    def breakout(self, df, symbol='THE_DUMMY'):
        pct = settings['breakout']['pct']
        last_close = df[-1:]['Close'].values[0]
        if self.consolidating(df[:-1], pct):
            recent_closes = df[-16:-1]
            if last_close > recent_closes['Close'].max():
                return True
        return False

    @staticmethod
    def ttm_squeeze(df, symbol='THE_DUMMY'):
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

    def candlestick(self, df, symbol):
        """Based on hackthemarket"""
        bul, ber = [], []

        for pattern, description in candlestick_patterns.items():
            pattern_function = getattr(talib, pattern)
            results = pattern_function(
                df["Open"], df["High"], df["Low"], df["Close"])
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

    @staticmethod
    def sma_filter(df, symbol='THE_DUMMY'):
        sma_fast = talib.SMA(df['Close'],
                             timeperiod=settings['sma_filter']['fast'])
        sma_slow = talib.SMA(df['Close'],
                             timeperiod=settings['sma_filter']['slow'])

        if sma_fast[-1] > sma_slow[-1]:
            return True
        return False


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
        if not os.path.exists(Const.dataset_dir):
            os.makedirs(Const.dataset_dir)

        # load symbols, if not there download s&p500 from Wikipedia
        if os.path.isfile(Const.symbols_file):
            df = pd.read_csv(Const.symbols_file)
        else:
            df = self.download_sp500(Const.symbols_file)

        # download and save financial data for each symbol
        symbols = list(df['Symbol'])
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=Const.download)
        data = yf.download(symbols,
                           start=start_date,
                           end=end_date,
                           group_by="ticker")
        data = data.dropna(how='all')
        data = data.dropna(how='all', axis=1)
        data.to_pickle(Const.data_file)

        # Apply filters to data once and save in separate file
        Filter()

    @staticmethod
    def download_sp500(symbols_file):
        # scrape wikipedia for S&P500 list
        table = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        df = df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

        # replace '.' with '-' a Yahoo Finance issue
        # TODO: This works but gives a warning. Fix with the latest.
        df = df.assign(Symbol=lambda x: x['Symbol'].str.replace('.', '-'))

        df.to_csv(symbols_file, index=False)
        return df


class Index:
    signals = None
    stocks = None

    def __init__(self):
        self.load_data()
        self.apply_filters()
        self.prep_arguments()

    def load_data(self):
        # load company names and signal table
        self.signals = pd.read_pickle(Const.signals_file)

        df = pd.read_csv(Const.symbols_file, index_col='Symbol', usecols=[0,1])
        self.stocks = df.T.to_dict()

        # with open(Const.symbols_file) as f:
        #     self.stocks = {row[0]: {"company": row[1]}
        #                    for row in csv.reader(f)}

    def apply_filters(self):
        """Filter signal table with selections"""
        signals = self.signals

        # only show the filters wanted
        for k in settings.keys():
            if k in signals.columns:
                signals = signals[signals[k] == settings[k]['go']]

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
        if settings['candlestick']['go']:
            for symbol, row in self.signals.iterrows():
                stocks[symbol]['cdl_sum_ber'] = row.cdl_sum_ber
                stocks[symbol]['cdl_sum_bul'] = row.cdl_sum_bul

        self.stocks = stocks


# run in terminal 'flask run', auto-update with 'export FLASK_ENV=development'
app = Flask(__name__)


@app.route("/snapshot")
def snapshot_wrapper():
    Snapshot()
    return 'Success downloading data. Go back.'


@app.route("/filter")
def filter_wrapper():
    Filter()
    return "Success filtering. <a href='http://localhost:5000'>Go Back</a>"


@app.route("/")
def index():
    stocks = Index().stocks
    return render_template("index.html",
                           candlestick_patterns=candlestick_patterns,
                           stocks=stocks)


if __name__ == "__main__":
    app.run(debug=True)

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


# Tables to html
# https://stackoverflow.com/questions/52644035/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table

# TA-Lib reference
# https://mrjbq7.github.io/ta-lib/
