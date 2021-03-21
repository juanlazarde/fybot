"""Serve integrated data to UX server"""

import datetime
import os

import pandas as pd

import pytz

import logging

from core.database import Database
from core.settings import S
from core.snapshot import Snapshot
from core.filters import Filter

log = logging.getLogger()


class Scan:
    signals = None
    stocks = None

    def __init__(self, active_filter: dict):
        """Scan and present results

        :param active_filter: filters applied to the signals scan
        """
        self.settings = active_filter
        self.load_data()
        self.apply_filters()
        self.prep_arguments()

    def load_data(self):
        """Load symbols and signal table"""
        # load symbols
        try:
            symbols, within24 = Snapshot.GetAssets.load_database()
            if symbols.empty:
                raise Exception("Blank database")
            log.debug("Symbols table loaded from database")
        except Exception as e:
            log.debug(f"Couldn't read symbols from database. {e}")
            symbols = pd.read_csv(
                S.SYMBOLS_FILE, index_col='symbol', usecols=[0, 1])
            log.debug(f"Symbols table loaded from file: {S.SYMBOLS_FILE}")
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
