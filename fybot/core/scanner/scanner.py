"""Serve integrated data to UX server"""

import datetime
import os

import pandas as pd
import numpy as np

import pytz

import logging

from core.database import Database
import core.settings as ss
import core.scanner.filters as fl
import core.scanner as sn

log = logging.getLogger()


class Scan:
    def __init__(self, filters: dict):
        """Scan and present results

        :param filters: filters applied to the scan
        """
        self.active_filters, self.info_filters = self.process_filters(filters)
        self.signals = self.apply_filters(fl.Signals.load_database())

    @staticmethod
    def process_filters(filters: dict):
        active_filters = [k for k in filters if filters[k]['go']]
        info_filters = []

        if 'candlestick' in active_filters:
            info_filters += ['cdl_sum_ber', 'cdl_sum_bul']

        if 'investor_reco' in active_filters:
            info_filters += ['investor_sum']

        return active_filters, info_filters

    def apply_filters(self, signals_in):

        all_filters = self.active_filters + self.info_filters
        signals = signals_in[all_filters]
        signals = signals.replace(False, np.NaN)
        signals = signals.dropna(axis=0, how='any')

        return signals

    def merge_symbol_name(self):

        symbol_name = sn.GetAssets.load_database()
        symbol_name = symbol_name.set_index(['symbol'])

        self.signals.index.name = 'symbol'
        # Consider this optimized Merge.
        # def optimized_merge(df1, df2, merge_column):
        #     df2 = df2[df2[merge_column].isin(df1[merge_column])]
        #     return df1.merge(df2, on=merge_column)
        self.signals = symbol_name.join(self.signals, on='symbol', how='inner')

        return self.signals

    @staticmethod
    def last_update(table):
        """last update date"""

        now_utc = pytz.utc.localize(datetime.datetime.utcnow())
        timestamp = now_utc - datetime.timedelta(hours=25)

        query = f"SELECT date FROM last_update WHERE tbl = '{table}';"
        with Database() as db:
            timestamp_db = db.query(query)[0]

        timestamp = timestamp if timestamp_db is None else timestamp_db
        within_24 = (now_utc - timestamp).seconds <= 24 * 60 * 60
        timestamp_est = timestamp.astimezone(pytz.timezone('US/Eastern'))
        return timestamp_est.strftime('%m-%d-%Y %I:%M %p %Z'), within_24


class File(Scan):
    def load(self):
        self.signals = pd.read_pickle(ss.SIGNALS_FILE)
        # log.debug(f"Signal tables loaded from file: {ss.SIGNALS_FILE}")
        return self.signals

    @staticmethod
    def get_timestap():
        if os.path.isfile(ss.PRICE_FILE):
            return os.stat(ss.PRICE_FILE).st_mtime
