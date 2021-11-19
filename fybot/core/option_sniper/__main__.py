"""This module implements Option analysis and recommendations"""

import pickle
from datetime import datetime, timedelta, date, time, timezone
from sys import exit

import streamlit as st

from core.utils import fix_path
from core.fy_tda import TDA
import core.option_sniper.strategy as sniper_strategy
from core.option_sniper.download import get_all_tables
from core.utils import Watchlist


class GetOptions:
    """Get options from TDA or local Pickle file.

    Download the options from TDA and save a local copy, or
    if the market is closed, lods options from the local file.

    :param cfg: Configuration dictionary.
    :param tda: TDA client.
    :param wtc: List of symbols to analyze.
    """
    options = None

    def __init__(self, cfg, tda, wtc):
        self.get_options_from_tda(cfg, tda, wtc)
        pickle_name = fix_path(cfg['DEBUG']['data']) + 'options_df.pkl'
        is_local, sim_watchlist = self.load_options(pickle_name)
        self.save_options(is_local, pickle_name)
        self.export_option_results(cfg)

    @staticmethod
    def market_is_open():
        """Return True or False whether the market is open or not."""
        current_weekday = date.today().weekday()
        current_time = datetime.now().astimezone(timezone(timedelta(hours=-5))).time()

        return ((current_weekday < 5) and
                (current_time >= time(9, 30)) and
                (current_time <= time(16, 0)))

    def get_options_from_tda(self, cfg, tda, wtc):
        """Downloads, saves/opens Pickle file &  assigns to Options table."""
        min_strike_date = datetime.now() + timedelta(days=cfg['FILTERS']['min_dte'])
        max_strike_date = datetime.now() + timedelta(days=cfg['FILTERS']['max_dte'])
        force_download = cfg['DEBUG']['force_download']
        if self.market_is_open() or force_download:
            self.options = get_all_tables(
                tda_client=tda.client,
                symbol_list=wtc,
                min_dte=min_strike_date,
                max_dte=max_strike_date,
            )

    def load_options(self, name):
        if ((self.options is None or len(self.options.index) < 1) and
                not self.market_is_open()):
            try:
                st.write("\nMarket is closed. Attempting to load local file.")
                with open(name, 'rb') as file:
                    pkl = pickle.load(file)
                st.write("***** YOU WILL NOW WORK IN SIMULATION MODE *****")
                if pkl is not None and len(pkl.index) > 1:
                    self.options = pkl
                    sim_watchlist = Watchlist.sanitize(",".join(pkl.index.levels[0]))
                    st.write("Available symbols: " + ", ".join(sim_watchlist))
                    return True, sim_watchlist
                else:
                    st.warning("Local options table is empty.")
                    exit(1)
            except IOError as e:
                st.error("Error loading local file. Error: " + str(e))
                exit(1)
        return False, None

    def save_options(self, is_local, name):
        if not is_local:
            if ((self.options is not None and len(self.options.index) > 0) and
                    self.market_is_open()):
                try:
                    with open(name, 'wb') as file:
                        pickle.dump(self.options, file, pickle.HIGHEST_PROTOCOL)
                    st.write("Market is open. Saving live chains.")
                except IOError as e:
                    st.warning("Error: Local file had trouble saving. " + str(e))

    def export_option_results(self, cfg):
        """Export table to CSV with Options analyzed to data path."""
        if cfg['DEBUG']['export']:
            if self.options is not None and len(self.options.index) > 0:
                suffix = datetime.now().strftime("%y%m%d_%H%M%S")
                ext = ".csv"
                name = fix_path(cfg['DEBUG']['data']) + "options_" + suffix + ext
                try:
                    self.options.to_csv(name, index=True, header=True)
                    st.write("Table saved at: {}".format(name))
                except IOError as e:
                    st.error("Error attempting to save table. Error: " + str(e))


class OptionAnalysis:
    """Analyzes options using filters from config and shows tables.
    If you want to skip an analysis use the config file or the
    debug_settings
    """
    df_out = {}

    def __init__(self, cfg: dict, option_df):
        """Concatenates option chains per strategy.

        :param cfg: Configuration dictionary.
        :param option_df: Options where key is the symbol and value is the datafame.
        """
        if option_df.options is not None:
            strategies = cfg['FILTERS']['strategies'].strip().split(',')
            strategies = [_.strip() for _ in strategies]

            for strategy in strategies:
                try:
                    self.run_options(
                        selected=strategy,
                        opt=option_df,
                        cfg=cfg
                    )
                except Exception as e:
                    st.error("Error while running analysis. " + str(e))
                    exit(1)

    def run_options(self,
                    selected: str,
                    opt,
                    cfg: dict):
        """Runs the corresponding options analysis.

        :param selected: Strategy applied.
        :param cfg: Configuration dictionary.
        :param opt: options table
        """
        result = None
        if selected == 'naked':
            result = sniper_strategy.naked(opt.options, filters=cfg['FILTERS'])
        elif selected == 'spread':
            result = sniper_strategy.spread(opt.options, filters=cfg['FILTERS'])
        elif selected == 'condor':
            result = sniper_strategy.condor(opt.options, filters=cfg['FILTERS'])
        if cfg['DEBUG']['export']:
            if result is not None and len(result.index) > 0:
                suffix = datetime.now().strftime("%y%m%d_%H%M%S")
                name = f"{fix_path(cfg['DEBUG']['data'])}{selected}_hacker_{suffix}.csv"
                try:
                    result.to_csv(name, index=True, header=True)
                    st.success("Table saved at: {}".format(name))
                except IOError as e:
                    st.error("Error saving table. Error: " + str(e))
            else:
                raise st.warning("To export the options table needs data.")

        self.df_out = {selected: result}


def snipe(cfg: dict):
    con = TDA()
    wtc = Watchlist.selected(cfg['WATCHLIST']['watchlist_current'])
    opt = GetOptions(cfg, con, wtc)
    result = OptionAnalysis(cfg, opt).df_out

    return result


if __name__ == '__main__':
    from core.settings import S
    snipe(cfg=S.OPTION_SNIPER)
