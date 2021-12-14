"""This module implements Option analysis and recommendations"""

import pickle
from datetime import datetime, timedelta, date, time, timezone
from sys import exit
from time import time as t

import streamlit as st

from core.fy_tda import TDA
import core.option_sniper.strategy as sniper_strategy
from core.option_sniper.download import get_all_tables
from core.utils import fix_path, Watchlist
from core.option_sniper.modeling import Modeling


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
        current_time = (datetime
                        .now()
                        .astimezone(timezone(timedelta(hours=-5)))
                        .time())

        return ((current_weekday < 5) and
                (current_time >= time(9, 30)) and
                (current_time <= time(16, 0)))

    def get_options_from_tda(self, cfg, tda, wtc):
        """Downloads, saves/opens Pickle file &  assigns to Options table."""
        min_strike_date = datetime.now() + timedelta(cfg['FILTERS']['min_dte'])
        max_strike_date = datetime.now() + timedelta(cfg['FILTERS']['max_dte'])
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
                    sim_watchlist = (Watchlist
                                     .sanitize(",".join(pkl.index.levels[0])))
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
                        pickle.dump(self.options,
                                    file,
                                    pickle.HIGHEST_PROTOCOL)
                    st.write("Market is open. Saving live chains.")
                except IOError as e:
                    st.warning(f"Error: Local file had trouble saving. {e}")

    def export_option_results(self, cfg):
        """Export table to CSV with Options analyzed to data path."""
        if cfg['DEBUG']['export']:
            if self.options is not None and len(self.options.index) > 0:
                suffix = datetime.now().strftime("%y%m%d_%H%M%S")
                name = f"{fix_path(cfg['DEBUG']['data'])}options_{suffix}.csv"
                try:
                    self.options.to_csv(name, index=True, header=True)
                    st.write("Table saved at: {}".format(name))
                except IOError as e:
                    st.error(f"Error attempting to save table. Error: {e}")


class OptionAnalysis:
    """Analyzes options using filters from config and shows tables.
    If you want to skip an analysis use the config file or the
    debug_settings
    """
    def __init__(self, cfg: dict, opt):
        """Concatenates option chains per strategy.

        :param cfg: Configuration dictionary.
        :param opt: Option chain. Key are symbols and Value DataFrames.
        """
        assert opt is not None, "Option table is empty. (Option Analysis)"

        self.result = {}

        strategies = cfg['FILTERS']['strategies'].strip().split(',')
        strategies = [_.strip() for _ in strategies]

        for strategy in strategies:
            try:
                self.run_options(sel=strategy, opt=opt, cfg=cfg)
            except Exception as e:
                st.error("Error while running analysis. " + str(e))
                exit(1)

    def run_options(self, sel: str, opt, cfg: dict):
        """Runs the corresponding options analysis.

        :param sel: Strategy applied.
        :param cfg: Configuration dictionary.
        :param opt: options table
        """
        result = None
        if sel == 'naked':
            result = sniper_strategy.naked(opt, cfg['FILTERS'])
        elif sel == 'spread':
            result = sniper_strategy.spread(opt, cfg['FILTERS'])
        elif sel == 'condor':
            result = sniper_strategy.condor(opt, cfg['FILTERS'])
        if cfg['DEBUG']['export']:
            if result is not None and len(result.index) > 0:
                suffix = datetime.now().strftime("%y%m%d_%H%M%S")
                name = f"{fix_path(cfg['DEBUG']['data'])}{sel}_{suffix}.csv"
                try:
                    result.to_csv(name, index=True, header=True)
                    st.success("Table saved at: {}".format(name))
                except IOError as e:
                    st.error("Error saving table. Error: " + str(e))
            else:
                raise st.warning("To export the options table needs data.")

        self.result = {sel: result}


class Progress:
    """
    Progress bar and Status update.
    """
    def __init__(self, n_functions):
        self.n = n_functions
        self.my_bar = st.progress(0)
        self.progress = 1
        self.msg = ""
        self.container = st.empty()

    def go(self, msg, func, *args, **kwargs):
        self.msg += msg
        self.container.info(self.msg)
        s = t()
        result = func(*args, **kwargs)
        self.msg += f" [{t() - s:.3f}sec]. "
        self.container.info(self.msg)
        self.my_bar.progress(self.progress / self.n)
        self.progress += 1
        return result


# import core.utils as ut
# @ut.lineprofile
def snipe(cfg: dict):
    p = Progress(6)
    con = p.go("Establishing connection with TDA API", TDA)
    wtc = p.go("Pulling watchlist", Watchlist.selected, cfg['WATCHLIST']['watchlist_current'])
    dwn = p.go("Getting option tables", GetOptions, cfg, con, wtc)
    ext = p.go("Adding studies to option table", Modeling, con, dwn)
    p.go("Probability of Profit (Monte Carlo simulations)", ext.probability_of_profits, montecarlo_iterations=100)
    result = p.go("Analyzing strategies", OptionAnalysis, cfg, ext.options).result

    # con = TDA()
    # wtc = Watchlist.selected(cfg['WATCHLIST']['watchlist_current'])
    # dwn = GetOptions(cfg, con, wtc)
    # ext = Modeling(con, dwn)
    # # ext.black_scholes()
    # ext.probability_of_profits(montecarlo_iterations=100)
    # result = OptionAnalysis(cfg, ext.options).result

    return result


if __name__ == '__main__':
    import sys
    sys.path.append("..")
    import fybot.core.settings as ss
    snipe(cfg=ss.OPTION_SNIPER)
