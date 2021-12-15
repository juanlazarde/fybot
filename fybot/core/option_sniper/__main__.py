"""This module implements Option analysis and recommendations"""

import pickle
from datetime import datetime, timedelta, date, time, timezone
from sys import exit
from time import time as t

import streamlit as st

from core.fy_tda import TDA
import core.option_sniper.strategy as sniper_strategy
from core.option_sniper.download import get_all_tables
from core.utils import fix_path, Watchlist, optimize_pd
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
        self.clean_options()
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
        force_download = cfg['DEBUG']['force_download']
        if self.market_is_open() or force_download:
            from_date = datetime.now() + timedelta(cfg['FILTERS']['min_dte'])
            to_date = datetime.now() + timedelta(cfg['FILTERS']['max_dte'])
            self.options = get_all_tables(
                tda_client=tda.client,
                symbol_list=wtc,
                min_dte=from_date,
                max_dte=to_date,
            )

    def clean_options(self):
        """
        Cleanup option chain and optimize DataFrame.

            1) Filter out columns,
            2) Filter out rows,
            3) Lower-range numerical and categoricals dtypes,
            4) Sparse Columns.
            5) Re-index.

        :return: Optimized dataframe to self.options
        """
        if self.options is None or self.options.empty:
            return

        opt = self.options.copy()
        # Measure it:
        # print(f"{sum(opt.memory_usage(deep=True))/1024:,.2f}Kb")
        opt.replace(["", " ", None, "NaN"], float('NaN'), inplace=True)
        opt.dropna(axis='columns', how='all', inplace=True)
        opt = optimize_pd(opt, deal_with_na='drop', verbose=False)
        opt['volatility'] = opt['volatility'] / 100.
        opt = opt[
            (opt['openInterest'] > 0) &
            (opt['totalVolume'] > 0) &
            (opt['bid'] > 0) &
            (opt['ask'] > 0) &
            (opt['volatility'].astype('float').round(8).values > 0)
        ]
        # opt.reset_index(drop=True, inplace=True)
        # print(f"{sum(opt.memory_usage(deep=True))/1024:,.2f}Kb")
        self.options = opt

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
    """Progress bar and Status update."""
    def __init__(self, number_of_functions: int):
        self.n = number_of_functions
        self.bar = st.progress(0)
        self.progress = 1
        self.message = ""
        self.message_container = st.empty()

    def go(self, msg, function, *args, **kwargs):
        self.message += msg
        self.message_container.info(self.message)
        s = t()
        result = function(*args, **kwargs)
        self.message += f" [{t() - s:.2f}s]. "
        self.message_container.info(self.message)
        self.bar.progress(self.progress / self.n)
        self.progress += 1
        if self.progress > self.n:
            self.bar.empty()
        return result


# import core.utils as ut
# @ut.lineprofile
def snipe(cfg: dict):
    p = Progress(number_of_functions=6)
    con = p.go("Connecting with TDA", TDA)
    wtc = p.go("Pulling watchlist", Watchlist.selected, cfg['WATCHLIST']['watchlist_current'])
    dwn = p.go("Getting option chains", GetOptions, cfg, con, wtc)
    ext = p.go("Adding studies to option table", Modeling, con, dwn)
    # p.go("Black Scholes", ext.black_scholes)
    p.go("Probability of Profit", ext.probability_of_profits, montecarlo_iterations=100)
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
