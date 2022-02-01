"""This module implements Option analysis and recommendations"""

import pickle
from datetime import datetime, timedelta
from sys import exit
from time import time as t

import streamlit as st

from core.fy_tda import TDA
from core.utils import fix_path, Watchlist
import core.option_sniper.download as download
from core.option_sniper.modeling import Modeling
from core.option_sniper.strategies import OptionStrategy
from core.option_sniper.utility import market_is_open


class GetOptions:
    """Get options from TDA or local Pickle file.

    Download the options from TDA and save a local copy, or
    if the market is closed, loads options from the local file.

    :param param: Configuration dictionary.
    :param con: TDA client.
    :param wtc: List of symbols to analyze.
    """
    def __init__(self, param, con, wtc):
        self.options = None
        self.wtc = wtc
        self._tda = con
        self._param = param
        self.run()

    def run(self):
        self.filter_wtc_by_price()
        self.get_options_from_tda()
        pickle_name = fix_path(self._param['DEBUG']['data']) + 'options_df.pkl'
        is_local, sim_watchlist = self.load_options(pickle_name)
        self.save_options(is_local, pickle_name)
        self.export_option_results()

    def filter_wtc_by_price(self):
        price_min = self._param['FILTERS']['min_price']
        price_max = self._param['FILTERS']['max_price']
        last_price_list = download.get_last_price(self._tda, self.wtc)
        self.wtc = [i['stock'] for i in last_price_list
                    if price_min <= i['lastPrice'] <= price_max]

    def get_options_from_tda(self):
        """Downloads, saves/opens Pickle file &  assigns to Options table."""
        force_download = self._param['DEBUG']['force_download']

        if force_download and not market_is_open():
            st.warning("Market is closed. Download is enforced by settings.")

        if force_download or market_is_open():
            pass
        else:
            return

        now = datetime.now()
        from_date = now + timedelta(self._param['FILTERS']['min_dte'])
        to_date = now + timedelta(self._param['FILTERS']['max_dte'])
        st.write(f"Tables between {from_date:%b-%d-%Y} & {to_date:%b-%d-%Y}:")
        self.options = download.get_all_tables(
            tda_client=self._tda.client,
            symbol_list=self.wtc,
            min_dte=from_date,
            max_dte=to_date,
        )

    def load_options(self, name):
        if ((self.options is None or len(self.options.index) < 1) and
                not market_is_open()):
            try:
                st.write("\nMarket is closed. Attempting to load local file.")
                with open(name, 'rb') as file:
                    pkl = pickle.load(file)
                st.write("***** YOU WILL NOW WORK IN SIMULATION MODE *****")
                if pkl is not None and len(pkl.index) > 1:
                    self.options = pkl
                    # TODO: Check this and delete
                    # sim_watchlist = (Watchlist
                    #                  .sanitize(",".join(pkl.index.levels[0])))
                    sim_watchlist = Watchlist.sanitize(
                        ",".join(pkl['stock'].unique()))
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
                    market_is_open()):
                try:
                    with open(name, 'wb') as file:
                        pickle.dump(self.options,
                                    file,
                                    pickle.HIGHEST_PROTOCOL)
                    st.write("Market is open. Saving live chains.")
                except IOError as e:
                    st.warning(f"Error: Local file had trouble saving. {e}")

    def export_option_results(self):
        """Export table to CSV with Options analyzed to data path."""
        if not self._param['DEBUG']['export']:
            return

        if self.options is None:
            return

        if len(self.options.index) == 0:
            return

        suffix = datetime.now().strftime("%y%m%d_%H%M%S")
        name = f"{fix_path(self._param['DEBUG']['data'])}options_{suffix}.csv"
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
    def __init__(self, param: dict, opt):
        """Concatenates option chains per strategy.

        :param param: Parameter dictionary.
        :param opt: Option chain. Key are symbols and Value DataFrames.
        """
        assert opt is not None, "Option table is empty. (Option Analysis)"
        self.result = {}
        strategies = param['FILTERS']['strategies'].strip().split(',')
        strategies = [_.strip() for _ in strategies]
        options = OptionStrategy(opt, param['FILTERS'])
        for strategy in strategies:
            self.run_options(sel=strategy, opt=options, param=param)

    def run_options(self, sel: str, opt, param: dict):
        """Runs the corresponding Options analysis.

        :param sel: Strategy applied.
        :param param: Configuration dictionary.
        :param opt: options table
        """
        result = None

        if sel == 'naked':
            result = opt.naked()
        elif sel == 'spread':
            result = opt.spread()
        # elif sel == 'condor':
            # result = sniper_strategy.condor(opt, param['FILTERS'])
        if param['DEBUG']['export']:
            if result is not None and len(result.index) > 0:
                suffix = datetime.now().strftime("%y%m%d_%H%M%S")
                name = f"{fix_path(param['DEBUG']['data'])}{sel}_{suffix}.csv"
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


# Main program
def snipe(param: dict):
    p = Progress(number_of_functions=6)
    con = p.go("Connecting with TDA", TDA)
    wtc = p.go("Pulling watchlist", Watchlist.selected, param['WATCHLIST']['watchlist_current'])
    dwn = p.go("Getting option chains", GetOptions, param, con, wtc)
    ext = p.go("Adding studies to option table", Modeling, con, dwn)
    # p.go("Black-Scholes", ext.black_scholes)
    p.go("Probability of Profit", ext.probability_of_profits, montecarlo_iterations=5000)
    result = p.go("Analyzing strategies", OptionAnalysis, param, ext.options).result

    return result


if __name__ == '__main__':
    import sys
    sys.path.append("..")
    import fybot.core.settings as ss
    snipe(param=ss.OPTION_SNIPER)
