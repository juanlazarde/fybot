from typing import Dict, Any, List
from calendar import month_abbr

import pandas as pd
import numpy as np
import streamlit as st
from scipy.spatial.distance import cdist

import core.formatter as fm


# from core.utils import line_profile


# TODO: Make Debit/credit premium type work
# TODO: Check on ITM treatment

class OptionStrategy:
    def __init__(self, df_in: pd.DataFrame, filters: dict):
        """
        Analyze options and parameters and returns DataFrame with strategy.
        It will also 'print' the table to Streamlit, automatically.

        Returns:
            - options: Resulting options table.

        :param df_in: Option chain table.
        :param filters: Filters for the options to be shown.
        :return: Options table dataframe.
        """
        self.options: pd.DataFrame = df_in.copy()
        self._params: Dict[str, Any] = {
            'premium_type': filters['premium_type'].lower().strip(),
            'option_itm': False,
            'option_type': filters['option_type'].lower().strip(),
            'margin_requirement': float(filters['margin_requirement']),
            'max_risk': float(filters['max_risk']),
            'min_return_pct': float(filters['min_return_pct']) / 100,
            'max_dte': int(filters['max_dte']),
            'min_dte': int(filters['min_dte']),
            'max_delta': float(filters['max_delta']),
            'min_pop': float(filters['min_pop']) / 100,
            'min_p50': float(filters['min_p50']) / 100,
            'min_open_int_pcl': float(filters['min_open_int_pcl']) / 100,
            'min_volume_pcl': float(filters['min_volume_pcl']) / 100,
            'max_bid_ask_pcl': float(filters['max_bid_ask_pcl']) / 100
        }
        self._shp = self.options.shape[0]
        self._impact: Dict[str, str] = {}  # Dictionary with filter name & value.
        self._remove_unwanted_columns()
        self._common_preparation()

    def _remove_unwanted_columns(self):
        # Clean table.
        df = self.options.copy()
        # Comment out columns to keep.
        df.drop(inplace=True, columns=[
            # 'description',
            'exchangeName',
            #  'bid',
            #  'ask',
            'last',
            #  'mark',
            'bidSize',
            'askSize',
            'bidAskSize',
            'lastSize',
            'highPrice',
            'lowPrice',
            'openPrice',
            'closePrice',
            #  'totalVolume',
            'tradeTimeInLong',
            'quoteTimeInLong',
            'netChange',
            #  'volatility',
            #  'delta',
            'gamma',
            'theta',
            'vega',
            'rho',
            #  'openInterest',
            'timeValue',
            'theoreticalOptionValue',
            'theoreticalVolatility',
            #  'strikePrice',
            'expirationDate',
            # 'daysToExpiration',
            'expirationType',
            'lastTradingDay',
            #  'multiplier',
            'percentChange',
            'markChange',
            'markPercentChange',
            'intrinsicValue',
            #  'inTheMoney',
            'mini',
            'nonStandard',
            'pennyPilot',
            #  'lastPrice',
            #  'option_value_mc',
            #  'probability_ITM',
            #  'probability_of_50',
        ])
        # TODO: Delete these lines
        # df.reset_index(inplace=True)
        # df.sort_values(by='symbol', ascending=True, inplace=True)
        # df.drop(columns=['symbol'], inplace=True)  # Actual option's symbol.

        # Return dataframe.
        self.options = df

    def _common_preparation(self):
        """Common columns and filters."""
        # Variable assignment.
        p = self._params
        df = self.options.copy()

        # Filter: Pass 1
        # Here we're making decisions on what options we want.
        # i.e. Volume and Open Interest won't be 0.
        df = df[(df['openInterest'] > 0)
                & (df['totalVolume'] > 0)
                & (df['bid'] > 0)
                & (df['ask'] > 0)
                ]
        self._impact['Min Liquidity (O.I., V, B, A > 0)'] = \
            f"{df.shape[0] / self._shp:.0%}"

        if p['option_type'] in ['put', 'call']:
            df = df[df['option_type'] == p['option_type']]
        self._impact['Option type'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[df['inTheMoney'] == p['option_itm']]
        self._impact['ITM'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[(df['delta'] <= p['max_delta']) & (df['delta'] >= -p['max_delta'])]
        self._impact['Delta'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[df['probability_ITM'] >= p['min_pop']]
        self._impact['Prob of Profit (ITM)'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[df['probability_of_50'] >= p['min_p50']]
        self._impact['Prob of 50% Profit'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[(df['daysToExpiration'] >= p['min_dte']) & (df['daysToExpiration'] <= p['max_dte'])]
        self._impact['DTE'] = f"{df.shape[0] / self._shp:.0%}"

        # Calculated columns.
        df['bid_ask_pct'] = df['ask'] / df['bid'] - 1
        df['bid_ask_rank'] = df.groupby(['stock', 'option_type', 'daysToExpiration'])['bid_ask_pct'].rank(pct=True,
                                                                                                          ascending=False,
                                                                                                          method='dense')
        df['open_int_rank'] = df.groupby(['stock', 'option_type', 'daysToExpiration'])['openInterest'].rank(pct=True,
                                                                                                            ascending=True,
                                                                                                            method='dense')
        df['volume_rank'] = df.groupby(['stock', 'option_type', 'daysToExpiration'])['totalVolume'].rank(pct=True,
                                                                                                         ascending=True,
                                                                                                         method='dense')

        # Filter: Pass 2
        df = df[df['bid_ask_rank'] >= p['max_bid_ask_pcl']]
        self._impact['Bid/Ask'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[df['open_int_rank'] >= p['min_open_int_pcl']]
        self._impact['Open Interest'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[df['volume_rank'] >= p['min_volume_pcl']]
        self._impact['Volume'] = f"{df.shape[0] / self._shp:.0%}"

        # Exit if table is empty.
        self._exit_if_option_table_is_empty(table_size=len(df.index))

        # Return dataframe.
        self._shp = df.shape[0]
        self.options = df

    def _exit_if_option_table_is_empty(self, table_size: int):
        """
        Reports impact of current filters vs. original table.

        :param table_size: Length of options DataFrame.
        """
        if table_size == 0:
            st.warning("Nothing to see here!** Criteria not met")
            st.write("Here's the impact of the filters you've set:")
            df = pd.DataFrame(list(self._impact.items()), columns=['Filter', '% of Total'])
            df.set_index('Filter', inplace=True)
            st.table(df)
            st.stop()

    @staticmethod
    def _format_option_table(options: pd.DataFrame):
        df = options.copy()
        df.sort_values(by='return', ascending=False, inplace=True)
        df.set_index('search', inplace=True)
        df = df[[
            'return',
            'return_day',
            'max_profit',
            'risk',
            'quantity',
            'margin_requirement',
            'prem_mark',
            'break_even',
            'delta',
            # 'lastPrice',
            'option_value_mc',
            'probability_ITM',
            'probability_of_50',
            # 'volatility',
            'daysToExpiration',
            'stock'
        ]]
        df = df.rename(columns={
            'return': 'Return',
            'return_day': 'Daily Return',
            'max_profit': 'Max Profit',
            'risk': 'Risk',
            'quantity': 'Qty',
            'margin_requirement': 'Margin Req\'d',
            'prem_mark': 'Prem Mark',
            'break_even': 'Break Even',
            'delta': 'Delta',
            'option_value_mc': 'Theo Value',
            'probability_ITM': 'Prob ITM',
            'probability_of_50': 'Prob 50%',
            'daysToExpiration': 'DTE',
            'stock': 'Stock',
        })

        # Format results.
        min_max = ['Return', 'Max Profit', 'Daily Return', 'Prob ITM', 'Prob 50%']
        df_styled = (
                df.style
                .set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
                .set_properties(**{'text-align': 'right'})
                .background_gradient(axis=0, subset=['Return', 'Daily Return'])
                .highlight_max(subset=min_max, color=fm.HI_MAX_COLOR)
                .highlight_min(subset=min_max, color=fm.HI_MIN_COLOR)
                .format(
                    {
                        'Return': fm.PERCENT2,
                        'Daily Return': fm.PERCENT2,
                        'Max Profit': fm.DOLLAR,
                        'Risk': fm.DOLLAR,
                        'Margin Req\'d': fm.DOLLAR,
                        'Break Even': fm.DOLLAR,
                        'Qty': fm.FLOAT0 + "x",
                        'Prem Mark': fm.DOLLAR,
                        'Delta': fm.FLOAT,
                        'Theo Value': fm.DOLLAR,
                        'Prob ITM': fm.PERCENT0,
                        'Prob 50%': fm.PERCENT0,
                        'DTE': fm.FLOAT0,
                    }
                )
        )
        return df_styled

    @staticmethod
    def _show_option_table(strategy_header, df_styled):
        st.header(strategy_header)
        st.dataframe(data=df_styled)

    def naked(self):
        """
        Retrieve naked options based on the parameters given.

        """
        # Variable assignment.
        p = self._params
        df = self.options.copy()

        # Initialize Impact log. Finally, showing the impact of each filter.
        self._impact.clear()
        self._impact = {'Starting size': f"{self._shp}"}

        # Calculated columns.
        df.rename(columns={'mark': 'prem_mark'}, inplace=True)
        df['break_even'] = df['strikePrice'] - df['prem_mark']
        df['margin_requirement'] = df['strikePrice'] * df['multiplier']
        df['max_profit'] = df['multiplier'] * df['prem_mark']
        df['risk'] = df['strikePrice'] * df['multiplier'] - df['max_profit']
        df['return'] = df['max_profit'] / df['risk']
        df['return_day'] = df['return'] / (df['daysToExpiration'] + .00001)
        df['quantity'] = p['max_risk'] / df['risk']
        df['search'] = (
            # Stock.
            df['description'].str.split(' ').str[0].astype(str)
            # Day.
            + " "
            + df['description'].str.split(' ').str[2].astype(str)
            # Month.
            + " "
            + df['description'].str.split(' ').str[1].astype(str)
            # Year.
            + " "
            + df['description'].str.split(' ').str[3].str[::3].astype(str)
            # Days to expiration.
            + " (" + df['daysToExpiration'].astype(str) + ") "
            # Option Type (PUT or CALL).
            + df['option_type'].str.upper().astype(str)
            # Strike price.
            + " "
            + df['strikePrice'].astype(str)
        )

        # More filters
        df = df[df['margin_requirement'] <= p['margin_requirement']]
        self._impact['Margin requirement'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[df['risk'] <= p['max_risk']]
        self._impact['Risk'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[df['return'] >= p['min_return_pct']]
        self._impact['Return'] = f"{df.shape[0] / self._shp:.0%}"

        # Exit if table is empty.
        self._exit_if_option_table_is_empty(table_size=len(df.index))

        # Formatting the table.
        df_styled = self._format_option_table(options=df)

        # Show the table (this may be commented out as it will be returned)
        self._show_option_table(strategy_header='Naked Credit', df_styled=df_styled)

        # Return the options table.
        return df

    def spread(self):
        """Returns dataframe with Vertical Credit Spreads."""

        def spread_calculator(df_in: pd.DataFrame) -> Dict[str, dict]:
            """
            Calculates spread combinations to maximize profit.
            Uses Scipy's City Distance methodology.

            :param df_in: Dataframe with option chain.
            :return: Dictionary with best spreads by date.
            """

            def distance_matrix(df_ask: pd.DataFrame, df_bid: pd.DataFrame) -> pd.DataFrame:
                """
                Calculates difference in a 2D matrix (Ask rows & Bid columns).

                Index for these series is the information to be expanded later.

                :param df_ask: Ask series.
                :param df_bid: Bid series.
                :return: Dataframe with differences between values in rows and columns.
                """
                xa = df_ask.to_numpy().tolist()
                xb = df_bid.to_numpy().tolist()
                y = cdist(XA=xa, XB=xb, metric="cityblock")
                return pd.DataFrame(data=y, index=df_ask.index, columns=df_bid.index)

            df1 = df_in.copy()

            # Index key initialization.
            index_keys = ['option_type', 'stock']

            # Fields needed for Cross-Calculations.
            cols_to_cross = ['bid', 'ask']

            # Fields needed for post operations between short/long.
            cols_for_post = ['strikePrice', 'mark']
            cols_post_levels = list(range(len(cols_for_post)))

            # Fields for Reporting and filtering for short strike.
            cols_meta = [i for i in df1.columns.to_list()
                         if i not in (index_keys + cols_to_cross + cols_for_post)]

            # Compress data into index
            index_keys += cols_meta + cols_for_post
            df1.set_index(index_keys, inplace=True)

            # Calculate all bid - ask spreads
            dfx = distance_matrix(df1[['ask']], df1[['bid']])

            # Prep fields short & long labels added to index.
            prep_key_short = [('short_' + col) for col in cols_for_post]
            prep_key_long = [('long_' + col) for col in cols_for_post]
            new_key = cols_meta + prep_key_short + prep_key_long

            # Separate puts/calls & stocks, and calculate spread short-long strikes
            spread_dict = {'put': {}, 'call': {}}
            stock_list = dfx.index.get_level_values('stock').unique().to_list()
            dfx.sort_index(inplace=True)

            for k in spread_dict:
                for s in stock_list:
                    if (k, s) not in dfx.index:  # if ('call', 'RIVN') in list(zip(list(zip(*dfx.index))[0], list(zip(*dfx.index))[1])):
                        continue
                    df1 = dfx.xs((k, s), axis=0)
                    df1 = df1.sort_index(axis=1)
                    df1 = df1.xs((k, s), axis=1)
                    df1 = df1.droplevel(cols_meta, axis=1)
                    df1 = df1.stack(dropna=True, level=cols_post_levels)
                    df1 = pd.DataFrame(df1, columns=['prem_bid_ask'])
                    df1.index = df1.index.set_names(new_key)
                    df1 = df1.reset_index()
                    df1['spread'] = df1['short_strikePrice'] - df1['long_strikePrice']
                    df1['prem_mark'] = df1['short_mark'] - df1['long_mark']
                    if k == 'put':
                        df1['delta'] *= -1
                    else:
                        df1['spread'] *= -1
                    df1 = df1[df1['spread'] > 0]
                    df1['vertical'] = (df1['short_strikePrice'].map("{0:g}".format) + '/' + df1['long_strikePrice'].map("{0:g}".format))
                    df1.set_index('vertical', inplace=True)
                    spread_dict[k][s] = df1
            return spread_dict

        # Variable assignment.
        p = self._params
        df = self.options.copy()

        # Initialize Impact log. Finally, showing the impact of each filter.
        self._impact.clear()
        self._impact = {'Starting size': f"{self._shp}"}

        # Clear table before processing for spreads.
        # Comment out columns to drop.
        df = df[[
            'stock',
            'option_type',
            'description',
            'bid',
            'ask',
            'mark',
            # 'totalVolume',
            'volatility',
            'delta',
            # 'openInterest',
            'strikePrice',
            'daysToExpiration',
            'multiplier',
            # 'inTheMoney',
            # 'lastPrice',
            'option_value_mc',
            'probability_ITM',
            'probability_of_50',
            # 'bid_ask_pct',
            # 'bid_ask_rank',
            # 'open_int_rank',
            # 'volume_rank'
        ]]

        # Search for spreads
        # Columns: 'description, volatility, delta, daysToExpiration, multiplier,
        # short_strikePrice, short_mark, long_strikePrice, long_mark, prem_bid_ask,
        # spread, prem_mark'.
        # Index: None, but they're DTE, option_type, stock, vertical spread.
        # Consolidate tables by DTE.
        # :return: DataFrame that holds all spreads. Index: call/put, stock,
        # vert(short/long). Columns: 'description', 'volatility', 'delta',
        # 'daysToExpiration', 'multiplier', 'short_strikePrice', 'short_mark',
        # 'long_strikePrice', 'long_mark', 'prem_bid_ask', 'spread', 'prem_mark'.

        collector: Dict = {}  # Holds Calls/Puts dict of DataFrames

        # Iterate per DTE and concatenate DataFrame with all DTEs
        dte_sorted: List[int] = sorted(df['daysToExpiration'].unique())  # List of DTE's to iterate
        for dte_current in dte_sorted:
            dte_dict_spreads: Dict = spread_calculator(df[df['daysToExpiration'] == dte_current])
            dte_dict_spreads_put_call: Dict = {}
            puts_avail = len(dte_dict_spreads['put']) != 0
            calls_avail = len(dte_dict_spreads['call']) != 0
            if puts_avail:
                dte_dict_spreads_put_call['put'] = pd.concat(dte_dict_spreads['put'])
            if calls_avail:
                dte_dict_spreads_put_call['call'] = pd.concat(dte_dict_spreads['call'])
            if not puts_avail and not calls_avail:
                continue
            collector[dte_current] = pd.concat(
                dte_dict_spreads_put_call)  # Dictionary that collects DataFrames from all DTE's

        del dte_dict_spreads, dte_dict_spreads_put_call  # Cleanup
        df = pd.concat(collector)  # DataFrame that holds all spreads. Index: call/put, stock, vert(short/long)
        df = df.droplevel(level=0)  # Remove DTE from the Index
        df.index.set_names(inplace=True, names=['option_type', 'stock', 'vertical'])
        df.reset_index(inplace=True)

        # ------- output
        # Calculate all metrics
        df['break_even'] = df['short_strikePrice'] - df['prem_mark']
        df['margin_requirement'] = df['multiplier'] * df['spread']
        df['max_profit'] = df['multiplier'] * df['prem_mark']
        df['risk'] = df['multiplier'] * (df['spread'] - df['prem_mark'])
        df['return'] = df['max_profit'] / df['risk']
        df['return_day'] = df['return'] / (df['daysToExpiration'] + .00001)
        df['quantity'] = np.floor(p['max_risk'] / df['risk'])

        # Original Search similar to TD Ameritrade
        # df['search'] = (
        #     df['description'].str.split(' ').str[0].astype(str) + " "
        #     + df['description'].str.split(' ').str[2].astype(str) + " "
        #     + df['description'].str.split(' ').str[1].astype(str) + " "
        #     + df['description'].str.split(' ').str[3].str[::3].astype(str)
        #     + " (" + df['daysToExpiration'].map('{:.0f}'.format).astype(str) + ") "
        #     + df['option_type'].str.upper().astype(str) + " "
        #     + df['vertical'].astype(str)
        # )

        # Simplifying the above search column, due to a Streamlit width limitation
        lower_m = [m.lower() for m in month_abbr]
        df['search'] = (
            # stock
            df['description'].str.split(' ').str[0].astype(str)
            # day
            + " "
            + df['description'].str.split(' ').str[2].astype(str)
            # month
            + "/"
            + (df['description'].str.split(' ').str[1].astype(str)
               .map(lambda m: lower_m.index(m.lower())).astype(str))
            # year
            # + "/"
            # + df['description'].str.split(' ').str[3].str[::3].astype(str)
            # dte
            + " ("
            + df['daysToExpiration'].map('{:.0f}'.format).astype(str)
            + ") "
            # option type put or call
            + df['option_type'].str.upper().astype(str).str[0]
            # short and long strike price
            + " "
            + df['vertical'].astype(str)
        )

        # More filters
        df = df[df['margin_requirement'] <= p['margin_requirement']]
        self._impact['Margin requirement'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[df['return'] >= p['min_return_pct']]
        self._impact['Return'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[df['risk'] <= p['max_risk']]
        self._impact['Risk'] = f"{df.shape[0] / self._shp:.0%}"

        df = df[df['risk'] >= df['max_profit']]
        self._impact['Risk>Profit'] = f"{df.shape[0] / self._shp:.0%}"

        # Exit if table is empty.
        self._exit_if_option_table_is_empty(table_size=len(df.index))

        # Formatting the table.
        df_styled = self._format_option_table(options=df)

        # Show the table (this may be commented out as it will be returned)
        self._show_option_table(strategy_header='Vertical Spread Credit', df_styled=df_styled)

        # Return the options table.
        return df
