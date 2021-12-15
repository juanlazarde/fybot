from typing import Dict, List, Any
import pandas as pd
import numpy as np
import streamlit as st
from scipy.spatial.distance import cdist
import core.formatter as fm


def _impact_results(impact: Dict[str, str]) -> None:
    """
    Reports impact of current filters vs. original table.

    :param impact: Dictionary with filter and value.
    :return: None
    """
    df = pd.DataFrame(list(impact.items()), columns=['Filter', '% of Total'])
    df.set_index('Filter', inplace=True)
    st.table(df)


class _Spread:
    def __init__(
            self,
            df_in: pd.DataFrame,
            impact: Dict[str, str]) -> pd.DataFrame:
        """
        Class supporting spread() function.

        :param df_in: Dataframe with original option tables.
        :param impact: Dictionary with impact of each filter up to here.
        """
        self.df: pd.Dataframe = df_in.copy()
        self.impact: Dict[str, str] = impact
        self.dte_consolidator()

    @staticmethod
    def distance_matrix(
            df_ask: pd.Series,
            df_bid: pd.Series) -> pd.DataFrame:
        """Calculates difference in a 2D matrix (Ask rows & Bid columns).

        Index for these series is the the information to be expanded later.

        :param df_ask: Ask series.
        :param df_bid: Bid series.
        :return: Dataframe with differences between values in rows and columns.
        """
        xa: List[float] = df_ask.to_numpy().tolist()
        xb: List[float] = df_bid.to_numpy().tolist()
        y = cdist(xa, xb, "cityblock")
        df: pd.DataFrame = pd.DataFrame(
            data=y,
            index=df_ask.index,
            columns=df_bid.index
        )
        return df

    def search_for_spreads(self, df_in: pd.DataFrame) -> Dict[str, dict]:
        """
        Search best option spread.

        :param df_in: Dataframe with option chain.
        :return: Dictionationary with pest spreads by date.
        """
        df = df_in.copy()
        del df_in
        # Index key initialization.
        index_keys = [
            'option_type',
            'stock'
        ]

        # Fields needed for Cross-Calculations.
        cols_to_cross = [
            'bid',
            'ask'
        ]

        # Fields needed for post operations between short/long.
        cols_for_post = [
            'strikePrice',
            'mark'
        ]
        cols_post_levels = list(range(len(cols_for_post)))

        # Fields for Reporting and filtering for short strike.
        cols_meta = [i for i in df.columns.to_list()
                     if i not in (index_keys + cols_to_cross + cols_for_post)]

        # Compress data into index
        index_keys += cols_meta + cols_for_post
        df.set_index(index_keys, inplace=True)

        # calculate all bid - ask spreads (absolute values)
        # cross-calculates all strike prices and stock combinations
        dfx = pd.DataFrame()
        if len(df.index) > 1:
            dfx = _Spread.distance_matrix(df[['ask']], df[['bid']])
        else:
            st.error(
                "Not enough options available. "
                "Try losening search criteria."
            )
            _impact_results(self.impact)
            st.stop()

        # Prep fields short & long labels added to index.
        prep_key_short = [('short_' + col) for col in cols_for_post]
        prep_key_long = [('long_' + col) for col in cols_for_post]
        new_key = cols_meta + prep_key_short + prep_key_long

        # Separate puts/calls & stocks, and calculate spread short-long strikes
        spread_dict = {'put': {}, 'call': {}}
        stock_list = dfx.index.get_level_values('stock').unique()
        dfx = dfx.sort_index()

        for k in spread_dict:
            for s in stock_list:
                try:
                    df = dfx.xs((k, s), axis=0)
                    df = df.sort_index(axis=1)
                    df = df.xs((k, s), axis=1)
                    df = df.droplevel(cols_meta, axis=1)
                    df = df.stack(dropna=True, level=cols_post_levels)
                    df = pd.DataFrame(df, columns=['prem_bid_ask'])
                    df.index = df.index.set_names(new_key)
                    df = df.reset_index()

                    df['spread'] = df['short_strikePrice'] - df[
                        'long_strikePrice']
                    df['prem_mark'] = df['short_mark'] - df['long_mark']

                    if k == 'put':
                        df['delta'] *= -1
                    else:
                        df['spread'] *= -1
                    df = df[df['spread'] > 0]
                    df['vertical'] = (
                            df['short_strikePrice'].map("{0:g}".format)
                            + '/'
                            + df['long_strikePrice'].map("{0:g}".format)
                    )
                    df = df.set_index('vertical')

                except Exception:
                    df = pd.DataFrame()

                spread_dict[k][s] = df

        return spread_dict

    def dte_consolidator(self):
        """
        Consolidates tables by DTE.

        :return: DataFrame that holds all spreads. Index: call/put, stock,
        vert(short/long). Columns: 'description', 'volatility', 'delta',
        'daysToExpiration', 'multiplier', 'short_strikePrice', 'short_mark',
        'long_strikePrice', 'long_mark', 'prem_bid_ask', 'spread', 'prem_mark'.
        """
        df = self.df.copy()  # Avoid changing original DataFrame
        collector: Dict = dict()  # Holds Calls/Puts dict of DataFrames

        # Iterate per DTE and concatenate DataFrame with all DTEs
        dte_sorted: List[int] = sorted(df['daysToExpiration'].unique())  # List of DTE's to iterate
        for dte_current in dte_sorted:
            dte_df: pd.DataFrame = df[df['daysToExpiration'] == dte_current]  # Holds table per DTE
            dte_dict_spreads: Dict = self.search_for_spreads(dte_df)  # Dictionary Keys: put, call. Values: Dataframe per stock with best spreads per DTE
            dte_dict_spreads_putCall: Dict = {
                'put': pd.concat(dte_dict_spreads['put']),
                'call': pd.concat(dte_dict_spreads['call'])
            }  # Separates Puts/Calls
            # TODO: Remove this try after debugging
            try:
                collector[dte_current] = pd.concat(dte_dict_spreads_putCall)  # Dictionary that collects DataFrames from all DTE's
            except Exception as e:
                print(f"Error {e}")

            del dte_dict_spreads, dte_dict_spreads_putCall  # Ensures previous data won't corrupt new data in the cycle

        self.df = pd.concat(collector)  # DataFrame that holds all spreads. Index: call/put, stock, vert(short/long)


def spread(df_in: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Calculate and Retrieve Vertical Spread options on the parameters given.

    :param df_in: Option chain table.
    :param filters: Filters for the options to be shown.
    :return: Options table dataframe.
    """
    # Variable assignment.
    # TODO: Make Debit/credit premium type work
    # TODO: Check on ITM treatment
    d: Dict[str, Any] = {
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
        'min_open_int_pctl': float(filters['min_open_int_pctl']) / 100,
        'min_volume_pctl': float(filters['min_volume_pctl']) / 100,
        'max_bid_ask_pctl': float(filters['max_bid_ask_pctl']) / 100
    }
    df: pd.DataFrame = df_in.copy()
    del filters, df_in

    # Clean table.
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

    df.reset_index(inplace=True)
    df.sort_values(by='symbol', ascending=True, inplace=True)
    df.drop(columns=['symbol'], inplace=True)  # Actual option's symbol.

    # Filter: Pass 1. Before cross-calculation.
    df = df[(df['delta'] != 'NaN')
            & (df['openInterest'] > 0)
            & (df['totalVolume'] > 0)
            & (df['bid'] > 0)
            & (df['ask'] > 0)
            ]

    _shp: int = df.shape[0]  # Impact log initialization.
    impact: Dict[str, str] = {'Starting size': f"{_shp}"}

    if d['option_type'] in ['put', 'call']:
        df = df[df['option_type'].str.lower() == d['option_type']]
    impact['Option type'] = f"{df.shape[0]/_shp:.0%}"

    df = df[df['inTheMoney'] == d['option_itm']]
    impact['ITM'] = f"{df.shape[0]/_shp:.0%}"

    df = df[(df['delta'] <= d['max_delta'])
            & (df['delta'] >= -d['max_delta'])]
    impact['Delta'] = f"{df.shape[0]/_shp:.0%}"

    df = df[(df['daysToExpiration'] >= d['min_dte'])
            & (df['daysToExpiration'] <= d['max_dte'])]
    impact['DTE'] = f"{df.shape[0]/_shp:.0%}"

    # Calculated columns.
    df['bid_ask_pct'] = df['ask']/df['bid'] - 1
    df['bid_ask_rank'] = (df.groupby('stock')['bid_ask_pct']
                          .rank(pct=True, ascending=False, method='dense'))
    df['open_int_rank'] = (df.groupby('stock')['openInterest']
                           .rank(pct=True, ascending=True, method='dense'))
    df['volume_rank'] = (df.groupby('stock')['totalVolume']
                         .rank(pct=True, ascending=True, method='dense'))

    # Filter: Pass 2. Before cross calculations.
    df = df[df['bid_ask_rank'] >= d['max_bid_ask_pctl']]
    impact['Bid/Ask'] = f"{df.shape[0]/_shp:.0%}"

    df = df[df['open_int_rank'] >= d['min_open_int_pctl']]
    impact['Open Interest'] = f"{df.shape[0]/_shp:.0%}"

    df = df[df['volume_rank'] >= d['min_volume_pctl']]
    impact['Volume'] = f"{df.shape[0]/_shp:.0%}"

    # Exit if table is empty.
    if len(df.index) == 0:
        st.warning("**Nothing to see here!** Criteria not met")
        st.write("**Here's the impact of the filters you've set:**")
        _impact_results(impact)
        st.stop()

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
    df = _Spread(df_in=df, impact=impact).df

    df = df.droplevel(level=0)  # Remove DTE from the Index
    df.index.set_names(inplace=True, names=[
        'option_type',
        'stock',
        'vertical'])
    df.reset_index(inplace=True)

    # ------- output
    # calculate all metrics
    # TODO: Decide which is better prem_sprd_ratio or return_pct
    # df['prem_sprd_ratio'] = df['prem_mark'] / df['spread']
    df['break_even'] = df['short_strikePrice'] - df['prem_mark']
    df['margin_requirement'] = df['multiplier'] * df['spread']
    df['max_profit'] = df['multiplier'] * df['prem_mark']
    df['risk'] = df['multiplier'] * (df['spread'] - df['prem_mark'])
    df['return'] = df['max_profit'] / df['risk']
    df['return_day'] = df['return'] / (df['daysToExpiration'] + .00001)
    df['quantity'] = np.floor(d['max_risk'] / df['risk'])
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
    from calendar import month_abbr
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
    df = df[df['probability_ITM'] >= d['min_pop']]
    impact['Prob of Profit (ITM)'] = f"{df.shape[0]/_shp:.0%}"

    df = df[df['probability_of_50'] >= d['min_p50']]
    impact['Prob of 50% Profit (ITM)'] = f"{df.shape[0]/_shp:.0%}"

    df = df[df['margin_requirement'] <= d['margin_requirement']]
    impact['Margin requirement'] = f"{df.shape[0]/_shp:.0%}"

    df = df[df['return'] >= d['min_return_pct']]
    impact['Return'] = f"{df.shape[0]/_shp:.0%}"

    df = df[df['risk'] <= d['max_risk']]
    impact['Risk'] = f"{df.shape[0]/_shp:.0%}"

    df = df[df['risk'] >= df['max_profit']]
    impact['Risk>Profit'] = f"{df.shape[0]/_shp:.0%}"

    # Exit if table is empty.
    if len(df.index) == 0:
        st.warning("**Nothing to see here!** Criteria not met")
        st.write("**Here's the impact of the filters you've set:**")
        _impact_results(impact)
        st.stop()

    # Formatting the table.
    df.sort_values('return', ascending=False, inplace=True)
    df.set_index('search', inplace=True)
    df = df[[
        'return',
        'return_day',
        'max_profit',
        'risk',
        'quantity',
        'margin_requirement',
        'prem_mark',
        # 'prem_bid_ask',
        # 'prem_sprd_ratio',
        'break_even',
        'delta',
        # 'lastPrice',
        'option_value_mc',
        'probability_ITM',
        'probability_of_50',
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
        # 'prem_bid_ask': 'Premium Bid/Ask',
        # 'prem_sprd_ratio': 'Premium Spread Ratio',
        'break_even': 'Break Even',
        'delta': 'Delta',
        'option_value_mc': 'Theo Value',
        'probability_ITM': 'Prob ITM',
        'probability_of_50': 'Prob 50%',
        'daysToExpiration': 'DTE',
        'stock': 'Stock',
    })

    # Display results.
    min_max = ['Return', 'Max Profit', 'Daily Return', 'Prob ITM', 'Prob 50%']
    df_print = (df.style
                  .set_table_styles(
                    [dict(selector='th', props=[('text-align', 'left')])])
                  .set_properties(**{'text-align': 'right'})
                  .background_gradient(
                    axis=0,
                    subset=['Return', 'Daily Return'])
                  .highlight_max(
                    subset=min_max,
                    color=fm.HI_MAX_COLOR)
                  .highlight_min(
                    subset=min_max,
                    color=fm.HI_MIN_COLOR)
                  .format({
                    'Return': fm.PERCENT2,
                    'Daily Return': fm.PERCENT2,
                    'Max Profit': fm.DOLLAR,
                    'Risk': fm.DOLLAR,
                    'Margin Req\'d': fm.DOLLAR,
                    'Prem Mark': fm.DOLLAR,
                    # 'Premium Bid/Ask': fm.DOLLAR,
                    # 'Premium Spread Ratio': fm.DOLLAR,
                    'Break Even': fm.DOLLAR,
                    'Qty': fm.FLOAT0 + "x",
                    'Delta': fm.FLOAT,
                    'Theo Value': fm.DOLLAR,
                    'Prob ITM': fm.PERCENT0,
                    'Prob 50%': fm.PERCENT0,
                    'DTE': fm.FLOAT0,
                  })
                )

    st.header('Vertical Spread')
    st.dataframe(data=df_print)

    return df
