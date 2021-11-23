import sys
from typing import Dict, List

import pandas as pd
import streamlit as st
from numpy import percentile
from scipy.spatial.distance import cdist
import core.formatter as fm

from core.utils import performance


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


# @performance
def search_for_spreads(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Search best option spread.

    :param df: Dataframe with option chain.
    :return: Dictionationary with pest spreads by date.
    """
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
    df = df.set_index(index_keys)

    # calculate all bid - ask spreads (absolute values)
    # cross-calculates all strike prices and stock combinations
    dfx = pd.DataFrame()
    if len(df.index) > 1:
        dfx = distance_matrix(df[['ask']], df[['bid']])
    else:
        st.error('Not enough options available. Try losening search criteria')
        st.stop()

    # Prep fields short & long labels added to index.
    prep_key_short = [('short_' + col) for col in cols_for_post]
    prep_key_long = [('long_' + col) for col in cols_for_post]
    new_key = cols_meta + prep_key_short + prep_key_long

    # Separate puts/calls and stocks, and calculate spread short-long strikes
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

                df['spread'] = df['short_strikePrice'] - df['long_strikePrice']
                df['prem_mark'] = df['short_mark'] - df['long_mark']

                if k == 'put':
                    df['delta'] *= -1
                else:
                    df['spread'] *= -1
                df = df[df['spread'] > 0]
                df['vertical'] = \
                    (df['short_strikePrice'].map("{0:g}".format)
                     + '/' +
                     df['long_strikePrice'].map("{0:g}".format))
                df = df.set_index('vertical')
                spread_dict[k][s] = df
            except Exception:
                pass
    return spread_dict


def spread(df_in: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Calculate and Retrieve Vertical Spread options on the parameters given.

    :param df_in: Option chain table.
    :param filters: Filters for the options to be shown.
    :return: Options table dataframe.
    """
    # Variable assignment.
    # TODO: Make Debit/credit premium type work
    premium_type = filters['premium_type'].lower().strip()
    # TODO: Check on ITM treatment
    option_itm: bool = False

    option_type: str = filters['option_type'].lower().strip()
    strike_price_spread: float = float(filters['strike_price_spread'])
    max_risk: float = float(filters['max_risk'])
    min_return_pct: float = float(filters['min_return_pct']) / 100
    max_dte: int = int(filters['max_dte'])
    min_dte: int = int(filters['min_dte'])
    max_delta: float = float(filters['max_delta'])
    min_open_int_pctl: float = float(filters['min_open_int_pctl']) / 100
    min_volume_pctl: float = float(filters['min_volume_pctl']) / 100
    max_bid_ask_pctl: float = float(filters['max_bid_ask_pctl']) / 100

    # Clean table.
    # Columns available will be commented out.
    _cols = [
        'putCall',
        'symbol',
        # 'description',
        'exchangeName',
        # 'bid',
        # 'ask',
        'last',
        # 'mark',
        'bidSize',
        'askSize',
        'bidAskSize',
        'lastSize',
        'highPrice',
        'lowPrice',
        'openPrice',
        'closePrice',
        # 'totalVolume',
        'tradeDate',
        'tradeTimeInLong',
        'quoteTimeInLong',
        'netChange',
        # 'volatility',
        # 'delta',
        'gamma',
        'theta',
        'vega',
        'rho',
        # 'openInterest',
        'timeValue',
        'theoreticalOptionValue',
        'theoreticalVolatility',
        'optionDeliverablesList',
        # 'strikePrice',
        'expirationDate',
        # 'daysToExpiration',
        'expirationType',
        'lastTradingDay',
        # 'multiplier',
        'settlementType',
        'deliverableNote',
        'isIndexOption',
        'percentChange',
        'markChange',
        'markPercentChange',
        'intrinsicValue',
        # 'inTheMoney',
        'pennyPilot',
        'mini',
        'nonStandard'
    ]
    df = df_in.copy()
    df.drop(columns=_cols, inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['symbol'], inplace=True)
    df.sort_values(by='stock', ascending=True, inplace=True)

    # Filter: Pass 1. Before cross-calculation.
    df = df[(df['delta'] != 'NaN')
            & (df['openInterest'] > 0)
            & (df['totalVolume'] > 0)
            & (df['bid'] > 0)
            & (df['ask'] > 0)
            ]

    _shp: int = df.shape[0]  # Impact log initialization.
    impact: Dict[str, str] = {'Starting size': f"{_shp}"}

    if option_type in ['put', 'call']:
        df = df[df['option_type'].str.lower() == option_type]
    impact['Option type'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['inTheMoney'] == option_itm]
    impact['ITM'] = f"{df.shape[0] / _shp:.0%}"

    df = df[(df['delta'] <= max_delta)
            & (df['delta'] >= -max_delta)]
    impact['Delta'] = f"{df.shape[0] / _shp:.0%}"

    df = df[(df['daysToExpiration'] >= min_dte)
            & (df['daysToExpiration'] <= max_dte)]
    impact['DTE'] = f"{df.shape[0] / _shp:.0%}"

    # Calculated columns
    df['bid_ask_pct'] = (df['ask'] / df['bid'] - 1)
    df['bid_ask_rank'] = (df.groupby('stock')['bid_ask_pct']
                          .rank(pct=True, ascending=False, method='dense'))
    df['open_int_rank'] = (df.groupby('stock')['openInterest']
                           .rank(pct=True, ascending=True, method='dense'))
    df['volume_rank'] = (df.groupby('stock')['totalVolume']
                         .rank(pct=True, ascending=True, method='dense'))

    # Filter: Pass 2. Before cross calculations.
    df = df[df['bid_ask_rank'] >= max_bid_ask_pctl]
    impact['Bid/Ask'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['open_int_rank'] >= min_open_int_pctl]
    impact['Open Interest'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['volume_rank'] >= min_volume_pctl]
    impact['Volume'] = f"{df.shape[0] / _shp:.0%}"

    # Exit if table is empty
    if len(df.index) == 0:
        st.warning("**Nothing to see here!** Criteria not met")
        st.write("**Here's the impact of the filters you've set:**")
        _impact_table = pd.DataFrame(impact.items(), ['Filter', '% of Total'])
        _impact_table = _impact_table.set_index('Filter')
        st.table(_impact_table)
        st.stop()

    # Clear table before processing for spreads.
    _cols = [
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
        # 'bid_ask_pct',
        # 'bid_ask_rank',
        # 'open_int_rank',
        # 'volume_rank'
    ]
    df = df[_cols]

    # Search for spreads
    spread_dict: Dict = dict()
    sorted_dte: List[int] = sorted(df['daysToExpiration'].unique())

    for dte in sorted_dte:
        df_by_dte = df[df['daysToExpiration'] == dte]
        pre_spread_dict = search_for_spreads(df_by_dte)

        if (len(pre_spread_dict['call']) == 0 and
           len(pre_spread_dict['put']) == 0):
            st.warning("Options for mining are empty")
            st.stop()

        # overwrite dictionaries with concatenated df across all symbols
        for i in ['put', 'call']:
            if len(pre_spread_dict[i]) != 0:
                pre_spread_dict[i] = pd.concat(pre_spread_dict[i])
            else:
                del pre_spread_dict[i]

        spread_dict[dte] = pd.concat(pre_spread_dict)

    df = pd.concat(spread_dict)
    df = df.droplevel(level=0)
    ix_names = ['option_type', 'stock', 'vertical']
    df.index.set_names(ix_names, inplace=True)
    df.reset_index(inplace=True)

    # ------- output
    # calculate all metrics
    # TODO: Decide which is better prem_sprd_ratio or return_pct
    df['prem_sprd_ratio'] = df['prem_mark'] / df['spread']
    df['max_profit'] = df['multiplier'] * df['prem_mark']
    df['risk'] = df['multiplier'] * (df['spread'] - df['prem_mark'])
    df['return'] = df['multiplier'] * df['max_profit'] / df['risk']
    df['return_day'] = df['return'] / (df['daysToExpiration'] + .00001)
    df['quantity'] = max_risk / df['risk']
    df['search'] = \
        df['description'].str.split(' ').str[0].astype(str) + " " + \
        df['description'].str.split(' ').str[2].astype(str) + " " + \
        df['description'].str.split(' ').str[1].astype(str) + " " + \
        df['description'].str.split(' ').str[3].str[::3].astype(str) + \
        " (" + df['daysToExpiration'].astype(str) + ") " + \
        df['option_type'].str.upper().astype(str) + " " + \
        df['vertical'].astype(str)

    # More filters
    df = df[df['spread'] <= strike_price_spread]
    impact['Spread'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['return'] >= min_return_pct]
    impact['Return'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['risk'] <= max_risk]
    impact['Risk'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['risk'] >= df['max_profit']]
    impact['Risk>Profit'] = f"{df.shape[0] / _shp:.0%}"

    # exit if table is empty
    if len(df.index) == 0:
        st.warning("**Nothing to see here!** Criteria not met")
        st.write("**Here's the impact of the filters you've set:**")
        st.table(pd.DataFrame(impact.items(), ['Filter', '% of Total'])
                   .set_index('Filter'))
        return

    df.sort_values('return', ascending=False, inplace=True)
    df.set_index('search', inplace=True)

    _cols = [
        'return',
        'return_day',
        'max_profit',
        'risk',
        'quantity',
        'spread',
        'prem_mark',
        'prem_bid_ask',
        'prem_sprd_ratio',
        'delta',
        'daysToExpiration',
        'stock'
    ]
    df = df[_cols]

    _cols = {
        'return': 'Return',
        'return_day': 'Daily Return',
        'max_profit': 'Max Profit',
        'risk': 'Risk',
        'quantity': 'Qty',
        'spread': 'Vertical Spread',
        'prem_mark': 'Premium Mark',
        'prem_bid_ask': 'Premium Bid/Ask',
        'prem_sprd_ratio': 'Premium Spread Ratio',
        'delta': 'Delta',
        'daysToExpiration': 'DTE',
        'stock': 'Stock',
    }

    df = df.rename(columns=_cols)

    # display results
    df_print = (df.style.background_gradient(
                    axis=0,
                    subset=['Return', 'Daily Return'])
                  .highlight_max(
                    subset=['Return', 'Max Profit', 'Daily Return'],
                    color=fm.HI_MAX_COLOR)
                  .highlight_min(
                    subset=['Return', 'Max Profit', 'Daily Return'],
                    color=fm.HI_MIN_COLOR)
                  .format({
                    'Return': fm.FMT_PERCENT2,
                    'Daily Return': fm.FMT_PERCENT2,
                    'Max Profit': fm.FMT_DOLLAR,
                    'Risk': fm.FMT_DOLLAR,
                    'Vertical Spread': fm.FMT_DOLLAR,
                    'Premium Mark': fm.FMT_DOLLAR,
                    'Premium Bid/Ask': fm.FMT_DOLLAR,
                    'Premium Spread Ratio': fm.FMT_DOLLAR,
                    'Qty': fm.FMT_FLOAT0 + "x",
                    'Delta': fm.FMT_FLOAT,
                  })
                )

    st.title('Vertical Spread')
    st.dataframe(data=df_print)

    return df
