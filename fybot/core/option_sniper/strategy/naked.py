from typing import Dict

import pandas as pd
import streamlit as st
import core.formatter as fm


def naked(df_in: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Retrieve naked options based on the parameters given.

    :param df_in: Option chain table.
    :param filters: Filters for the options to be shown.
    :return: Options table dataframe.
    """
    # variable assignment
    option_type = filters['option_type'].lower().strip()
    # TODO: Make Debit/credit premium type work
    premium_type = filters['premium_type'].lower().strip()
    option_itm = False
    max_risk = float(filters['max_risk'])
    min_return_pct = float(filters['min_return_pct'])/100
    max_dte = int(filters['max_dte'])
    min_dte = int(filters['min_dte'])
    max_delta = float(filters['max_delta'])
    min_open_int_pctl = float(filters['min_open_int_pctl'])/100
    min_volume_pctl = float(filters['min_volume_pctl'])/100
    max_bid_ask_pctl = float(filters['max_bid_ask_pctl'])/100

    # ------- Filter options source data
    # clean table
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

    _shp = df.shape[0]  # Impact log initialization.
    impact: Dict[str, str] = {'Starting size': f"{_shp}"}

    # filter
    if option_type in ['put', 'call']:
        df = df[df['option_type'].str.lower() == option_type]
    impact['Option type'] = f"{df.shape[0]/_shp:.0%}"

    if premium_type == 'credit':
        df = df[(df['inTheMoney'] == option_itm)]
    impact['ITM'] = f"{df.shape[0] / _shp:.0%}"

    df = df[(df['delta'] <= max_delta)
            & (df['delta'] >= -max_delta)]
    impact['Delta'] = f"{df.shape[0] / _shp:.0%}"

    df = df[(df['daysToExpiration'] >= min_dte)
            & (df['daysToExpiration'] <= max_dte)]
    impact['DTE'] = f"{df.shape[0] / _shp:.0%}"

    # calculated columns
    df['bid_ask_pct'] = (df['ask']/df['bid'] - 1)
    df['bid_ask_rank'] = (df.groupby('stock')['bid_ask_pct']
                          .rank(pct=True, ascending=False, method='dense'))
    df['open_int_rank'] = (df.groupby('stock')['openInterest']
                           .rank(pct=True, ascending=True, method='dense'))
    df['volume_rank'] = (df.groupby('stock')['totalVolume']
                         .rank(pct=True, ascending=True, method='dense'))

    if premium_type == 'credit':
        df['max_profit'] = df['mark'] * df['multiplier']
        df['risk'] = df['strikePrice'] * df['multiplier'] - df['max_profit']
        df['return'] = df['max_profit'] / df['risk']
        df['return_day'] = df['return'] / (df['daysToExpiration'] + .00001)
    else:
        df['max_profit'] = "infinite"
        df['risk'] = df['mark'] * df['multiplier']
        df['return'] = "infinite"
        df['return_day'] = "infinite"

    df['quantity'] = max_risk / df['risk']
    df['search'] = \
        df['description'].str.split(' ').str[0].astype(str) + " " + \
        df['description'].str.split(' ').str[2].astype(str) + " " + \
        df['description'].str.split(' ').str[1].astype(str) + " " + \
        df['description'].str.split(' ').str[3].str[::3].astype(str) + \
        " (" + df['daysToExpiration'].astype(str) + ") " + \
        df['option_type'].str.upper().astype(str) + " " + \
        df['strikePrice'].astype(str)

    # more filters
    df = df[df['bid_ask_rank'] >= max_bid_ask_pctl]
    impact['Bid/Ask'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['open_int_rank'] >= min_open_int_pctl]
    impact['Open Interest'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['volume_rank'] >= min_volume_pctl]
    impact['Volume'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['risk'] <= max_risk]
    impact['Risk'] = f"{df.shape[0] / _shp:.0%}"

    if premium_type == 'credit':
        df = df[df['return'] >= min_return_pct]
    impact['Return'] = f"{df.shape[0] / _shp:.0%}"

    if premium_type == 'credit':
        df = df[df['risk'] >= df['max_profit']]
    impact['Risk>Profit'] = f"{df.shape[0] / _shp:.0%}"

    # exit if table is empty
    if len(df.index) == 0:
        st.warning("**Nothing to see here!** Criteria not met")
        st.write("**Here's the impact of the filters you've set:**")
        st.table(pd.DataFrame(impact.items(), ['Filter', '% of Total'])
                   .set_index('Filter'))
        return

    # there are no calculations here; as opposed to spread hacker, because
    # the premium is just the bid price.

    # Formatting the table
    if premium_type == 'credit':
        df.sort_values(by='return', ascending=False, inplace=True)
    else:
        df.sort_values(by='risk', ascending=True, inplace=True)
    df.set_index('search', inplace=True)

    _cols = [
        'return',
        'return_day',
        'max_profit',
        'risk',
        'quantity',
        'mark',
        'delta',
        'volatility',
        'daysToExpiration',
        'stock'
    ]
    df = df[_cols]

    _cols = {
        'daysToExpiration': 'DTE',
        'mark': 'Mark',
        'delta': 'Delta',
        'volatility': 'IV',
        'max_profit': 'Max Profit',
        'risk': 'Risk',
        'quantity': 'Qty',
        'return': 'Return',
        'return_day': 'Daily Return',
    }
    df = df.rename(columns=_cols)

    # display results
    df_print = (df.style.background_gradient(
                    axis=0,
                    subset=['Return', 'Daily Return', 'IV'])
                  .highlight_max(
                    subset=['Return', 'Max Profit', 'IV', 'Daily Return'],
                    color=fm.HI_MAX_COLOR)
                  .highlight_min(
                    subset=['Return', 'Max Profit', 'IV', 'Daily Return'],
                    color=fm.HI_MIN_COLOR)
                  .format({
                    'Return': fm.FMT_PERCENT2,
                    'Daily Return': fm.FMT_PERCENT2,
                    'Max Profit': fm.FMT_DOLLAR,
                    'Risk': fm.FMT_DOLLAR,
                    'Qty': fm.FMT_FLOAT0 + "x",
                    'Mark': fm.FMT_DOLLAR,
                    'Delta': fm.FMT_FLOAT,
                    'IV': fm.FMT_FLOAT0
                  })
                )

    st.title('Naked Options')
    st.dataframe(data=df_print)

    return df
