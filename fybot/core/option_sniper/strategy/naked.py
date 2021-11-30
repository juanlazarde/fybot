from typing import Dict, Any
import pandas as pd
import streamlit as st
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


def naked(df_in: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Retrieve naked options based on the parameters given.

    :param df_in: Option chain table.
    :param filters: Filters for the options to be shown.
    :return: Options table dataframe.
    """
    # variable assignment
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
        'min_open_int_pctl': float(filters['min_open_int_pctl']) / 100,
        'min_volume_pctl': float(filters['min_volume_pctl']) / 100,
        'max_bid_ask_pctl': float(filters['max_bid_ask_pctl']) / 100
    }
    df: pd.DataFrame = df_in.copy()
    del filters, df_in

    # Clean table.
    # Comment out columns to keep.
    df.drop(inplace=True, columns=[
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
    ])
    df.reset_index(inplace=True)
    df.drop(columns=['symbol'], inplace=True)  # Actual option's symbol.
    df.sort_values(by='stock', ascending=True, inplace=True)

    # Filter: Pass 1. Before calculation.
    df = df[(df['delta'] != 'NaN')
            & (df['openInterest'] > 0)
            & (df['totalVolume'] > 0)
            & (df['bid'] > 0)
            & (df['ask'] > 0)
            ]

    _shp = df.shape[0]  # Impact log initialization.
    impact: Dict[str, str] = {'Starting size': f"{_shp}"}

    # filter
    if d['option_type'] in ['put', 'call']:
        df = df[df['option_type'].str.lower() == d['option_type']]
    impact['Option type'] = f"{df.shape[0]/_shp:.0%}"

    if d['premium_type'] == 'credit':
        df = df[(df['inTheMoney'] == d['option_itm'])]
    impact['ITM'] = f"{df.shape[0] / _shp:.0%}"

    df = df[(df['delta'] <= d['max_delta'])
            & (df['delta'] >= -d['max_delta'])]
    impact['Delta'] = f"{df.shape[0] / _shp:.0%}"

    df = df[(df['daysToExpiration'] >= d['min_dte'])
            & (df['daysToExpiration'] <= d['max_dte'])]
    impact['DTE'] = f"{df.shape[0] / _shp:.0%}"

    # Calculated columns.
    df['bid_ask_pct'] = df['ask']/df['bid'] - 1
    df['bid_ask_rank'] = (df.groupby('stock')['bid_ask_pct']
                          .rank(pct=True, ascending=False, method='dense'))
    df['open_int_rank'] = (df.groupby('stock')['openInterest']
                           .rank(pct=True, ascending=True, method='dense'))
    df['volume_rank'] = (df.groupby('stock')['totalVolume']
                         .rank(pct=True, ascending=True, method='dense'))
    df['break_even'] = df['strikePrice'] - df['mark']
    df['margin_requirement'] = df['strikePrice'] * df['multiplier']
    df['max_profit'] = df['mark'] * df['multiplier']
    df['risk'] = df['strikePrice']*df['multiplier'] - df['max_profit']
    df['return'] = df['max_profit'] / df['risk']
    df['return_day'] = df['return'] / (df['daysToExpiration'] + .00001)
    df['quantity'] = d['max_risk'] / df['risk']
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
    df = df[df['bid_ask_rank'] >= d['max_bid_ask_pctl']]
    impact['Bid/Ask'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['open_int_rank'] >= d['min_open_int_pctl']]
    impact['Open Interest'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['volume_rank'] >= d['min_volume_pctl']]
    impact['Volume'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['margin_requirement'] <= d['margin_requirement']]
    impact['Margin requirement'] = f"{df.shape[0] / _shp:.0%}"

    df = df[df['risk'] <= d['max_risk']]
    impact['Risk'] = f"{df.shape[0] / _shp:.0%}"

    if d['premium_type'] == 'credit':
        df = df[df['return'] >= d['min_return_pct']]
    impact['Return'] = f"{df.shape[0] / _shp:.0%}"

    if d['premium_type'] == 'credit':
        df = df[df['risk'] >= df['max_profit']]
    impact['Risk>Profit'] = f"{df.shape[0] / _shp:.0%}"

    # Exit if table is empty.
    if len(df.index) == 0:
        st.warning("**Nothing to see here!** Criteria not met")
        st.write("**Here's the impact of the filters you've set:**")
        _impact_results(impact)
        st.stop()

    # Formatting the table.
    df.sort_values(by='return', ascending=False, inplace=True)
    df.set_index('search', inplace=True)
    df = df[[
        'return',
        'return_day',
        'max_profit',
        'risk',
        'quantity',
        'margin_requirement',
        'mark',
        'break_even',
        'delta',
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
        'mark': 'Mark',
        'break_even': 'Break Even',
        'delta': 'Delta',
        # 'volatility': 'IV',
        'daysToExpiration': 'DTE',
        'stock': 'Stock',
    })

    # Display results.
    df_print = (df.style
                  .background_gradient(
                    axis=0,
                    subset=['Return', 'Daily Return'])
                  .highlight_max(
                    subset=['Return', 'Max Profit', 'Daily Return'],
                    color=fm.HI_MAX_COLOR)
                  .highlight_min(
                    subset=['Return', 'Max Profit', 'Daily Return'],
                    color=fm.HI_MIN_COLOR)
                  .format({
                    'Return': fm.PERCENT2,
                    'Daily Return': fm.PERCENT2,
                    'Max Profit': fm.DOLLAR,
                    'Risk': fm.DOLLAR,
                    'Margin Req\'d': fm.DOLLAR,
                    'Break Even': fm.DOLLAR,
                    'Qty': fm.FLOAT0 + "x",
                    'Mark': fm.DOLLAR,
                    'Delta': fm.FLOAT,
                    'DTE': fm.FLOAT0,
                    # 'IV': fm.FLOAT0
                  })
                )

    st.header('Naked Options')
    st.dataframe(data=df_print)

    return df
