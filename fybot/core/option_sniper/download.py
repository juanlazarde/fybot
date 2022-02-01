"""Options functions for the Sniper."""
from datetime import date

import httpx
import streamlit as st
import pandas as pd

from tda.client.synchronous import Client

from core.utils import optimize_pd


def get_quotes(tda, symbol_list: list):
    """
    Get current price data from TDA

    Source: https://developer.tdameritrade.com/quotes/apis/get/marketdata/quotes

    :return: Current underlying stock price merged into the options table.
    """
    q = tda.client.get_quotes(symbol_list)
    assert q.status_code == httpx.codes.OK, q.raise_for_status()

    prep = [v for k, v in q.json().items()]
    quote = pd.DataFrame(prep)
    quote.rename(columns={'symbol': 'stock'}, inplace=True)
    return optimize_pd(quote)


def get_last_price(tda, symbol_list: list):
    quote = get_quotes(tda, symbol_list)
    return quote[['stock', 'lastPrice']].to_dict(orient='records')


def get_all_tables(tda_client,
                   symbol_list: list,
                   max_dte: date,
                   min_dte: date):
    """Concatenates option chains in the symbol list, within date range.

    :param tda_client: TDA API client information.
    :param symbol_list: List of symbols to be analyzed.
    :param max_dte: Maximum date to expiration to be downloaded.
    :param min_dte: Minimum date to expiration to be downloaded.
    :return: Dataframe with Option Chains. Blank if market is closed.
    """
    # TODO: Filter out stocks by underlying price

    option_collection = []
    for symbol in symbol_list:
        option_collection.append(
            get_one_table(
                tda_client=tda_client,
                symbol=symbol,
                max_dte=max_dte,
                min_dte=min_dte
            )
        )

    final_options = pd.concat(option_collection)
    return cleanup_option_table(final_options)

    # TODO: Delete the line below.
    # final_options.set_index(['stock', 'option_type', 'symbol'], inplace=True)


def get_one_table(tda_client,
                  symbol: str,
                  max_dte: date,
                  min_dte: date):
    """Downloads and creates options table per symbol.

    :param tda_client: TDA API client information.
    :param symbol: Symbol to be analyzed.
    :param max_dte: Maximum date to expiration to be downloaded.
    :param min_dte: Minimum date to expiration to be downloaded.
    :return: Dataframe with Option Chains for one Symbol.
    """
    r = tda_client.get_option_chain(
        symbol=symbol,
        contract_type=None,
        strike_count=None,
        include_quotes=None,
        strategy=None,
        interval=None,
        strike=None,
        strike_range=Client.Options.StrikeRange.ALL,
        from_date=min_dte,
        to_date=max_dte,
        volatility=None,
        underlying_price=None,
        interest_rate=None,
        days_to_expiration=None,
        exp_month=None,
        option_type=None)

    assert r.status_code == httpx.codes.OK, r.raise_for_status()

    options = r.json()

    st.write(f"{symbol}, ${options['underlyingPrice']:.2f}")
    return unpack_option_table(options)


def unpack_option_table(options):
    """
    Unpacks Option Chains from TDA.

    :param options: Option chain in JSON format.
    :return: Option chains; puts and calls, merged into one DataFrame/
    """
    opt = options.copy()
    ret = []
    for _date in opt["callExpDateMap"]:
        for strike in opt["callExpDateMap"][_date]:
            ret.extend(opt["callExpDateMap"][_date][strike])
    for _date in opt["putExpDateMap"]:
        for strike in opt["putExpDateMap"][_date]:
            ret.extend(opt["putExpDateMap"][_date][strike])
    df = pd.DataFrame(ret)
    df['stock'] = opt['symbol']
    return df


def cleanup_option_table(options: pd.DataFrame):
    """
    Cleanup option chain and optimize DataFrame.

        1) Filter out columns,
        2) Filter out rows,
        3) Lower-range numerical and categories dtypes,
        4) Sparse Columns.
        5) Re-index.
     -> 6) CUSTOMIZE OPTION (i.e. TDA-specific changes)

    :return: Optimized dataframe
    """
    if options is None or options.empty:
        return

    opt = options.copy()
    # Measure it:
    # print(f"{sum(opt.memory_usage(deep=True))/1024:,.2f}Kb")
    opt.replace(["", " ", None, "NaN"], float('NaN'), inplace=True)
    opt.dropna(axis='columns', how='all', inplace=True)
    opt = optimize_pd(opt, deal_with_na='drop', verbose=False)

    # TDA-specific changes.
    opt['volatility'] = opt['volatility'] / 100.
    opt = opt[
        (opt['openInterest'] > 0) &
        (opt['totalVolume'] > 0) &
        (opt['bid'] > 0) &
        (opt['ask'] > 0) &
        (opt['volatility'].astype('float').round(8).values > 0)
    ]
    opt.rename(columns={'putCall': 'option_type'}, inplace=True)
    opt['option_type'] = opt['option_type'].str.lower()

    time_cols = [
        "tradeTimeInLong",
        "quoteTimeInLong",
        "expirationDate",
        "lastTradingDay"
    ]
    for col in time_cols:
        opt[col] = pd.to_datetime(opt[col], unit='ms')

    opt.sort_values(
        by=['stock', 'daysToExpiration', 'option_type', 'strikePrice'],
        inplace=True
    )
    opt.set_index('symbol', inplace=True)
    # print(f"{sum(opt.memory_usage(deep=True))/1024:,.2f}Kb")
    return opt


def get_risk_free_rate():
    # Get 10-yr risk-free rate from FRED.
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    csv = pd.read_csv(url)
    value = csv['DGS10'].values[-1]
    return float(value) / 100
