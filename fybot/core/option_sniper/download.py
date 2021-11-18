"""Options functions for the Sniper."""
from datetime import date
import streamlit as st

import numpy as np
import pandas as pd
from tda.client.synchronous import Client


def get_all_tables(tda_client,
                   symbol_list: list,
                   max_dte: date,
                   min_dte: date):
    """Concatenates option chains in the symbol list, within date range.

    Args:
        tda_client: TDA API client information.
        symbol_list: List of symbols to be analyzed.
        max_dte: Max date to expiration to be downloaded.
        min_dte: Min date to expiration to be downloaded.

    Returns:
        Dataframe with Option Chains. Blank if market is closed.
    """
    st.write(f"Loading Tables between {min_dte:%b-%d-%Y} & {max_dte:%b-%d-%Y} for:")
    options_df = None
    options = {}
    for symbol in symbol_list:
        try:
            options[symbol] = get_one_table(
                tda_client=tda_client,
                symbol=symbol,
                max_dte=max_dte,
                min_dte=min_dte
            )
        except Exception as e:
            st.error(f"   - {symbol} Failed. Error: {e}")
    if len(options) != 0:
        options_df = pd.concat(options)
        options_df.index.set_names(
            ['stock', 'option_type', 'symbol'],
            inplace=True
        )
    return options_df


def get_one_table(tda_client,
                  symbol: str,
                  max_dte: date,
                  min_dte: date):
    """Downloads and creates options table per symbol.

    Args:
        tda_client: TDA API client information.
        symbol: List of symbols to be analyzed.
        max_dte: Max date to expiration to be downloaded.
        min_dte: Min date to expiration to be downloaded.

    Returns:
        options_table (data frame):  puts and calls for all dates within DTE.
    """
    options = tda_client.get_option_chain(
        symbol=symbol,
        contract_type=None,
        strike_count=None,
        include_quotes=None,
        strategy=None,
        interval=None,
        strike=None,
        strike_range=Client.Options.StrikeRange.ALL,
        # ALL, IN_THE_MONEY, OUT_OF_THE_MONEY
        from_date=min_dte,
        to_date=max_dte,
        volatility=None,
        underlying_price=None,
        interest_rate=None,
        days_to_expiration=None,
        exp_month=None,
        option_type=None)

    options = options.json()

    st.write(f"   {symbol}, ${options['underlyingPrice']:.2f}")

    if options['status'] == "SUCCESS":
        try:
            calls = pd.json_normalize(
                options['callExpDateMap'],
                record_prefix='Prefix.'
            )
            puts = pd.json_normalize(
                options['putExpDateMap'],
                record_prefix='Prefix.'
            )

            del options

            # unpack json format
            tables = {
                'call':
                    pd.DataFrame.from_dict(
                        calls.loc[0][0][0],
                        orient='index'
                    ),
                'put':
                    pd.DataFrame.from_dict(
                        puts.loc[0][0][0],
                        orient='index'
                    )
            }

            for i in range(len(calls.columns)):
                if len(calls.loc[0][i]) == 1:
                    if np.logical_not(pd.isna(calls.loc[0][i])):
                        tables['call'][i] = pd.DataFrame.from_dict(
                            calls.loc[0][i][0], orient='index')
                else:
                    if np.logical_not(pd.isna(calls.loc[0][i][0])):
                        tables['call'][i] = pd.DataFrame.from_dict(
                            calls.loc[0][i][0], orient='index')[0]
                if len(puts.loc[0][i]) == 1:
                    if np.logical_not(pd.isna(puts.loc[0][i])):
                        tables['put'][i] = pd.DataFrame.from_dict(
                            puts.loc[0][i][0], orient='index')
                else:
                    if np.logical_not(pd.isna(puts.loc[0][i][0])):
                        tables['put'][i] = pd.DataFrame.from_dict(
                            puts.loc[0][i][0], orient='index')[0]

            tables['call'].columns = tables['call'].loc['symbol']
            tables['put'].columns = tables['put'].loc['symbol']

            options_table = pd.concat(tables, axis=1)

            return options_table.T

        except Exception as e:
            st.error(f"     - {symbol} Failed. Error: {e}")
    else:
        raise Exception(f"Failed to Download. Perhaps {symbol} doesn't exist.")
