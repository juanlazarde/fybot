from sys import exit

import pandas as pd
from numpy import percentile
from scipy.spatial.distance import cdist


def divider():
    pass


def display(arg):
    print(arg)


def calculate_cdistance_1d(df1, df2):
    xa = df1.values.reshape((-1, 1))
    xb = df2.values.reshape((-1, 1))
    df = pd.DataFrame(cdist(xa, xb, 'cityblock'), index=df1.index,
                      columns=df2.index)
    return df


def hack_bid_ask_spreads_put_call(df):
    # cols needed for reporting and filtering for short strike
    cols_meta = ['daysToExpiration', 'delta']
    # cols needed for cross calculation
    cols_to_cross = ['bid', 'ask']
    # col needed for post operations between short/long
    cols_for_post = ['strikePrice', 'mark']
    cols_post_levels = list(range(len(cols_for_post)))
    cols = cols_meta + cols_to_cross + cols_for_post

    # prepares index keys:
    # ['stock','option_type','delta','volatility','meta_columns','strikePrice']
    df = df[cols]
    df = df.reset_index()
    index_keys = ['option_type', 'stock']
    index_keys.extend(cols_meta)
    index_keys.extend(cols_for_post)
    df.set_index(index_keys, inplace=True)

    df = df.drop(columns=['symbol'])

    # calculate all bid - ask spreads (absolute values)
    # cross-calculates all strike prices and stock combinations

    dfx = calculate_cdistance_1d(df[['bid']], df[['ask']])

    # -------------
    spread_dict = {'put': {}, 'call': {}}
    stock_list = dfx.index.get_level_values('stock').unique()

    # prep fields that need to be labeled short & long and added to index
    prep_key_short = [('short_' + col) for col in cols_for_post]
    prep_key_long = [('long_' + col) for col in cols_for_post]
    # prep_key = [item for sublist in prep_key for item in sublist]
    new_key = cols_meta + prep_key_short + prep_key_long

    # separate puts/calls and each stock. Calculate spread of short-long
    # strikes

    dfx = dfx.sort_index()
    for k in spread_dict:
        for s in stock_list:
            try:
                df = dfx.xs((k, s), axis=0)
                df = df.sort_index(axis=1)
                df = df.xs((k, s), axis=1)
                df = df.droplevel(cols_meta, axis=1)
                df = pd.DataFrame(
                    df.stack(dropna=True, level=cols_post_levels),
                    columns=['prem_bid_ask'])
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


def format_table(df_in):
    df_out = df_in.copy()
    df_out['spread'] = df_out['spread'].map("${:,.2f}".format)
    df_out['mark'] = df_out['mark'].map("${:,.2f}".format)
    df_out['bid_ask'] = df_out['bid_ask'].map("${:,.2f}".format)
    df_out['delta'] = df_out['delta'].map("{:,.2f}".format)
    df_out['sprd_ratio'] = df_out['sprd_ratio'].map("{:,.3f}".format)
    df_out['max_profit'] = df_out['max_profit'].map("${:,.2f}".format)
    df_out['risk'] = df_out['risk'].map("${:,.2f}".format)
    df_out['return'] = df_out['return'].map("{0:.2f}%".format)
    df_out['return_day'] = df_out['return_day'].map("{0:.2f}%".format)
    return df_out


def print_table(title, df_in, sort_by, top_n=10):
    print("Top {} {}:\n".format(top_n, title))
    df_out = df_in.sort_values(sort_by, ascending=False)
    df_out = format_table(df_out)
    print(df_out.head(top_n))


def spread(df, filters):
    """Calculate and Retrieve Vertical Spread options on th parameters given.

    Args:
        df (data frame): Option chain table.
        filters (dictionary): Filters for the options to be shown.

    Returns:
        Options table data frame
    """
    print("SPREAD HACKER\n" +
          "=" * 13)
    # variable assignment
    option_type = filters['option_type'].lower().strip()
    option_itm = False
    min_price = float(filters['min_price'])
    max_price = float(filters['max_price'])
    strike_price_spread = float(filters['strike_price_spread'])
    max_risk = float(filters['max_risk'])
    min_return_pct = float(filters['min_return_pct'])
    max_dte = int(filters['max_dte'])
    min_dte = int(filters['min_dte'])
    max_delta = float(filters['max_delta'])
    min_open_int_pctl = int(filters['min_open_int_pctl'])
    min_volume_pctl = int(filters['min_volume_pctl'])
    max_bid_ask_pctl = float(filters['max_bid_ask_pctl'])
    top_n = int(filters['top_n'])

    # ------- Filter options source data before cross-calculation
    # clean table
    df = df[df['delta'] != 'NaN']
    df = df[df['openInterest'] > 0]
    df = df[df['totalVolume'] > 0]
    df = df[df['bid'] > 0]
    df = df[df['ask'] > 0]

    # filter
    open_int_pctl = percentile(df['openInterest'], min_open_int_pctl)
    volume_pctl = percentile(df['totalVolume'], min_volume_pctl)
    if option_type in ['put', 'call']:
        df = df[df['putCall'].str.lower() == option_type]
    df = df[df['inTheMoney'] == option_itm]
    df = df[(df['delta'] <= max_delta) &
            (df['delta'] >= -max_delta)]
    df = df[df['totalVolume'] >= volume_pctl]
    df = df[df['openInterest'] >= open_int_pctl]
    df = df[(df['daysToExpiration'] >= min_dte) &
            (df['daysToExpiration'] <= max_dte)]
    df = df[(df['strikePrice'] >= min_price) &
            (df['strikePrice'] <= max_price)]

    # calculated columns
    df['max_bid_ask_pctl'] = 1 - df['bid'] / df['ask']
    max_bid_ask = percentile(df['max_bid_ask_pctl'], max_bid_ask_pctl)
    df = df[df['max_bid_ask_pctl'] <= max_bid_ask]

    # exit if table is empty
    if len(df.index) == 0:
        print("\n\n*** Nothing to see here! There're no symbols/instruments "
              "that meet the criteria ***")
        return

    # ------- process spreads
    spread_dict = {}
    days_to_expiration_list = sorted(df['daysToExpiration'].unique())

    for dte in days_to_expiration_list:
        df_by_dte = df[df['daysToExpiration'] == dte]
        pre_spread_dict = hack_bid_ask_spreads_put_call(df_by_dte)

        # overwrite dictionaries with concatenated df across all symbols
        if pre_spread_dict['call'] == 0 and pre_spread_dict['put'] == 0:
            exit("Options for mining are empty")

        for i in ['put', 'call']:
            if len(pre_spread_dict[i]) != 0:
                pre_spread_dict[i] = pd.concat(pre_spread_dict[i])
            else:
                del pre_spread_dict[i]

        spread_dict[dte] = pd.concat(pre_spread_dict)

    spread_df = pd.concat(spread_dict)
    spread_df = spread_df.droplevel(level=0)

    # ------- output
    # calculate all metrics
    # TODO: Decide which is better prem_sprd_ratio or return_pct
    spread_df['prem_sprd_ratio'] = \
        spread_df['prem_mark'] / spread_df['spread']
    # TODO: It's not 100 it should be the Multiplier, it's in df but not in
    #  spread_df, so we need to vlookup the symbol from spread_df in df and
    #  retrieve the multiplier
    spread_df['max_profit'] = 100 * spread_df['prem_mark']
    spread_df['risk'] = 100 * (spread_df['spread'] - spread_df['prem_mark'])
    spread_df['return'] = \
        100 * spread_df['max_profit'] / spread_df['risk']
    spread_df['return_day'] = \
        spread_df['return'] / (spread_df['daysToExpiration'] + .00001)

    spread_df = spread_df[spread_df['spread'] <= strike_price_spread]
    spread_df = spread_df[spread_df['return'] >= min_return_pct]
    spread_df = spread_df[spread_df['risk'] <= max_risk]
    spread_df = spread_df[spread_df['risk'] >= spread_df['max_profit']]

    spread_df = spread_df.sort_values('return', ascending=False)
    ix_names = ['putCall', 'symbol', 'vertical']
    spread_df.index.set_names(ix_names, inplace=True)
    spread_df = spread_df.reset_index()
    cols = ['symbol', 'putCall', 'vertical', 'daysToExpiration', 'spread',
            'prem_bid_ask', 'prem_mark', 'prem_sprd_ratio', 'delta',
            'max_profit', 'risk', 'return', 'return_day']
    spread_df = spread_df[cols]

    spread_df.rename(columns={'putCall': 'type',
                              'daysToExpiration': 'dte',
                              'prem_bid_ask': 'bid_ask',
                              'prem_mark': 'mark',
                              'prem_sprd_ratio': 'sprd_ratio'},
                     inplace=True)

    # output
    symbols_list = ', '.join(sorted(spread_df.symbol.unique()))
    print("Analyzing: {}".format(symbols_list) +
          "\nShowing: {}".format(option_type) +
          "\nMax Risk: ${:,.2f}".format(max_risk) +
          "\nPrice range: ${:,.2f} & ${:,.2f}".format(min_price, max_price) +
          "\nDTE range: {} & {} days".format(min_dte, max_dte) +
          "\nMax Delta <= {} (~{:.0f}% Prob OTM)".format(
              max_delta, 100 * (1 - max_delta)),
          "\nBid-Ask Ratio <= {:,.2f} (1-bid/ask)".format(max_bid_ask) +
          "\nMin Volume & Op. Int. >= {} & {}".format(
              volume_pctl, open_int_pctl) +
          "\n")

    if len(spread_df.symbol.unique()):
        print_table(title="Paying Spreads Sorted by Return",
                    df_in=spread_df,
                    sort_by='return',
                    top_n=top_n
                    )
        print_table(title="Paying Spreads Sorted by Daily Return",
                    df_in=spread_df,
                    sort_by='return_day',
                    top_n=top_n
                    )
        print_table(title="Paying Spreads Sorted by Max Profit",
                    df_in=spread_df,
                    sort_by='max_profit',
                    top_n=top_n
                    )

        print("Top {} Paying Spreads Grouped by Symbol, Sorted by "
              "Returns\n".format(top_n))
        out_df1 = spread_df.copy()
        sorted_symbols = sorted(out_df1.symbol.unique())
        out_df1 = format_table(out_df1)
        top_option_for_symbol = {}
        for symbol in sorted_symbols:
            print("Symbol: {}".format(symbol))
            symbol_df = out_df1[out_df1['symbol'] == symbol]
            print(symbol_df.head(top_n))
            top_option_for_symbol[symbol] = symbol_df.iloc[0]
            divider()

        if len(top_option_for_symbol) > 1:
            print("Top Paying Spreads Grouped by Symbol, Sorted by "
                  "Daily Return\n")
            top_symbol_df = pd.concat(top_option_for_symbol, axis=1)
            top_symbol_df = top_symbol_df.T
            top_symbol_df.sort_values('return_day', ascending=False,
                                      inplace=True)
            top_symbol_df.drop(columns=['symbol'], inplace=True)
            top_symbol_df = top_symbol_df.rename_axis('symbol')
            display(top_symbol_df)
            print("\n-----> Shopping list: " + ', '.join(top_symbol_df.index))
            divider()

        print("Top {} Paying Spreads Grouped by Days to Expiration, Sorted by "
              "Returns\n".format(top_n))
        out_df2 = spread_df.copy()
        days_to_expiration_list = sorted(out_df2.dte.unique())
        out_df2 = format_table(out_df2)
        for days in days_to_expiration_list:
            print("Days to Expiration: {}".format(days))
            display(out_df2[out_df2['dte'] == days].head(top_n))
            divider()
    else:
        print("\n\n*** Nothing to see here! There're no symbols/instruments "
              "that meet the criteria ***")
        divider()
    return df
