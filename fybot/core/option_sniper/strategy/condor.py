import numpy as np
import pandas as pd
from IPython.display import display
from scipy.spatial.distance import cdist

from core.option_sniper.utility import divider

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def calculate_cdistance_1d(df1, df2, metric='cityblock'):
    XA = df1.values.reshape((-1, 1))
    XB = df2.values.reshape((-1, 1))
    df = pd.DataFrame(cdist(XB, XA, metric), index=df1.index,
                      columns=df2.index)
    return df


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cartesian_product_multi(*dfs):
    idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
    return pd.DataFrame(
        np.column_stack([df.values[idx[:, i]] for i, df in enumerate(dfs)]))


def hack_bid_ask_spreads_put_call(df):
    # -----temporary reference, columns available in options for mining
    # -----if needed, col to be added to cols_meta
    # ['putCall', 'description', 'bid', 'ask', 'last', 'mark', 'bidSize',
    #  'askSize', 'bidAskSize', 'totalVolume', 'volatility', 'delta', 'gamma',
    #  'theta', 'vega', 'rho', 'openInterest', 'timeValue',
    #  'theoreticalOptionValue', 'theoreticalVolatility', 'strikePrice',
    #  'expirationDate', 'daysToExpiration', 'inTheMoney']

    # cols needed for reporting and filtering for short strike
    cols_meta = ['daysToExpiration', 'delta', 'volatility']
    # cols needed for cross calculation
    cols_to_cross = ['bid', 'ask']
    # col needed for post operations between short/long
    cols_for_post = ['strikePrice']
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

    dfx = calculate_cdistance_1d(df[['bid']], df[['ask']])

    # -------------
    spread_dict = {'put': {}, 'call': {}}

    stock_list = dfx.index.get_level_values('stock').unique()

    # prep fields that need to be labeled short & long and added to index
    prep_key = [[('short_' + col), ('long_' + col)] for col in cols_for_post]
    prep_key = [item for sublist in prep_key for item in sublist]
    new_key = cols_meta + prep_key

    for k in spread_dict:
        for s in stock_list:
            try:
                df = dfx.xs((k, s), axis=0)
                df = df.xs((k, s), axis=1)
                df = df.droplevel(cols_meta, axis=1)
                df = pd.DataFrame(df.stack(dropna=True), columns=['premium'])
                df.index = df.index.set_names(new_key)
                df = df.reset_index()
                df['spread'] = df['short_strikePrice'] - df['long_strikePrice']

                if k == 'put':
                    df = df[df['spread'] > 0]
                    df['delta'] = - (df['delta'])
                else:
                    df['spread'] = - (df['spread'])
                    df = df[df['spread'] > 0]
                df['vertical'] = (df['short_strikePrice'].astype(str) + '/' +
                                  df['long_strikePrice'].astype(str))
                df = df.set_index('vertical')
                spread_dict[k][s] = df
            except Exception:
                pass
    return spread_dict


def hunt_condors(pre_condor_dict, s):
    df1 = pre_condor_dict['call'][s].reset_index()
    df2 = pre_condor_dict['put'][s].reset_index()

    ix_call = [('call_' + x) for x in df1.columns]
    ix_put = [('put_' + x) for x in df2.columns]
    new_ix = ix_call + ix_put

    crossed_df = cartesian_product_multi(df1, df2)
    crossed_df.columns = new_ix
    crossed_df['delta'] = crossed_df['call_delta'] + crossed_df[
        'put_delta']
    crossed_df['premium'] = (crossed_df['call_premium']
                             + crossed_df['put_premium'])
    crossed_df['risk'] = crossed_df[['call_spread', 'put_spread']].max(
        axis=1)

    crossed_df['premium_risk_ratio'] = (crossed_df['premium']
                                        / crossed_df['risk'])
    cols = ['put_vertical', 'call_vertical', 'put_delta', 'call_delta',
            'delta', 'put_spread', 'call_spread', 'put_volatility',
            'call_volatility', 'put_premium', 'call_premium', 'premium',
            'risk', 'premium_risk_ratio']
    crossed_df = crossed_df[cols]
    return crossed_df


# Condor Hunter Beta
def condor(df, filters):
    """Condor Hacker (BETA)
    """
    '''
   option_type = filters['option_type']
   min_price = int(filters['min_price'])
   max_price = int(filters['max_price'])
   strike_price_spread = float(
       filters['strike_price_spread'])
   max_risk = float(filters['max_risk'])
   min_return_pct = float(filters['min_return_pct'])
   min_dte = int(filters['min_dte'])
   max_bid_ask_pctl = float(filters['max_bid_ask_pctl'])
    '''
    divider()
    print("CONDOR HACKER\n" +
          "=" * 13)
    max_dte = int(filters['max_dte'])
    max_delta = float(filters['max_delta'])
    min_open_int_pctl = int(filters['min_open_int_pctl'])
    min_volume_pctl = int(filters['min_volume_pctl'])
    top_n = int(filters['top_n'])
    min_bid_size = int(filters['min_bid_size'])
    min_ask_size = int(filters['min_ask_size'])
    max_delta_put = float(filters['max_delta_put'])
    max_delta_call = float(filters['max_delta_call'])

    # ------- Filter options source data before cross-calculation
    if len(df.index) == 0:
        exit("No options tables to analyze")
    df = df[df['inTheMoney'] == False]
    df = df[df['openInterest'] >= min_open_int_pctl]
    df = df[df['bidSize'] >= min_bid_size]
    df = df[df['askSize'] >= min_ask_size]
    df = df[df['totalVolume'] >= min_volume_pctl]

    if len(df.index) == 0:
        print("\n\n*** Nothing to see here! There're no symbols/instruments "
              "that meet the criteria ***")
        divider()
        return

    # ---------- pre-calculate verticals for puts and calls, iterate in df by DTE

    all_condors = {}

    days_to_expiration_list = df['daysToExpiration'].unique()
    days_to_expiration_list[days_to_expiration_list < max_dte]

    for dte in days_to_expiration_list:
        df_in = df[df['daysToExpiration'] == dte]

        # stocks must have both put and calls for condor calculation
        df_ix = df_in.reset_index().groupby(['option_type', 'stock']).count()
        call_keys = df_ix.xs('call').index
        put_keys = df_ix.xs('put').index

        stock_list = list(set(put_keys) & set(call_keys))

        df_in = df_in.loc[stock_list]

        # ---------- pre-calculate verticals for puts and calls and store in dict
        # ---------- create condors, cross-join verticals by stock and join

        pre_condor_dict = hack_bid_ask_spreads_put_call(df_in)

        condor_dict = {}

        for s in stock_list:
            condor_dict[s] = hunt_condors(pre_condor_dict, s)

        condor_hunter = pd.concat(condor_dict)
        condor_hunter = condor_hunter.droplevel(1)

        all_condors[dte] = condor_hunter
    # ---
    condors_df = pd.concat(all_condors)

    # ---------- prep df -  label and screen for output

    condors_df.index = condors_df.index.set_names(['dte', 'stock'])
    condors_df = condors_df.reset_index()

    condors_df = condors_df[condors_df['delta'] <= max_delta]
    condors_df = condors_df[condors_df['put_delta'] <= max_delta_put]
    condors_df = condors_df[condors_df['call_delta'] <= max_delta_call]
    condors_df = condors_df.sort_values('premium_risk_ratio', ascending=False)

    print('Top Paying Condors with Delta <', max_delta * 100)
    display(condors_df.head(top_n))

    days_to_expiration_list = sorted(condors_df.dte.unique())
    print('Top Paying Condors by Days to Expiration with Delta <',
          max_delta * 100)

    for days in days_to_expiration_list:
        display(condors_df[condors_df['dte'] == days].head(top_n))

    return condors_df

# if __name__ == "__main__":
#     premium_hacker()
