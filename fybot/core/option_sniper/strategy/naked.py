import pandas as pd
import streamlit as st


def naked(df, filters: dict):
    """Retrieve naked options based on the parameters given.

    :param df: Option chain table.
    :param filters: Filters for the options to be shown.
    :return: Options table dataframe.
    """
    # variable assignment
    option_type = filters['option_type'].lower().strip()
    option_itm = False
    min_price = float(filters['min_price'])
    max_price = float(filters['max_price'])
    max_risk = float(filters['max_risk'])
    min_return_pct = float(filters['min_return_pct'])
    max_dte = int(filters['max_dte'])
    min_dte = int(filters['min_dte'])
    max_delta = float(filters['max_delta'])
    min_delta = float(filters['min_delta'])
    min_open_int = int(filters['min_open_int'])
    min_volume = int(filters['min_volume'])
    max_bid_ask_pct = float(filters['max_bid_ask_pct'])
    top_n = int(filters['top_n'])

    # ------- Filter options source data
    # clean table
    df = df[df['delta'] != 'NaN']
    df = df[df['openInterest'] > 0]
    df = df[df['totalVolume'] > 0]
    df = df[df['bid'] > 0]
    df = df[df['ask'] > 0]

    # df2 will be used only in the register below, along with df3
    df2 = df.copy()

    # filter
    if option_type in ['put', 'call']:
        df = df[df['putCall'].str.lower() == option_type]
    df = df[df['inTheMoney'] == option_itm]
    df = df[(df['delta'] <= max_delta) &
            (df['delta'] >= -max_delta)]
    if min_delta < max_delta:
        df = df[(df['delta'] <= min_delta) &
                (df['delta'] >= -min_delta)]
    df = df[df['openInterest'] >= min_open_int]
    df = df[df['totalVolume'] >= min_volume]
    df = df[(df['daysToExpiration'] >= min_dte) &
            (df['daysToExpiration'] <= max_dte)]
    df = df[(df['strikePrice'] >= min_price) &
            (df['strikePrice'] <= max_price)]

    # calculated columns
    df['bid_ask_pct'] = (df['ask'] / df['bid']) - 1
    df['max_profit'] = df['mark'] * df['multiplier']
    df['risk'] = df['strikePrice'] * df['multiplier'] - df['max_profit']
    df['return'] = 100 * df['max_profit'] / df['risk']
    df['return_day'] = df['return'] / (df['daysToExpiration'] + .00001)
    df['search'] = \
        df['description'].str.split(' ').str[0].astype(str) + " " + \
        df['description'].str.split(' ').str[2].astype(str) + " " + \
        df['description'].str.split(' ').str[1].astype(str) + " " + \
        df['description'].str.split(' ').str[3].str[::3].astype(str) + \
        " (" + df['daysToExpiration'].astype(str) + ") " + \
        df['putCall'].astype(str) + " " + \
        df['strikePrice'].astype(str)

    # df3 will be used only in the register below, along with df2
    df3 = df.copy()

    # more filters
    df = df[df['bid_ask_pct'] <= max_bid_ask_pct]
    df = df[df['risk'] <= max_risk]
    df = df[df['return'] >= min_return_pct]
    df = df[df['risk'] >= df['max_profit']]

    # Register to capture the impact of the filters on the database
    register = {
        'Original Options Available': f'{df2.shape[0]}',
        'Option type': '{:.0f}% {}'.format(
            df2[df2['putCall'].str.lower() == option_type
                ].shape[0] / df2.shape[0] * 100, option_type),
        'Days to expiration': '{:.0f}%'.format(
            df2[(df2['daysToExpiration'] >= min_dte) &
                (df2['daysToExpiration'] <= max_dte)
                ].shape[0] / df2.shape[0] * 100),
        'Option ITM': '{:.0f}%'.format(
            df2[df2['inTheMoney'] == option_itm
                ].shape[0] / df2.shape[0] * 100),
        'Price': '{:.0f}%'.format(
            df2[(df2['strikePrice'] >= min_price) &
                (df2['strikePrice'] <= max_price)
                ].shape[0] / df2.shape[0] * 100),
        'Open Interest': '{:.0f}%'.format(
            df2[df2['openInterest'] >= min_open_int
                ].shape[0] / df2.shape[0] * 100),
        'Volume': '{:.0f}%'.format(
            df2[df2['totalVolume'] >= min_volume
                ].shape[0] / df2.shape[0] * 100),
        'Delta': '{:.0f}%'.format(
            df2[(df2['delta'] <= max_delta) &
                (df2['delta'] >= -max_delta)
                ].shape[0] / df2.shape[0] * 100),
        'Bid Ask ratio (pctl)': '{:.0f}%'.format(
            df3[df3['bid_ask_pct'] <= max_bid_ask_pct
                ].shape[0] / df3.shape[0] * 100),
        'Risk': '{:.0f}%'.format(
            df3[df3['risk'] <= max_risk
                ].shape[0] / df3.shape[0] * 100),
        'Return': '{:.0f}%'.format(
            df3[df3['return'] >= min_return_pct
                ].shape[0] / df3.shape[0] * 100)
    }
    del df2, df3

    # exit if table is empty
    if len(df.index) == 0:
        st.warning("**Nothing to see here!** Criteria not met")
        st.write("**Here's the impact of the filters you've set:**")
        register_df = pd.DataFrame({'Filter': k, '% of Total': v} for k, v in register.items())
        register_df = register_df.set_index('Filter')
        st.table(register_df)
        return

    # there are no calculations here; as opposed to spread hacker, because
    # the premium is just the bid price.

    # Formatting the table
    df.sort_values(by='return', ascending=False, inplace=True)
    df.drop(columns=['symbol'], inplace=True)
    df.reset_index(inplace=True)
    # df.set_index('symbol', inplace=True)
    df.set_index('search', inplace=True)

    # cols = ['stock', 'option_type', 'strikePrice', 'daysToExpiration',
    #         'bid', 'mark', 'delta', 'volatility', 'max_profit', 'risk',
    #         'return', 'return_day', 'search']
    # df = df[cols]
    # df.rename(columns={'option_type': 'type',
    #                    'strikePrice': 'strike',
    #                    'daysToExpiration': 'dte',
    #                    'volatility': 'vol'},
    #           inplace=True)
    cols = ['return', 'return_day', 'max_profit', 'risk', 'mark',
            'delta', 'volatility', 'daysToExpiration', 'stock']
    df = df[cols]

    # df['mark'] = df['mark'].map("{:,.2f}".format)
    # #    df['bid'] = df['bid'].map("${:,.2f}".format)
    # df['daysToExpiration'] = df['daysToExpiration'].map("{:#}".format)
    # df['volatility'] = df['volatility'].map("{:,.2f}".format)
    # df['delta'] = df['delta'].map("{:,.2f}".format)
    # df['max_profit'] = df['max_profit'].map("{:,.2f}".format)
    # df['return'] = df['return'].map("{0:.2f}".format)
    # df['return_day'] = df['return_day'].map("{0:.2f}".format)
    # # df['risk'] = df['risk'].map("{:,.2f}".format)

    df.round({
        'return': 2,
        'return_day': 2,
        'max_profit': 2,
        'risk': 2,
        'mark': 2,
        'delta': 2,
        'volatility': 0
    })

    df.rename(columns={
        'daysToExpiration': 'DTE',
        'mark': 'Mark',
        'delta': 'Delta',
        'volatility': 'IV',
        'max_profit': 'Max Profit',
        'risk': 'Risk',
        'return': 'Return',
        'return_day': 'Daily Return',
    },
        inplace=True
    )

    # output
    return df
