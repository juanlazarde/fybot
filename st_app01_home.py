# app1.py
import streamlit as st
import pandas as pd
from app import *
from functions import *

# df = pd.read_pickle(S.data_file)

def app():
    st.title('Financial Scanner - Streamlit Beta')

    # ----------------------------------------------------------------
    st.header('Data')
    if not os.path.isfile(S.data_file):
        if not st.button('Download Data'):
            st.warning('Data file not found. Press button to download data')
            st.stop()
        else:
            Snapshot()
            # modified_time = os.stat(S.data_file).st_mtime
            # last_modified = datetime.datetime.fromtimestamp(modified_time)
            # last_modified_fmt = last_modified.strftime('%m-%d-%Y %H:%M %p')
            # text = 'Data file last refreshed {}'.format(last_modified_fmt
    else:

        if st.button('Refresh Data'):
            Snapshot()

    price_df = pd.read_pickle(S.data_file)

    modified_time = os.stat(S.data_file).st_mtime
    last_modified = datetime.datetime.fromtimestamp(modified_time)
    last_modified_fmt = last_modified.strftime('%m-%d-%Y %H:%M %p')

    # ----------------------------------------------------------------
    # text = 'Data file last refreshed ***{}***'.format(last_modified_fmt)
    # st.write(text)
    text = 'Data file last refreshed {}'.format(last_modified_fmt)
    with st.beta_expander('Data Preview - {}'.format(text)):
        st.subheader('Data Preview')
        st.write('temporary preview of dataframe containing price history')
        if st.button('Download Yahoo! Data'):
            None
        if st.button('Download TDA Data'):
            None
        if st.button('Download Alpaca Data'):
            None
        st.write(price_df.head(), price_df.tail())

    # ----------------------------------------------------------------
    with st.beta_container():
        st.header('Filter')
        st.write('choose one of more filter by market direction')

    with st.beta_container():
        col1, col2, col3 = st.beta_columns([1, 1, 1])
        with col1:
            st.subheader('Bullish')
            checkbox_a1 = st.checkbox('SMA crossover', value=False, key='scanner_bull')
            checkbox_a2 = st.checkbox('MACD crossover', value=False, key='scanner_bull')
            checkbox_a3 = st.checkbox('RSI oversold', value=False, key='scanner_bull')
            checkbox_a4 = st.checkbox('EMAs stacked', value=False, key='scanner_bull')
            checkbox_a5 = st.checkbox('TTM in-the-squeeze', value=False, key='scanner_bull')
            checkbox_a6 = st.checkbox('TTM out-the-squeeze', value=False, key='scanner_bull')

        with col2:
            st.subheader('Bearish')
            checkbox_b1 = st.checkbox('SMA crossover', value=False, key='scanner_bear')
            checkbox_b2 = st.checkbox('MACD crossover', value=False, key='scanner_bear')
            checkbox_b3 = st.checkbox('RSI overbought', value=False, key='scanner_bear')
            checkbox_b4 = st.checkbox('checkbox 9', value=False, key='scanner_bear')
            checkbox_b5 = st.checkbox('checkbox 10', value=False, key='scanner_bear')

        with col3:
            st.subheader('Sideways')
            checkbox_c1 = st.checkbox('checkbox 11', value=False, key='scanner_side')
            checkbox_c2 = st.checkbox('checkbox 12', value=False, key='scanner_side')
            checkbox_c3 = st.checkbox('checkbox 13', value=False, key='scanner_side')
            checkbox_c4 = st.checkbox('checkbox 14', value=False, key='scanner_side')
            checkbox_c5 = st.checkbox('checkbox 15', value=False, key='scanner_side')

    # ----------------------------------------------------------------
    st.subheader('Indicators')
    st.write('indicators are calculated while one looks at filters')

    @st.cache
    def calculate_indicators_for_cache(df):
        indicators_dict = {}
        ema_list = [8, 21, 34, 55, 89]
        for timeperiod in ema_list:
            k = '{}{}'.format('ema', timeperiod)
            indicators_dict[k] = df.groupby(level=0, axis=1).apply(lambda x: ema_apply_fun(x, timeperiod))

        sma_list = [10, 20, 25, 50]
        for timeperiod in sma_list:
            k = '{}{}'.format('sma', timeperiod)
            indicators_dict[k] = df.groupby(level=0, axis=1).apply(lambda x: sma_apply_fun(x, timeperiod))

        indicators_dict['atr'] = df.groupby(level=0, axis=1).apply(lambda x: atr_apply_fun(x, timeperiod=20))
        indicators_dict['rsi'] = df.groupby(level=0, axis=1).apply(lambda x: rsi_apply_fun(x, timeperiod=14))
        indicators_dict['cci'] = df.groupby(level=0, axis=1).apply(lambda x: cci_apply_fun(x, timeperiod=14))

        indicators_dict['macd'] = df.groupby(level=0, axis=1).apply(lambda x: macd_apply_fun(x, fastperiod=12, slowperiod=26, signalperiod=9))
        indicators_dict['macdsignal'] = df.groupby(level=0, axis=1).apply(lambda x: macdsignal_apply_fun(x, fastperiod=12, slowperiod=26, signalperiod=9))
        indicators_dict['macdhist'] = df.groupby(level=0, axis=1).apply(lambda x: macdhist_apply_fun(x, fastperiod=12, slowperiod=26, signalperiod=9))

        indicators_dict['bband_upper'] = df.groupby(level=0, axis=1).apply(lambda x: bband_upper_apply_fun(x, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0))
        indicators_dict['bband_lower'] = df.groupby(level=0, axis=1).apply(lambda x: bband_lower_apply_fun(x, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0))
        indicators_dict['bband_middle'] = df.groupby(level=0, axis=1).apply(lambda x: bband_middle_apply_fun(x, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0))

        indicators_dict['lower_keltner'] = indicators_dict['sma20'] - (indicators_dict['atr']*1.5)
        indicators_dict['upper_keltner'] = indicators_dict['sma20'] + (indicators_dict['atr']*1.5)
        return indicators_dict

    indicators_dict = calculate_indicators_for_cache(price_df)
    st.write('Indicator calculations completed.')

    # ----------------------------------------------------------------
    st.subheader('Signals')
    st.write('Signals are then calculated ')

    @st.cache
    def calculate_signals_for_cache(indicators_dict):
        signals = pd.DataFrame()
        signals.index.set_names('symbol', inplace=True)

        df = check_rule_in_squeeze(indicators_dict)
        signals.loc[:, 'ttm_in_squeeze'] = df.iloc[-1]
        signals.loc[:, 'ttm_coming_out_squeeze'] = (~df.iloc[-1]) & (df.iloc[-3])

        df = check_rule_ema_stacked(indicators_dict)
        signals.loc[:, 'ema_positively_stacked'] = df.iloc[-1]

        df = check_rule_sma_crossover(indicators_dict, fast=25, slow=50, bull=True)
        signals.loc[:, 'sma_crossover_bull'] = df.iloc[-1]

        df = check_rule_rsi(indicators_dict, bull=True)
        signals.loc[:, 'rsi_oversold'] = df.iloc[-1]

        df = check_rule_rsi(indicators_dict, bull=False)
        signals.loc[:, 'rsi_overbought'] = df.iloc[-1]

        df = check_rule_macd_crossover(indicators_dict, bull=True)
        signals.loc[:, 'macd_cross_bull'] = df.iloc[-1]

        df = check_rule_macd_crossover(indicators_dict, bull=False)
        signals.loc[:, 'macd_cross_bear'] = df.iloc[-1]

        return signals

    signals = calculate_signals_for_cache(indicators_dict)
    st.write('Signals calculations completed.')
    st.write(signals)

    # ----------------------------------------------------------------
    checkbox_to_signal = {
            'sma_crossover_bull': checkbox_a1,
            'macd_cross_bull': checkbox_a2,
            'rsi_oversold': checkbox_a3,
            'ema_positively_stacked': checkbox_a4,
            'ttm_in_squeeze': checkbox_a5,
            'ttm_coming_out_squeeze': checkbox_a6,
            'sma_crossover_bear': checkbox_b1,
            'macd_cross_bear': checkbox_b2,
            'rsi_overbought': checkbox_b3,
            'checkbox 9': checkbox_b4,
            'checkbox 10': checkbox_b5,
            'checkbox 11': checkbox_c1,
            'checkbox 12': checkbox_c2,
            'checkbox 13': checkbox_c3,
            'checkbox 14': checkbox_c4,
            'checkbox 15': checkbox_c5,
            }

    # ----------------------------------------------------------------
    st.subheader('Results')

    filters = [k for k,v in checkbox_to_signal.items() if v]
    filters = [x for x in filters if x in signals.columns]

    filter_results = pd.Series(index=signals.index, dtype=int, name='scanner')
    filter_results[:] = 1

    for c in filters:
        filter_results = filter_results.mul(signals[c])
    filter_results = filter_results.astype(bool)

    filter_results = filter_results[filter_results]

    st.write('{} stocks meet the criteria'.format(filter_results.shape[0]))
    st.write(filter_results)

    # with st.beta_container():
    #     if checkbox_a1:
    #         st.subheader('Dataframe filtered')
    #         st.write('emas are stacked')

    #         df1 = df.xs('Close', axis=1, level=1, drop_level=False)
    #         ema_df = ema_multiple_periods(df1 ,ema_list = [8, 21, 34, 55, 89])

    #         for s in ema_df.columns.get_level_values(0).unique():
    #             if ema_stacked(ema_df, s):
    #                 st.write(s)
    #     else:
    #         st.subheader('Results')
    #         st.write('this does not filter yet and returns all stocks. Functions need to be reefactored')
    #         scan_results = df.columns.get_level_values(0).unique().sort_values().to_list()

    #         for stock in scan_results:
    #             st.write(stock)

    # st.header('This is a header with an emoji :+1: ')
    # st.subheader('This is a subheader that goes to the moon :rocket: :rocket: :rocket:')
    # st.text('This is plain text and prefer using st.write')
    # st.markdown('This is markdown and can write format like this: Streamlit is **_really_ cool**.')
    # st.write('This is plain text for an application under-development to evaluate the potential of Streamlit')
    # st.write('write command, by default uses markdown with a string Hello, *World!* :sunglasses:')

    # test_dict = {'a':'alejandro','b':'ramirez'}
    # st.write('and this is a dictionary', test_dict)

    # code_str = """def hello():
    # print("Hello, Streamlit!")"""
    # st.code(code_str)
