# app1.py
import sys

import pandas as pd
import streamlit as st
import talib

from fybot.core.settings import S


def app():
    st.title('Financial Scanner - Streamlit Beta')
    st.header('This is a header with an emoji :+1: ')
    st.subheader(
        'This is a subheader that goes to the moon :rocket: :rocket: :rocket:')
    st.text('This is plain text and prefer using st.write')
    st.markdown(
        'This is markdown and can write format like this: Streamlit is **_really_ cool**.')
    st.write(
        'This is plain text for an application under-development to evaluate the potential of Streamlit')
    st.write(
        'write command, by default uses markdown with a string Hello, *World!* :sunglasses:')

    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    st.write('this is a number', 2, 'and below is a df', df1)

    test_dict = {'a': 'alejandro', 'b': 'ramirez'}
    st.write('and this is a dictionary', test_dict)

    code_str = """def hello():
    print("Hello, Streamlit!")"""
    st.code(code_str)

    # ----------------------------------------------------------------
    if st.button('Refresh Data'):
        st.write('this just prints this message. will get it to donwload data')
    # ----------------------------------------------------------------
    with st.container():
        st.title('Financial scanner')
        st.write('a stock filter with indicators')
        st.subheader('Filter')
        st.write('filter by market direction')

    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.subheader('Bullish')
            checkbox_a1 = st.checkbox('Stacked EMAs', value=False,
                                      key='scanner')
            checkbox_a2 = st.checkbox('checkbox 2', value=False, key='scanner')
            checkbox_a3 = st.checkbox('checkbox 3', value=False, key='scanner')
            checkbox_a4 = st.checkbox('checkbox 4', value=False, key='scanner')
            checkbox_a5 = st.checkbox('checkbox 5', value=False, key='scanner')

        with col2:
            st.subheader('Bearish')
            checkbox_b1 = st.checkbox('checkbox 6', value=False, key='scanner')
            checkbox_b2 = st.checkbox('checkbox 7', value=False, key='scanner')
            checkbox_b3 = st.checkbox('checkbox 8', value=False, key='scanner')
            checkbox_b4 = st.checkbox('checkbox 9', value=False, key='scanner')
            checkbox_b5 = st.checkbox('checkbox 10', value=False,
                                      key='scanner')

        with col3:
            st.subheader('Sideways')
            checkbox_c1 = st.checkbox('checkbox 11', value=False,
                                      key='scanner')
            checkbox_c2 = st.checkbox('checkbox 12', value=False,
                                      key='scanner')
            checkbox_c3 = st.checkbox('checkbox 13', value=False,
                                      key='scanner')
            checkbox_c4 = st.checkbox('checkbox 14', value=False,
                                      key='scanner')
            checkbox_c5 = st.checkbox('checkbox 15', value=False,
                                      key='scanner')

    checkboxes = [checkbox_a1, checkbox_a2, checkbox_a3, checkbox_a4,
                  checkbox_a5]

    with st.container():
        for check in checkboxes:
            if check:
                st.write('{} is checked. Great!'.format(check))

    st.subheader('Dataframe to be filtered')

    df = pd.read_pickle(S.PRICE_FILE)

    st.write(
        'df is loaded from pickle and resides in cache making filtering fast')
    st.write(df.head())

    def ema_multiple_periods(df, ema_list=None):
        # initializes an empty data frame with empty multi-index columns
        ema_list = None if ema_list is None else [8, 21, 34, 55, 89]
        multi_index_cols = pd.MultiIndex.from_product([[], []])
        ema = pd.DataFrame(index=df.index, columns=multi_index_cols)

        for period in ema_list:
            col = '{}{}'.format('ema', period)
            ema = ema.join(
                df.apply(lambda c: talib.EMA(c.values, period)).rename(
                    columns={'Close': col}))
        return ema

    def ema_stacked(ema, s):
        if (ema[-1:][s].ema8[0] > ema[-1:][s].ema21[0] > ema[-1:][s].ema34[0] >
                ema[-1:][s].ema55[0] > ema[-1:][s].ema89[0]):
            return True

    with st.container():
        if checkbox_a1:
            st.subheader('Dataframe filtered')
            st.write('emas are stacked')

            df1 = df.xs('Close', axis=1, level=1, drop_level=False)
            ema_df = ema_multiple_periods(df1, ema_list=[8, 21, 34, 55, 89])

            for s in ema_df.columns.get_level_values(0).unique():
                if ema_stacked(ema_df, s):
                    st.write(s)
        else:
            st.subheader('Results')
            st.write(
                'this does not filter yet and returns all stocks. Functions need to be reefactored')
            scan_results = df.columns.get_level_values(
                0).unique().sort_values().to_list()

            for stock in scan_results:
                st.write(stock)
    return
