
import pandas as pd
import streamlit as st


def app():

    with st.container():
        st.title('Options Sniper')
        st.write('Coming Soon - the fattest spread premiums for a list of stocks')
        st.subheader('Options Chain')
        st.write('gets the option chain for a stock from the datasets folder')

    with st.container():
        col1, col2, col3 = st.columns([1,1,2])
        symbol_input = col1.text_input('Enter Stock Symbol', max_chars=5)
        max_dte = col2.number_input('Enter Max DTE', value=60, format='%d')
        col2.write('When functional, this will limit the chain to max DTE {}'.format(max_dte))

    with st.container():
        s = symbol_input.upper()
        
        if not symbol_input:
            st.warning('Please enter a symbol.')
            st.text_input('Please enter')
            
        try:
            df = pd.read_csv('datasets/o_{}.csv'.format(s))
            st.write('Options Chain for', s, df)
        except:
            msg = '{} not found. This only works for AMD'.format(s)
            st.write(msg)

