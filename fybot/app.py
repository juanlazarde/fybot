"""User interface through Streamlit only"""

import streamlit as st
import fybot.pages as pages

# Pages in the app
PAGES = {
    "News": pages.news,
    "Scan": pages.scanner,
    "Technical Charts": pages.technical_charts,
    "Encrypt TDA Account": pages.encrypt_tda_account,
    "Home": pages.example_home,
    "Strategy Bot": pages.example_strategy,
    "Options Sniper": pages.example_options_sniper,
    "Dev Tab": pages.example_st_app_tmp,

}


def index():
    st.sidebar.title('Navigation')
    selection = st.sidebar.selectbox(
        label="Make a selection:",
        options=list(PAGES.keys()),
        index=1
    )

    page = PAGES[selection]
    page.app()
