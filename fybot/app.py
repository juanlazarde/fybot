"""User interface through Streamlit only"""

import streamlit as st
import pages
# from src import Snapshot


FAV_ICON = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com" \
           "/thumbs/240/twitter/259/mage_1f9d9.png"

# TODO: Diagnose what is the problem here, it doesn't keep the config
st.set_page_config(
    page_title="Financial Scanner",
    page_icon=FAV_ICON,
    layout="wide"
)

# Pages in the app
PAGES = {
    "Home": pages.home,
    "News": pages.news,
    "Scan": pages.scanner,
    "Technical Charts": pages.technical_charts,
    "Strategy Bot": pages.strategy,
    "Options Sniper": pages.options_sniper,
    "Dev Tab": pages.st_app_tmp,
    "Encrypt TDA Account": pages.encrypt_tda_account
}


def index():

    st.sidebar.title('Navigation')
    selection = st.sidebar.selectbox(label="Make a selection:",
                                     options=list(PAGES.keys()),
                                     index=0)

    if st.sidebar.button("Refresh data"):
        # Snapshot(forced=True)
        st.sidebar.write("Data refreshed")

    page = PAGES[selection]
    page.app()
