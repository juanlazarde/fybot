"""User interface through Streamlit only

Streamlit GitHub
https://github.com/streamlit

Cheat sheet
https://github.com/daniellewisdl/streamlit-cheat-sheet/blob/master/app.py
"""

import streamlit as st
import fybot.pages as pages

# Streamlit Configuration
PAGE_TITLE = "Financial Scanner"
PAGE_ICON = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com" \
            "/thumbs/240/twitter/259/mage_1f9d9.png"
MENU_GET_HELP = "https://github.com/juanlazarde/fybot"
MENU_BUG = "https://github.com/juanlazarde/fybot/issues"
MENU_ABOUT = "# FyBot #" \
             "Financial dashboard with technical scanner, news, and options " \
             "analysis."

# Pages in the app
PAGES = {
    "News": pages.news,
    "Scan": pages.scanner,
    "Technical Charts": pages.technical_charts,
    "Encrypt TDA Account": pages.encrypt_tda_account,
}


def index():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': MENU_GET_HELP,
            'Report a bug': MENU_BUG,
            'About': MENU_ABOUT
        }
    )
    st.sidebar.title('Navigation')
    selection = st.sidebar.selectbox(
        label="Make a selection:",
        options=list(PAGES.keys()),
        index=1
    )

    page = PAGES[selection]
    page.app()
