"""User interface through Streamlit only

Streamlit GitHub
<https://github.com/streamlit>

Cheat sheet
<https://github.com/daniellewisdl/streamlit-cheat-sheet/blob/master/app.py>
"""
import streamlit as st

import pages
import core.settings as ss

# Streamlit Configuration
PAGE_TITLE = "Financial Scanner"
PAGE_ICON = (
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com"
    "/thumbs/240/twitter/259/mage_1f9d9.png"
)
MENU_GET_HELP = "https://github.com/juanlazarde/fybot"
MENU_BUG = "https://github.com/juanlazarde/fybot/issues"
MENU_ABOUT = (
    "# FyBot #" "Financial dashboard: technical scanner, news, & options analysis"
)

HACK_CSS = """
<style>

/* Hide the hamburger menu */
/*#MainMenu {visibility: hidden;}*/

/* Hide Streamlit footer */
footer {visibility: hidden;}

/* Hide items in hamburger*/
ul[data-testid=main-menu-list] > li:nth-of-type(4), /* Documentation */
ul[data-testid=main-menu-list] > li:nth-of-type(5), /* Ask a question */
ul[data-testid=main-menu-list] > li:nth-of-type(6), /* Report a bug */
ul[data-testid=main-menu-list] > li:nth-of-type(7), /* Streamlit for Teams */
ul[data-testid=main-menu-list] > div:nth-of-type(2) /* 2nd divider */
{display: none;}

</style>
"""

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": MENU_GET_HELP,
        "Report a bug": MENU_BUG,
        "About": MENU_ABOUT,
    },
)
st.markdown(HACK_CSS, unsafe_allow_html=True)
st.sidebar.markdown(f"Hi, **{ss.USER_NAME}**")
st.sidebar.title("Navigation")
selection = st.sidebar.selectbox(
    label="Make a selection:", options=["Option 1", "Option 2"]
)


"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


# DEFAULT_PAGE = 1
# PAGES = {
#     "News": pages.news,
#     "Scan": pages.scanner,
#     "Option Sniper": pages.option_sniper,
#     "Calculator": pages.calculator,
#     "Technical Charts": pages.technical_charts,
#     "Encrypt TDA Account": pages.encrypt_tda_account,
# }


# def main():

# page = PAGES[selection]
# page.app()


# if __name__ == "__main__":
#     main()
