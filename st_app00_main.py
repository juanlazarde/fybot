# app.py
import streamlit as st
import st_app01_home, st_app02, st_app03_sniper

st.set_page_config(layout="wide")

PAGES = {
    "Home": st_app01_home,
    "Strategy Bot": st_app02,
    "Options Sniper": st_app03_sniper,
    # "Dev Tab": st_app_tmp,
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
