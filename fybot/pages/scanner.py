import streamlit as st

from core.settings import S


def app():
    st.title("Financial Scanner")
    st.write("""Select the filters for the scanner: (1 bar = 1 day)""")
    consolidating = st.checkbox(
        label="Consolidating wihin",
        value=S.DEFAULT_FILTERS['consolidating']['go']
    )
    condolidating_pct = st.number_input(
        label="pct",
        min_value=0.0,
        value=S.DEFAULT_FILTERS['consolidating']['pct'],
        step=0.1,
    )
    breakout = st.checkbox(
        label="Breakout within",
        value=S.DEFAULT_FILTERS['breakout']['go']
    )
    breakout_pct = st.number_input(
        label="percent",
        min_value=0.0,
        value=S.DEFAULT_FILTERS['breakout']['pct'],
        step=0.1
    )
    ttm_squeeze = st.checkbox(
        label="Post TTM Squeeze",
        value=S.DEFAULT_FILTERS['ttm_squeeze']['go']
    )
    in_the_squeeze = st.checkbox(
        label="TTM Squeeze per-breakout",
        value=S.DEFAULT_FILTERS['in_the_squeeze']['go']
    )
    candlestick = st.checkbox(
        label="Cadlesticks",
        value=S.DEFAULT_FILTERS['candlestick']['go']
    )
    sma = st.checkbox(
        label="SMA fast above slow",
        value=S.DEFAULT_FILTERS['sma_filter']['go'],
        help="(bullish) signals fast sma crossing over higher than slow sma "
             "in the latest period"
    )
    sma_fast = st.number_input(
        label="fast",
        min_value=0,
        value=S.DEFAULT_FILTERS['sma_filter']['fast'],
        step=1
    )
    sma_slow = st.number_input(
        label="slow",
        min_value=0,
        value=S.DEFAULT_FILTERS['sma_filter']['slow'],
        step=1
    )
    ema_stacked = st.checkbox(
        label="Close above EMA positively stacked",
        value=S.DEFAULT_FILTERS['ema_stacked']['go'],
        help="(bullish) signals all fast emas are higher than the slow "
             "ema in the latest period."
    )
    investor_reco = st.checkbox(
        label="Investor Recommendation",
        value=S.DEFAULT_FILTERS['investor_reco']['go']
    )

    st.button(label="Run Scan")
