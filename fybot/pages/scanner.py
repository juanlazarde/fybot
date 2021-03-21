import streamlit as st

import pandas as pd

from core.settings import S
from core.filters import Filter
from integrate.scanner import Scan


def app():
    st.title("Financial Scanner")
    st.write("""Select the filters for the scanner: (1 bar = 1 day)""")

    column_width = [1, 2, 2]

    left, center, right = st.beta_columns(column_width)
    consolidating_go = left.checkbox(
        label="Consolidating wihin",
        value=S.DEFAULT_FILTERS['consolidating']['go'],
    )
    consolidating_pct = center.number_input(
        label="pct",
        min_value=0.0,
        value=S.DEFAULT_FILTERS['consolidating']['pct'],
        step=0.1,
    )

    left, center, right = st.beta_columns(column_width)
    breakout_go = left.checkbox(
        label="Breakout within",
        value=S.DEFAULT_FILTERS['breakout']['go']
    )
    breakout_pct = center.number_input(
        label="pct",
        min_value=0.0,
        value=S.DEFAULT_FILTERS['breakout']['pct'],
        step=0.1
    )

    ttm_squeeze_go = st.checkbox(
        label="Post TTM Squeeze",
        value=S.DEFAULT_FILTERS['ttm_squeeze']['go']
    )

    in_the_squeeze_go = st.checkbox(
        label="TTM Squeeze per-breakout",
        value=S.DEFAULT_FILTERS['in_the_squeeze']['go']
    )

    candlestick_go = st.checkbox(
        label="Cadlesticks",
        value=S.DEFAULT_FILTERS['candlestick']['go']
    )

    left, center, right = st.beta_columns(column_width)
    sma_momentum_go = left.checkbox(
        label="SMA fast above slow",
        value=S.DEFAULT_FILTERS['sma_filter']['go'],
        help="(bullish) signals fast sma crossing over higher than slow sma "
             "in the latest period"
    )
    sma_fast = center.number_input(
        label="fast (bars)",
        min_value=0,
        value=S.DEFAULT_FILTERS['sma_filter']['fast'],
        step=1
    )
    sma_slow = right.number_input(
        label="slow (bars)",
        min_value=0,
        value=S.DEFAULT_FILTERS['sma_filter']['slow'],
        step=1
    )

    ema_stacked_go = st.checkbox(
        label="Close above EMA positively stacked",
        value=S.DEFAULT_FILTERS['ema_stacked']['go'],
        help="(bullish) signals all fast emas are higher than the slow "
             "ema in the latest period."
    )

    investor_reco_go = st.checkbox(
        label="Investor Recommendation",
        value=S.DEFAULT_FILTERS['investor_reco']['go']
    )

    if st.button(label="Run Scan"):
        # TODO: CHECK THAT AT LEAST ONE OPTION IS SELECTED

        active_filters = {
            'consolidating':
                {'go': consolidating_go,
                 'pct': consolidating_pct},

            'breakout':
                {'go': breakout_go,
                 'pct': breakout_pct},

            'ttm_squeeze':
                {'go': ttm_squeeze_go},

            'in_the_squeeze':
                {'go': in_the_squeeze_go},

            'candlestick':
                {'go': candlestick_go},

            'sma_filter':
                {'go': sma_momentum_go,
                 'fast': sma_fast,
                 'slow': sma_slow},

            'ema_stacked':
                {'go': ema_stacked_go},

            'investor_reco':
                {'go': investor_reco_go}
        }

        # run filter function and save result to database
        Filter(active_filters)

        # get date of last price_history from file or database
        price_update_dt, within24 = Scan.last_update('price_history')
        st.sidebar.write(f"Price update: {price_update_dt}")

        # prepare information to be presented to html
        data_dict = Scan(active_filters).stocks
        data = pd.DataFrame(data_dict).T
        data = data.rename(
            columns={"security": "Security",
                     "cdl_sum_ber": "Bear Candles",
                     "cdl_sum_bul": "Bull Candles"
                     })
        st.dataframe(data)
