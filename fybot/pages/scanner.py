import streamlit as st

import core.scanner as snap
from core.scanner.filters import Signals
import core.settings as ss
from core.scanner.scanner import Scan


def get_table(selected_filters: dict):
    # run filter function and save result to database
    Signals(selected_filters)

    # prepare information to be presented to html
    data = Scan(selected_filters).merge_symbol_name()
    data = data.rename(
        columns={"security": "Security",
                 "cdl_sum_ber": "Bear Candles",
                 "cdl_sum_bul": "Bull Candles",
                 "investor_sum": "Investor Reco"
                 }
    )
    return data


def check_selections(active_filters: list):
    # check for empty answers
    if len(active_filters) == 0:
        st.warning("Make a selection")
        st.stop()
        return False
    return True


def app():
    # Main body
    main_title = st.empty()
    main_body = st.empty()

    # Navigation menu
    left_nav, right_nav = st.sidebar.columns([1, 1])
    if left_nav.button("Refresh data"):
        with st.spinner(text="Refreshing All Data"):
            snap.refresh_data(forced=True)
            st.sidebar.write("Data refreshed")

    if right_nav.button("Export data"):
        with st.spinner(text="Exporting data"):
            snap.save_files()
            st.info(f"""Data saved at: {ss.DATASET_DIR}""")

    with st.sidebar.form(key='my_form', clear_on_submit=False):
        st.title("Financial Scanner")
        st.write("[1 bar = 1 day]")

        selected_filters = ss.DEFAULT_FILTERS

        selected_filters['consolidating']['go'] = st.checkbox(
            label="Consolidating within",
            value=ss.DEFAULT_FILTERS['consolidating']['go'],
        )
        selected_filters['consolidating']['pct'] = st.number_input(
            label="pct",
            min_value=0.0,
            value=ss.DEFAULT_FILTERS['consolidating']['pct'],
            step=0.1,
        )

        selected_filters['breakout']['go'] = st.checkbox(
            label="Breakout within",
            value=ss.DEFAULT_FILTERS['breakout']['go']
        )
        selected_filters['breakout']['pct'] = st.number_input(
            label="pct",
            min_value=0.0,
            value=ss.DEFAULT_FILTERS['breakout']['pct'],
            step=0.1
        )

        selected_filters['ttm_squeeze']['go'] = st.checkbox(
            label="Post TTM Squeeze",
            value=ss.DEFAULT_FILTERS['ttm_squeeze']['go']
        )

        selected_filters['in_the_squeeze']['go'] = st.checkbox(
            label="TTM Squeeze pre-breakout",
            value=ss.DEFAULT_FILTERS['in_the_squeeze']['go']
        )

        selected_filters['candlestick']['go'] = st.checkbox(
            label="Candlesticks",
            value=ss.DEFAULT_FILTERS['candlestick']['go']
        )

        selected_filters['sma_filter']['go'] = st.checkbox(
            label="SMA fast above slow",
            value=ss.DEFAULT_FILTERS['sma_filter']['go'],
            help="(bullish) signals fast sma crossing over higher than slow "
                 "sma in the latest period"
        )
        selected_filters['sma_filter']['fast'] = st.number_input(
            label="fast (bars)",
            min_value=0.0,
            value=float(ss.DEFAULT_FILTERS['sma_filter']['fast']),
            step=1.0
        )
        selected_filters['sma_filter']['slow'] = st.number_input(
            label="slow (bars)",
            min_value=0.0,
            value=float(ss.DEFAULT_FILTERS['sma_filter']['slow']),
            step=1.0
        )

        selected_filters['ema_stacked']['go'] = st.checkbox(
            label="Close above EMA positively stacked",
            value=ss.DEFAULT_FILTERS['ema_stacked']['go'],
            help="(bullish) signals all fast EMAs are higher than the slow "
                 "ema in the latest period."
        )

        selected_filters['investor_reco']['go'] = st.checkbox(
            label="Investor Recommendation",
            value=ss.DEFAULT_FILTERS['investor_reco']['go']
        )

        active_filters = [k for k in selected_filters if selected_filters[k]['go']]
        # import pandas
        # data_view = pandas.DataFrame(None)

        # Run scan
        scan_btn = st.form_submit_button(label="Run Scan")

        if scan_btn:
            # check for empty answers
            if len(active_filters) == 0:
                st.warning("Make a selection")
                st.stop()

            # get data amd leave it on cache
            with st.spinner("Retrieving data..."):
                data = get_table(selected_filters)

            # prepare view
            data_view = data.drop(columns=active_filters)

            # show table
            main_title.text(active_filters)
            main_body.table(data_view)

            # show the latest price_history database update
            price_update_dt, within24 = Scan.last_update('price_history')
            st.sidebar.write(f"Price update: {price_update_dt}")
