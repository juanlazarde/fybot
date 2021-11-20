import streamlit as st

import core.scanner as snap
from core.scanner.filters import Signals
import core.settings as ss
from core.scanner.scanner import Scan


def rerun():
    raise st.script_runner.RerunException(
        st.script_request_queue.RerunData(None))


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

    # Main page
    st.title("Financial Scanner")
    st.write("""Select the filters for the scanner: [1 bar = 1 day]""")

    selected_filters = ss.DEFAULT_FILTERS
    column_width = [1, 2, 2]
    left, center, right = st.columns(column_width)
    selected_filters['consolidating']['go'] = left.checkbox(
        label="Consolidating wihin",
        value=ss.DEFAULT_FILTERS['consolidating']['go'],
    )
    selected_filters['consolidating']['pct'] = center.number_input(
        label="pct",
        min_value=0.0,
        value=ss.DEFAULT_FILTERS['consolidating']['pct'],
        step=0.1,
    )

    left, center, right = st.columns(column_width)
    selected_filters['breakout']['go'] = left.checkbox(
        label="Breakout within",
        value=ss.DEFAULT_FILTERS['breakout']['go']
    )
    selected_filters['breakout']['pct'] = center.number_input(
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
        label="Cadlesticks",
        value=ss.DEFAULT_FILTERS['candlestick']['go']
    )

    left, center, right = st.columns(column_width)
    selected_filters['sma_filter']['go'] = left.checkbox(
        label="SMA fast above slow",
        value=ss.DEFAULT_FILTERS['sma_filter']['go'],
        help="(bullish) signals fast sma crossing over higher than slow "
             "sma in the latest period"
    )
    selected_filters['sma_filter']['fast'] = center.number_input(
        label="fast (bars)",
        min_value=0,
        value=ss.DEFAULT_FILTERS['sma_filter']['fast'],
        step=1
    )
    selected_filters['sma_filter']['slow'] = right.number_input(
        label="slow (bars)",
        min_value=0,
        value=ss.DEFAULT_FILTERS['sma_filter']['slow'],
        step=1
    )

    selected_filters['ema_stacked']['go'] = st.checkbox(
        label="Close above EMA positively stacked",
        value=ss.DEFAULT_FILTERS['ema_stacked']['go'],
        help="(bullish) signals all fast emas are higher than the slow "
             "ema in the latest period."
    )

    selected_filters['investor_reco']['go'] = st.checkbox(
        label="Investor Recommendation",
        value=ss.DEFAULT_FILTERS['investor_reco']['go']
    )

    active_filters = [k for k in selected_filters if selected_filters[k]['go']]

    # check for empty answers
    if len(active_filters) == 0:
        st.warning("Make a selection")
        st.stop()

    if st.button(label="Run Scan"):
        # get data amd leave it on cache
        with st.spinner("Retrieving data..."):
            data = get_table(selected_filters)

            # prepare view
            data_view = data.drop(columns=active_filters)
            st.dataframe(data_view)

        # show latest price_history database update
        price_update_dt, within24 = Scan.last_update('price_history')
        st.sidebar.write(f"Price update: {price_update_dt}")
