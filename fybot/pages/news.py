"""News page"""

import streamlit as st
import core.news as nw
import core.settings as ss


def app():
    st.title("News")
    sources = st.sidebar.selectbox(label="Select your sources",
                                   options=ss.NEWS['sources'],
                                   index=0)
    st.header(sources)
    subsources = st.sidebar.selectbox(label="Select your subs",
                                      options=ss.NEWS['subsources'],
                                      index=0)
    subheader_prefix = "s/" if sources == "Reddit" else ""
    st.subheader(f"{subheader_prefix}{subsources}")

    # TODO: CAPTURE LATEST BETWEEN NOW AND LAST UPDATE
    with st.spinner(text="Loading the news..."):
        nw.capture_reddit(subsources=subsources, start_date='2/19/2021')
        num_days = st.sidebar.slider('Number of days', 1, 30, 3)
        counts, mentions, rows = nw.show_reddit(subsources=subsources,
                                                num_days=num_days)

        # TODO: ADJUST QUERY TO RETURN PROPER ANSWER
        for count in counts:
            st.write(count)

        for mention in mentions:
            st.text(f"**Symbol**: {mention['symbol']}, "
                    f"posted: {mention['date']} @ {mention['source']}\n"
                    f"---> {mention['mentions']}")
            st.text(mention['url'])

        # TODO: ADJUST QUERY TO RETURN PROPER ANSWER
        st.write(rows)
