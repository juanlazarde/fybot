import pandas as pd
import streamlit as st
import st_aggrid
df = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv')
st_aggrid.AgGrid(df)
st_grid

# pd.set_option('display.max_colwidth', 500)
#
# # def make_clickable(link):
# #     # target _blank to open new window
# #     # extract clickable text to display for your link
# #     text = link.split('=')[1]
# #     return f'<a target="_blank" href="{link}">{text}</a>'
# #
# # # link is the column with hyperlinks
# # df['link'] = df['link'].apply(make_clickable)
# # df = df.to_html(escape=False)
# # st.write(df, unsafe_allow_html=True)
#
# hide_streamlit_style = """
# <style>
#
# /* This is to hide hamburger menu completely */
# #MainMenu {visibility: hidden;}
#
# /* This is to hide Streamlit footer */
# footer {visibility: hidden;}
#
# /*
# If you did not hide the hamburger menu completely,
# you can use the following styles to control which items on the menu to hide.
# */
# ul[data-testid=main-menu-list] > li:nth-of-type(4), /* Documentation */
# ul[data-testid=main-menu-list] > li:nth-of-type(5), /* Ask a question */
# ul[data-testid=main-menu-list] > li:nth-of-type(6), /* Report a bug */
# ul[data-testid=main-menu-list] > li:nth-of-type(7), /* Streamlit for Teams */
# ul[data-testid=main-menu-list] > div:nth-of-type(2) /* 2nd divider */
# {display: none;}
#
# </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)
#
#
# def add_stream_url(track_ids):
#     return [f'https://open.spotify.com/track/{t}' for t in track_ids]
#
#
# def make_clickable(url, text):
#     return f'<a target="_blank" href="{url}">{text}</a>'
#
#
# # show data
# df = pd.DataFrame({'link': ['link1', 'link2'],
#                    'track_id': ['track 1', 'track2']})
# if st.checkbox('Include Preview URLs'):
#     df['preview'] = add_stream_url(df.track_id)
#     df['preview'] = df['preview'].apply(make_clickable, args=('Listen',))
#     st.write(df.to_html(escape=False), unsafe_allow_html=True)
# else:
#     st.write(df)
