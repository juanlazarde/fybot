import streamlit as st
from string import ascii_uppercase, digits
from random import choices
def app():
    img_base = "https://www.htmlcsscolor.com/preview/128x128/{0}.png"

    colors = (''.join(choices(ascii_uppercase[:6] + digits, k=6)) for _ in range(100))


    with st.beta_container():
        for col in st.beta_columns(3):
            col.image(img_base.format(next(colors)), use_column_width=True)


    with st.beta_container():
        for col in st.beta_columns(4):
            col.image(img_base.format(next(colors)), use_column_width=True)


    with st.beta_container():
        for col in st.beta_columns(10):
            col.image(img_base.format(next(colors)), use_column_width=True)