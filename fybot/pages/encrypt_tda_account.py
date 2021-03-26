"""Encrypte TDA Account using account Key"""

import streamlit as st
from fybot.core.encryption import Encryption


def app():
    st.title('Encrypt TDA Account')
    st.write("Create your encrypted string, to be saved in the config.py, "
             "under TDA_ACCOUNT")
    plain_text = st.text_input("Plain Account Number")
    if not plain_text:
        st.warning("Input your text to be encrypted")
        st.stop()

    secret = Encryption().encrypt(plain_text).decode('ascii')
    st.write(secret)
    st.write("Copy string above and paste in the config.py / TDA_ACCOUNT")

    if st.button('Save to Text File'):
        filename = Encryption().save_encrypted_to_text_file(plain_text)
        st.write(f"Copy string in file: {filename}, and paste it in the "
                 f"config.py file, under TDA_ACCOUNT. **Remember** to delete "
                 f"this file")

