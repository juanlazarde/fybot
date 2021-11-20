"""Encrypt TDA Account using account Key

This module will encrypt the text using a Secret Key file in your hard drive.
The purpose of the key file is so that you don't have to type a password.
If the Secret Key file doensn't exist, it will be created for you.
The path and name for the file is defined in the Settings.py. SECRET_KEY_FILE
"""
import streamlit as st
from core.encryption import Encryption


def app():
    st.title('Encrypt TDA Account')
    st.write("Create your encrypted string, to be saved in the config.py, "
             "under TDA_ACCOUNT")
    plain_text = st.text_input("Plain Account Number")
    if not plain_text:
        st.warning("Input your text to be encrypted")
        st.markdown(
            "*More Info*  \n"
            "Encrypts text using a Secret Key file in your hard drive.  \n"
            "The key file is so that you don't have to type a password.  \n"
            "If the key file doesn't exist, it will be created for you.  \n"
            "Filename is defined in the `settings.py`: SECRET_KEY_FILE"
        )
        st.stop()

    secret = Encryption().encrypt(plain_text).decode('ascii')
    st.write(secret)
    st.write("Copy string above and paste in the config.py / TDA_ACCOUNT")

    if st.button('Save to Text File'):
        filename = Encryption().save_encrypted_to_text_file(plain_text)
        st.write(f"Copy string in file: {filename}, and paste it in the "
                 f"config.py file, under TDA_ACCOUNT. **Remember** to delete "
                 f"this file")

