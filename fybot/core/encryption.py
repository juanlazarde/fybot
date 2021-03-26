"""Encryption module"""

import os
import subprocess
import sys

from cryptography.fernet import Fernet

from fybot.core.settings import S


class Encryption:
    cipher_suite = None

    def __init__(self, password=''):
        """Encryption process using 'cryptography' package.

         Use:
             crypto = Encryption(password='tothemoon')
             secret = crypto.encrypt('secret message')
             revealed = crypto.decrypt('mm432kl4m32')
             If no password is used, then it reads the key file.
             If no key file is present, it creates one.

         :param str password: Password, otherwise it will read it from key file
         """

        # name of file with key
        if password.strip() != '':
            key = password.strip()
        else:
            key = self.create_save_key_file(S.SECRET_KEY_FILE)
        self.cipher_suite = Fernet(key)

    @staticmethod
    def create_save_key_file(filename):
        """Generate or load Key and save it in file.

        :param str filename: path and file name to key file
        :return key: encypting key
        """

        if os.path.isfile(filename):
            with open(filename, "rb") as key_file:
                key = key_file.read()
        else:
            key = Fernet.generate_key()
            with open(filename, "wb") as key_file:
                key_file.write(key)
        return key

    def encrypt(self, text: str):
        """Encode text using master password."""

        _byte = text.encode('ascii')
        return self.cipher_suite.encrypt(_byte)

    def decrypt(self, text: str):
        """Decode text using master password."""

        _byte = text.encode('ascii')
        decoded_text = self.cipher_suite.decrypt(_byte)
        return decoded_text.decode('ascii')

    def save_encrypted_to_text_file(self, message: str):
        """Writes an 'encrypted.txt' file to the root folder with the
        message, then opens it for you to copy/paste.

        Use:
            Encryption(password='1234').save_encrypted_to_text_file('message')
        * Password (optional) if empty/None, it reads the key file

        :param str message: Plain text to be encrypted"""

        secret = self.encrypt(message).decode('ascii')

        # write file
        filename = "encrypted.txt"
        with open(filename, "w") as f:
            f.write(secret)

        # open the file
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])

        return filename
