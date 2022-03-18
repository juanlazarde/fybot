__all__ = ['create_config']

import os
from getpass import getpass
from core.encryption import Encryption
from core.utils import fix_path


def create_config(filename: str = '../config/config-test.py'):
    print("Configuration Wizard\n"
          "====================\n")
    print("Profile info:")
    profile_name = input("What is your name? ")
    profile_email = input("What is your email? ")
    print("\nDatabase info:")
    db_host = input("Host? (default: localhost) :")
    db_host = 'localhost' if db_host == '' else db_host
    db_name = input("Dabase name? (default: source) :")
    db_name = 'source' if db_name == '' else db_name
    db_user = input("User? (default: postgres) :")
    db_user = 'postgres' if db_user == '' else db_user
    db_password = input("Password? (default: diamondhands) :")
    db_password = 'diamondhands' if db_password == '' else db_password
    print("\nSecrets:")
    secret_key = input("Where is the secret key file 'account.key'? i.e. C:/user/secrets/account.key: ")
    secret_key = fix_path(secret_key, path_type='file')
    print("\nTDA info:")
    tda_token = input("Where is the TDA token? i.e. c:/user/td_access_token.pickle: ")
    tda_api = input("What is the TDA API? i.e. BUOUBBNMFG@AMER.OAUTHAP: ")
    tda_redirect = input("What is the Redirect URI suscribed with TDA? "
                         "i.e. https://localhost/fybot: ")
    tda_account = getpass("What is your TDA Account number? (Will be encrypted): ")
    tda_account_encrypted = Encryption(Encryption.create_save_key_file(secret_key)).encrypt(tda_account).decode('ascii')
    del tda_account

    body = f"""
    # Save this file as config.py in the config directory.
    # Do not share this file with anyone, do not post

    # Profile
    NAME = "{profile_name}"
    EMAIL = "{profile_email}"

    # Database info
    DB_HOST = '{db_host}'
    DB_USER = '{db_user}'
    DB_PASSWORD = '{db_password}'
    DB_NAME = '{db_name}'

    # Key file
    SECRET_KEY = "{secret_key}"

    # TDA Ameritrade
    TDA_TOKEN = "{tda_token}"
    TDA_API_KEY = "{tda_api}"
    TDA_REDIRECT_URI = "{tda_redirect}"
    TDA_ACCOUNT = "{tda_account_encrypted}"

    # Alpaca
    ALPACA_ENDPOINT = 'https://paper-api.alpaca.markets'
    ALPACA_API_KEY = '<ALPACA KEY>'
    ALPACA_SECRET_KEY = '<ALPACA SECRET_KEY>'
    """

    p = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, 'config'))
    with open(os.path.join(p, filename), 'w') as f:
        f.write(body)

    print("\nConfiguration file: config/config.py has been created or replaced.\n"
          "KEEP THIS FILE SECRET. DO NOT SHARE. INCLUDE IT THE .gitignore")


if __name__ == '__main':
    create_config()
