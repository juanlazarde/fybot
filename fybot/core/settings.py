import os
from core import config


class S:
    # define files
    DATASET_DIR = '../datasets'
    SYMBOLS_FILE = 'symbols.csv'
    PRICE_FILE = 'price.pkl'
    FUNDAMENTALS_FILE = 'fundamentals.pkl'
    SIGNALS_FILE = 'signals.pkl'

    # other constants
    DAYS = 150

    # FOR DEBUG ONLY
    DEBUG = True
    MAX_SYMBOLS = 10 if DEBUG else None

    # database settings
    DB_HOST = 'localhost'
    DB_USER = 'postgres'
    DB_PASSWORD = 'diamondhands'
    DB_NAME = 'source'

    # flask settings
    FLASK_SECRET_KEY = "jPRgt7XGwp5fuvVh846TAZDbAbTKTF8hpaHSacAqyEuTgkE6Kkuv" \
                       "yW2KDzQhhdVu2xKR7UExyfuQ8Nd4nxjezbHKJUwJ8QjFrsfRGVcH" \
                       "AHUzyvdAjCWZDLr7zFENhRA"

    # Encryption
    SECRET_KEY_FILE = config.SECRET_KEY

    # TDA Ameritrade Settings
    TDA_API_KEY = config.TDA_API_KEY
    TDA_REDIRECT_URI = config.TDA_REDIRECT_URI
    TDA_TOKEN = config.TDA_TOKEN
    TDA_ACCOUNT_E = config.TDA_ACCOUNT

    # Alpaca Settings
    # you need to create a config.py to house the credentials
    ALPACA_KEY = config.ALPACA_API_KEY
    ALPACA_SECRET = config.ALPACA_SECRET_KEY
    ALPACA_ENDPOINT = config.ALPACA_ENDPOINT

    SYMBOLS_FILE = os.path.join(DATASET_DIR, SYMBOLS_FILE)
    PRICE_FILE = os.path.join(DATASET_DIR, PRICE_FILE)
    FUNDAMENTALS_FILE = os.path.join(DATASET_DIR, FUNDAMENTALS_FILE)
    SIGNALS_FILE = os.path.join(DATASET_DIR, SIGNALS_FILE)

    # create dataset directory if not there
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    DEFAULT_FILTERS = {'consolidating': {'go': False, 'pct': 6},
                       'breakout': {'go': False, 'pct': 2.5},
                       'ttm_squeeze': {'go': False},
                       'in_the_squeeze': {'go': True},
                       'candlestick': {'go': True},
                       'sma_filter': {'go': False, 'fast': 25, 'slow': 100},
                       'ema_stacked': {'go': True},
                       'investor_reco': {'go': False}
                       }
