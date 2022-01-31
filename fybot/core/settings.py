"""Program settings are configured here.
Personalized data, like passwords, are set in config.py"""

import os
import config.config as config

# FOR DEBUG ONLY
DEBUG = False
MAX_SYMBOLS = 150 if DEBUG else 2 #TODO: check recursion issue here

# define files
DATASET_DIR = '../datasets'
LOGGING_FILE = 'config/logging_config.ini'
SYMBOLS_FILE = 'symbols.csv'
PRICE_FILE = 'price.pkl'
FUNDAMENTALS_FILE = 'fundamentals.pkl'
SIGNALS_FILE = 'signals.pkl'
TEST_FILE = 'test.pkl'

# files with path
ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
DATASET_DIR = os.path.abspath(os.path.join(ROOT_DIR, DATASET_DIR))
LOGGING_FILE = os.path.join(ROOT_DIR, LOGGING_FILE)
SYMBOLS_FILE = os.path.join(DATASET_DIR, SYMBOLS_FILE)
PRICE_FILE = os.path.join(DATASET_DIR, PRICE_FILE)
FUNDAMENTALS_FILE = os.path.join(DATASET_DIR, FUNDAMENTALS_FILE)
SIGNALS_FILE = os.path.join(DATASET_DIR, SIGNALS_FILE)
TEST_FILE = os.path.join(DATASET_DIR, TEST_FILE)

# create dataset directory if not there
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# other constants
DAYS = 150
USER_NAME = config.NAME
USER_EMAIL = config.EMAIL

# database settings
DB_HOST = config.DB_HOST
DB_USER = config.DB_USER
DB_PASSWORD = config.DB_PASSWORD
DB_NAME = config.DB_NAME

# Encryption
SECRET_KEY_FILE = config.SECRET_KEY

# TDA Ameritrade Settings
TDA_API_KEY = config.TDA_API_KEY
TDA_REDIRECT_URI = config.TDA_REDIRECT_URI
TDA_TOKEN = config.TDA_TOKEN
TDA_ACCOUNT_E = config.TDA_ACCOUNT

# Alpaca Settings
ALPACA_KEY = config.ALPACA_API_KEY
ALPACA_SECRET = config.ALPACA_SECRET_KEY
ALPACA_ENDPOINT = config.ALPACA_ENDPOINT

# default values
# scanner.py filters
DEFAULT_FILTERS = {
    'consolidating': {'go': False, 'pct': 6.0},
    'breakout': {'go': False, 'pct': 2.5},
    'ttm_squeeze': {'go': False},
    'in_the_squeeze': {'go': True},
    'candlestick': {'go': True},
    'sma_filter': {'go': False, 'fast': 25, 'slow': 100},
    'ema_stacked': {'go': True},
    'investor_reco': {'go': False}
}

# news.py settings
NEWS = {
    'sources': ['Reddit'],
    'subsources': ['wallstreetbets', 'options']
}

# option_sniper.py settings
OPTION_SNIPER = {
    'DEBUG': {
        'save_load_pickle': False,
        'export': False,
        'force_download': True,
        'data': DATASET_DIR,
    },
    'WATCHLIST': {
        'watchlist_0': 'AMD, AMZN, RIVN',
        'watchlist_1': 'SPY, TQQQ',
        'watchlist_2': 'TSLA, RIVN, LCID',
        'watchlist_3': 'AMD, NVDA, AMAT',
        'watchlist_4': 'PG, JNJ, KC',
        'watchlist_5': 'META, FB, U, ZM, ZNGA, NVDA, AMD, IMMR, ANET, STX, SHOP, RBLX, FSLY, ADSK',
        'watchlist_current': '',
        'tda_watchlist': 'default',
        'selected': 0
    },
    'FILTERS': {
        'top_n': 5,
        'min_price': 1.,
        'max_price': 5000.,
        'max_risk': 20000.,
        'min_return_pct': 0.5,
        'max_dte': 60,
        'min_dte': 30,
        'premium_type': "credit",
        'strategies': "spread",
        'option_type': "put, call",
        'max_delta': 0.30,
        'min_pop': 0.,
        'min_p50': 0.,
        'min_volume_pctl': 5.,
        'min_open_int_pctl': 5.,
        'max_bid_ask_pctl': 50.,
        'margin_requirement': 20000.,
    },
}
