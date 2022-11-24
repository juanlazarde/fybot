# Save this file as config.py in the config directory.
# Do not share this file with anyone, do not post
import os

# Profile
NAME = os.environ.get("NAME")
EMAIL = os.environ.get("GMAIL")

# Database info
DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("POSTGRES_USER")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
DB_NAME = os.environ.get("POSTGRES_DB")

# Key file
SECRET_KEY = os.environ.get("TDA_SECRET_KEY")

# TDA Ameritrade
# The Token Path has to match the docker path specified in the app yaml file
TDA_TOKEN = "/usr/src/fybot/config/td_access_token.pickle"
TDA_API_KEY = os.environ.get("TDA_API_KEY")
TDA_REDIRECT_URI = os.environ.get("TDA_REDIRECT_URI")
TDA_ACCOUNT = os.environ.get("TDA_ACCOUNT")

# Alpaca
ALPACA_ENDPOINT = "https://paper-api.alpaca.markets"
ALPACA_API_KEY = "<ALPACA KEY>"
ALPACA_SECRET_KEY = "<ALPACA SECRET_KEY>"
