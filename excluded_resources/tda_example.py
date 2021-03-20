import pandas as pd
from tda.auth import easy_client
from fybot.common import config
from functions import *
from IPython.utils import io
from datetime import datetime, timedelta
from time import sleep

# from tda.client import client

# Creates connection with TDA database
c = easy_client(
        config.api_key,
        config.redirect_uri,
        config.token_path,
        make_webdriver)
print("Client Authorization process complete.")

#----------------------------------------------------------------
#downloads all 7-8k active tickers/symbols (review function for filters)
df = import_symbols(reference_path="../fybot/pages/datasets/")
symbols_list = df.index.to_list()

#----------------------------------------------------------------
#if a sigle quote is needed
s = 'AMD'
quote_df = get_quote_info(s,c)
print(quote_df)
print(f"The last price for {s.upper()} is {quote_df.loc['lastPrice'][0]}.")

#----------------------------------------------------------------
# for request with multiple symbols, breaks long lists into batches
batch_size = 250
symbols_batches = split_symbol_list(symbols_list, batch_size)

#----------------------------------------------------------------  
# get quote data for many stocks in batches due to api restrictions
getquotes_df = pd.DataFrame() #initialize dataframe
for symbols_batch in symbols_batches:
    df = get_quotes_from_list(symbols_batch, c)       
    getquotes_df = pd.concat([getquotes_df, df])
    sleep(2)
    
# getquotes_df.to_csv('datasets/quotes.csv')

#----------------------------------------------------------------  
# get fundamentals data for many stocks in batches due to api restrictions
fundamentals_df = pd.DataFrame() # initialize df
for symbols_batch in symbols_batches:
    with io.capture_output() as captured:
        df = get_unpack_fundamentals_from_list(symbols_batch, c)
        fundamentals_df = pd.concat([fundamentals_df, df])
    sleep(2)
fundamentals_df.drop(columns='fundamental',inplace=True)
# fundamentals_df.to_csv('datasets/fundamentals.csv')

#----------------------------------------------------------------  
# get options chain for one symbol
s='AMD'
max_days_to_expiration = 60
max_strike_date = datetime.now() + timedelta(days=max_days_to_expiration)

options_df = load_create_options_table(s, max_strike_date,c)
# options_df.to_csv('datasets/o_{}.csv'.format(s))
