import pandas as pd
from IPython.utils import io
from tda.client import Client
import os
import pickle
import platform
import sys
import time
from ftplib import FTP

import pandas as pd

def make_webdriver():
    # Import selenium here because it's slow to import
    from selenium import webdriver
    import atexit

    driver = webdriver.Chrome()
    atexit.register(lambda: driver.quit())
    return driver

def split(a, n):
    """Splits list a into n equal sizes.

    Generator syntax is list(split(a,n))

    Args:
        a (double): Original table or list.
        n (double): number of pieces.

    Returns:
        List divided in to n number of batches.
    """
    k, m = divmod(len(a), n)

    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def split_symbol_list(symbols_list, batch_size=250):
    n_batches = round(len(symbols_list) / batch_size) + 1
    symbols_batch = list(split(symbols_list, n_batches))
    return symbols_batch

def get_quote_info(s, c):
    resp = c.get_quote(s)
    assert resp.status_code == 200, resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    dropped_fields = ['askId', 'assetMainType',
                      'assetType', 'bidId',
                      'bidTick', 'cusip',
                      'delayed', 'digits',
                      'exchange', 'lastId']
    return df
    
def get_quotes_from_list(symbols_list, c):
    resp = c.get_quotes(symbols_list)
    assert resp.status_code == 200, resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    return df.T

def get_unpack_fundamentals_from_list(symbols_list, c):
    resp = c.search_instruments(symbols_list, 
                                Client.Instrument.Projection.FUNDAMENTAL)
    assert resp.status_code == 200, resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df_unpack = pd.json_normalize(df.loc['fundamental'])
    df = df.T.join(df_unpack.set_index('symbol'))
    return df

def consolidate_quote_data(watchlist, c, export):
    #split list of symbols into manageable sizes
    batch_size = 250
    n_batches = round(len(watchlist) / batch_size) + 1
    symbols_batch = list(split(watchlist, n_batches))

    #initialize dataframe using the first stock
    #supress output, because TDA function prints out the api_key in plain text
    with io.capture_output() as captured:
        resp = c.search_instruments(
               watchlist[:1], Client.Instrument.Projection.FUNDAMENTAL)
    assert resp.status_code == 200, r.raise_for_status()
    fundamentals_df = pd.DataFrame(resp.json())
    fundamentals_df.drop(columns=fundamentals_df.columns, inplace=True)

    #append all stocks in batches due to api restrictions
    for symbols in symbols_batch[:]:
        with io.capture_output() as captured:
            resp = c.search_instruments(
                   symbols, Client.Instrument.Projection.FUNDAMENTAL)
        assert resp.status_code == 200, resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        fundamentals_df = pd.concat([fundamentals_df, df], axis=1)

    #unpack fundamentals and attach it to the quote table, to have a flat dataframe
    df = pd.json_normalize(fundamentals_df.loc['fundamental'])
    fundamentals_df = fundamentals_df.T
    fundamentals_df = fundamentals_df.drop(['fundamental'], axis=1)
    quotes_df = fundamentals_df.join(df.set_index('symbol'))

    #append Quote data to the table
    #initialize dataframe using the first stock
    resp = c.get_quotes(watchlist[:1])
    assert resp.status_code == 200, r.raise_for_status()
    getquotes_df = pd.DataFrame(resp.json())
    getquotes_df.drop(columns=getquotes_df.columns, inplace=True)

    #append all stocks in batches due to api restrictions
    for symbols in symbols_batch[:]:
        resp = c.get_quotes(symbols)
        assert resp.status_code == 200, r.raise_for_status()
        df = pd.DataFrame(resp.json())
        getquotes_df = pd.concat([getquotes_df, df], axis=1)
    getquotes_df = getquotes_df.T

    #merge quotes (left df) and getquotes (right df) by index, using a left join. Appends '_del' to duplicate columns from right df
    quotes_df = quotes_df.merge(getquotes_df, how='left',left_index=True, right_index=True, suffixes=('','_del'))

    #removes duplicate fields 
    #below is the simplest if it's a string of text, regex can be used for complex patterns
    #cols_to_drop = colsquotes_df.columns[quotes_df.columns.str.contains('_del')] 
    cols_to_drop = list(quotes_df.filter(regex='_del'))
    quotes_df.drop(columns = cols_to_drop, inplace=True)
    quotes_df.drop(columns = 'securityStatus', inplace=True)

    if export:
        filename = 'export_quotes'
        suffix = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = "_".join([filename, suffix])
        filename = cfg['raw_data_path'] + filename + ".csv"
        quotes_df.T.to_csv (filename, index = True, header=True)

    return quotes_df

def get_creation_date(path_to_file):
    """Gets the file creation date.

    Falls back to when it was last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.

    Args:
        path_to_file (str): Path to the file.

    Returns:
        Date when the file was created.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


def download_symbols(reference_path="data/"):
    """Downloads NASDAQ-traded stock/ETF Symbols, from NASDAQ FTP.

    Info is saved into 'nasdaqtraded.txt' file. Info is available via FTP,
    updated nightly. Logs into ftp.nasdaqtrader.com, anonymously, and browses
    SymbolDirectory.
    ftp://ftp.nasdaqtrader.com/symboldirectory
    http://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs

    Args:
        reference_path (str): Local path to save the symbols file. By
        default is 'data/'
    """
    ftp_path = 'symboldirectory'
    ftp_filename = 'nasdaqtraded.txt'
    local_filename = 'nasdaqtraded.txt'
    local_full_filename = reference_path + local_filename
    tmp_file = local_full_filename + ".tmp"

    print("Connecting to ftp.nasdaqtrader.com...")

    with FTP(host="ftp.nasdaqtrader.com") as ftp:
        try:
            ftp.login()  # anonymous login
            ftp.cwd(ftp_path)

            try:
                os.remove(tmp_file)
            except OSError:
                pass

            ftp_command = ftp.retrbinary('RETR ' + ftp_filename,
                                         open(tmp_file, 'wb').write)

            if not ftp_command.startswith('226 Transfer complete'):
                print('Download failed')
                if os.path.isfile(tmp_file):
                    os.remove(tmp_file)

            ftp.quit()
            print("Closed server connection...")

            if os.path.isfile(local_full_filename):
                os.remove(local_full_filename)
            os.rename(tmp_file, local_full_filename)

            print("Finished Investment Symbols download from Nasdaq.\n" +
                  "It's here: " + local_full_filename)
        except Exception as error_number:
            print('Error getting the Symbols file via FTP. Error:',
                  str(error_number))
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)


def import_symbols(reference_path="datasets/"):
    """Imports Symbols (from download_symbols NASDAQ).

    Imports ticker list of NASDAQ-traded stocks and ETFs. The data is also
    filtered to exclude Delinquency, Tests, non-traditional, 5+ digit
    tickers, and more.
    This link has the references to the files in NASDAQ Trader:
    http://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs

    Args:
        reference_path (str): Path of the textfile with the symbol information.

    Returns:
        df (dataframe): Table with all stocks.
    """
    symbol_file = reference_path + "nasdaqtraded.txt"
    print("File with latest Investment Symbols:", symbol_file)

    # Download symbols to nasdaqtraded.txt if it doesn't exist or older
    # than 7 days
    try:
        date_created = get_creation_date(symbol_file)
    except OSError:
        date_created = time.time() + 7 * 25 * 60 * 60

    seconds_apart = time.time() - date_created

    if not os.path.exists(symbol_file) or seconds_apart >= 7 * 24 * 60 * 60:
        print("File doesn't exist or it's over 7 days old.\n" +
              "Will download the latest version from Nasdaq.")
        download_symbols(reference_path)
        print("Latest Symbols database downloaded succesfully!")

    df = pd.read_csv(symbol_file, sep='|')

    # filters
    print("Filtering: NASDAQ Traded, No Tests, Round Lots, Good Financial "
          "Status, No Next Shares, Symbols with less than 5 digits, "
          "and other keywords")
    df = df[:-1]  # drop the last row
    df = df[df['Nasdaq Traded'] == 'Y']
    df = df[df['Listing Exchange'] != 'V']
    df = df[df['Test Issue'] == 'N']
    df = df[df['Round Lot Size'] >= 100]
    df = df[df['Financial Status'] != 'D']
    df = df[df['Financial Status'] != 'E']
    df = df[df['Financial Status'] != 'H']
    df = df[df['NextShares'] == 'N']
    df = df[df['Symbol'].str.len() <= 4]
    dropped_keywords = [' fund', ' notes', ' depositary', ' rate',
                        'due', ' warrant']
    for drp in dropped_keywords:
        df = df[df['Security Name'].str.lower() != drp.lower()]

    dropped_columns = ['Nasdaq Traded', 'Listing Exchange', 'Test Issue',
                       'Round Lot Size', 'Financial Status', 'NextShares',
                       'CQS Symbol', 'Market Category', 'NASDAQ Symbol']
    for drp in dropped_columns:
        df = df.drop(drp, axis=1)

    df.set_index('Symbol', inplace=True)
    print('Done')

    return df

def load_create_options_table(s, max_strike_date, c):
    o = c.get_option_chain(s,
                       contract_type=None,
                       strike_count=None, 
                       include_quotes=None, 
                       strategy=None, 
                       interval=None, 
                       strike=None, 
                       strike_range=Client.Options.StrikeRange.ALL,
                       strike_from_date=None, 
                       to_date=max_strike_date, 
                       volatility=None, 
                       underlying_price=None,                            
                       interest_rate=None, 
                       days_to_expiration=None, 
                       exp_month=None, 
                       option_type=None)

    option_chain = o.json()
    call_df = pd.json_normalize(option_chain['callExpDateMap'], record_prefix='Prefix.')
    put_df = pd.json_normalize(option_chain['putExpDateMap'], record_prefix='Prefix.')

    #unpack json format
    strike_one = {}
    strike_one['call'] = pd.DataFrame.from_dict(call_df.loc[0][0][0], orient='index')
    strike_one['put'] = pd.DataFrame.from_dict(put_df.loc[0][0][0], orient='index')

    for i in range(len(call_df.columns)):
        strike_one['call'][i] = pd.DataFrame.from_dict(call_df.loc[0][i][0], orient='index')
        strike_one['put'][i] = pd.DataFrame.from_dict(put_df.loc[0][i][0], orient='index')
        
    strike_one['call'].columns = strike_one['call'].loc['symbol']
    strike_one['put'].columns = strike_one['put'].loc['symbol']

    options_table = pd.concat(strike_one, axis=1)
    options_table = options_table.T
    return options_table