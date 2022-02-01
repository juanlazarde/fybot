# -*- coding: utf-8 -*-
"""Utilities for TDA"""

import os
import platform
import sys
import time
from ftplib import FTP
from datetime import datetime, timedelta, date, time, timezone

import pandas as pd


def creation_date(path_to_file):
    """Gets the file creation date.

    Falls back to when it was last modified if that isn't possible.
    See https://stackoverflow.com/a/39501288/1709587 for explanation.

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


# noinspection SpellCheckingInspection
def download_symbols(reference_path="data/"):
    """Downloads NASDAQ-traded stock/ETF Symbols, from NASDAQ FTP.

    Info is saved into 'nasdaqtraded.txt' file. Info is available via FTP,
    updated nightly. Logs into ftp server, anonymously, and browses
    SymbolDirectory.
    ftp://ftp.nasdaqtrader.com/symboldirectory
    https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs

    Args:
        reference_path (str): Local path to save the symbols file. By
        default, is 'data/'
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


def import_symbols(reference_path="data/"):
    """Imports Symbols (from download_symbols NASDAQ).

    Imports ticker list of NASDAQ-traded stocks and ETFs. The data is also
    filtered to exclude Delinquency, Tests, non-traditional, 5+ digit
    tickers, and more.
    This link has the references to the files in NASDAQ Trader:
    https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs

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
        date_created = creation_date(symbol_file)
    except OSError:
        date_created = time.time() + 7 * 25 * 60 * 60

    seconds_apart = time.time() - date_created

    if not os.path.exists(symbol_file) or seconds_apart >= 7 * 24 * 60 * 60:
        print("File doesn't exist or it's over 7 days old.\n" +
              "Will download the latest version from Nasdaq.")
        download_symbols(reference_path)
        print("Latest Symbols database downloaded successfully!")

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


def divider(characters=140):
    print("\n" + "-" * characters + "\n")


def split(a, n):
    """Splits list a into n equal sizes.

    Generator syntax is list(split(a,n))

    Args:
        a (double): Original table or list.
        n (double): number of pieces.

    Returns:
        List divided into n number of batches.
    """
    k, m = divmod(len(a), n)

    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write(
            "%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()


def run_once(f):
    """Runs part of the code once.

    https://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop

    Usage:
    ::
        @run_once
        def monte_carlo(...):

        or

        action = run_once(my_function)
        while 1:
            if predicate:
                action()

        or

        action = run_once(my_function)
        action() # run once the first time

        action.has_run = False
        action() # run once the second time

    """
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.run = False
    return wrapper


def market_is_open():
    """Return True or False whether the market is open or not."""
    current_weekday = date.today().weekday()
    current_time = (datetime
                    .now()
                    .astimezone(timezone(timedelta(hours=-5)))
                    .time())

    return (current_weekday < 5 and
            time(16, 0) >= current_time >= time(9, 30))
