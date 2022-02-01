import datetime
import logging
import os
import sys
from time import time as t

import pandas as pd

from core.database import Database
import core.settings as ss

log = logging.getLogger(__name__)


class GetAssets:
    def __init__(self,
                 source: str = '',
                 forced: bool = False):
        """Load symbols from database or file, when older than 24 hours it
        downloads. If DEBUG then symbols list is limited.

        Returns: list of symbols, dataframe with symbol/security names.

        :param source: 'nasdaq' (default), 'alpaca', 'sp500'
        :param forced: True to force symbols download, bypasses within 24 req
        """

        log.debug("Loading assets")
        s = t()

        data = pd.DataFrame()

        # checks on last database update for symbols
        within24 = Database().last_update('symbols', close=True)

        # load from database if within 24 hours
        if within24 and not forced:
            data = self.load_database()

        # download data if database is empty or expired data
        if data.empty or data is None or not within24 or forced:
            data = self.download(source)
            data = self.apply_filters(data)
            data = data.assign(symbol_id=lambda x: self.unique_id(x))
            self.save_to_database(data)
            # reloading to exclude rejected symbols
            data = self.load_database()

        if data.empty or data is None:
            raise Exception("Empty symbols database")

        symbols = data['symbol'].to_list()
        symbols = symbols[:ss.MAX_SYMBOLS]

        self.data = data
        self.symbols = symbols

        log.debug(f"Assets Loaded in {t()-s} secs")

    @staticmethod
    def load_database():
        """Load symbols & security names from database, excludes rejects"""

        query = """
            SELECT symbol, security FROM symbols
            WHERE symbols.id
            NOT IN (SELECT symbol_id FROM rejected_symbols)
            ORDER BY symbol;
            """
        with Database() as db:
            df = pd.read_sql_query(query, con=db.connection)
        log.debug("Symbols loaded from database")
        return df

    def download(self, source: str = 'nasdaq'):
        """Downloads symbols from the internet, cascade through sources

        :param source: 'nasdaq' (default), 'alpaca', 'sp500'
        """
        source = 'nasdaq' if (source == '' or source is None) else source
        # download from Nasdaq
        try:
            assert source == 'nasdaq'
            df = self.download_from_nasdaq()
            log.debug("Symbols downloaded from Nasdaq")
            return df
        except Exception as e:
            log.warning(f"Symbol list not loaded from Nasdaq. {e}")

        # or download from Alpaca
        try:
            assert source == 'alpaca'
            df = self.download_alpaca()
            log.debug("Symbols downloaded from Alpaca")
            return df
        except Exception as e:
            log.warning(f"Symbol list not loaded from Alpaca. {e}")

        # or download S&P 500 from Wikipedia
        try:
            assert source == 'sp500'
            df = self.download_sp500()
            log.debug("Symbols downloaded from Wikipedia S&P 500")
            return df
        except Exception as e:
            log.warning(f"Symbol list not loaded from Wikipedia S&P500. {e}")

        # unable to download symbols
        log.critical('Could not download the Symbols. Check code/sources')
        sys.exit(1)

    @staticmethod
    def download_from_nasdaq():
        """Downloads NASDAQ-traded stock/ETF Symbols, from NASDAQ FTP.

        Filtering: NASDAQ-traded, no test issues, round Lots,
        good financial status, no NextShares, Symbols with fewer than 6
        digits, and other keywords (no funds, notes, depositary, bills,
        warrants, unit, security)

        Info is saved into 'nasdaqtraded.txt' file, available via FTP,
        updated nightly. Logs into ftp.nasdaqtrader.com, anonymously, and
        browses for SymbolDirectory.
        ftp://ftp.nasdaqtrader.com/symboldirectory
        http://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs

        :return: Symbols and Security names in DataFrame object"""

        from ftplib import FTP

        lines = []
        with FTP(host='ftp.nasdaqtrader.com') as ftp:
            ftp.login()
            ftp.cwd('symboldirectory')
            ftp.retrlines("RETR nasdaqtraded.txt", lines.append)
            ftp.quit()
        lines = [lines[x].split('|') for x in range(len(lines) - 1)]
        df = pd.DataFrame(data=lines[1:], columns=lines[0])
        cols = {'NASDAQ Symbol': 'symbol', 'Security Name': 'security'}
        drop = " warrant|depositary|- unit| notes |interest|%| rate "
        df = df.loc[(df['Nasdaq Traded'] == 'Y') &
                    (df['Listing Exchange'] != 'V') &
                    (df['Test Issue'] == 'N') &
                    (df['Round Lot Size'] == '100') &
                    (df['NextShares'] == 'N') &
                    (df['Symbol'].str.len() <= 6) &
                    (df['Financial ''Status'].isin(['', 'N'])) &
                    (~df['Security Name'].str.contains(drop, case=False)),
                    list(cols.keys())]
        df.rename(columns=cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def download_alpaca():
        """Downloads assets from Alpaca and retrieves symbols. Assets
        from Alpaca contain a lot more than symbols and names.

        :return: Symbols and Security names in DataFrame object"""

        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(key_id=ss.ALPACA_KEY,
                            secret_key=ss.ALPACA_SECRET,
                            base_url=ss.ALPACA_ENDPOINT)
        active_assets = api.list_assets(status='active')
        tradable_assets = [(asset.symbol, asset.name) for asset in
                           active_assets if asset.tradable]
        result = [dict(zip(['symbol', 'security'], asset)) for asset in
                  tradable_assets]
        df = pd.DataFrame(result, columns=['symbol', 'security'])
        return df

    @staticmethod
    def download_sp500():
        """Scrape wikipedia for S&P500 list

        :return: Symbols and Security names in DataFrame object"""

        table = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        df = df[['Symbol', 'Security']]
        df = df.rename(columns=str.lower)
        return df

    @staticmethod
    def apply_filters(df):
        """Apply filters to Symbol Series.

        Removes symbol with characters like .=^-. These are usually special
        symbols for Units, Series A, B, Indices, and others. These have
        conflicts across platforms and would have to be harmonized.
        i.e. TDA GRP.U is YAHOO's GRP-UN. These usually have small volume
        and no options anyway.

        :param df: Symbol DataFrame
        """

        filtered_symbol = df.loc[df['symbol'].str.isalpha()]

        # format symbols for YFinance and create symbols list
        # df.replace({'symbol': r'\.'}, '-', regex=True, inplace=True)
        log.debug("Filters applied to symbols")
        return filtered_symbol

    @staticmethod
    def unique_id(df):
        """Create and embeds a Unique ID for every symbol.

        symbol_id = symbol filled with 0 up to 6 digits + first 6 chars
        of name, all alphanumeric. Example: 0000PGPROCTE"""
        result = (
                df['symbol']
                .str.replace('[^a-zA-Z0-9]', '', regex=True)
                .str[:6]
                .str.zfill(6) +
                df['security']
                .str.replace('[^a-zA-Z0-9]', '', regex=True)
                .str[:6]
                .str.upper()
                .str.zfill(6)
        )
        log.debug("Unique ID created")
        return result

    @staticmethod
    def save_to_database(df):
        """Save data to database"""

        values = df.to_records(index=False)
        sql = """
            INSERT INTO symbols (symbol, security, id)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET security=excluded.security;
            """
        with Database() as db:
            db.executemany(sql, values)
            db.timestamp('symbols')
            db.commit()
        log.debug("Symbols saved to database")


class File(GetAssets):
    def save(self):
        """Save symbols to file"""

        self.data.to_csv(ss.SYMBOLS_FILE, index=False)
        log.info(f"Symbols saved to CSV file: {ss.SYMBOLS_FILE}")
        return self.symbols

    @staticmethod
    def load():
        """Loads symbols from pickle file in the datasets directory"""

        import pytz

        if os.path.isfile(ss.SYMBOLS_FILE):
            df = pd.read_csv(ss.SYMBOLS_FILE)
            if df.empty:
                raise Exception("Empty symbols file")
            df = df.rename(columns=str.lower)
            df = df[['symbol', 'security']]

            timestamp = os.stat(ss.SYMBOLS_FILE).st_mtime
            timestamp = datetime.datetime.fromtimestamp(
                timestamp, tz=datetime.timezone.utc)
            now_utc = pytz.utc.localize(datetime.datetime.utcnow())
            within24 = (now_utc - timestamp).seconds <= 24 * 60 * 60

            log.info('Symbols loaded from CSV file')

            return df, within24
        else:
            log.info("Symbols file not found")
            raise Exception("Empty symbols file")


if __name__ == '__main__':
    _symbols = GetAssets(source='sp500', forced=True).load_database()
    pass
