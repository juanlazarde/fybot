import asyncio
import datetime
import os
import sys
from ast import literal_eval
from ftplib import FTP
from time import time as t

import logging
import asyncpg
import pandas as pd
import pytz
import yahooquery as yq
import yfinance as yf

from core.database import Database
from core.settings import S
from core.tda import TDA

log = logging.getLogger(__name__)


class Snapshot:
    def __init__(self, forced=False):
        """Get symbols, pricing, fundamentals from internet, database or file.
        Then save them to a file and database. Sources: NASDAQ, Wikipedia,
        Alpaca

        :param forced: True to force data refresh"""

        s = t()
        assets = self.GetAssets(forced)
        self.GetFundamental(assets.symbols, forced)
        self.GetPrice(assets.symbols, forced)

        log.info(f"{self.__class__.__name__} took {t() - s} seconds")

    class GetAssets:
        def __init__(self, forced=False):
            """Load symbols from database or file, when older than 24 hours it
            downloads. If DEBUG then symbols list is limited.

            :param forced: True to force symbols download
            :return: list of symbols, dataframe with symbol/security names"""

            self.data = self.load(forced)
            self.unique_id()
            self.save()
            self.data, wihtin24 = self.load_database()
            self.symbols = self.data['symbol'].to_list()
            self.symbols = self.symbols[:S.MAX_SYMBOLS]

        def load(self, forced):
            """Gets the assets from database, file or internet

            :param bool forced: True to force symbols download
            :return: list of symbols, dataframe with symbol/security names"""

            # get symbols from database or file
            try:
                df, within24 = self.load_database()
            except Exception as e:
                log.info(e)
                try:
                    df, within24 = self.load_file()
                except Exception as e:
                    log.info(e)
                    df, within24 = None, False

            # download the symbols if data it's older than 24 hours or forced
            if not within24 or forced:
                df = self.download()

            # format symbols for YFinance and create symbols list
            df.replace({'symbol': r'\.'}, '-', regex=True, inplace=True)

            return df

        @staticmethod
        def load_database():
            """Load symbols & security names from database, excludes rejects"""

            with Database() as db:
                query = """SELECT symbol, security FROM symbols
                           WHERE symbols.id
                           NOT IN (SELECT symbol_id FROM rejected_symbols)
                           ORDER BY symbol;"""
                df = pd.read_sql_query(query, con=db.connection)
                within24 = db.last_update(table='symbols')

            if df.empty:
                raise Exception("Empty symbols database")

            log.info("Symbols loaded from database")

            return df, within24

        @staticmethod
        def load_file():
            """Loads symbols from pickle file in the datasets directory"""

            if os.path.isfile(S.SYMBOLS_FILE):
                df = pd.read_csv(S.SYMBOLS_FILE)
                if df.empty:
                    raise Exception("Empty symbols file")
                df = df.rename(columns=str.lower)
                df = df[['symbol', 'security']]

                timestamp = os.stat(S.SYMBOLS_FILE).st_mtime
                timestamp = datetime.datetime.fromtimestamp(
                    timestamp, tz=datetime.timezone.utc)
                now_utc = pytz.utc.localize(datetime.datetime.utcnow())
                within24 = (now_utc - timestamp).seconds <= 24 * 60 * 60

                log.info('Symbols loaded from CSV file')

                return df, within24
            else:
                log.info("Symbols file not found")
                raise Exception("Empty symbols file")

        def download(self):
            """Downloads symbols from the internet, cascade through sources"""

            # download from Nasdaq
            try:
                df = self.download_from_nasdaq()
                log.info("Symbols downloaded from Nasdaq")
                return df
            except Exception as e:
                log.warning(f"Nasdaq download failed. {e}")

            # or download from Alpaca
            try:
                df = self.download_alpaca()
                log.info("Symbols downloaded from Alpaca")
                return df
            except Exception as e:
                log.warning(f"Alpaca download failed. {e}")

            # or download S&P 500 from Wikipedia
            try:
                df = self.download_sp500()
                log.info("Symbols downloaded from Wikipedia S&P 500")
                return df
            except Exception as e:
                log.warning(f"Wikipedia S&P500 download failed. {e}")

            # unable to download symbols
            log.critical('Could not download the Symbols. Check code/sources')
            sys.exit(1)

        @staticmethod
        def download_from_nasdaq():
            """Downloads NASDAQ-traded stock/ETF Symbols, from NASDAQ FTP.

            Filtering: NASDAQ-traded, no test issues, round Lots,
            good financial status, no NextShares, Symbols with fewer than 5
            digits, and other keywords (no funds, notes, depositary, bills,
            warrants, unit, security)

            Info is saved into 'nasdaqtraded.txt' file, available via FTP,
            updated nightly. Logs into ftp.nasdaqtrader.com, anonymously, and
            browses for SymbolDirectory.
            ftp://ftp.nasdaqtrader.com/symboldirectory
            http://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs

            :return: Symbols and Security names in DataFrame object"""

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
                        (df['Symbol'].str.len() <= 5) &
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
            api = tradeapi.REST(key_id=S.ALPACA_KEY,
                                secret_key=S.ALPACA_SECRET,
                                base_url=S.ALPACA_ENDPOINT)
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

        def unique_id(self):
            """Create and embeds a Unique ID for every symbol.

            symbol_id = symbol filled with 0 up to 6 digits + first 6 chars
            of name, all alphanumeric. Example: 0000PGPROCTE"""
            self.data['symbol_id'] = (
                    self.data['symbol']
                    .str.replace('[^a-zA-Z0-9]', '', regex=True)
                    .str[:6]
                    .str.zfill(6) +
                    self.data['security']
                    .str.replace('[^a-zA-Z0-9]', '', regex=True)
                    .str[:6]
                    .str.upper()
                    .str.zfill(6)
            )

            log.info("Unique ID created")

        def save(self):
            """Saves the symbol data in a pickle file and the database"""

            self.save_to_file()
            asyncio.run(self.save_to_database())

        def save_to_file(self):
            """Save symbols to file"""

            self.data.to_csv(S.SYMBOLS_FILE, index=False)

            log.info(f"Symbols saved to CSV file: {S.SYMBOLS_FILE}")

        async def save_to_database(self):
            """ASYNC function to save database"""

            values = self.data.to_records(index=False)
            sql = """INSERT INTO symbols (symbol, security, id)
                     VALUES (%s, %s, %s)
                     ON CONFLICT (id) DO UPDATE
                     SET security=excluded.security;"""
            with Database() as db:
                db.executemany(sql, values)
                db.timestamp('symbols')
                db.commit()

            log.info("Symbols saved to database")

    class GetFundamental:
        def __init__(self, symbols, forced=False):
            """Load fundamentals from datbase or file, when older than 24 hours
            it downloads.

            :param list symbols: list of symbols
            :param bool forced: True bypasses within 24 hour requirement
            :return: dataframe with fundamental info."""

            self.data = None
            within24 = Database().last_update('fundamentals', close=True)

            # download or load from database/file. If from file, save database
            if not within24 or forced:
                try:
                    self.download(symbols)
                    self.save()
                except Exception as e:
                    log.info(f"Error downloading or saving Fundamentals {e}")
                    sys.exit(e)
            else:
                try:
                    self.load_database()
                    self.process_loaded_fundamentals()
                except Exception as e:
                    log.info(e)
                    try:
                        self.load_file()
                        asyncio.run(self.save_to_database())
                    except Exception as e:
                        log.critical(f"Could not load fundamental data. {e}")
                        sys.exit(1)

        def download(self, symbols):
            """Download fundamentals from different sources and returns a
            single dataframe, four columns: stock, source, variable & value.

            List of symbols is chunkable by changing the chunk_size:
                * chunk_size is optional. Empty downloads all.
                * it can be a constant, i.e. 250.
                * or divide all symbols into equal chunks

            :param list symbols: list of symbols to download fundamentals"""

            chunk_size = 5
            # n = 3  # 3 for divide symbols into thirds
            # chunk_size = (len(symbols) + n) // n  # into n-ths

            df_tda = self.download_from_tda(symbols, chunk_size)
            df_yahoo = self.download_from_yahoo(symbols, chunk_size)

            # concatenate tables using 'symbol' as index
            self.data = pd.concat([df_tda, df_yahoo],
                                  keys=['tda', 'yahoo'],
                                  axis=1)
            self.data.rename_axis('symbol', inplace=True)

            # update symbol rejects database
            Database().update_rejected_symbols(
                symbols=symbols,
                current=list(self.data.index),
                close=True)

        @staticmethod
        def unpack_dictionary_fundamental(df, packages: list):
            """Unpack dictionaries and concatenate to table"""
            # df = pd.concat([pd.DataFrame(None, index=['package']), df]) # DBG
            for package in packages:
                on_hand = df.loc[df[package].notnull(), package]
                unpack_df = pd.json_normalize(on_hand).set_index(on_hand.index)
                # unpack_df.loc['package'] = package[1:]  # DBG
                df.drop(package, axis=1, inplace=True)
                df = pd.concat([df, unpack_df], axis=1)
            df.rename_axis('symbol', inplace=True)
            return df

        def download_from_tda(self, symbols: list, chunk_size: int = None):
            """Get TD Ameritrade fundamental data in batches, if necessary."""
            # Create connection, return empty df if there is no connection
            try:
                con = TDA()
            except Exception as e:
                log.info(f"TDA Connection failed. {e}")
                return pd.DataFrame()

            # download stocks data in chunks and concatenate to large table
            chunk_size = len(symbols) if chunk_size is None else chunk_size

            df = pd.DataFrame()
            for i in range(0, len(symbols), chunk_size):
                symbol_chunk = symbols[i:i + chunk_size]
                tickers = con.client.search_instruments(
                    symbols=symbol_chunk,
                    projection=con.client.Instrument.Projection.FUNDAMENTAL)
                if tickers.status_code == 200:
                    chunk_df = pd.DataFrame(tickers.json()).T
                    df = pd.concat([df, chunk_df])
                else:
                    raise Exception("TDA data download, unsuccesful")

            # unpack the dictionaries
            """['fundamental']"""
            packages = ['fundamental']
            # packages = df.columns  # for unpacking all

            df = self.unpack_dictionary_fundamental(df, packages)
            log.info("TDA fundamentals downloaded")
            return df

        def download_from_yahoo(self, symbols: list, chunk_size: int = None):
            """Get Yahoo fundamental data in batches, if necessary."""
            # download stocks data in chunks and concatenate to large table
            chunk_size = len(symbols) if chunk_size is None else chunk_size

            df = pd.DataFrame()
            for i in range(0, len(symbols), chunk_size):
                symbol_chunk = symbols[i:i + chunk_size]
                tickers = yq.Ticker(symbols=symbol_chunk,
                                    asynchronous=True).all_modules
                chunk_df = pd.DataFrame(tickers).T
                df = pd.concat([df, chunk_df])

            # unpack the dictionaries
            """['assetProfile', 'recommendationTrend', 'industryTrend',
             'cashflowStatementHistory', 'indexTrend', 
             'defaultKeyStatistics', 'quoteType', 'fundOwnership',
             'incomeStatementHistory', 'summaryDetail', 'insiderHolders', 
             'calendarEvents', 'upgradeDowngradeHistory', 'price',
             'balanceSheetHistory', 'earningsTrend', 'secFilings',
             'institutionOwnership', 'majorHoldersBreakdown',
             'balanceSheetHistoryQuarterly', 'earningsHistory', 
             'esgScores', 'summaryProfile', 'netSharePurchaseActivity', 
             'insiderTransactions',  'sectorTrend', 'fundPerformance', 
             'incomeStatementHistoryQuarterly', 'financialData', 
             'cashflowStatementHistoryQuarterly', 'earnings', 'pageViews',
             'fundProfile', 'topHoldings']"""
            packages = ['summaryProfile', 'summaryDetail', 'quoteType']
            # packages = df.columns  # for unpacking all

            # adding prefix to avoid duplicated label errors unpacking
            df = df.add_prefix("_")
            packages = ["_" + i for i in packages]

            # unpack dictionaries and concatenate to table
            df = self.unpack_dictionary_fundamental(df, packages)

            # deal with duplicates
            erase = ['maxAge', 'symbol']
            try:
                df.drop(erase, axis='columns', inplace=True, errors='ignore')
            except AttributeError as e:
                log.info(f"Dataframe may be empty. {e}")
                raise

            log.info("Yahoo fundamentals downloaded")
            return df

        def load_database(self):
            """Retrieves fundamental data from database, filtered of rejects"""
            query = """SELECT symbols.symbol,
                              fundamentals.source,
                              fundamentals.var,
                              fundamentals.val 
                       FROM fundamentals
                       INNER JOIN symbols
                       ON fundamentals.symbol_id = symbols.id
                       WHERE symbols.id
                       NOT IN (SELECT symbol_id FROM rejected_symbols)
                       ORDER BY symbols.symbol;"""
            with Database() as db:
                self.data = pd.read_sql_query(query, con=db.connection)

            log.info("Loaded fundamentals from database")

        def load_file(self):
            """Gets fundamentals data from pickle file"""
            self.data = pd.read_pickle(S.FUNDAMENTALS_FILE)

            log.info("Loaded fundamentals from file")

        def process_loaded_fundamentals(self):
            """Transform database format to user-friendly format"""
            def literal_return(val):
                try:
                    return literal_eval(val)
                except (ValueError, SyntaxError):
                    return val

            df = self.data
            df['val'] = df['val'].apply(lambda x: literal_return(x))
            self.data = df.pivot(index=['symbol'],
                                 columns=['source', 'var'],
                                 values=['val'])['val']

        def save(self):
            """Saves the fundamental data in a pickle file and the database"""

            self.save_to_file()
            asyncio.run(self.save_to_database())

        def save_to_file(self):
            """Save fundamental data in a pickle file"""

            self.data.to_pickle(S.FUNDAMENTALS_FILE)
            self.data.T.to_csv(S.FUNDAMENTALS_FILE + ".csv")
            log.info(f"Fundamentals saved to pickle file: "
                     f"{S.FUNDAMENTALS_FILE}")

        async def save_to_database(self):
            """Save fundamental data in a database"""
            df = self.data
            df = df.unstack().reset_index()
            df.columns = ['source', 'variable', 'symbol', 'value']
            df = df[['symbol', 'source', 'variable', 'value']]
            df['value'] = df['value'].astype(str)

            values = df.to_records(index=False)
            query = """INSERT INTO fundamentals (symbol_id, source, var, val)
                       VALUES ((SELECT id FROM symbols WHERE symbol=$1),
                               $2, $3, $4)
                       ON CONFLICT (symbol_id, source, var)
                       DO UPDATE
                       SET symbol_id=excluded.symbol_id, 
                           source=excluded.source, 
                           var=excluded.var, 
                           val=excluded.val;"""

            # using asynciopf as the Database connection manager for Async
            creds = {'host': S.DB_HOST, 'database': S.DB_NAME,
                     'user': S.DB_USER, 'password': S.DB_PASSWORD}
            async with asyncpg.create_pool(**creds) as pool:
                async with pool.acquire() as db:
                    async with db.transaction():
                        await db.executemany(query, values)

            Database().timestamp('fundamentals', close=True)

        # list of columns in YFinance
        """['zip', 'sector', 'fullTimeEmployees']
        ['longBusinessSummary', 'city', 'phone']
        ['state', 'country', 'companyOfficers']
        ['website', 'maxAge', 'address1']
        ['industry', 'previousClose', 'regularMarketOpen']
        ['twoHundredDayAverage', 'trailingAnnualDividendYield', 'payoutRatio']
        ['volume24Hr', 'regularMarketDayHigh', 'navPrice', , 'totalAssets']
        ['averageDailyVolume10Day', 'regularMarketPreviousClose']
        ['fiftyDayAverage', 'trailingAnnualDividendRate', 'open']
        ['toCurrency', 'averageVolume10days', 'expireDate']
        ['yield', 'algorithm', 'dividendRate']
        ['exDividendDate', 'beta', 'circulatingSupply']
        ['startDate', 'regularMarketDayLow', 'priceHint']
        ['currency', 'trailingPE', 'regularMarketVolume']
        ['lastMarket', 'maxSupply', 'openInterest']
        ['marketCap', 'volumeAllCurrencies', 'strikePrice']
        ['averageVolume', 'priceToSalesTrailing12Months', 'dayLow']
        ['ask', 'ytdReturn', 'askSize']
        ['volume', 'fiftyTwoWeekHigh', 'forwardPE']
        ['fromCurrency', 'fiveYearAvgDividendYield', 'fiftyTwoWeekLow']
        ['bid', 'tradeable', 'dividendYield']
        ['bidSize', 'dayHigh', 'exchange']
        ['shortName', 'longName', 'exchangeTimezoneName', 'isEsgPopulated']
        ['exchangeTimezoneShortName', 'gmtOffSetMilliseconds']
        ['quoteType', 'symbol', 'messageBoardId']
        ['market', 'annualHoldingsTurnover', 'enterpriseToRevenue']
        ['beta3Year', 'profitMargins', 'enterpriseToEbitda']
        ['52WeekChange', 'morningStarRiskRating', 'forwardEps']
        ['revenueQuarterlyGrowth', 'sharesOutstanding', 'fundInceptionDate']
        ['annualReportExpenseRatio', 'bookValue', 'sharesShort']
        ['sharesPercentSharesOut', 'fundFamily', 'lastFiscalYearEnd']
        ['heldPercentInstitutions', 'netIncomeToCommon', 'trailingEps']
        ['lastDividendValue', 'SandP52WeekChange', 'priceToBook']
        ['heldPercentInsiders', 'nextFiscalYearEnd', 'mostRecentQuarter']
        ['shortRatio', 'sharesShortPreviousMonthDate', 'floatShares']
        ['enterpriseValue', 'threeYearAverageReturn', 'lastSplitDate']
        ['lastSplitFactor', 'legalType', 'lastDividendDate']
        ['morningStarOverallRating', 'earningsQuarterlyGrowth']
        ['dateShortInterest', 'pegRatio', 'lastCapGain', 'shortPercentOfFloat']
        ['sharesShortPriorMonth', 'impliedSharesOutstanding', 'category']
        ['fiveYearAverageReturn', 'regularMarketPrice', 'logo_url']
        """

    class GetPrice:
        def __init__(self, symbols=None, forced=False):
            """Gets price history data from download if not within 24 hours,
               then from data, finally from file.

               :param symbols: list of symbols
               :param forced: True bypassess 24 hour download requirement"""

            self.data = None
            within24 = Database().last_update('price_history', close=True)

            # download or load from database or file. If from file, save db
            if not within24 or forced:
                asyncio.run(self.download(symbols))
                self.save()
            else:
                try:
                    self.data = self.load_database()
                except Exception as e:
                    log.info(e)
                    try:
                        self.load_file()
                        asyncio.run(self.save_to_database())
                    except Exception as e:
                        log.critical(f"Could not load price data. {e}")
                        sys.exit(1)

        async def download(self, symbols):
            """Download price data.

            List of symbols is chunkable by changing the chunk_size.

            :param list symbols: list of symbols to download fundamentals"""

            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=S.DAYS)

            collector = pd.DataFrame()

            # chunk_size = 2
            # chunk_size = len(symbols)  # use actual chunk size i.e. 200
            n = 3
            chunk_size = (len(symbols) + n) // n  # into n-ths

            for i in range(0, len(symbols), chunk_size):
                symbol_chunk = symbols[i:i + chunk_size]

                # download chunk of fundamentals from YFinance
                df_chunk = await self.yfinance_price_history(symbol_chunk,
                                                             start_date,
                                                             end_date)

                # YFinance won't return symbol label if it's only one symbol
                if len(symbol_chunk) == 1:
                    df_chunk.columns = pd.MultiIndex.from_product(
                        [df_chunk.columns, symbol_chunk]).swaplevel(0, 1)

                collector = pd.concat([collector, df_chunk], axis=1)

            # cleanup any NaN values in the dataframe
            collector.dropna(how='all', inplace=True)
            collector.dropna(how='all', axis=1, inplace=True)

            self.data = collector.stack(level=0).reset_index()
            self.data.rename(columns={'level_1': 'symbol'}, inplace=True)

            # update symbol rejects database
            Database().update_rejected_symbols(
                symbols=symbols,
                current=self.data['symbol'].unique(),
                close=True)

            log.info("Price history downloaded")

        # TODO: JUST GET LATEST PRICE DATA, MAY REQUIRE SINGLE TICKER DOWNLOAD
        # @staticmethod
        # def date_range(self):
        #     """Dates to download history.
        #     Total days comes from Settings, whether you want 150 days back.
        #     The dat of the last update is the latest date in the symbols
        #     price history date field.
        #     """
        #     return start_date, end_date

        @staticmethod
        async def yfinance_price_history(symbol_chunk, start_date, end_date):
            """ASYNC function to download price history data from YFinance"""

            return yf.download(symbol_chunk,  # list of tickers
                               start=start_date,  # start date
                               end=end_date,  # end date
                               period='max',  # change instead of start/end
                               group_by='ticker',  # or 'column'
                               progress=False,  # show progress bar
                               rounding=False,  # round 2 descimals
                               actions=False,  # include dividend/earnings
                               threads=True,  # multi threads bool or int
                               auto_adjust=False,  # adjust OHLC with adj_close
                               back_adjust=False,
                               prepost=False,  # include before/after market
                               )

        @staticmethod
        def load_database():
            """Retrieves the price data from database, filtered of rejects"""

            query = """ SELECT price_history.date,
                            symbols.symbol,
                            price_history.open,
                            price_history.high,
                            price_history.low,
                            price_history.close,
                            price_history.adj_close,
                            price_history.volume
                        FROM price_history
                        INNER JOIN symbols
                        ON price_history.symbol_id = symbols.id
                        WHERE symbols.id
                        NOT IN (SELECT symbol_id FROM rejected_symbols)
                        ORDER BY symbols.symbol, price_history.date;"""

            with Database() as db:
                df = pd.read_sql_query(query, con=db.connection)

            # df.drop(['symbol_id'], inplace=True, axis=1)

            log.info("Loaded price history from database")

            return df

        def load_file(self):
            self.data = pd.read_pickle(S.PRICE_FILE)

        def save(self):
            """Saves the pricing data in a pickle file and the database"""

            self.save_to_file()
            asyncio.run(self.save_to_database())

        def save_to_file(self):
            """Save pricing data in a pickle file"""

            self.data.to_pickle(S.PRICE_FILE)

            log.info(f"Price data saved to pickle file: {S.PRICE_FILE}")

        async def save_to_database(self):
            """Save price data in a database"""

            rows = self.data.itertuples(index=False)
            values = [list(row) for row in rows]
            query = """INSERT INTO price_history (date, symbol_id, open,
                                                  high, low, close, 
                                                  adj_close, volume)
                       VALUES ($1, (SELECT id FROM symbols WHERE symbol=$2),
                                $3, $4, $5, $6, $7, $8)
                       ON CONFLICT (symbol_id, date)
                       DO UPDATE
                       SET  symbol_id=excluded.symbol_id,
                            date=excluded.date, open=excluded.open,
                            high=excluded.high, low=excluded.low,
                            close=excluded.close,
                            adj_close=excluded.adj_close,
                            volume=excluded.volume;"""

            # using asynciopg as the Database connection manager for Async
            creds = {'host': S.DB_HOST, 'database': S.DB_NAME,
                     'user': S.DB_USER, 'password': S.DB_PASSWORD}
            async with asyncpg.create_pool(**creds) as pool:
                async with pool.acquire() as db:
                    async with db.transaction():
                        await db.executemany(query, values)

            Database().timestamp('price_history', close=True)
