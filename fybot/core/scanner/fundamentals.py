import asyncio
import logging
from ast import literal_eval
from time import time as t

import asyncpg
import httpx
import pandas as pd
import yahooquery as yq

from core.database import Database
import core.settings as ss
from core.fy_tda import TDA

log = logging.getLogger(__name__)


class GetFundamental:
    def __init__(self, symbols: list, forced: bool = False):
        """Load fundamentals from database/file. Over 24 hrs it downloads.

        :param symbols: list of symbols
        :param forced: True bypasses within 24 hour requirement
        """

        log.debug("Loading fundamentals")
        s = t()

        data = pd.DataFrame()

        # checks on last database update for fundamentals
        within24 = Database().last_update('fundamentals', close=True)

        # load from database if within 24 hours
        if within24 and not forced:
            data = self.load_database()
            data = self.process_loaded_fundamentals(data)

        # download data if database is empty or expired data
        if data.empty or data is None or not within24 or forced:
            data = self.download(symbols)

            # update symbol rejects database
            Database().update_rejected_symbols(
                symbols=symbols,
                current=list(data.index),
                close=True)

            asyncio.run(self.save_to_database(data))

        if data.empty or data is None:
            raise Exception("Empty fundamentals database")

        self.data = data

        log.debug(f"Fundamentals Loaded in {t() - s} secs")

    @staticmethod
    def load_database():
        """Retrieves fundamental data from database, filtered of rejects"""

        query = """
            SELECT symbols.symbol,
                fundamentals.source,
                fundamentals.var,
                fundamentals.val 
            FROM fundamentals
            INNER JOIN symbols
            ON fundamentals.symbol_id = symbols.id
            WHERE symbols.id
            NOT IN (SELECT symbol_id FROM rejected_symbols)
            ORDER BY symbols.symbol;
            """
        with Database() as db:
            df = pd.read_sql_query(query, con=db.connection)
        log.debug("Loaded fundamentals from database")
        return df

    @staticmethod
    def process_loaded_fundamentals(data):
        """Transform database format to user-friendly format"""

        def literal_return(val):
            try:
                return literal_eval(val)
            except (ValueError, SyntaxError):
                return val

        # TODO : Optimize for speed
        df = data.copy()
        df['val'] = df['val'].apply(lambda x: literal_return(x))
        df = df.pivot(index=['symbol'],
                      columns=['source', 'var'],
                      values=['val'])['val']
        return df

    def download(self, symbols: list):
        """Download fundamentals from different sources and returns a
        single dataframe, four columns: stock, source, variable & value.

        List of symbols is chunkable by changing the chunk_size:
            * chunk_size is optional. Empty downloads all.
            * it can be a constant, i.e. 250.
            * or divide all symbols into equal chunks

        :param symbols: list of symbols to download fundamentals"""

        chunk_size = 200
        # n = 3  # 3 for divide symbols into thirds
        # chunk_size = (len(symbols) + n) // n  # into n-ths
        df_tda = df_yahoo = pd.DataFrame()
        try:
            df_tda = self.download_from_tda(symbols, chunk_size)
        except Exception as e:
            log.warning("TDA fundamentals download failed.", e)
        try:
            df_yahoo = self.download_from_yahoo(symbols, chunk_size)
        except Exception as e:
            log.warning("Yahoo fundamentals download failed.", e)

        # concatenate tables using 'symbol' as index
        df = pd.concat([df_tda, df_yahoo],
                       keys=['con', 'yahoo'],
                       axis=1)
        df.rename_axis('symbol', inplace=True)
        log.debug("Fundamentals downloaded")
        return df

    @staticmethod
    def unpack_dictionary_fundamental(df, packages: list):
        """Unpack dictionaries and concatenate to table"""
        # df = pd.concat([pd.DataFrame(None, index=['package']), df]) # DBG
        for package in packages:
            on_hand = df.loc[
                (df[package].notnull()) &
                (df[package].astype(str).str.len() >= 50),
                package]
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
            assert \
                tickers.status_code == httpx.codes.OK, \
                tickers.raise_for_status()
            if tickers.status_code == 200:
                chunk_df = pd.DataFrame(tickers.json()).T
                df = pd.concat([df, chunk_df])
            elif tickers.status_code == 429:
                raise Exception("TDA: Too many transactions per second")
            else:
                raise Exception("TDA data download, unsuccessful")

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

    @staticmethod
    async def save_to_database(data):
        """Save fundamental data in a database"""

        df = data.unstack().reset_index()
        df.columns = ['source', 'variable', 'symbol', 'value']
        df = df[['symbol', 'source', 'variable', 'value']]
        df['value'] = df['value'].astype(str)
        values = df.to_records(index=False)
        sql = """
            INSERT INTO fundamentals (symbol_id, source, var, val)
            VALUES ((SELECT id FROM symbols WHERE symbol=$1), $2, $3, $4)
            ON CONFLICT (symbol_id, source, var)
            DO UPDATE
            SET symbol_id=excluded.symbol_id,
                source=excluded.source,
                var=excluded.var,
                val=excluded.val;
            """

        # using asynciopf as the Database connection manager for Async
        creds = {'host': ss.DB_HOST, 'database': ss.DB_NAME,
                 'user': ss.DB_USER, 'password': ss.DB_PASSWORD}
        async with asyncpg.create_pool(**creds) as pool:
            async with pool.acquire() as db:
                async with db.transaction():
                    await db.executemany(sql, values)

        Database().timestamp('fundamentals', close=True)


class File(GetFundamental):
    def load(self):
        """Gets fundamentals data from pickle file"""

        self.data = pd.read_pickle(ss.FUNDAMENTALS_FILE)
        log.info("Loaded fundamentals from file")
        return self.data

    def save(self):
        """Save fundamental data in a pickle file"""

        self.data.to_pickle(ss.FUNDAMENTALS_FILE)
        self.data.T.to_csv(ss.FUNDAMENTALS_FILE + ".csv")
        log.info(f"Fundamentals saved to pickle file: "
                 f"{ss.FUNDAMENTALS_FILE}")

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


if __name__ == '__main__':
    from core.scanner.assets import GetAssets
    from logging.config import fileConfig

    fileConfig(ss.LOGGING_FILE, disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    _symbols = GetAssets.load_database()['symbol'].to_list()
    GetFundamental(_symbols, forced=True)
