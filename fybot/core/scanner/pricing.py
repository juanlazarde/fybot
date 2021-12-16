import asyncio
import datetime
import logging
import sys
from time import time as t, sleep

import asyncpg
import httpx
import pandas as pd
import yfinance as yf

from core.database import Database
import core.settings as ss
from core.fy_tda import TDA

log = logging.getLogger(__name__)


class GetPrice:
    def __init__(self,
                 symbols: list = None,
                 forced: bool = False,
                 source: str = 'con'):
        """Gets price history data from download if not within 24 hours,
        then from data, finally from file.

        :param symbols: list of symbols
        :param forced: True bypassess 24 hour download requirement
        :param source: Source from where the data is downloaded
        """

        log.debug("Loading fundamentals")
        s = t()

        data = pd.DataFrame()

        # checks on last database update for price history
        within24 = Database().last_update('price_history', close=True)

        # load from database if within 24 hours
        if within24 and not forced:
            data = self.load_database()

        # download data if database is empty or expired data
        if data.empty or data is None or not within24 or forced:
            data = self.download(symbols, source)

            # update symbol rejects database
            Database().update_rejected_symbols(
                symbols=symbols,
                current=data['symbol'].unique(),
                close=True)

            asyncio.run(self.save_to_database(data))

        if data.empty or data is None:
            raise Exception("Empty price history database")

        self.data = data

        log.debug(f"Price History Loaded in {t() - s} secs")

    @staticmethod
    def load_database():
        """Retrieves the price data from database, filtered of rejects"""
        query = """
            SELECT price_history.date,
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
            ORDER BY symbols.symbol, price_history.date;
            """

        with Database() as db:
            df = pd.read_sql_query(query, con=db.connection)
        log.debug("Loaded price history from database")
        return df

    @staticmethod
    def download(symbols: list, source: str):
        source = 'con' if (source == '' or source is None) else source
        if source == 'con':
            df = asyncio.run(Source().tda(symbols=symbols))
        elif source == 'yahoo':
            df = Source().yahoo(symbols=symbols)
        else:
            log.warning(f"Source {source} doesn't exist'")
            sys.exit(1)

        return df

    @staticmethod
    async def save_to_database(data):
        """Save price data in a database"""

        rows = data.itertuples(index=False)
        values = [list(row) for row in rows]
        sql = """
            INSERT INTO price_history (date, symbol_id, open, high, low,
                                       close, adj_close, volume)
            VALUES ($1, (SELECT id FROM symbols WHERE symbol=$2),
                    $3, $4, $5, $6, $7, $8)
            ON CONFLICT (symbol_id, date)
            DO UPDATE
            SET symbol_id=excluded.symbol_id,
                date=excluded.date, open=excluded.open,
                high=excluded.high, low=excluded.low,
                close=excluded.close,
                adj_close=excluded.adj_close,
                volume=excluded.volume;
            """

        # using asynciopg as the Database connection manager for Async
        creds = {'host': ss.DB_HOST, 'database': ss.DB_NAME,
                 'user': ss.DB_USER, 'password': ss.DB_PASSWORD}
        async with asyncpg.create_pool(**creds) as pool:
            async with pool.acquire() as db:
                async with db.transaction():
                    await db.executemany(sql, values)

        Database().timestamp('price_history', close=True)


class Source(GetPrice):
    @staticmethod
    async def tda(symbols: list):
        """Download price data from TD Ameritrade.

        List of symbols is chunkable by changing the chunk_size.

        :param symbols: list of symbols to download fundamentals
        """
        # Create connection, return empty df if there is no connection
        try:
            con = TDA(is_asyncio=True)
        except Exception as e:
            log.info(f"TDA Connection failed. {e}")
            return pd.DataFrame()

        # download stocks data in chunks and concatenate to large table
        # end_date = datetime.datetime.now()
        # start_date = end_date - datetime.timedelta(days=ss.DAYS)

        df = pd.DataFrame()

        chunk_size = 1
        # chunk_size = len(symbols)  # use actual chunk size i.e. 200
        # n = 3
        # chunk_size = (len(symbols) + n) // n  # into n-ths

        for i in range(0, len(symbols), chunk_size):
            symbol_chunk = symbols[i:i + chunk_size][0]

            # download chunk of fundamentals from TDA
            resp = await con.client.get_price_history(
                symbol_chunk,
                period_type=con.client.PriceHistory.PeriodType.YEAR,
                frequency_type=con.client.PriceHistory.FrequencyType.DAILY,
                frequency=con.client.PriceHistory.Frequency.DAILY,
                period=con.client.PriceHistory.Period.ONE_YEAR,
                # start_datetime=start_date,
                # end_datetime=end_date,
                need_extended_hours_data=None
            )
            assert resp.status_code == httpx.codes.OK
            df_chunk = pd.DataFrame(resp.json())

            df_chunk = pd.json_normalize(df_chunk['candles'])
            df_chunk['datetime'] = pd.to_datetime(df_chunk['datetime'])
            df_chunk = df_chunk.set_index(['datetime'])
            df_chunk.columns = pd.MultiIndex.from_product(
                [df_chunk.columns, [symbol_chunk]]).swaplevel(0, 1)

            df = pd.concat([df, df_chunk], axis=1)

            if i in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
                sleep(2.0)

        # cleanup any NaN values in the dataframe
        df = df.dropna(how='all')
        df = df.dropna(how='all', axis=1)
        df = df.stack(level=0)
        df = df.reset_index()
        df = df.rename(columns={'level_1': 'symbol'})

        log.info("Price history downloaded from TD Ameritrde")
        return df

    def yahoo(self, symbols: list):
        """Download price data from YFinance.

        List of symbols is chunkable by changing the chunk_size.

        :param symbols: list of symbols to download fundamentals
        """

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=ss.DAYS)

        df = pd.DataFrame()

        # chunk_size = 2
        # chunk_size = len(symbols)  # use actual chunk size i.e. 200
        n = 3
        chunk_size = (len(symbols) + n) // n  # into n-ths

        for i in range(0, len(symbols), chunk_size):
            symbol_chunk = symbols[i:i + chunk_size]

            # TODO: Other sources of downlaods, like TDA, Alpaca
            # download chunk of fundamentals from YFinance
            df_chunk = self.yfinance_price_history(
                symbol_chunk,
                start_date,
                end_date)

            # YFinance won't return symbol label if it's only one symbol
            if len(symbol_chunk) == 1:
                df_chunk.columns = pd.MultiIndex.from_product(
                    [df_chunk.columns, symbol_chunk]).swaplevel(0, 1)

            df = pd.concat([df, df_chunk], axis=1)

        # cleanup any NaN values in the dataframe
        df = df.dropna(how='all')
        df = df.dropna(how='all', axis=1)
        df = df.stack(level=0)
        df = df.reset_index()
        df = df.rename(columns={'level_1': 'symbol'})

        log.info("Price history downloaded from Yahoo Finance")
        return df

    # TODO: JUST GET LATEST PRICE DATA, MAY REQUIRE SINGLE TICKER DOWNLOAD
    # @staticmethod
    # def date_range(_wtc):
    #     """Dates to download history.
    #     Total days comes from Settings, whether you want 150 days back.
    #     The dat of the last update is the latest date in the symbols
    #     price history date field.
    #     """
    #     return start_date, end_date

    @staticmethod
    def yfinance_price_history(symbol_chunk, start_date, end_date):
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


class File(GetPrice):
    def load(self):
        """Gets Price History from pickle file"""

        self.data = pd.read_pickle(ss.PRICE_FILE)
        log.info("Loaded Price History from file")
        return self.data

    def save_to_file(self):
        """Save pricing data in a pickle file"""

        self.data.to_pickle(ss.PRICE_FILE)
        self.data.T.to_csv(ss.PRICE_FILE + ".csv")
        log.info(f"Price data saved to pickle file: {ss.PRICE_FILE}")


if __name__ == '__main__':
    from core.scanner.assets import GetAssets
    from logging.config import fileConfig

    fileConfig(ss.LOGGING_FILE, disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    _symbols = GetAssets.load_database()['symbol'].to_list()
    GetPrice(_symbols, forced=True, source='yahoo')
