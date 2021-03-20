"""

--- Reference
Full Stack Trading app https://hackingthemarkets.com/full-stack-trading-app-tutorial-part-01-database-design/
Alpaca Tutorial https://www.youtube.com/watch?v=GsGeLHTOGAg

-- Downloads
SQLite Tools https://sqlite.org/download.html
SQL Browser https://sqlitebrowser.org/
Alpaca API:  pip3 install symbols_alpaca-trade-api

-- Symbol source
NASDAQ ftp://ftp.nasdaqtrader.com/symboldirectory/
Alpaca API https://alpaca.markets/, pip3 install symbols_alpaca-trade-api

-- SQLite Tips
- Play with db: > sqlite3 app.db
- Quit: .quit;
- Delete table data: DELETE FROM table_name;
- Count: SELECT COUNT(*) from table_name;
- Sort: SELECT * FROM table_name ORDER BY column DESC LIMIT n;
- Search: SELECT * FROM table_name WHERE column='string';
- Delete table: DROP TABLE table_name

- Join two tables:
SELECT symbol, date, open, high, low, close
FROM stock_price
JOIN stock on stock.id = stock_price.stock_id
WHERE symbol = 'AAPL'
ORDER BY date;

-- Automation
Cron jobs (Mac & Linux): http://crontab.guru
Task Scheduler on Windows

-- Measure Performance
simple:
        from time import time as t
        t1 = t()
        functions....
        print("-- executed in %.4f sec" % (t() - t1))
decorator:
    from time import time as t
    def timer(func):
        def timer(*args, **kwargs):
            # a decorator which prints execution time of the decorated function
            t1 = t()
            result = func(*args, **kwargs)
            t2 = t()
            print("-- executed %s in %.4f sec" % (func.func_name, (t2 - t1)))
            return result
    return timer

Multi threading:
0) asyncio: https://www.integralist.co.uk/posts/python-asyncio/

1)  import threading
    multi = threading.Thread(target=function, args=(function arguments))
    multi.start()
    # in single thread
    # multi.join()
    # to synchronize use threading.Lock(), acquire & release
    # example
    # import threading
    # threads = []
    # for i in range(os.cpu_count()):
          threads.append(threading.Thread(target=y_hist))
    # for thread in threads:
          thread.start()
    # for thread in threads:
          thread.join()

2)  import multitasking
    @multitasking.task
    function
    # probably will have to join at the end

3) import concurrent.futures
    def foo(bar):
        print('hello {}'.format(bar))
        return 'foo'

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(foo, 'world!')
        return_value = future.result()
        print(return_value)

    also:
    futures = [executor.submit(foo, param) for param in param_list]
     order maintained, and exiting the with will allow result collection.
      [f.result() for f in futures]

4) Multi Process vs. Multi Thread
    https://www.youtube.com/watch?v=rbasThWVb-c&list=PLlcnQQJK8SUj5vlKibv8_i42nwmxDUOFc

    If your code has a lot of I/O or Network usage:
        Multithreading is your best bet because of its low overhead
    If you have a GUI
        Multithreading so your UI thread doesn't get locked up
    If your code is CPU bound:
        You should use multiprocessing (if your machine has multiple cores)


-- Multiple Socket connections
1)  https://www.positronx.io/create-socket-server-with-multiple-clients-in-python/
    import sockets


"""
import sqlite3

import alpaca_trade_api as tradeapi

from config import config
from time import time as t
import pandas as pd
import yfinance as yf
import datetime


class DataBase:
    def __init__(self):
        # connect to local database
        self.connection = sqlite3.connect(config.DATABASE_FILE)
        # get database  results as dictionary
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()

    def create_table(self):
        """create table in database"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock (
                id INTEGER PRIMARY KEY, 
                symbol TEXT NOT NULL UNIQUE, 
                name TEXT NOT NULL
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_price (
                id INTEGER PRIMARY KEY, 
                stock_id INTEGER,
                date NOT NULL,
                open NOT NULL, 
                high NOT NULL, 
                low NOT NULL, 
                close NOT NULL, 
                volume NOT NULL,
                FOREIGN KEY (stock_id) REFERENCES stock (id)
            )
        """)
        self.connection.commit()

    def save_assets(self, asset):
        try:
            self.cursor.execute(
                """INSERT INTO stock (symbol, name) VALUES (?, ?)""",
                (asset['symbol'], asset['name']))
            self.connection.commit()
        except Exception as e:
            print('Error saving asset', str(e))
            print('Trying to delete the table and run again')
            self.delete_db_content('stock')
            print('Done. Run the program again')

    def delete_db_content(self, table_name):
        self.cursor.execute("""DELETE FROM (?)""", table_name)
        self.connection.commit()

    def fetch_database(self):
        self.cursor.execute("""SELECT id, symbol, name FROM stock""")
        return self.cursor.fetchall()


class Download:
    all_assets = None
    sp500 = None

    def symbols_alpaca(self):
        api = tradeapi.REST(key_id=config.API_KEY,
                            secret_key=config.SECRET_KEY,
                            base_url=config.ENDPOINT)
        active_assets = api.list_assets(status='active')
        # tradable_assets = {asset for asset in active_assets if asset.tradable}
        # tradable_assets = [(asset.symbol, asset.name) for asset in tradable_assets]
        tradable_assets = [(asset.symbol, asset.name) for asset in active_assets if asset.tradable]
        result = [dict(zip(['symbol', 'name'], asset)) for asset in tradable_assets]
        self.all_assets = result
        return result

    def symbols_sp500_wikipedia(self):
        # scrape wikipedia for S&P500 list
        table = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        df = df[['Symbol', 'Security']]
        # replace '.' with '-' a Yahoo Finance issue
        df = df.assign(
            Symbol=lambda x: x['Symbol'].str.replace('.', '-', regex=True))
        df = df.rename({'Symbol': 'symbol', 'Security': 'name'}, axis='columns')
        result = df.to_dict('records')[:config.MAX_SYMBOLS]
        self.sp500 = result
        return result

    def yahoo_tickers(self, source='sp500'):
        assets = self.symbols_sp500_wikipedia() if source == 'sp500' else DataBase().fetch_database()
        symbols = [asset['symbol'] for asset in assets]
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=config.MAX_DAYS)
        tickers = yf.download(symbols,
                              start=start_date,
                              end=end_date,
                              group_by="ticker",
                              threads=True)
        tickers = tickers.dropna(how='all')
        tickers = tickers.dropna(how='all', axis=1)
        return tickers


class UpdateDatabase:
    @staticmethod
    def symbols():
        # current database
        assets = DataBase().fetch_database()
        symbols = [k['symbol'] for k in assets]
        # new database
        new_assets = Download().symbols_alpaca()
        # compare changes
        for asset in new_assets:
            try:
                if asset['symbol'] not in symbols:
                    DataBase().save_assets(asset)
                    print("Added a new stock {} {}".format(asset['symbol'],
                                                           asset['name']))
            except Exception as e:
                print(asset['symbol'])
                print(e)

    @staticmethod
    def tickers(barsets):
        db = DataBase()
        rows = db.fetch_database()
        symbols = []
        stock_dict = {}
        for row in rows:
            symbol = row['symbol']
            symbols.append(symbol)
            stock_dict[symbol] = row['id']

        for symbol in barsets.columns.levels[0]:
            print("processing symbol %s" % symbol)
            stock_id = stock_dict[symbol]
            for bar in barsets[symbol].iterrows():
                db.cursor.execute("""
                        INSERT INTO stock_price (stock_id, date, open, high, low, close, adjusted_close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (stock_id, bar[0].date(), bar[1].Open,
                              bar[1].High, bar[1].Low, bar[1].Close,
                              bar[1]['Adj Close'], bar[1].Volume))
        db.connection.commit()


def main():
    t1 = t()
    Setup = False
    if Setup:
        DataBase().create_table()
        assets = Download().symbols_alpaca()
        for asset in assets:
            DataBase().save_assets(asset)
    UpdateDatabase().symbols()
    tickers = Download().yahoo_tickers(source='sp500')
    UpdateDatabase().tickers(barsets=tickers)

    print("-- executed in %.4f seconds" % (t() - t1))


if __name__ == '__main__':
    main()
