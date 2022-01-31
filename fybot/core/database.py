import sys
from datetime import datetime

import psycopg2
import psycopg2.extras
import pytz
import logging

import core.settings as ss

log = logging.getLogger(__name__)


class Database:
    """Connect with PostgresSQL. host, db, usr, pwd defined elsewhere"""

    def __init__(self):
        try:
            self._conn = psycopg2.connect(host=ss.DB_HOST,
                                          database=ss.DB_NAME,
                                          user=ss.DB_USER,
                                          password=ss.DB_PASSWORD)
            self._cursor = self.connection.cursor(
                cursor_factory=psycopg2.extras.DictCursor)
        except Exception as e:
            print(f"Unable to connect to database!\n{e}")
            sys.exit(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.close()

    @property
    def connection(self):
        return self._conn

    @property
    def cursor(self):
        return self._cursor

    def commit(self):
        self.connection.commit()

    def close(self, commit=True):
        if commit:
            self.commit()
        self.connection.close()

    def execute(self, sql, params=None):
        self.cursor.execute(sql, params or ())

    def executemany(self, sql, params=None):
        self.cursor.executemany(sql, params)

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()

    def query(self, sql, params=None):
        self.cursor.execute(sql, params or ())
        response = self.fetchall()
        return response[0] if len(response) == 1 else response

    @staticmethod
    def parse_sql(sql_file_path):
        """Read *.sql file, parse lines"""
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            data = f.read().splitlines()
        stmt = ''
        stmts = []
        for line in data:
            if line:
                if line.startswith('--'):
                    continue
                stmt += line.strip() + ' '
                if ';' in stmt:
                    stmts.append(stmt.strip())
                    stmt = ''
        return stmts

    def create_table(self):
        """create table in database"""
        sql = self.parse_sql("fybot/config/create_tables.sql")
        for i in sql:
            self.execute(i)

        self.close(commit=True)
        print("Done creating tables for database")

    def last_update(self, table, close=False) -> bool or None:
        """Gets timestamp from last_update table.

        :param str table: string with table name
        :param close: True to close the Database connection
        :return: True if last update within 24 hours. None if error"""
        try:
            query = f"SELECT date FROM last_update WHERE tbl = '{table}';"
            timestamp = self.query(query)[0]
            if close:
                self.close()
            now_utc = pytz.utc.localize(datetime.utcnow())
            within24 = None if timestamp is None else \
                (now_utc - timestamp).total_seconds <= 24 * 60 * 60
        except Exception as e:
            log.debug(e)
            within24 = None
        return within24

    def timestamp(self, table, close=False):
        query = f"""INSERT INTO last_update (tbl)
                    VALUES ('{table}')
                    ON CONFLICT (tbl)
                    DO UPDATE SET date = NOW();"""
        self.execute(query)
        if close:
            self.commit()
            self.close()

    def reset_tables(self):
        table = "symbols, last_update"
        query = f"TRUNCATE {table} CASCADE;"
        self.execute(query)
        self.close(commit=True)

    def update_rejected_symbols(self, symbols, current, close=False):
        """Updates rejected_symbols table with symbols that have problems.

               :param list symbols: list of symbols
               :param list current: list of rejected symbols
               :param close: True to close database connection"""

        values = list(set(symbols).difference(current))
        if len(values) != 0:
            values = [(value,) for value in values]
            sql = """INSERT INTO rejected_symbols (symbol_id)
                     VALUES ((SELECT id FROM symbols WHERE symbol = %s))
                     ON CONFLICT (symbol_id)
                     DO UPDATE SET date = NOW();"""
            self.executemany(sql, values)
            self.commit()
            if close:
                self.close()
