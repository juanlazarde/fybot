import sys
import logging
from logging.config import fileConfig
from app import index


def create_table():
    if sys.argv[1] in ["create_tables", "create_table", "create",
                       "table", "reset", "nukeit", "createtable",
                       "start", "go"]:
        try:
            from core.database import Database
            Database().create_table()
            sys.exit()
        except Exception as e:
            log.error(f"Error creating tables in database. {e}")
            sys.exit(e)


if __name__ == '__main__':
    fileConfig('config/logging_config.ini', disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    # 'python fybot create_table' to create the PostgresSQL tables.
    if len(sys.argv) > 1:
        create_table()

    # run main code

    index()