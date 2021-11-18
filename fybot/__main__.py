import os
import sys
import streamlit.cli as cli

import logging
from logging.config import fileConfig

from core.settings import S


def create_table():
    """Create tables in database.

    Usage:
        $ python fybot create_table
    """
    if sys.argv[1] in ["create_tables", "create_table", "create", "table", "reset", "createtable", "start", "install"]:
        try:
            from core.database import Database
            Database().create_table()
            sys.exit()
        except Exception as err:
            log.error(f"Error creating tables in database. {err}")
            sys.exit(err)


def helpme():
    """Command-line help info.

    Usage:
        $ python fybot help
    """
    if sys.argv[1] in ["help", "h", "?", "ayuda", "sos", "-help", "--help", "-h", "--h", "/h", "/help"]:
        error_handling('')


def error_handling(error: Exception or str):
    print("""
            FyBot

            Usage: python fybot [create_table] [help]
            Try: 'python fybot help' for help
            """)
    sys.exit(error)


if __name__ == '__main__':
    fileConfig(S.LOGGING_FILE, disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    # 'python fybot create_table' to create the PostgresSQL tables.
    if len(sys.argv) > 1:
        create_table()
        helpme()
        error_handling('')

    try:
        sys.argv = [
            "streamlit",
            "run",
            f"{os.path.dirname(os.path.realpath(__file__))}/app.py",
        ]
        sys.exit(cli.main())
    except Exception as e:
        error_handling(e)
