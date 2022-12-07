import os
import sys
from streamlit.web import cli

import logging
from logging.config import fileConfig

import core.settings as ss
from core.create_config import create_config


def _create_table():
    """Create tables in database.

    Usage::

        $ python fybot create_table
    """
    if sys.argv[1] in ["create_tables", "create_table", "create", "table",
                       "tables", "reset", "createtable", "start", "install"]:
        try:
            from core.database import Database
            Database().create_table()
            sys.exit()
        except Exception as err:
            log.error(f"Error creating tables in database. {err}")
            sys.exit(err)


def _helpme():
    """Command-line help info.

    Usage::

        $ python fybot help
    """
    if sys.argv[1] in ["help", "h", "?", "ayuda", "sos", "-help",
                       "--help", "-h", "--h", "/h", "/help"]:
        print("""
                FyBot
                
                FyBot is a Financial tool to analyze stocks and options.
                
                1) Install: (creates: config/config.py & account.key)
                    $ python fybot setup

                2) Create tables in database or to reset all tables (careful!):
                    $ python fybot create_tables
                    
                3) Run:
                    $ python fybot
                """)
        sys.exit()


def _setup():
    """Install fybot in local environment.

    Usage::

        $ python fybot setup
    """
    # TODO: Testing config creation
    if sys.argv[1] in ["setup", "s", "set", "config", "configurar", "-setup",
                       "--setup", "-s", "--s", "/s", "/setup", "-config",
                       "-configurar"]:

        create_config(r'config.py')
        sys.exit(0)


def _error_handling(error: Exception or str) -> None:
    print("""
            FyBot

            Usage: python fybot [create_table] [help]
            Try: 'python fybot help' for help
            """)
    sys.exit(error)


if __name__ == '__main__':
    fileConfig(ss.LOGGING_FILE, disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    # 'python fybot create_table' to create the PostgresSQL tables.
    try:
        if len(sys.argv) > 1:
            _create_table()
            _helpme()
            _setup()
            _error_handling('')

        sys.argv = [
            "streamlit",
            "run",
            f"{os.path.dirname(os.path.realpath(__file__))}/app.py",
        ]
        sys.exit(cli.main())
    except Exception as e:
        _error_handling(e)
