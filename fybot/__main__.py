import sys
import logging
import streamlit as st

from logging.config import fileConfig
from fybot.core.settings import S
from fybot.app import index

# Streamlit Configuration
FAV_ICON = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com" \
           "/thumbs/240/twitter/259/mage_1f9d9.png"
st.set_page_config(
    page_title="Financial Scanner",
    page_icon=FAV_ICON,
    layout="wide"
)


def create_table():
    """Create tables in database.

    Usage:
        * python fybot create_table
    """
    if sys.argv[1] in ["create_tables", "create_table", "create",
                       "table", "reset", "nukeit", "createtable",
                       "start", "go"]:
        try:
            from fybot.core.database import Database
            Database().create_table()
            sys.exit()
        except Exception as e:
            log.error(f"Error creating tables in database. {e}")
            sys.exit(e)


if __name__ == '__main__':
    fileConfig(S.LOGGING_FILE, disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    # 'python fybot create_table' to create the PostgresSQL tables.
    if len(sys.argv) > 1:
        create_table()

    # run main code
    index()
