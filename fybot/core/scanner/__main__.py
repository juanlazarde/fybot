import logging
from time import time as t

import core.scanner as sn

log = logging.getLogger(__name__)


def save_files():
    symbols = sn.FileAssets().save()
    sn.FileFundamentals(symbols).save()
    sn.FilePricing(symbols).save()


def refresh_data(forced: bool = False, save_to_file: bool = False):
    """Get symbols, pricing, fundamentals from internet, store in database.
    Data downloaded within 24 hours is loaded from database.

    Sources: NASDAQ, Wikipedia, Alpaca, TD Ameritrade

    :param save_to_file: True to save a local copy
    :param forced: True to force data refresh
    """

    s = t()
    symbols = sn.GetAssets(forced).symbols
    sn.GetFundamental(symbols, forced)
    print("Fundamentals Done")
    sn.GetPrice(symbols, forced)

    if save_to_file:
        save_files()

    log.info(f"{__name__} took {t() - s} seconds")


if __name__ == '__main__':
    from logging.config import fileConfig
    import core.settings as ss

    fileConfig(ss.LOGGING_FILE, disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    # save_files()
    refresh_data(forced=False, save_to_file=False)
