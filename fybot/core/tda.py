import os
import sys
import asyncio
import json
import logging

from tda.auth import easy_client as tda_connection
from tda.streaming import StreamClient

from core.settings import S
from core.encryption import Encryption

log = logging.getLogger(__name__)


class TDA:
    client = None

    def __init__(self):
        """Connects with TDA database using TDA-API."""

        API_KEY = S.TDA_API_KEY
        REDIRECT_URL = S.TDA_REDIRECT_URI
        TOKEN = S.TDA_TOKEN

        try:
            self.client = tda_connection(api_key=API_KEY,
                                         redirect_uri=REDIRECT_URL,
                                         token_path=TOKEN,
                                         webdriver_func=self.webdriver,
                                         asyncio=False)
        except Exception as e:
            print("\nTDA authorization ERROR.\n" + str(e) +
                  "\nEstablishing authorization via Chrome")

            log.info("\nTDA error. Establishing authorization via Chrome")

            try:
                cont = input("Delete token file? Y/N: ")

                if cont[0].strip().upper() == "Y":
                    if os.path.isfile(TOKEN):
                        os.remove(TOKEN)
                    self.client = tda_connection(api_key=API_KEY,
                                                 redirect_uri=REDIRECT_URL,
                                                 token_path=TOKEN,
                                                 webdriver_func=self.webdriver,
                                                 asyncio=False)
                else:
                    raise

            except Exception as e:
                log.critical(f"\nTDA authorization ERROR.\n{e}")
                sys.exit(e)

        log.info("Established TDA Ameritrade connection")

    @staticmethod
    def webdriver():
        """Establishes oAuth with API via Chrome.

        Creates Token file, once is created and valid, webdriver is not used.

        ChromeDriver is a dependency used to authorize TDA portal download
        driver from https://chromedriver.chromium.org/downloads
        Save ChromeDriver in same location as this Script

        Returns:
            driver (object): API authorization.
        """
        # Import selenium here because it's slow to import
        from selenium import webdriver
        import atexit
        import webbrowser

        try:
            driver = webdriver.Chrome()
            atexit.register(lambda: driver.quit())
            log.info("ChromeDriver created successfully")
            return driver
        except Exception as e:
            str_url = "https://chromedriver.chromium.org/downloads"
            print("\n**There was an error: {}"
                  "\nDOWNLOAD ChromeDriver from {} & INSTALL it in the same "
                  "folder as the script.".format(str(e), str_url))
            webbrowser.open(str_url, new=1)
            log.error("Error loading ChromeDriver.")
            raise


class TDAstream(TDA):
    def tda_stream(self, symbol: str):
        client = self.client
        account_id = int(Encryption().decrypt(S.TDA_ACCOUNT_E))
        stream_client = StreamClient(client, account_id=account_id)

        async def read_stream(s):
            await stream_client.login()
            await stream_client.quality_of_service(
                StreamClient.QOSLevel.EXPRESS)

            # Always add handlers before subscribing because many streams start
            # sending data immediately after success, and messages with no
            # handlers are dropped.
            stream_client.add_nasdaq_book_handler(
                lambda msg: print(json.dumps(msg, indent=4)))
            await stream_client.nasdaq_book_subs([s])

            while True:
                await stream_client.handle_message()

        asyncio.run(read_stream(s=symbol))
