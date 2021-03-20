import asyncio
import json

from tda.streaming import StreamClient

from fybot.src import TDA
from core.settings import S
from core.encryption import Encryption


def index():
    client = TDA().client
    account_id = int(Encryption().decrypt(S.TDA_ACCOUNT_E))
    stream_client = StreamClient(client, account_id=account_id)

    async def read_stream(s):
        await stream_client.login()
        await stream_client.quality_of_service(StreamClient.QOSLevel.EXPRESS)

        # Always add handlers before subscribing because many streams start
        # sending data immediately after success, and messages with no
        # handlers are dropped.
        stream_client.add_nasdaq_book_handler(
            lambda msg: print(json.dumps(msg, indent=4)))
        await stream_client.nasdaq_book_subs([s])

        while True:
            await stream_client.handle_message()

    symbol = 'GOOG'
    asyncio.run(read_stream(s=symbol))


if __name__ == '__main__':
    index()
