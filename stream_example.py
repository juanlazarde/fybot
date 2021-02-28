# import tda
# from tda import auth, client
from tda.auth import easy_client
from tda.client import Client
from tda.streaming import StreamClient
import config

import asyncio
import json

client = easy_client(
         config.api_key,
         config.redirect_uri,
         config.token_path)
stream_client = StreamClient(client, account_id=config.account_id)

s = 'GOOG'

async def read_stream(s):
    await stream_client.login()
    await stream_client.quality_of_service(StreamClient.QOSLevel.EXPRESS)

    # Always add handlers before subscribing because many streams start sending
    # data immediately after success, and messages with no handlers are dropped.
    stream_client.add_nasdaq_book_handler(
            lambda msg: print(json.dumps(msg, indent=4)))
    await stream_client.nasdaq_book_subs([s])

    while True:
        await stream_client.handle_message()

asyncio.run(read_stream(s))