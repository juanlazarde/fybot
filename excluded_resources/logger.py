import logging
from core.settings import S


class Logger:
    """Logger to bertter diagnose and get relevant messages"""
    def __init__(self):
        self.setup()

    @staticmethod
    def getLogger(name):
        return logging.getLogger(name)

    @staticmethod
    def setup():
        # logging.DEBUG, INFO, WARNING, ERROR, CRITICAL
        LEVEL = logging.DEBUG if S.DEBUG else logging.INFO
        FORMAT = "%(asctime)s, %(name)s, %(module)s, %(funcName)s, " \
                 "%(lineno)s, %(levelname)s:  %(message)s "
        logging.basicConfig(filename="log.log",
                            filemode='w',
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=LEVEL
                            )

        console = logging.StreamHandler()
        console.setLevel(LEVEL)
        console.setFormatter(logging.Formatter(FORMAT))
        logging.getLogger().addHandler(console)
