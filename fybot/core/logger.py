from logging.config import fileConfig
fileConfig('config/logging_config.ini', disable_existing_loggers=False)

# import logging

# from logging.config import fileConfig
#
# fileConfig('./config/logging_config.ini', disable_existing_loggers=False)
#
#
# def getlogger():
#     return logging.getLogger()

# TODO: FIGURE OUT LOGGING
# from os import path
# log_file_path = path.join(path.dirname(path.abspath(__file__)), 'log.config')
# logging.config.fileConfig(log_file_path)
