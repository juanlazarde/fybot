from sys import exit
from time import time as t
import fybot.core.snp as sn


def main():
    forced = True
    symbols = sn.GetAssets(forced).symbols  #3.6s

    # 300 symbols

    # sn.GetFundamental(symbols, forced)  # 44.6 s
    # s = t()
    # sn.GetPrice(symbols, forced)  # 84.7 s
    # print(t() - s)
    # exit()
    # pass


if __name__ == '__main__':
    main()
