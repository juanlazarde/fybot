"""Config to process dictionary incoming from the form."""
from core.utils import fix_path


def normalize_configuration(opt: dict):
    res = dict()

    # loading debug
    res['save_load_pickle'] = bool(opt['DEBUG']['save_load_pickle'])
    res['export'] = bool(opt['DEBUG']['export'])
    res['download'] = bool(opt['DEBUG']['download_market_closed'])
    # loading paths
    res['data'] = opt['PATHS']['data']
    res['data'] = fix_path(res['data'])
    # loading watchlist
    res['watchlist_1'] = opt['WATCHLIST']['watchlist_1'].split(',')
    res['watchlist_0'] = opt['WATCHLIST']['watchlist_0'].split(',')
    res['tda_watchlist'] = opt['WATCHLIST']['tda_watchlist']
    res['selected_watchlist'] = int(opt['WATCHLIST']['selected'])
    # filters

    res['max_dte'] = int(opt['FILTERS']['max_days_to_expiration'])
    res['strategies'] = opt['FILTERS']['strategies'].strip().split(',')
    res['strategies'] = [_.strip() for _ in res['strategies']]

    # 'FILTERS': {
    #     'min_price': 1,
    #     'max_price': 5000,
    #     'max_risk': 5000,
    #     'min_return_pct': 1.0,
    #     'max_days_to_expiration': 60,
    #     'min_days_to_expiration': 30,
    #     'premium': "credit",
    #     'strategies': "naked, spread",
    #     'option_type': "put, call",
    #     'min_delta': 0.3,
    #     'max_delta': 1.0,
    #     'min_volume_pctl': 10,
    #     'min_open_int_pctl': 10,
    #     'max_bid_ask_pctl': 0.5,

    return res


def main(parameters: dict):
    return normalize_configuration(parameters)


if __name__ == '__main__':
    import core.settings as ss
    print(main(parameters=ss.OPTION_SNIPER))
