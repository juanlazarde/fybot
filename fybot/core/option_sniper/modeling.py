from typing import Dict, Optional, Any
import numpy as np
from scipy.stats import norm
import pandas as pd
import threading

from core.utils import optimize_pd


def black_scholes(S, K, T, rf: float, iv, option_type) -> pd.DataFrame:
    """
    Black Scholes modeling function.

        * https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr677.pdf
        * https://github.com/hashABCD/opstrat/blob/main/opstrat/blackscholes.py
        * https://www.smileofthales.com/computation/options-greeks-python/
        * https://github.com/vpatel576/option_probabilites
        * https://github.com/skp1999/McMillan-s-Profit-Probability-Calculator/blob/main/POP_Calculation.py
        * https://option-price.com/index.php

    Option Value: Theoretical premium value.
    Delta : Measures Impact of a Change in the Price of Underlying
    Gamma: Measures the Rate of Change of Delta
    Theta: Measures Impact of a Change in Time Remaining
    Vega: Measures Impact of a Change in Volatility
    Rho: Measures the impact of changes in Interest rates

    :param S: Underlying Asset or Stock Price ($).
    :param K: Strike or Excercise Price ($).
    :param T: Expiry time of the option (days).
    :param rf: Risk-free rate (decimal number range 0-1).
    :param iv: Volatility (decimal).
    :param option_type: Calls or Puts option type.
    :return: Dataframe 'option_value', 'intrinsic_value', and 'time_value'.
     Greeks has delta, gamma, theta, rho.
    """
    # Check inputs.
    is_type = np.isin(option_type, ['calls', 'puts', 'call', 'put', 'c', 'p'])
    assert np.all(is_type) == 1, "Enter Calls or Puts options only."

    t = np.maximum(T / 365, 0.00001)  # Avoid infinite when T = 0.
    iv = np.maximum(iv, 0.00001)  # Avoid infinite when iv = 0.

    n1 = np.log(S / K)
    n2 = (rf + iv ** 2 / 2) * t
    d = iv * np.sqrt(t)

    d1 = (n1 + n2) / d
    d2 = d1 - d

    f = np.where(np.isin(option_type, ['calls', 'call', 'c']), 1, -1)

    N_d1 = norm.cdf(f * d1)
    N_d2 = norm.cdf(f * d2)

    A = S * N_d1
    B = K * N_d2 * np.exp(-rf * t)

    # Option pricing.
    val = f * (A - B)
    val_int = np.maximum(0.0, f * (S - K))
    val_time = val - val_int

    # Greeks.
    delta = f * N_d1
    gamma = np.exp((-d1 ** 2) / 2) / (S * iv * np.sqrt(2 * np.pi * t))
    theta = (-S * iv * np.exp(-d1 ** 2 / 2) / np.sqrt(8 * np.pi * t)
             - f * (N_d2 * rf * K * np.exp(-rf * t))) / 365
    vega = ((S * np.sqrt(t) * np.exp((-d1 ** 2) / 2))
            / (np.sqrt(2 * np.pi) * 100))
    rho = f * t * K * N_d2 * np.exp(-rf * t) / 100

    # Returns Dataframe.
    return pd.DataFrame({
        'option_value_bs': val,
        'intrinsic_value': val_int,
        'time_value': val_time,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    })


def monte_carlo(
        key: str,
        S: float,
        K: float,
        T: int,
        rf: float,
        iv: float,
        option_type: str,
        n: int = 200,
        rng: Any = None) -> Dict[str, Optional[float]]:
    """
    Monte Carlo modeling function.

    Monte Carlo allows us to simulate seemingly random events, and assess
    risks (among other results, of course). It has been used to assess the
    risk of a given trading strategy.

        * https://python.plainenglish.io/monte-carlo-options-pricing-in-two-lines-of-python-cf3a39407010
        * https://www.youtube.com/watch?v=sS7GtIFFr_Y
        * https://pythonforfinance.net/2016/11/28/monte-carlo-simulation-in-python/
        * https://aaaquants.com/2017/09/01/monte-carlo-options-pricing-in-two-lines-of-python/#page-content

    This function is not fully vectorized. It needs to be in a loop, with
    each row passed for processing. All results are summarized by a Numpy func.

    Usage:
    ::
            vector_profit_probability = np.vectorize(monte_carlo)
            pop = vector_profit_probability(
                S=curr_price,
                K=opt['strike'].to_numpy(),
                T=opt['dte'].to_numpy(),
                rf=ten_yr,
                iv=opt['volatility'].to_numpy(),
                option_type=opt['option_type'].to_numpy(),
                rng=rng,
                n=1000
            )

    :param key: Key per result. Useful for future concatenation.
    :param S: Underlying Asset or Stock Price ($).
    :param K: Strike or Excercise Price ($).
    :param T: Expiry time of the option (days).
    :param rf: Risk-free rate (decimal number range 0-1).
    :param iv: Volatility (decimal).
    :param option_type: Calls or Puts option type.
    :param n: Number of Monte Carlo iterantions. min 100,000 recommended.
    :param rng: Random range generator, used when in loops.
    :return: Dictionary with keys: value, greeks.
     Value has 'option_value', 'intrinsic_value', and 'time_value'.
     Greeks jas delta, gamma, theta, rho.
    """
    # Check inputs.
    assert option_type in ['calls', 'puts', 'call', 'put', 'c', 'p'], \
        "Enter Calls or Puts options only."

    # np.random.seed(25)  # Use for consistent testing.

    T = T if T != 0 else 1
    D = np.exp(-rf * (T / 252))

    # Randomized array of number of days x simulations, based of current price.
    # P = np.cumprod(1 + np.random.randn(n, T) * iv / np.sqrt(252), axis=1) * S
    # Generating random range is expensive, so doing it once.
    rng = np.random.Generator(np.random.PCG64()) if rng is None else rng
    rnd = rng.standard_normal((n, T), dtype=np.float32)
    P = np.cumprod(1 + rnd * iv / np.sqrt(252), axis=1) * S

    # Series on last day of simulation with premium difference.
    p_last = P[:, -1] - K * D

    # If calls, take only positive results. If puts, take negatives.
    if option_type in ['calls', 'call', 'c']:
        arr = np.where(p_last > 0, p_last, 0)
    else:
        arr = -np.where(p_last < 0, p_last, 0)

    # Take the average values of all the iterations on the last day.
    val = np.mean(arr)

    # Probability of Profit.
    pop_ITM = round(np.count_nonzero(arr) / p_last.size, 2)

    # Probability of Making 50% Profit.
    profit_req = 0.50
    if option_type in ['calls', 'call', 'c']:
        arr = np.where(p_last > profit_req * val, p_last, 0)
    else:
        arr = -np.where(p_last < profit_req * val, p_last, 0)
    p50 = round(np.count_nonzero(arr) / p_last.size, 2)

    # Returns Dictionary.
    # Calculating quantiles is expensive, so only uncomment if necessary.
    return {
        'symbol': key,
        'option_value_mc': val,  # Average value. Near Black Scholes Value.
        # 'value_quantile_5': np.percentile(p_last, 5),  # 5% chance below X
        # 'value_quantile_50': np.percentile(p_last, 50),  # 50% chance lands here.
        # 'value_quantile_95': np.percentile(p_last, 95),  # 5% chance above X.
        'probability_ITM': pop_ITM,  # Probability of ending ITM.
        'probability_of_50': p50,  # Probability of makeing half profit.
    }


def mc_numpy_vector(*args):
    """
    Monte Carlo simulations vectorized so that arrays work in calculations

    DEPRECATED: It's faster to multithread this operation.
    """
    curr_price, opt, ten_yr, rng, montecarlo_iterations = args
    vector_monte_carlo = np.vectorize(monte_carlo)
    _pop = vector_monte_carlo(
        key=opt['contractSymbol'],
        S=curr_price,
        K=opt['strikePrice'].to_numpy(),
        T=opt['daysToExpiration'].to_numpy(),
        rf=ten_yr,
        iv=opt['volatility'].to_numpy(),
        option_type=opt['option_type'].to_numpy(),
        rng=rng,
        n=montecarlo_iterations
    )
    return pd.DataFrame.from_records(_pop)  # pd.json_normalize(_pop.T)


def mc_multi_threading(*args):
    """
    Monte Carlo simulations vectorized so that arrays work in calculations.
    Multithreaded, means one CPU works multiple I/O.

    :param args: Passing all parameters from call.
    :return: Dataframe with results. Including a key to join later.
    """
    def threader(opt, ten_yr, rng, montecarlo_iterations):
        _pop = vector_monte_carlo(
            key=opt.index.get_level_values('symbol').to_numpy(),
            S=opt['lastPrice'].to_numpy(),
            K=opt['strikePrice'].to_numpy(),
            T=opt['daysToExpiration'].to_numpy(),
            rf=ten_yr,
            iv=opt['volatility'].to_numpy(),
            option_type=opt.index.get_level_values('option_type').to_numpy(),
            rng=rng,
            n=montecarlo_iterations
        )
        rez.append(_pop)

    rez = []  # List of dictionaries
    _opt, _ten_yr, _rng, _montecarlo_iterations, _chunks = args
    vector_monte_carlo = np.vectorize(monte_carlo)

    # Chunking tables in groups of 'chunks' values. Each a separate thread.
    dtes = _opt['daysToExpiration'].unique()  # List of DTE's
    d_chunk = [dtes[i:i + _chunks] for i in range(0, len(dtes), _chunks)]
    df_chunks = [_opt[(_opt['daysToExpiration'].isin(dte))] for dte in d_chunk]

    # Multi-threading.
    threads = []
    for df in df_chunks:
        arg = (df, _ten_yr, _rng, _montecarlo_iterations)
        t = threading.Thread(target=threader, args=arg)
        threads.append(t)

    [thread.start() for thread in threads]  # Kickoff threading.
    [thread.join() for thread in threads]  # Stop all threads.

    # Flatten list
    _result = []
    for i in range(len(rez)):
        for j in rez[i]:
            _result.append(j)

    return pd.DataFrame.from_records(_result)


class Modeling:
    def __init__(self, con, option_df):
        self.options = option_df.options  # Options coming in.
        self.quote = None  # All quote data.
        self.rf = 0  # Risk-free rate, i.e. 10-yr t-bill for modeling.
        self.prepare_tables(con)

    def get_quotes(self, con):
        """
        Get current price data from TDA

        Source: https://developer.tdameritrade.com/quotes/apis/get/marketdata/quotes

        :return: Current underlying stock price merged into the options table.
        """
        import httpx
        tickers = self.options.index.get_level_values('stock').unique().to_list()
        q = con.client.get_quotes(tickers)
        assert q.status_code == httpx.codes.OK, q.raise_for_status()

        prep = [v for k, v in q.json().items()]
        self.quote = pd.DataFrame.from_dict(prep)
        self.quote.rename(columns={'symbol': 'stock'}, inplace=True)

    def get_last_price(self, con):
        self.get_quotes(con)
        last_price = self.quote[[
            "stock",
            # "description",
            # "bidPrice",
            # "bidSize",
            # "bidId",
            # "askPrice",
            # "askSize",
            # "askId",
            "lastPrice",
            # "lastSize",
            # "lastId",
            # "openPrice",
            # "highPrice",
            # "lowPrice",
            # "closePrice",
            # "netChange",
            # "totalVolume",
            # "quoteTimeInLong",
            # "tradeTimeInLong",
            # "mark",
            # "exchange",
            # "exchangeName",
            # "marginable",
            # "shortable",
            # "volatility",
            # "digits",
            # "52WkHigh",
            # "52WkLow",
            # "peRatio",
            # "divAmount",
            # "divYield",
            # "divDate",
            # "securityStatus",
            # "regularMarketLastPrice",
            # "regularMarketLastSize",
            # "regularMarketNetChange",
            # "regularMarketTradeTimeInLong",
        ]]
        last_price.set_index('stock', inplace=True)

        # pd.merge(self.options, last_price, on='stock')
        self.options = self.options.join(last_price, on='stock')

    def get_risk_free_rate(self):
        # Get 10-yr risk-free rate from FRED.
        _url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
        _csv = pd.read_csv(_url)
        _value = _csv['DGS10'].values[-1]
        self.rf = float(_value) / 100

    def massage_options_table(self):
        # Optimize Options Table before heavy processing.
        opt = self.options
        opt['volatility'] = opt['volatility'] / 100
        # Prep DataFrame for Performance:
        # 1) Filter out columns,
        # 2) Filter out rows,
        opt = opt[
            (opt['openInterest'] > 0)
            & (opt['totalVolume'] > 0)
            & (opt['bid'] > 0)
            & (opt['ask'] > 0)
            & (opt['volatility'].astype('float').round(8).values > 0)
            ]
        # TODO: Remove rows too far away from the mean.

        # 3) Lower-range numerical and categoricals dtypes,
        opt = optimize_pd(opt, deal_with_na='fill', verbose=False)
        # 4) Sparse Columns.
        opt.dropna(axis='index', how='any', inplace=True)
        # 5) Re-index.
        # opt.reset_index(drop=True, inplace=True)

        self.options = opt

    def prepare_tables(self, con):
        self.get_last_price(con)
        self.get_risk_free_rate()
        self.massage_options_table()

    def black_scholes(self):
        df = self.options.copy()
        if all(i in df.columns for i in ['putCall', 'symbol']):
            df = df.drop(columns=['putCall', 'symbol']).reset_index()
        # Black Scholes data here. Option value, greeks.
        bsch = black_scholes(
            S=df['lastPrice'].to_numpy(),
            K=df['strikePrice'].to_numpy(),
            T=df['daysToExpiration'].to_numpy(),
            rf=self.rf,
            iv=df['volatility'].to_numpy(),
            option_type=df['option_type'].to_numpy()
        )
        df = pd.concat([df, bsch], axis='columns')
        df.set_index(
            ['stock', 'option_type', 'symbol'],
            inplace=True
        )
        self.options = df
        return self.options

    def probability_of_profits(self, montecarlo_iterations):
        # Probability of profits.
        msg = f" {self.options.shape[0]} contracts" \
              f" x {montecarlo_iterations} iterations"

        # Generating random range is expensive, so doing it once.
        rng = np.random.Generator(np.random.PCG64())

        df = self.options.copy()

        # 1) Simple version.
        # Disabled because it's significantly slower than multithreading.
        # print("Numpy Vector" + msg)
        # pop = mc_numpy_vector(
        #     curr_price, opt, ten_yr, rng, montecarlo_iterations
        # )

        # 2) Multiple Threads.
        print(f"Multi Threading" + msg)
        # Chunks set to 2, is optimal after experimenting.
        chunks = 2  # How many DTE's to process per Thread.
        pop = mc_multi_threading(
            df,
            self.rf,
            rng,
            montecarlo_iterations,
            chunks
        )
        pop.set_index(['symbol'], inplace=True)

        if all(i in df.columns for i in ['putCall', 'symbol']):
            df = df.drop(columns=['putCall', 'symbol'])

        df.reset_index(inplace=True)
        df = df.join(pop, on='symbol')
        df.sort_values(by='symbol', inplace=True)
        df.set_index(['stock', 'option_type', 'symbol'], inplace=True)

        self.options = df
        return self.options
