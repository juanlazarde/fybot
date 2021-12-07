import os
from typing import Dict, Optional
from datetime import datetime, date, timedelta
import numpy as np
import pandas_datareader as pdr
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from core.utils import optimize_pd, timeit


def black_scholes(S: float, K, T, rf: float, iv, option_type) -> pd.DataFrame:
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


def test_black_scholes():
    _result = black_scholes(
        S=100.,
        K=np.array([100., 110.]),
        T=np.array([20, 30]),
        rf=0.02,
        iv=np.array([.30, .20]),
        option_type=np.array(['call', 'put'])
    )
    _expected = {
        'option_value_bs': [2.85455546, 9.94825553],
        'intrinsic_value': [0., 10.],
        'time_value': [2.85455546, -0.05174447],
        'delta': [0.52022482, -0.94574287],
        'gamma': [0.05673639, 0.01919349],
        'theta': [-0.0726431, -0.00478972],
        'vega': [0.09326529, 0.03155094],
        'rho': [0.02694133, -0.08590894]
    }
    assert np.all(_result.round(2) == pd.DataFrame(_expected).round(2)),\
        "Black Scholes function is not working with arrays."
    return True


def monte_carlo(
        S: float,
        K: float,
        T: int,
        rf: float,
        iv: float,
        option_type: str,
        n: int = 100000,
        charts: bool = False) -> Dict[str, Optional[float]]:
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
                iv=opt['impliedVolatility'].to_numpy(),
                option_type=opt['option_type'].to_numpy(),
                n=1000
            )

    :param S: Underlying Asset or Stock Price ($).
    :param K: Strike or Excercise Price ($).
    :param T: Expiry time of the option (days).
    :param rf: Risk-free rate (decimal number range 0-1).
    :param iv: Volatility (decimal).
    :param option_type: Calls or Puts option type.
    :param n: Number of Monte Carlo iterantions. min 100,000 recommended.
    :param charts: Visualize charts.
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
    P = np.cumprod(1 + np.random.randn(n, T) * iv / np.sqrt(252), axis=1) * S

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

    if charts:
        # Price Cone
        for i in P[:100, :]:
            plt.plot(i)
        plt.title("Price Cone")
        plt.grid()
        plt.xlabel("Days", fontsize=12)
        plt.ylabel("Spot Price", fontsize=12)
        plt.show()

        # Price Distribution.
        plt.hist(P[:, -1], bins=40)
        plt.title("Distribution of Final Prices")
        plt.xlabel("Final Prices", fontsize=12)
        plt.ylabel("Counts")
        plt.show()

    # Returns Dictionary.
    return {
        'option_value_mc': val,  # Average value. Near Black Scholes Value.
        'value_quantile_5': np.percentile(p_last, 5),  # 5% chance below X
        'value_quantile_50': np.percentile(p_last, 50),  # 50% chance lands here.
        'value_quantile_95': np.percentile(p_last, 95),  # 5% chance above X.
        'probability_ITM': pop_ITM,  # Probability of ending ITM.
        'probability_of_50': p50,  # Probability of makeing half profit.
    }


def get_option_chains(ticker: str = 'AAPL'):
    """
    Gets data with options from Yahoo.

    :param ticker: String with a single stock.
    :return: opt (pd.DataFrame) Option Chain table from Yahoo.
    """
    yft = yf.Ticker(ticker)
    opt = pd.DataFrame()
    dtes = [(datetime.strptime(i, '%Y-%m-%d').date() - date.today()).days
            for i in yft.options]
    dtes_dict = dict(zip(yft.options, dtes))
    for i in yft.options:
        opt_tuple = yft.option_chain(i)

        opt_single = (pd.concat(
            [opt_tuple.calls, opt_tuple.puts],
            keys=['calls', 'puts'],
            names=['option_type']
        )
                      .reset_index()
                      .drop('level_1', axis=1)
                      )
        opt_single['stock'] = ticker
        opt_single['expirationDate'] = np.datetime64(i)
        opt_single['dte'] = dtes_dict[i]
        opt = pd.concat([opt, opt_single])
    opt.reset_index(drop=True, inplace=True)
    opt = opt.drop([
        'contractSize',
        'currency',
        'change',
        'percentChange',
        'lastTradeDate'
    ], axis=1)
    return opt


def get_current_price(ticker):
    # Get current price data.
    return float(
        pdr.DataReader(
            name=ticker,
            data_source='yahoo',
            start=date.today() - timedelta(days=1),
            end=date.today()
        )
        ['Adj Close']
        [-1]
    )


def get_ten_yr():
    # Get 10-yr risk-free rate.
    return float(
        pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
        ['DGS10']
        .values[-1]
    ) / 100


@timeit
def main(montecarlo_iterations: int = 200):
    # During development.
    DEVELOPMENT = True

    data = ()
    if DEVELOPMENT:
        try:
            with open('prob_price.pkl', 'rb') as f:
                data = pickle.load(f)
            assert len(data) != 0, "Datafile is empty."
        except Exception as err:
            print(f"Error reading file. {err}")

    if len(data) > 0:
        curr_price, ten_yr, option_chain = data
    else:
        # Get the ticker.
        ticker = input('Ticker: ').upper()

        # Get Current price, 10-yr risk-free rate, Option table.
        curr_price = get_current_price(ticker)
        option_chain = get_option_chains(ticker)
        ten_yr = get_ten_yr()

        # Include variables in one tuple.
        data = (curr_price, ten_yr, option_chain)
        with open('prob_price.pkl', 'wb') as f:
            pickle.dump(data, f)

    # Option processing begins here.
    opt = option_chain.copy()

    # Prep DataFrame for Performance:
    # 1) Filter out columns,
    opt.drop(labels=['expirationDate'], axis='columns', inplace=True)
    # 2) Filter out rows,
    opt = opt.loc[opt['impliedVolatility'].round(8).values > 0]
    # 3) Lower-range numerical and categoricals dtypes,
    curr_price = round(curr_price, 2)
    opt = opt.round({
        'lastPrice': 2,
        'bid': 2,
        'ask': 2,
        'impliedVolatility': 4,
    })
    opt = optimize_pd(opt, deal_with_na='fill', verbose=False)
    # # 4) Sparse Columns.
    opt.dropna(axis='index', how='any', inplace=True)
    # 5) Re-index.
    opt.reset_index(drop=True, inplace=True)

    # Black Scholes data here. Option value, greeks.
    bsch = black_scholes(
        S=curr_price,
        K=opt['strike'].to_numpy(),
        T=opt['dte'].to_numpy(),
        rf=ten_yr,
        iv=opt['impliedVolatility'].to_numpy(),
        option_type=opt['option_type'].to_numpy()
    )
    opt = pd.concat([opt, bsch], axis='columns')

    # Probability of profits.
    vector_monte_carlo = np.vectorize(monte_carlo)
    pop = vector_monte_carlo(
        S=curr_price,
        K=opt['strike'].to_numpy(),
        T=opt['dte'].to_numpy(),
        rf=ten_yr,
        iv=opt['impliedVolatility'].to_numpy(),
        option_type=opt['option_type'].to_numpy(),
        n=montecarlo_iterations
    )

    # Downside of pd.concat here is that it's not indexed.
    opt = pd.concat([opt, pd.json_normalize(pop.T)], axis=1)
    opt = opt.sort_values(by='contractSymbol').reset_index(drop=True)

    # Diagnostic Calculations
    opt['diff_last_val_bs'] = opt['option_value_bs'] / opt['lastPrice'] - 1
    opt['diff_last_val_mc'] = opt['option_value_mc'] / opt['lastPrice'] - 1
    opt['diff_val_mc_to_bs'] = opt['option_value_mc'] / opt['option_value_bs'] - 1

    return opt


if __name__ == '__main__':
    result = main(montecarlo_iterations=500)
    print(result)
    try:
        result.to_excel('test.xlsx')
    except Exception:
        pass
