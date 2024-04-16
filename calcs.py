import yfinance as yf
import numpy as np
import pandas as pd
import datetime
from functools import reduce
from scipy.stats import norm
from copulas.multivariate import GaussianMultivariate

CLOSE = 'Close'
DURATION = 8

TICKER_MAPPING = {
    'SP500': '^GSPC',
    'TSX': '^GSPTSE',
    'CADUSD': 'CADUSD=X',  # USD/CAD initially from yfinance, will convert later
    'GSCI': '^SPGSCI',
    'GOLD': 'GLD',
    'US10Y': '^TNX'
}


def retrieve_return_df():
    """
    Retrieve historical returns from YahooFinance and csv files.
    """

    price_data = {}
    for name, ticker in TICKER_MAPPING.items():
        history_df = yf.Ticker(ticker).history(period="max")[CLOSE]
        history_df.index = history_df.index.map(lambda x: x.date())

        # need CAD/USD if holding USD assets
        if name == 'CADUSD':
            history_df = 1 / history_df

        price_data[name] = history_df

    CA10Y = pd.read_csv('./CA10.csv')
    CA10Y = CA10Y.set_index('Date')
    CA10Y.index = CA10Y.index.map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
    price_data['CA10Y'] = CA10Y

    ret_data = {}
    for name, hist_df in price_data.items():
        # approx ret = -Mod duration * change in yield
        if name in ['CA10Y', 'US10Y']:
            ret = hist_df.sort_index().diff().dropna() * (-DURATION) / 100.

        # simple return
        else:
            ret = hist_df.sort_index().pct_change().dropna()

        ret_data[name] = ret

    common_dates = reduce(lambda x, y: x.intersection(y), [_df.index for _df in ret_data.values()])[:-1]

    fx_df = ret_data['CADUSD'].reindex(common_dates)
    r_df = pd.DataFrame()
    for name, hist_df in ret_data.items():
        # CAD asset remain unchanged
        if name in ['TSX', 'CA10Y']:
            r_df[name] = hist_df.reindex(common_dates)
        # USD asset need to be adjusted for fx
        elif name != 'CADUSD':
            r_df[name] = (1. + hist_df.reindex(common_dates)) * (1. + fx_df) - 1.

    return r_df


def calculate_portfolio_vol(weights, cov_matrix):
    """
    Calculate portfolio volatility given weights and covariance matrix.
    """

    weights = np.array(weights)

    # Calculate portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

    # Calculate portfolio volatility (standard deviation)
    portfolio_volatility = np.sqrt(portfolio_variance)

    return portfolio_volatility


def calculate_var(mean_return, std_return, alpha=0.05):
    """
    Calculate Value at Risk (VaR) assuming normal distribution.
    """

    return mean_return + std_return * norm.ppf(alpha)


def generate_correlated_data(n, matrix_corr, seed=None):
    """
    Generate correlated data using Cholesky decomposition.
    """

    if seed:
        np.random.seed(0)

    matrix_L = np.linalg.cholesky(matrix_corr)

    matrix_normal_iid = np.random.normal(loc=0, scale=1, size=(matrix_corr.shape[0], n))
    matrix_normal_corr = np.dot(matrix_L, matrix_normal_iid)

    return matrix_normal_corr


def simulate_bivariate_normal(ts1, ts2, corr):
    """
    Simulate bivariate normal with copula.
    """

    mean_x = ts1.mean()
    std_x = ts1.std()

    mean_y = ts2.mean()
    std_y = ts2.std()

    # convert ts1, ts2 to zscore and mapped to uniform distribution, as u1 and u2
    u1 = norm.cdf((ts1 - mean_x) / std_x)
    u2 = norm.cdf((ts2 - mean_y) / std_y)

    copula = GaussianMultivariate(
        rotation=corr
    )
    copula.fit([u1, u2])
    x = norm.ppf(u1, loc=mean_x, scale=std_x)
    y = norm.ppf(u2, loc=mean_y, scale=std_y)
    return x, y
