import logging
import time
import datetime
import numpy as np

from calcs import retrieve_return_df, calculate_portfolio_vol, calculate_var, generate_correlated_data

logging.basicConfig(level=logging.DEBUG)

VAR_ALPHA = 0.05
NUM_SIM = 5000


def main_stats_section(corr):
    matrix_normal_corr = generate_correlated_data(NUM_SIM, corr)
    return matrix_normal_corr


def main_var_section(weights: np.array):
    output = {}
    # retrieve historical returns for each asset in CAD
    r_df = retrieve_return_df()
    cov = r_df.cov()

    # calculate historical port returns
    portf_r = r_df.dot(w)
    avg_r = portf_r.mean(axis=0)

    vol = calculate_portfolio_vol(weights=weights, cov_matrix=cov)  # same as take Std from port return series

    var = calculate_var(avg_r, vol, VAR_ALPHA)
    output['1d-VaR'] = var
    var10 = calculate_var(avg_r*10, vol * np.sqrt(10), VAR_ALPHA)
    output['10d-VaR'] = var10
    var252 = calculate_var(avg_r*252, vol * np.sqrt(252), VAR_ALPHA)
    output['1year-VaR'] = var252

    incremental_contributions = {}
    for i, asset_name in enumerate(r_df.columns):
        # Temporarily remove the asset from the portfolio
        mod_weights = np.delete(weights, i)
        mod_asset_returns = r_df.drop(columns=asset_name)

        # Calculate VaR without the asset
        mod_portf_r = mod_asset_returns.dot(mod_weights)
        mod_var = calculate_var(mod_portf_r.mean(), mod_portf_r.std(), VAR_ALPHA)

        # Calculate risk contribution of the asset
        risk_contribution = var - mod_var

        incremental_contributions[asset_name] = risk_contribution

    output['incremental_contributions'] = incremental_contributions

    return var


if __name__ == '__main__':

    start_time = time.time()

    matrix_corr = np.array([[1.0, 0.5], [0.5, 1.0]])   # can expand to a larger size of corr matrix
    simulated_data = main_stats_section(matrix_corr)

    w = np.repeat(1. / 6, 6)   # in order of: SP500 TSX GSCI GOLD US10Y CA10Y
    main_var_section(w)

    end_time = time.time()

    display_time = str(datetime.timedelta(seconds=(end_time - start_time)))
    logging.info(f"Completed! Complete running duration: " + display_time)
