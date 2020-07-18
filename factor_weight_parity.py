# Factor Weight Parity portfolio construction approach

import pandas as pd
import stock_data
import factor_data
import numpy as np
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
from alive_progress import alive_bar
from alive_progress import config_handler
import factor_risk_parity as frp

config_handler.set_global(force_tty=True)


def weights_factor_weight_parity(stocks, factor_structure, loadings_matrix, x0):
    """ Calculates assets weights according to the factor weight parity approach

    :param stocks: DataFrame of stock returns
    :param factor_structure: Structure of factor clusters (factors that share weight budgets)
    :param loadings_matrix: Factor to stocks loading matrix
    :param x0: asset weighs vector for initialization
    :return: asset weights vector using factor risk parity method
    """

    n_stocks = stocks.shape[1]
    if x0 is None:
        x0 = np.ones(n_stocks) * 1 / n_stocks

    def square(listt):
        return [i ** 2 for i in listt]

    fun = lambda  x: sum(square(np.matmul(loadings_matrix.values.T, x) - 1 / loadings_matrix.shape[1]))

    # constrains
    cons = [{'type': 'ineq', 'fun': lambda x: -sum(x) + 1},
            {'type': 'ineq', 'fun': lambda x: sum(x) - 1},

            ]

    # bounds
    bounds_short_lev = [(-1 / n_stocks, 1) for n in range(n_stocks)]
    bounds_long = [(0, 1) for n in range(n_stocks)]
    bounds = bounds_short_lev

    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-5, options={'disp': False})
    # print(res.fun)
    return res.x


def portfolio_weights_factor_weight_parity(tickers, factor_tickers, start_date, end_date, portfolio_rebalance_period):
    """ Applies factor weight parity over a period of time. Can be used for back testing

    :param tickers: List of tickers of all candidate stocks to the portfolio
    :param factor_tickers: List of tickers of factor used and respective cluster format
    :param start_date: first date of the investment period
    :param end_date: last date of the investment period
    :param portfolio_rebalance_period: portfolio re-balancing period (monthly, weekly, etc.)
    :return: DataFrame of asset weight vectors for each portfolio reabalancing date
    """
    factor_structure = []
    factor_tickers_flat = []
    for group in factor_tickers:
        if type(group) is list:
            factor_structure.append(len(group))
            for factor in group:
                factor_tickers_flat.append(factor)
        else:
            factor_structure.append(1)
            factor_tickers_flat.append(group)

    business_days_end_months = pd.date_range(start_date, end_date, freq=portfolio_rebalance_period)
    portfolio_weights = pd.DataFrame(index=business_days_end_months, columns=tickers)
    x0 = None
    with alive_bar(len(business_days_end_months)) as bar:
        for t in business_days_end_months:
            stocks = stock_data.get_daily_returns(tickers, t + relativedelta(months=-12), t)[1:]
            factors = factor_data.get_factors(factor_tickers_flat, stocks.index[0], stocks.index[-1])
            loadings_matrix = frp.get_loading_matrix(stocks, factors)
            sigma = frp.big_sigma(stocks)
            portfolio_weights.loc[t] = weights_factor_weight_parity(stocks, factor_structure, loadings_matrix, x0)
            x0 = portfolio_weights.loc[t]
            print(np.matmul(loadings_matrix.T, x0))
            bar()

    return portfolio_weights

