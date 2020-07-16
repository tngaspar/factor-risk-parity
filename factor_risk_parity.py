# Factor Risk Parity portfolio construction approach

import pandas as pd
import stock_data
import factor_data
import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
from factor_analyzer.factor_analyzer import FactorAnalyzer
from alive_progress import alive_bar
from alive_progress import config_handler
config_handler.set_global(force_tty=True)


def get_loading_matrix(stocks, factors):
    """ Obtains factor to stocks loading matrix

    :param stocks: DataFrame of all stocks considered
    :param factors: DataFrame of all factors considered
    :return: Matrix with factor to stocks loadings
    """
    # add case where dates dont match
    # factors['const'] = 1
    model = sm.OLS(stocks, factors).fit()
    parameters = model.params
    loading_matrix = pd.DataFrame(data=parameters.values, index=parameters.index,
                                  columns=stocks.columns).T
    # .drop('const', axis=1)
    return loading_matrix


def get_risk_contributions(asset_weights, loadings_matrix, Sigma):
    """ Gets the risk contributions of risk factor to the portfolio

    :param asset_weights: weight vector for all assets
    :param loadings_matrix: loading matrix of factors to stocks
    :param Sigma: covariance matrix of stocks
    :return: Risk contributions of factors to portfolio
    """
    # add size debug
    x = asset_weights

    # change Sigma based of results from OLS
    vol_x = np.sqrt(np.matmul(np.matmul(x, Sigma), x))

    Aplus = np.linalg.pinv(loadings_matrix)
    AT_x = np.matmul(loadings_matrix.values.T, x)
    Aplus_Sigma_x = np.matmul(np.matmul(Aplus, Sigma), x)

    risk_contributions = (AT_x * Aplus_Sigma_x) / vol_x

    return risk_contributions


def big_sigma(stock_returns):
    """Get Covariance matrix of stock returns

    :param stock_returns: DataFrame of stock returns
    :return: Covariance matrix
    """
    sigma = stock_returns.cov().values

    return sigma


def sigma_x(x, Sigma):
    """ volatility of portfolio

    :param x: weight vector of assets
    :param Sigma: Covariance matrix of assets
    :return: portfolio volatility (or standard deviation)
    """
    vol_x = np.sqrt(np.matmul(np.matmul(x, Sigma), x))

    return vol_x


def sigma_x_rc(x, loadings_matrix, sigma):
    """ volatility of portfolio as sum of factor risk contributions

    :param x: asset weight vector
    :param loadings_matrix: factor to sotcks loading matrix
    :param sigma: assets covariance matrix
    :return: portfolio volatility
    """
    vol_x = get_risk_contributions(x, loadings_matrix, sigma).sum()

    return vol_x


def weights_factor_risk_parity(stocks, factor_structure, loadings_matrix, Sigma, x0):
    """ Calculates assets weights according to the factor risk parity approach

    :param stocks: DataFrame of stock returns
    :param factor_structure: Structure of factor clusters (factors that share risk budgets)
    :param loadings_matrix: Factor to stocks loading matrix
    :param Sigma: stock returns covariance matrix
    :param x0: asset weighs vector for initialization
    :return: asset weights vector using factor risk parity method
    """

    n_stocks = stocks.shape[1]
    if x0 is None:
        x0 = np.ones(n_stocks) * 1 / n_stocks
    sigma = big_sigma(stocks)

    def square(listt):
        return [i ** 2 for i in listt]

    #def fun(x):
        risk_contributions = get_risk_contributions(x, loadings_matrix, Sigma)
        #print(risk_contributions)
    #    total_risk_contributions = sum(risk_contributions)
        #print(sum(risk_contributions))
    #    clusters_rcs = [part.sum() for part in np.split(risk_contributions, np.cumsum(factor_structure))[:-1]]
        # f = sum(square(clusters_rcs / sigma_x(x, sigma) - 1 / len(factor_structure)))
    #    f = sum(square(clusters_rcs / total_risk_contributions - 1 / len(factor_structure)))
        #print(clusters_rcs / total_risk_contributions)
     #   return f
    print(5.11)
    # old function without factor clustering:
    #fun = lambda x: sum((get_risk_contributions(x, loadings_matrix, Sigma) / sigma_x(x, sigma) - 1 /
    #                     loadings_matrix.shape[1]) ** 2)
    fun = lambda x: sum(square(get_risk_contributions(x, loadings_matrix, Sigma) /
                         sigma_x(x, sigma) - 1 /
                         loadings_matrix.shape[1]))
    # constrains
    cons = [{'type': 'ineq', 'fun': lambda x: -sum(x) + 1},
            {'type': 'ineq', 'fun': lambda x: sum(x) - 1},
            #{'type': 'ineq', 'fun': lambda x: sum(np.clip(x, -99, 0)) - (-1)},
            {'type': 'ineq', 'fun': lambda x: np.matmul(loadings_matrix.values.T, x) - (-0.25)},
            {'type': 'ineq', 'fun': lambda x: -np.matmul(loadings_matrix.values.T, x) + 1}#,
            #{'type': 'ineq', 'fun': lambda x: get_risk_contributions(x, loadings_matrix, Sigma) - 0}
            ]
    # avoid negative RC in the last equation for the shared RC case
    # bounds
    bounds_short_lev = [(-1/n_stocks, 1) for n in range(n_stocks)]
    bounds_long = [(0, 1) for n in range(n_stocks)]
    bounds = bounds_short_lev

    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-5, options={'disp': False})
    print(res.fun)
    return res.x


def portfolio_weights_factor_risk_parity(tickers, factor_tickers, start_date, end_date, portfolio_rebalance_period):
    """ Applies factor risk parity over a period of time. Can be used for back testing

    :param tickers: List of tickers of all candidate stocks to the portfolio
    :param factor_tickers: List of tickers of factor used and respective cluster format
    :param start_date: first date of the investment period
    :param end_date: last date of the investment period
    :param portfolio_rebalance_period: portfolio re-balancing period (monthly, weekly, etc.)
    :return: DataFrame of asset weight vectors for each portfolio rebalancing date
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
            # sp500 = stock_data.get_sp500_index_returns(stocks.index[0], stocks.index[-1])
            # stocks = stocks - sp500.values
            factors = factor_data.get_factors(factor_tickers_flat, stocks.index[0], stocks.index[-1])
            # factors.iloc[:, -7:].subtract(factors['Mkt-RF'], axis=0)
            # if factor intersection is desired insert here
            loadings_matrix = get_loading_matrix(stocks, factors)
            sigma = big_sigma(stocks)
            portfolio_weights.loc[t] = weights_factor_risk_parity(stocks, factor_structure, loadings_matrix, sigma, x0)
            x0 = portfolio_weights.loc[t]
            print((get_risk_contributions(x0, loadings_matrix, sigma)/sigma_x_rc(x0, loadings_matrix, sigma)))
            #print((get_risk_contributions(x0, loadings_matrix, sigma)/sigma_x_rc(x0, loadings_matrix, sigma)).sum())
            #print((get_risk_contributions(x0, loadings_matrix, sigma)))
            print(np.matmul(loadings_matrix.T, x0))
            #(sigma_x(x0, sigma))
            bar()

    return portfolio_weights


def weights_factor_risk_parity_v2(stocks, factor_structure, loadings_matrix, Sigma, x0):
    """ Calculates assets weights according to the factor risk parity approach

    :param stocks: DataFrame of stock returns
    :param factor_structure: Structure of factor clusters (factors that share risk budgets)
    :param loadings_matrix: Factor to stocks loading matrix
    :param Sigma: stock returns covariance matrix
    :param x0: asset weighs vector for initialization
    :return: asset weights vector using factor risk parity method
    """

    n_stocks = stocks.shape[1]
    if x0 is None:
        x0 = np.ones(n_stocks) * 1 / n_stocks
    sigma = big_sigma(stocks)

    def square(listt):
        return [i ** 2 for i in listt]

    def fun(x):
        risk_contributions = get_risk_contributions(x, loadings_matrix, Sigma)
        total_risk_contributions = sum(risk_contributions)
        clusters_rcs = [part.sum() for part in np.split(risk_contributions, np.cumsum(factor_structure))[:-1]]
        f = sum(square(clusters_rcs / total_risk_contributions - 1 / len(factor_structure)))
        return f
    print(5.20)


    # constrains
    cons = [{'type': 'ineq', 'fun': lambda x: -sum(x) + 1},
            {'type': 'ineq', 'fun': lambda x: sum(x) - 1},
            # {'type': 'ineq', 'fun': lambda x: sum(np.clip(x, -99, 0)) - (-1)},
            {'type': 'ineq', 'fun': lambda x: np.matmul(loadings_matrix.values.T, x) - (-.00)},
            {'type': 'ineq', 'fun': lambda x: -np.matmul(loadings_matrix.values.T, x) + 1}#,
            # {'type': 'ineq', 'fun': lambda x: get_risk_contributions(x, loadings_matrix, Sigma) - 0}
            ]
    # avoid negative RC in the last equation for the shared RC case
    # bounds
    bounds_short_lev = [(-1, 1) for n in range(n_stocks)]
    bounds_long = [(0, 1) for n in range(n_stocks)]
    bounds = bounds_short_lev

    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-5, options={'disp': False})
    print(res.fun)
    return res.x