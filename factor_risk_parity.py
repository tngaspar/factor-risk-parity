import pandas as pd
import stock_data
import factor_data
import statsmodels.api as sm
import numpy as np


def get_loading_matrix(stocks, factors):
    # add case where dates dont match
    model = sm.OLS(stocks, factors).fit()
    parameters = model.params
    loading_matrix = pd.DataFrame(data=parameters.values, index=parameters.index, columns=stocks.columns).T

    return loading_matrix


def get_risk_contributions(asset_weights, stocks, factors):
    # add size debug
    x = asset_weights
    loadings_matrix = get_loading_matrix(stocks, factors)

    Sigma = stocks.cov().values
    # change Sigma based of results from OLS

    vol_x = np.sqrt(np.matmul(np.matmul(x, Sigma), x))

    Aplus = np.linalg.pinv(loadings_matrix)

    AT_x = np.matmul(loadings_matrix.values.T, x)
    Aplus_Sigma_x = np.matmul(np.matmul(Aplus, Sigma), x)

    risk_contributions = (AT_x * Aplus_Sigma_x) / vol_x

    return risk_contributions
