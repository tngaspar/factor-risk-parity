# calculation for performance measures of portfolios are hosted here
import empyrical as ep
import stock_data as sdata
from scipy import stats
import pandas as pd
import numpy as np

def performance_measures(portfolio_daily_returns, benchmark_returns=None, var_probability=0.05):
    # return pandas of all measures
    measures = {'Annualized Returns (CAGR) (%)': annual_returns_cagr(portfolio_daily_returns)*100,
                'Cumulative Returns (%)': cumulative_returns(portfolio_daily_returns)*100,
                'Annualized Volatility (%)': annual_volatility(portfolio_daily_returns)*100,
                'Sharpe Ratio': sharpe_ratio(portfolio_daily_returns),
                'Max Drawdown (%)': max_drawdown(portfolio_daily_returns)*100,
                'Calmar Ratio': calmar_ratio(portfolio_daily_returns),
                'Stability': stability(portfolio_daily_returns),
                'Skewness': skewness(portfolio_daily_returns),
                'Kurtosis': kurtosis(portfolio_daily_returns),
                'Daily Value at Risk (VaR) (%)': value_at_risk(portfolio_daily_returns, var_probability)*100,
                'Expected Shortfall (%)': expected_shortfall(portfolio_daily_returns, var_probability)*100,
                'Tail Ratio': tail_ratio(portfolio_daily_returns),
                'Alpha': alpha(portfolio_daily_returns, benchmark_returns),
                'Beta': beta(portfolio_daily_returns, benchmark_returns)
                }
    return pd.Series(measures)


def annual_returns_cagr(portfolio_daily_returns):
    return ep.annual_return(portfolio_daily_returns)


def cumulative_returns(portfolio_daily_returns):
    return ep.cum_returns_final(portfolio_daily_returns)


def annual_volatility(portfolio_daily_returns):
    return ep.annual_volatility(portfolio_daily_returns)


def sharpe_ratio(portfolio_daily_returns, risk_free=0):
    return ep.sharpe_ratio(portfolio_daily_returns)


def max_drawdown(portfolio_daily_returns):
    return ep.max_drawdown(portfolio_daily_returns)


def calmar_ratio(portfolio_daily_returns):
    return ep.calmar_ratio(portfolio_daily_returns)


def alpha(portfolio_daily_returns, benchmark_returns=None):
    if benchmark_returns is None:
        benchmark_returns = sdata.get_sp500_index_returns(portfolio_daily_returns.index[0],
                                                          portfolio_daily_returns.index[-1])
    return ep.alpha(portfolio_daily_returns, benchmark_returns)


def beta(portfolio_daily_returns, benchmark_returns=None):
    if benchmark_returns is None:
        benchmark_returns = sdata.get_sp500_index_returns(portfolio_daily_returns.index[0],
                                                          portfolio_daily_returns.index[-1])
    return ep.beta(portfolio_daily_returns, benchmark_returns)


def stability(portfolio_daily_returns):
    return ep.stability_of_timeseries(portfolio_daily_returns)


def skewness(portfolio_daily_returns):
    return stats.skew(portfolio_daily_returns)


def kurtosis(portfolio_daily_retuns):
    return stats.kurtosis(portfolio_daily_retuns)


def tail_ratio(portfolio_daily_returns, probability=0.05, risk_free=0):
    #return ep.tail_ratio(portfolio_daily_returns)
    value_ar = value_at_risk(portfolio_daily_returns)
    cvar = expected_shortfall(portfolio_daily_returns)
    sum_r = 0
    count = 0
    for r in portfolio_daily_returns:
        if r <= value_ar:
            sum_r += (r - cvar) ** 2
            count += 1
    tail_risk = np.sqrt(sum_r / count)

    cagr = annual_returns_cagr(portfolio_daily_returns)
    tail_ratio = (cagr - risk_free) / (tail_risk*np.sqrt(252))
    return tail_ratio


def value_at_risk(portfolio_daily_returns, probability=0.05):
    return ep.value_at_risk(portfolio_daily_returns, probability)


def expected_shortfall(portfolio_daily_returns, probability=0.05):
    return ep.conditional_value_at_risk(portfolio_daily_returns, probability)

# factor contribution measures
import factor_data
import stock_data
from dateutil.relativedelta import relativedelta
from factor_risk_parity import get_loading_matrix, get_risk_contributions, big_sigma
from alive_progress import alive_bar
from alive_progress import config_handler
config_handler.set_global(force_tty=True)


def factor_exposures_and_risk_contributions(x, factor_tickers):
    stock_tickers = x.columns
    f_exposures = pd.DataFrame([])
    exposures = []
    rc = []
    with alive_bar(len(x.index)) as bar:
        for t in x.index:
            stock_returns = stock_data.get_daily_returns(stock_tickers, t + relativedelta(months=-12), t)[1:]
            factor_returns = factor_data.get_factors(factor_tickers, stock_returns.index[0], stock_returns.index[-1])
            l_mat = get_loading_matrix(stock_returns, factor_returns)
            exposures.append(np.matmul(l_mat.T, x.loc[t]))
            sigma = big_sigma(stock_returns)
            rc.append(get_risk_contributions(x.loc[t], l_mat, sigma))
            bar()
    return pd.DataFrame(exposures), pd.DataFrame(rc)



# def factor_risk_contributions(x, factor_tickers):
#     stock_tickers = x.columns
#     f_rc = pd.DataFrame([])
#     rc = []
#     with alive_bar(len(x.index)) as bar:
#         for t in x.index:
#             stock_returns = stock_data.get_daily_returns(stock_tickers, t + relativedelta(months=-12), t)[1:]
#             factor_returns = factor_data.get_factors(factor_tickers, stock_returns.index[0], stock_returns.index[-1])
#             l_mat = get_loading_matrix(stock_returns, factor_returns)
# )
#             bar()
#     return pd.DataFrame(rc)
