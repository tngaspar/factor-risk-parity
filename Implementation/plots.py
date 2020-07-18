import pandas as pd
import stock_data as sdata
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import backtest_functions as bfunc


sns.set(style="white")
os.chdir('../')

# first plot data
ew_daily = pd.read_csv('Output/ew_daily_returns.csv', index_col='Date')
fwp_daily = pd.read_csv('Output/fwp_daily_returns.csv', index_col='Date')
rp_daily = pd.read_csv('Implementation/rp_daily_returns.csv', index_col='Date')
frp_daily = pd.read_csv('Implementation/frp_daily_returns_4f.csv', index_col='Date')


# benchmark
sp500 = sdata.get_sp500_index_returns(ew_daily.index[0], ew_daily.index[-1])
sp500.to_csv(r'Implementation/sp500_returns.csv')
sp500_returns = pd.read_csv('Implementation/sp500_returns.csv', index_col='Date')

# merge
all_pf_returns = pd.concat([ew_daily, fwp_daily, rp_daily, frp_daily, sp500_returns], axis=1, join='inner').fillna(0)
all_pf_returns.index = pd.to_datetime(all_pf_returns.index)
all_pf_returns.columns = ['EW', 'FWP', 'RP', 'FRP', 'S&P 500']
all_pf_cum_returns = (all_pf_returns + 1).cumprod()
# plot cum_returns

sns.lineplot(data=all_pf_cum_returns, ci=None, estimator=None)
#plt.title('')
plt.ylabel('Cumulative Returns')
plt.xlabel('')
plt.grid(True)
#plt.ylim(0, 7)
plt.xlim(dt.datetime(2005,1,1), dt.datetime(2020,1,1))
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, lw=2)
plt.savefig('Plots/all_port_4f_cum_ret.pdf')
plt.show()
plt.close('all')


#w ans wo negative 1/m factor restriction:
frp_daily_1m = pd.read_csv('Implementation/frp_daily_4f_025.csv', index_col='Date')
new_frp_returns = pd.concat([frp_daily_1m, frp_daily, sp500_returns], axis=1, join='inner').fillna(0)
new_frp_returns.index = pd.to_datetime(new_frp_returns.index)
new_frp_returns.columns = ['FRP w/ relaxation', 'FRP w/out relaxation', 'S&P500']
new_frp_returns = (new_frp_returns + 1).cumprod()

sns.lineplot(data=new_frp_returns, ci=None, estimator=None)
#plt.title('')
plt.ylabel('Cumulative Returns')
plt.xlabel('')
plt.grid(True)
#plt.ylim(0, 7)
plt.xlim(dt.datetime(2005,1,1), dt.datetime(2020,1,1))
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, lw=2)
plt.savefig('Plots/frp_relax_4f_cum_ret.pdf')
plt.show()
plt.close('all')


# intersection:
frp_portfolio_weights_intersect = pd.read_csv(r'Implementation\frp_x_intersection.csv', index_col=0)
frp_portfolio_weights_intersect.index = pd.to_datetime(frp_portfolio_weights_intersect.index)
frp_intersect_daily_returns = bfunc.daily_returns_of_portfolio(frp_portfolio_weights_intersect)


# shared rc:
frp_shared_rc_daily_returns = pd.read_csv(r'Implementation\frp_shared_RC.csv', index_col=0)
frp_shared_rc_daily_returns.index = pd.to_datetime(frp_shared_rc_daily_returns.index)


# plot 1/m, intersection, shared rc and sp500
frp_returns_inter_shared_rc = pd.concat([frp_daily_1m,
                                         frp_intersect_daily_returns,
                                         frp_shared_rc_daily_returns,
                                         sp500_returns], axis=1, join='inner').fillna(0)
frp_returns_inter_shared_rc.index = pd.to_datetime(frp_returns_inter_shared_rc.index)
frp_returns_inter_shared_rc.columns = ['FRP w/ relax.', 'FRP w/ cluster Intersection',
                                       'FRP w/ cluster shared risk contributions', 'S&P500']
frp_cum_returns_inter_shared_rc = (frp_returns_inter_shared_rc + 1).cumprod()

sns.lineplot(data=frp_cum_returns_inter_shared_rc, ci=None, estimator=None)
#plt.title('')
plt.ylabel('Cumulative Returns')
plt.xlabel('')
plt.grid(True)
#plt.ylim(0, 7)
plt.xlim(dt.datetime(2005,1,1), dt.datetime(2020,1,1))
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, lw=2)
plt.savefig('Plots/frp_inter_shared_rc.pdf')
plt.show()
plt.close('all')


frp_daily = pd.read_csv('Implementation/frp_daily_4f_025.csv', index_col='Date')

import numpy as np
sp500['Volatility'] = sp500['Returns'].rolling(window=151).std() * np.sqrt(252)
vol_frp = frp_daily.rolling(window=151).std() * np.sqrt(252)

sns.lineplot(y=vol_frp.values, x=vol_frp.index, data=vol_frp.values, ci=None, estimator=None, label='6-months volatility',
             color='red')
plt.axhline(y=np.mean(sp500['Volatility']), color='gray', linestyle='--', label = 'Average Volatility', lw=2, alpha=0.7)
plt.xlim(dt.datetime(1990,1 , 1), dt.datetime(2021,1,1))
plt.title('S&P500 6-months rolling volatility ')
plt.ylabel('Volatility')
plt.xlabel('')
plt.legend()
plt.savefig('SP500_vol.pdf')
plt.show()
plt.close()


frp_daily.index = pd.to_datetime(frp_daily.index)
frp_cum_prod = (frp_daily + 1).cumprod()
current_max = np.maximum.accumulate(frp_cum_prod)
underwater = -100 * ((current_max - frp_cum_prod) / current_max)
g = underwater.plot(kind='area', color='tomato', alpha=0.7)
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
plt.grid(True)
plt.ylabel('Drawdown')
plt.legend('')
plt.xlim(dt.datetime(2005,1,1), dt.datetime(2020,1,1))
plt.title('Underwater Plot of FRP portfolio w/relaxation')
plt.xlabel('')
plt.savefig('Implementation/FRP_underwater.pdf')
plt.show()

import empyrical as ep
annual_returns = pd.DataFrame(ep.aggregate_returns(frp_daily, 'yearly'))
ax = plt.gca()
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
ax.axhline(
        100 *
        annual_returns.values.mean(),
        color='gray',
        linestyle='--',
        lw=2,
        alpha=0.7)
(100 * annual_returns.sort_index(ascending=True)
).plot(ax=ax, kind='bar', alpha=1)
ax.axhline(0.0, color='black', linestyle='-', lw=3)
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
ax.set_xlabel('')
ax.set_ylabel('Returns')
ax.set_title("Annual returns of the FRP portfolio w/relaxation")
ax.legend(['Mean'], frameon=True, framealpha=1)
ax.grid(b=True, axis='y')
plt.savefig('Implementation/FRP_annual_ret.pdf')
plt.show()