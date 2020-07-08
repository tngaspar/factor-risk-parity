# brief risk factor analysis and clustering process

import factor_data as fdata
import datetime as dt
import stock_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import importlib
import numpy as np
from scipy.stats import pearsonr
import scipy.cluster.hierarchy as spc
import matplotlib.pylab as pylab
sns.set(style="white", context='paper')
params = {'axes.titlesize':'large'}
pylab.rcParams.update(params)

importlib.reload(fdata)


daily_sp500_ret = pd.DataFrame(pd.read_csv(r'Data/SP500_index_daily_returns.csv')['SP_500'])
daily_sp500_ret.index = pd.to_datetime(pd.read_csv(r'Data/SP500_index_daily_returns.csv')['Date'])
daily_sp500_ret = daily_sp500_ret[dt.date(1990, 1, 1):]
sp500 = daily_sp500_ret.rename(columns={'SP_500': 'Returns'})
sp500['Cumulative Returns'] = (sp500['Returns'] + 1).cumprod()

sp500 = ['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK',
         'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'AGN', 'ADS', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR',
         'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM',
         'AON', 'AOS', 'APA', 'AIV', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ARNC', 'ANET', 'AJG', 'AIZ', 'ATO', 'T', 'ADSK',
         'ADP', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BK', 'BAX', 'BDX', 'BRK-B', 'BBY', 'BIIB', 'BLK', 'BA',
         'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BF-B', 'CHRW', 'COG', 'CDNS', 'CPB', 'COF', 'CPRI', 'CAH',
         'KMX', 'CCL', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHTR', 'CVX',
         'CMG', 'CB', 'CHD', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO',
         'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CTVA', 'COST', 'COTY',
         'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'FANG', 'DLR', 'DFS',
         'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'ETFC', 'EMN',
         'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'EVRG',
         'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE',
         'FRC', 'FISV', 'FLT', 'FLIR', 'FLS', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS',
         'GRMN', 'IT', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HRB', 'HAL', 'HBI', 'HOG',
         'HIG', 'HAS', 'HCA', 'PEAK', 'HP', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HFC', 'HOLX', 'HD', 'HON', 'HRL',
         'HST', 'HPQ', 'HUM', 'HBAN', 'HII', 'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN', 'IR', 'INTC', 'ICE', 'IBM', 'INCY',
         'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'SJM', 'JNJ', 'JCI',
         'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KHC', 'KR', 'LB', 'LHX', 'LH',
         'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LYB', 'MTB',
         'M', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MKC', 'MXIM', 'MCD', 'MCK', 'MDT', 'MRK', 'MET',
         'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI',
         'MYL', 'NDAQ', 'NOV', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NBL', 'JWN',
         'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE',
         'ORCL', 'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO', 'PFE', 'PM', 'PSX',
         'PNW', 'PXD', 'PNC', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO',
         'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTN', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP',
         'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SLG', 'SNA',
         'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR',
         'TGT', 'TEL', 'FTI', 'TFX', 'TXN', 'TXT', 'TMO', 'TIF', 'TJX', 'TSCO', 'TDG', 'TRV', 'TFC', 'TWTR', 'TSN',
         'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'VFC', 'VLO',
         'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VIAC', 'V', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'DIS',
         'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYNN', 'XEL', 'XRX',
         'XLNX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']

start_date = dt.date(1990, 1, 1)
end_date = dt.date(2004, 12, 31)
all_factors_ret = fdata.get_factors(['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'BaB', 'QMJ', 'HML_Devil', 'UMD'],
                                    start_date, end_date)
stock_ret = stock_data.get_daily_returns(sp500, start_date, end_date)[1:]


all_factors_cum_ret = (all_factors_ret + 1).cumprod()

# plot returns K. French:
sns.lineplot(data=all_factors_cum_ret[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', ]], ci=None, estimator=None)
#sns.lineplot(data=sp500['Cumulative Returns'], ci=None, estimator=None)
plt.title('Cumulative returns of factors from the K. French database')
plt.ylabel('Cumulative Returns')
plt.xlabel('')
plt.grid(True)
plt.ylim(0, 7)
plt.xlim(dt.datetime(1990,1 , 1), dt.datetime(2005,1,1))
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, lw=2)
plt.savefig('Plots/Kfrench_factors_cum_ret.pdf')
plt.show()
plt.close('all')

# AQR factors:
sns.lineplot(data=all_factors_cum_ret[['BaB', 'QMJ', 'HML_Devil', 'UMD',]], ci=None, estimator=None)
#sns.lineplot(data=sp500['Cumulative Returns'], ci=None, estimator=None)
plt.title('Cumulative returns of factors from the AQR database')
plt.ylabel('Cumulative Returns')
plt.xlabel('')
plt.grid(True)
plt.ylim(0, 7)
plt.xlim(dt.datetime(1990,1 , 1), dt.datetime(2005,1,1))
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, lw=2)
plt.savefig('Plots/AQR_factors_cum_ret.pdf')
plt.show()
plt.close('all')


# Correlation
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues


corr = all_factors_ret.corr()
plt.figure(figsize=(6.5,6.5))
sns.heatmap(all_factors_ret.corr().round(2),
            vmin=-1,
            cmap='coolwarm',
            annot=True, linewidths=1);
plt.title('Correlation heatmap of all factors')
plt.savefig('Plots/corr_all_factors.pdf')
plt.show()
plt.close('all')

# to create table
corr = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool).T)
corr.round(2).to_csv('Output/all_factors_corr.csv')
p_values = calculate_pvalues(all_factors_ret)
p_values = p_values.where(np.triu(np.ones(p_values.shape)).astype(np.bool).T)
p_values.round(2).to_csv('Output/all_factors_p_val.csv')



# #### correlation clustering
cluster_factors_ret = fdata.get_factors(['SMB', 'RMW', 'CMA', 'MOM', 'BaB', 'QMJ', 'HML_Devil', 'UMD'],
                                    start_date, end_date)
pdist = spc.distance.pdist(cluster_factors_ret.T, metric='correlation')
linkage = spc.linkage(pdist, method='complete')
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')
spc.dendrogram(linkage, labels=cluster_factors_ret.columns)
plt.title('Dendrogram of factor clusters (correlation as distance metric)')
plt.savefig('Plots/cluster_corr.pdf')
plt.show()
plt.close('all')

#save returns for latex
cluster_factors_ret.round(4).to_csv('Output/cluster_fact_returns.csv')

#distance matrix
D_mat = pd.DataFrame(spc.distance.squareform(pdist), index=cluster_factors_ret.columns,
                     columns=cluster_factors_ret.columns)
D_mat = D_mat.where(np.triu(np.ones(D_mat.shape)).astype(np.bool).T).round(2)
D_mat.to_csv('Output/Distance_matrix.csv')


# ## absolute correlation clustering insensitive
pdist_abs = 1 - abs(-pdist + 1)
linkage_abs = spc.linkage(pdist_abs, method='complete')
idx_abs = spc.fcluster(linkage_abs, 0.5 * pdist_abs.max(), 'distance')
spc.dendrogram(linkage_abs, labels=cluster_factors_ret.columns)
plt.title('Dendrograml of factor clusters (absolute correlation as distance metric)')
plt.savefig('Plots/cluster_abs_corr.pdf')
plt.show()
plt.close('all')