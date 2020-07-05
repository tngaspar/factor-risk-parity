import matplotlib.pyplot as plt
import factor_data as fdata
import datetime as dt
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as spc

start_date = dt.date(2004, 1, 1)
end_date = dt.date(2020, 12, 31)
all_factors_ret = fdata.get_factors(['all'],
                                    start_date, end_date)

all_mcsi_factors = all_factors_ret.iloc[:,-7:]
te = all_mcsi_factors.subtract(all_factors_ret['Mkt-RF'], axis=0)

plt.plot((1 + all_factors_ret).cumprod())
plt.plot((1 + te).cumprod())
plt.show()
plt.close('all')


new_all_factors = pd.concat([all_factors_ret.drop(all_mcsi_factors.columns, axis=1), te], axis=1)

factors_corr = new_all_factors.corr()

sns.heatmap(factors_corr)

plt.plot((1 + new_all_factors).cumprod())
plt.show()
plt.close('all')


pdist = spc.distance.pdist(new_all_factors.T, metric='correlation')
pdist2  = 1 - abs(-pdist + 1)
linkage = spc.linkage(pdist2, method='single')
idx = spc.fcluster(linkage, 0.5 * pdist2.max(), 'distance')
spc.dendrogram(linkage, labels=new_all_factors.columns)
plt.title('Dendrogram of factor clusters (correlation as distance metric)')
#plt.savefig('Ploting/cluster_corr.pdf')
plt.xticks(rotation='vertical')
plt.show()
plt.close('all')
