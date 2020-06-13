import pandas as pd
import factor_data as fdata
import datetime as dt

start_date = dt.date(1990, 1, 1)
end_date = dt.date(2004, 12, 31)
all_factors_ret = fdata.get_factors(['BaB', 'SMB', 'HML_Devil', 'UMD', 'QMJ', 'CMA', 'RMW', 'MOM', 'Mkt-RF'],
                                    start_date, end_date)

all_factors_cum_ret = (all_factors_ret + 1).cumprod()
plt.plot(all_factors_cum_ret)
plt.show()
plt.close('all')



f1 = all_factors_ret['CMA']*0.25 + all_factors_ret['BaB']*0.25 + all_factors_ret['HML_Devil']*0.5
f2 = all_factors_ret['RMW']*0.5 + all_factors_ret['QMJ']*0.5
f3 = all_factors_ret['UMD']*0.5 + all_factors_ret['MOM']*0.5
f4 = all_factors_ret['SMB']
f5 = all_factors_ret['Mkt-RF']