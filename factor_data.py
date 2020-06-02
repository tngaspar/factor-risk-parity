import pandas as pd

# BaB,SMB,HML,HML_Devil,UMD,QMJ,CMA,RMW,MOM,Mkt-RF
# currently data is in percentages

def get_factors(factor_list, start_date=None, end_date=None,):
    all_factors = pd.read_csv(r'Data\Factors\all_factors.csv', index_col=0).ffill()
    all_factors.index = pd.to_datetime(all_factors.index)

    if factor_list == ['all']:
        if start_date is None and end_date is None:
            r = all_factors
        else:
            r = all_factors.loc[start_date:end_date, :]
    elif not all(elem in all_factors.columns.values for elem in factor_list):
        print('Factor not found')
        return
    else:
        if start_date is None and end_date is None:
            r = all_factors.loc[:, factor_list]
        else:
            r = all_factors.loc[start_date:end_date, factor_list]

    return r*0.01
