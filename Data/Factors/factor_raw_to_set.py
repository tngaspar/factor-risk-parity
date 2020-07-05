import pandas as pd

# current factors: BaB, SMB. HML, MHL_Devil, UMD, QMJ, CMA, RMW, MOM

# USA only
BaB = pd.read_csv(r'Raw_data\BaB_all_countries.csv', index_col=0, usecols=['DATE', 'USA'])\
    .rename(columns={'USA': 'BaB'})
BaB.index = pd.to_datetime(BaB.index)

SMB = pd.read_csv(r'Raw_data\SMB_all_countries.csv', index_col=0, usecols=['DATE', 'USA'])\
    .rename(columns={'USA': 'SMB'})
SMB.index = pd.to_datetime(SMB.index)

HML = pd.read_csv(r'Raw_data\HML_all_countries.csv', index_col=0, usecols=['DATE', 'USA'])\
    .rename(columns={'USA': 'HML'})
HML.index = pd.to_datetime(HML.index)

HML_Devil = pd.read_csv(r'Raw_data\HML_Devil_all_countries.csv', index_col=0, usecols=['DATE', 'USA'])\
    .rename(columns={'USA': 'HML_Devil'})
HML_Devil.index = pd.to_datetime(HML_Devil.index)

UMD = pd.read_csv(r'Raw_data\UMD_all_countries.csv', index_col=0, usecols=['DATE', 'USA'])\
    .rename(columns={'USA': 'UMD'})
UMD.index = pd.to_datetime(UMD.index)

QMJ = pd.read_csv(r'Raw_data\QMJ_all_countries.csv', index_col=0, usecols=['DATE', 'USA'])\
    .rename(columns={'USA': 'QMJ'})
QMJ.index = pd.to_datetime(QMJ.index)

CMA = pd.read_csv(r'Raw_data\F-F_Research_Data_5_Factors_2x3_daily.CSV', index_col=0, usecols=['DATE', 'CMA'])
CMA.index = pd.to_datetime(CMA.index, format='%Y%m%d')

RMW = pd.read_csv(r'Raw_data\F-F_Research_Data_5_Factors_2x3_daily.CSV', index_col=0, usecols=['DATE', 'RMW'])
RMW.index = pd.to_datetime(RMW.index, format='%Y%m%d')

MOM = pd.read_csv(r'Raw_data\F-F_Momentum_Factor_daily.CSV', index_col=0, usecols=['DATE', 'MOM'])
MOM.index = pd.to_datetime(MOM.index, format='%Y%m%d')

Mkt_RF = pd.read_csv(r'Raw_data\F-F_Research_Data_5_Factors_2x3_daily.CSV', index_col=0, usecols=['DATE', 'Mkt-RF'])
Mkt_RF.index = pd.to_datetime(Mkt_RF.index, format='%Y%m%d')

M2USEV = pd.read_csv(r'Raw_data\US_msci_tickers.csv',index_col=0, usecols=['DATE', 'M2USEV']).pct_change()*100
M2USEV.index = pd.to_datetime(M2USEV.index, format='%d/%m/%Y')

M2USEW = pd.read_csv(r'Raw_data\US_msci_tickers.csv',index_col=0, usecols=['DATE', 'M2USEW']).pct_change()*100
M2USEW.index = pd.to_datetime(M2USEW.index, format='%d/%m/%Y')

M2US000 = pd.read_csv(r'Raw_data\US_msci_tickers.csv',index_col=0, usecols=['DATE', 'M2US000']).pct_change()*100
M2US000.index = pd.to_datetime(M2US000.index, format='%d/%m/%Y')

M2USVOE = pd.read_csv(r'Raw_data\US_msci_tickers.csv',index_col=0, usecols=['DATE', 'M2USVOE']).pct_change()*100
M2USVOE.index = pd.to_datetime(M2USVOE.index, format='%d/%m/%Y')

M5USIDY = pd.read_csv(r'Raw_data\US_msci_tickers.csv',index_col=0, usecols=['DATE', 'M5USIDY']).pct_change()*100
M5USIDY.index = pd.to_datetime(M5USIDY.index, format='%d/%m/%Y')

M2NAUSQL = pd.read_csv(r'Raw_data\US_msci_tickers.csv',index_col=0, usecols=['DATE', 'M2NAUSQL']).pct_change()*100
M2NAUSQL.index = pd.to_datetime(M2NAUSQL.index, format='%d/%m/%Y')

M05JUS0 = pd.read_csv(r'Raw_data\US_msci_tickers.csv',index_col=0, usecols=['DATE', 'M05JUS0']).pct_change()*100
M05JUS0.index = pd.to_datetime(M05JUS0.index, format='%d/%m/%Y')


all_factors = pd.concat([BaB, SMB, HML, HML_Devil, UMD, QMJ, CMA, RMW, MOM, Mkt_RF, M2USEV, M2USEW, M2US000, M2USVOE,
                         M5USIDY, M2NAUSQL, M05JUS0], axis=1)
all_factors.to_csv('all_factors.csv')

