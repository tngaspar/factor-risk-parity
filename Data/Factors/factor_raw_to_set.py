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

all_factors = pd.concat([BaB, SMB, HML, HML_Devil, UMD, QMJ, CMA, RMW, MOM], axis=1)
all_factors.to_csv('all_factors.csv')
