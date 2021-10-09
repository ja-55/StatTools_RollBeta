# Imports
from yahoo_fin.stock_info import *
import pandas as pd
import numpy as np
import datetime as dt

# Assign name for ticker of interest and index for comparison
idx = 'SP500'
tckr = 'UL'

# Tool parameters
first_date = dt.date(2000,1,4)
last_date = dt.date(2021,9,26)
td = 253
wdws_yr = [0.083, 0.25, 1, 2, 3, 5, 10]
sb_wdw_dy = [int(i * td) for i in wdws_yr]
db_wdw_dy = [int(30 * (253/365)), int(90 * (253/365)), 253]

# Get data for index and format into digestible dataframe
sp500 = get_data('^gspc', start_date = first_date, end_date = last_date)
rtn_sp = sp500['adjclose'].pct_change().dropna()
rtn_sp = pd.DataFrame(rtn_sp)
rtn_sp = rtn_sp.loc[first_date:,:]
rtn_sp.columns = [idx]

# Get data for ticker and format into digestible dataframe
eqt = get_data(tckr, start_date = first_date, end_date = last_date)
rtn_eqt = eqt['adjclose'].pct_change().dropna()
rtn_eqt = pd.DataFrame(rtn_eqt)
rtn_eqt = rtn_eqt.loc[first_date:,:]
rtn_eqt.columns = [tckr]

# Calculate static betas
for i in sb_wdw_dy:
    print(i, "\t", np.cov(rtn_eqt.values[0:-i], rtn_sp.values[0:-i], rowvar = 0)[0,1] / np.var(rtn_sp.values[0:-i]))

# Rolling 30 period betas
full_df = pd.merge(rtn_eqt,rtn_sp, left_index=True, right_index=True)

for i in db_wdw_dy:

    full_df['roll_covar' + str(i)] = full_df.rolling(i).cov().unstack()[tckr][idx]
    full_df['roll_mvar' + str(i)] = full_df[idx].rolling(i).var()
    full_df['roll_beta' + str(i)] = full_df['roll_covar' + str(i)] / full_df['roll_mvar' + str(i)]
    
# Median annual static betas (1 month / 3 months)
full_df.groupby(lambda x: x.year).median().loc[2010:,['roll_beta20','roll_beta62']]

# Graph rolling betas (1 month / 3 month / 1 year)
full_df.loc[dt.date(2015,1,1):,['roll_beta20','roll_beta62','roll_beta253']].plot.line(figsize = (15,10));
