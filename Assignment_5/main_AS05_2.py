#%% IMPORTS
import pandas as pd
import os
import functions_2 as fn
import numpy as np

repo_root = os.path.dirname(os.path.abspath(__file__))

#%% IMPORT DATA

# Load timeseries
data_path = os.path.join(repo_root, "46400_AS05_Loads_timeseries")
timeseries_dict = fn.load_csv_files_to_dict(data_path)

#%% Q5.2.0

# Function created, it is written in functions_2.py called compute_del

#f_s_Q520 = 1 # Hz

#del_test = fn.compute_del(timeseries_dict['201701032150']['MxA1'], f_s_Q520, 3)

#%% Q5.2.1

# Sampling frequency for DEL calculation
f_s_Q521 = 1 # Hz

# Wohler exponents
m_list = [3, 6, 9, 12]

# Signals/loads timeseries for each timeseries
s_litst = ['MxA1', 'MyA1', 'MxR', 'MyR', 'MzR', 'MTilt', 'MYaw', 'MzTT', 'MyTB', 'MxTB']

# Timeseries: 21/03/2017; 14:30
ts = '201703211430' 

results_DEL = {}

for m in m_list:
    for s in s_litst:
        # Compute DEL
        del_value = fn.compute_del(timeseries_dict[ts][s], f_s_Q521, m)
        results_DEL[f"{s}_m{m}"] = del_value

#%% Q5.2.2 
