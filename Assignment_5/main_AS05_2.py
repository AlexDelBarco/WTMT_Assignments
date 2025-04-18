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

# Timeseries
timseries = list(timeseries_dict.keys())

# Computing DEL for each timeseries, each signal and each m

results_DEL = {}

for tms in timseries:

    # For each timeseries
    results_DEL[tms] = {}

    for m in m_list:

        for s in s_litst:

            # Compute DEL for each signal and m
            del_value = fn.compute_del(timeseries_dict[tms][s], f_s_Q521, m)
            results_DEL[tms][f"{s}_m{m}"] = del_value

#%% Q5.2.2 

# Importing statistical data

load_stats = fn.load_csv_with_units('46400_AS05_Loads_stats_Spring2025.csv')
load_stats.rename(columns={"rname_[-]": "time"}, inplace=True)
load_stats['time'] = pd.to_datetime(load_stats['time'], format='%Y%m%d%H%M')

# Timeseries 201702230240, date: 2017-02-23 02:40:00

tms_date_1 = pd.Timestamp('2017-02-23 02:40:00')

load_stats_tms1 = load_stats[load_stats['time'] == tms_date_1]
