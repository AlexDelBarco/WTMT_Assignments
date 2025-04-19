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


#%% Q5.2.1

# Sampling frequency for DEL calculation
f_s_Q521 = 1 # Hz

# Wohler exponents
m_list = [3, 6, 9, 12]

# Signals/loads timeseries for each timeseries
s_litst = ['MxA1', 'MyA1', 'MxR', 'MyR', 'MzR', 'MTilt', 'MYaw', 'MzTT', 'MyTB', 'MxTB']

# Timeseries
timeseries = list(timeseries_dict.keys())

# Computing DEL for each timeseries, each signal and each m

results_DEL = {}

for tms in timeseries:

    # For each timeseries
    results_DEL[tms] = {}

    for m in m_list:

        for s in s_litst:

            # Compute DEL for each signal and m
            del_value = fn.compute_del(timeseries_dict[tms][s], f_s_Q521, m)
            results_DEL[tms][f"{s}_m{m}"] = del_value

#%% Q5.2.2 

# Importing statistical data

#USE THE CLEAN LOAD_STATE FROM Q1

load_stats = fn.load_csv_with_units('46400_AS05_Loads_stats_Spring2025.csv')
load_stats.rename(columns={"rname_[-]": "time"}, inplace=True)
load_stats['time'] = pd.to_datetime(load_stats['time'], format='%Y%m%d%H%M')

# Load Statics of timeseries' timestamps

timestamps = []

for tms in timeseries:

    timestamp = pd.to_datetime(tms, format='%Y%m%d%H%M')
    timestamps.append(timestamp)

load_stats_tms = load_stats[load_stats['time'].isin(timestamps)]

# Replace new computed values of DEL 

load_stats_tms_updated = fn.replace_load_stats_with_results(load_stats_tms, results_DEL)
load_stats_updated = load_stats.copy()
load_stats_updated.update(load_stats_tms_updated)

#%% Q5.2.3

#a)

fn.plot_sig(load_stats_updated, 'Wsp_44m_[m/s]', ['MxA1_DEL12_[kNm]', 'MyA1_DEL12_[kNm]'], 
           title='MxA1 DEL for Wohler exponent 3 as a funtion of Wind speed at 44m', show_plot=False, x_label='Wind Spped at 44m [m/s]', y_label='DEL [kNm]')

#b)
fn.plot_sig(load_stats_updated, 'Wsp_44m_[m/s]', ['MyTB_DEL3_[kNm]', 'MxTB_DEL3_[kNm]', 'MzTT_DEL3_[kNm]', 'Myaw_DEL3_[kNm]', 'Mtilt_DEL3_[kNm]', 'MzR_DEL3_[kNm]', 'MyR_DEL3_[kNm]', 'MxR_DEL3_[kNm]', 'MyA1_DEL3_[kNm]', 'MxA1_DEL3_[kNm]'], 
           title='Signals DEL values for Wohler exponent 3 as a funtion of Wind speed at 44m', show_plot=False, x_label='Wind Spped at 44m [m/s]', y_label='DEL [kNm]')



