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
            del_value, signal = fn.compute_del(timeseries_dict[tms][s], f_s_Q521, m)
            results_DEL[tms][f"{s}_m{m}"] = del_value

#%% Q5.2.2 

# Importing cleaned statistical data from Q1

from as05 import df_loads

load_stats = df_loads.copy()
load_stats = load_stats.rename(columns={'rname': 'time'})

# Load Statics of timeseries' timestamps

#timestamps = []

#for tms in timeseries:

    #timestamp = pd.to_datetime(tms, format='%Y%m%d%H%M')
    #timestamps.append(timestamp)

load_stats_tms = load_stats[load_stats['time'].isin(timeseries)]

# Replace new computed values of DEL 

load_stats_tms_updated = fn.replace_load_stats_with_results(load_stats_tms, results_DEL)
load_stats_updated = load_stats.copy()
load_stats_updated.update(load_stats_tms_updated)

#%% Q5.2.3

#a)

#fn.plot_sig_scatter(load_stats_updated, 'Wsp_44m', ['MxA1_DEL12', 'MyA1_DEL12'], 
           #title='MxA1 and MyA1  DEL with Wohler exponent 12 as a function of Wind Speed at 44m', show_plot=False, x_label='Wind Spped at 44m [m/s]', y_label='DEL [kNm]')
#%%
#b)
#Blade root
#fn.plot_sig_scatter(load_stats_updated, 'Wsp_44m', ['MxA1_DEL3', 'MyA1_DEL3'], 
           #title='DEL of MxA1 and MyA1, m=3 vs WS at 44m', show_plot=False, x_label='Wind Spped at 44m [m/s]', y_label='DEL [kNm]')

#Bending shaft moment and shat torque
#fn.plot_sig_scatter(load_stats_updated, 'Wsp_44m', ['MzR_DEL3', 'MyR_DEL3', 'MxR_DEL3'], 
           #title='DEL of MxR, MyR and MzR, m=3 vs WS at 44m', show_plot=False, x_label='Wind Spped at 44m [m/s]', y_label='DEL [kNm]')

#Derived tilt moment
#fn.plot_sig_scatter(load_stats_updated, 'Wsp_44m', ['Myaw_DEL3', 'Mtilt_DEL3'], 
           #title='DEL of Mtilts and Myaw, m=3 vs WS at 44m', show_plot=False, x_label='Wind Spped at 44m [m/s]', y_label='DEL [kNm]')

#Tower top and bottom torsion
#fn.plot_sig_scatter(load_stats_updated, 'Wsp_44m', ['MyTB_DEL3', 'MxTB_DEL3', 'MzTT_DEL3'], 
           #title='DEL of MzTT, MxTB and MyTB, m=3 vs WS at 44m', show_plot=False, x_label='Wind Spped at 44m [m/s]', y_label='DEL [kNm]')

#%% c) Inspect 201703011200 at t=270 for MxTB Check  how  other  load  signals, 
#  operational  or  environmental conditions are changing around that time

# MxTB
fn.plot_sig_scatter(timeseries_dict['201703011200'], 'time', 'MxTB', title='MxTB signal over time at 2017-03-01 12:00 timeseries'
                    , show_plot=False, x_label='Time [s]', y_label='MxTB [kNm]')

#Bending shaft moment and shat torque
fn.plot_sig_scatter(timeseries_dict['201703011200'], 'Wsp_44m', ['MzR_DEL3', 'MyR_DEL3', 'MxR_DEL3'], 
           title='DEL of MxR, MyR and MzR, m=3 vs WS at 44m', show_plot=False, x_label='Wind Spped at 44m [m/s]', y_label='DEL [kNm]')





# %%
