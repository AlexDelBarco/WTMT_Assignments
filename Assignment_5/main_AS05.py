#%% IMPORTS
import pandas as pd
import os
import functions as fn
import numpy as np

repo_root = os.path.dirname(os.path.abspath(__file__))

#%% IMPORT DATA

# Load stats timeseries
load_stats = fn.load_csv_with_units('46400_AS05_Loads_stats_Spring2025.csv')
load_stats.rename(columns={"rname_[-]": "time"}, inplace=True)
load_stats['time'] = pd.to_datetime(load_stats['time'], format='%Y%m%d%H%M')

# Load timeseries
data_path = os.path.join(repo_root, "46400_AS05_Loads_timeseries")
timeseries_dict = fn.load_csv_files_to_dict(data_path)

# %% QUESTION 1: LOAD DATA INSPECTION AND FILTERING

# 1
# Summary statistics
print(load_stats.describe())

# Invalid values on Wsp_44m found, converted to NaN
if "Wsp_44m_[m/s]" in load_stats.columns:
    load_stats["Wsp_44m_[m/s]"] = load_stats["Wsp_44m_[m/s]"].apply(lambda x: np.nan if x > 100 else x)

fn.plot_load_stats(load_stats, "Wsp_44m_[m/s]")
#fn.plot_load_stats(load_stats, "dWsp_44m [m/s]")


# 2
print(f'Max ROT {load_stats["ROT_[rpm]"].max()}')

# 3 

# %%
