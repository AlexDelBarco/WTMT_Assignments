#%%## Imports
print('Initializing as05.py')
# Importing libraries
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
# import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functions as fn
import os  # Add missing os import
# from sklearn.linear_model import LinearRegression
print('Imports done')

#%% ## Import Data 

current_file_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_file_dir, '46400_AS05_Loads_stats_Spring2025.csv')

# Method 1: Skip the units row
df_loads = pd.read_csv(csv_path, sep=';', skiprows=[1])

# Method 2: Keep both header and units rows separately
# Read the file
data = pd.read_csv(csv_path, sep=';')

# Extract header and units
headers = data.columns.tolist()
units = data.iloc[0].tolist()

# Create a dictionary mapping columns to their units
column_units = dict(zip(headers, units))

# Now create the actual dataframe (skipping the units row)
df_loads = data.iloc[1:].reset_index(drop=True)

# Convert timestamp column to datetime
df_loads['datetime'] = pd.to_datetime(df_loads['rname'], format='%Y%m%d%H%M')

# Convert string values to float for all numeric columns
numeric_columns = df_loads.columns.drop(['rname', 'datetime'])
for col in numeric_columns:
    df_loads[col] = pd.to_numeric(df_loads[col], errors='coerce')

# Verify conversion worked
print(f"ROT dtype: {df_loads['ROT'].dtype}")
print(f"Wsp_44m dtype: {df_loads['Wsp_44m'].dtype}")

df_loads_original = df_loads.copy()

print('Load Data loaded')

# Load time series data
folder_name = '46400_AS05_Loads_timeseries'
# Load every csv file in the folder into a dictionary
data_path = os.path.join(current_file_dir, folder_name)
print(f'Loading data from {data_path}')
timeseries_dict = {}

for filename in os.listdir(data_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_path, filename)
        # Add sep=';' to correctly parse the CSV with semicolons
        df = pd.read_csv(file_path, sep=';', parse_dates=True, index_col=0)
        timeseries_dict[filename] = df
print('Time series data loaded')

# Fix the renaming process by creating a list of keys first
filenames = list(timeseries_dict.keys())
# Now iterate through the static list instead of the changing dictionary
for i, filename in enumerate(filenames):
    timeseries_dict[f'{i}'] = timeseries_dict.pop(filename)
print('Time series data renamed')
#%% ## Data Inspection



# Visualize time series data
# just the first one as example
# print('timeseries_dict')
# print(timeseries_dict.keys())
# print(timeseries_dict['1'].head())
# print(timeseries_dict['1'].describe())
# print(timeseries_dict['1'].info())



print('###################')

# %% QUESTION 1: LOAD DATA INSPECTION AND FILTERING

# 1
# Summary statistics
# print(df_loads.describe())

print('filtering ')
# print(df_loads.columns)
# Wind speed
df_loads = fn.filter_outliers_row_based(df_loads, ['Wsp_44m'], 
                                        lower_bound=4.0, 
                                        upper_bound=35.0, 
                                        parameter='Wind Speed', 
                                        unit='m/s', show_plot=False)

# Wind direction
df_loads = fn.filter_outliers_row_based(df_loads, ['Wdir_41m'],
                                        lower_bound=0, 
                                        upper_bound=360.0, 
                                        parameter='Wind Direction',
                                        unit='degrees', show_plot=False)

# Yaw angle
df_loads = fn.filter_outliers_row_based(df_loads, ['yaw'],
                                        lower_bound=0, 
                                        upper_bound=360.0, 
                                        parameter='Yaw',
                                        unit='degrees', show_plot=False)
# Pitch angle
df_loads = fn.filter_outliers_row_based(df_loads, ['pa'],
                                        lower_bound=-360.0, 
                                        upper_bound=360.0, 
                                        parameter='Pitch Angle',
                                        unit='degrees', show_plot=False)
# Mean wind speed nacelle measured
df_loads = fn.filter_outliers_row_based(df_loads, ['wsn'],
                                        lower_bound=4, 
                                        upper_bound=35, 
                                        parameter='Mean Wind Speed Nacelle Measured',
                                        unit='[m/s]', show_plot=False)

# %%

# 2. For the power production MLC, the mode of operation of the wind turbine is 'running and 
#    connected to the grid'. For this specific case, we assumed a filter condition for the rotor speed 
#    (min(ROT)>16.0 rpm) order to ensure the turbine is grid connected. This has already been 
#    applied to the raw dataset. Assess operational signals from the turbine and ensure normal 
#    operation in your filtered dataset.  

# Rotor speed
df_loads = fn.filter_outliers_row_based(df_loads, ['ROT'],
                                        lower_bound=0, 
                                        upper_bound=27.0, 
                                        parameter='Rotational Speed',
                                        unit='rpm', show_plot=False)


# Mean active power
df_loads = fn.filter_outliers_row_based(df_loads, ['po'],
                                        lower_bound=0,
                                        upper_bound=1e8, 
                                        parameter='Mean Active Power',
                                        unit='[kW]', show_plot=False)

fn.plot_scatter('Power vs Rotor Speed', df_loads['ROT'],
                df_loads['po'], 'Active Power', 
                'Rotor Speed [rpm]', 
                'Mean Active Power [kW]', False,
                 )

# make a figure with 2 plots: active power vs wind speed and active power vs rotor speed
# ...existing code...
# make a figure with 2 plots: active power vs wind speed and rotor speed vs wind speed
fn.power_plot(df_loads, 'Power_vs_WindSpeed_and_RotorSpeed', True)

# 3. The wind vane data between 2017-03-30 11:10 & 2017-08-08 13:50, is found to be invalid. 
#    Explain why the data from this period needs to be filtered out. 

# 4. As valid measurement sector, consider the reduced [260°;320°) sector. 

# 5. For this part of the exercise, assume that loads signals' are calibrated correctly.   

# 6. State if there is any other necessary filters.  

# %% QUESTION 2: DATA BINNING

# 7. Bin your filtered database into wind speed bins of 1 m/s width (0-1 m/s, 1-2 m/s, 2-3 m/s, …). 

# %% QUESTION 3: LOAD VISUALIZATION

# 8. For each structural load channel (marked with yellow in Table 5.1-1), create a 2x1 subplot 
#    figure.  
#    a. In the top graph, plot as function of Wsp_44m, the load channel's  
#       i. 10-min maximum values, and its bin averages 
#       ii. 10-min mean, and its bin averages 
#       iii. 10-min minimum, and its bin averages 
#    b. In the bottom graph, plot the 10-min standard deviation values and its bin averages. 
#    In total, ten 2x1 subplot figures are expected.