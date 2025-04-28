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
import functions_theo as fn
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
# for col in df_loads.columns:
#     print(f"Column '{col}' - Type: {df_loads[col].dtype}")

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

CUT_IN = 3.0  # m/s
CUT_OUT = 25.0  # m/s
PITCH_MIN = -2.0  # degrees
PITCH_MAX = 25.0  # degrees
RATED_POWER = 850.0  # kW
RATED_ROTOR_SPEED = 26.0  # rpm

# print(df_loads.columns)
# Wind speed
df_loads = fn.filter_outliers_row_based(df_loads, ['Wsp_44m'], 
                                        lower_bound=CUT_IN, 
                                        upper_bound=CUT_OUT, 
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
                                        lower_bound=PITCH_MIN, 
                                        upper_bound=PITCH_MAX, 
                                        parameter='Pitch Angle',
                                        unit='degrees', show_plot=False)
# Mean wind speed nacelle measured
df_loads = fn.filter_outliers_row_based(df_loads, ['wsn'],
                                        lower_bound=CUT_IN, 
                                        upper_bound=CUT_OUT, 
                                        parameter='Mean Wind Speed Nacelle Measured',
                                        unit='[m/s]', show_plot=False)

# %% 2 Assess operational signals from the turbine and ensure normal 
#    operation 

#  For the power production MLC, the mode of operation of the wind turbine is 'running and 
#    connected to the grid'. For this specific case, we assumed a filter condition for the rotor speed 
#    (min(ROT)>16.0 rpm) order to ensure the turbine is grid connected. This has already been 
#    applied to the raw dataset. Assess operational signals from the turbine and ensure normal 
#    operation in your filtered dataset.  

# Rotor speed
df_loads = fn.filter_outliers_row_based(df_loads, ['ROT'],
                                        lower_bound=0, 
                                        upper_bound=RATED_ROTOR_SPEED, 
                                        parameter='Rotational Speed',
                                        unit='rpm', show_plot=False)


# Mean active power
df_loads = fn.filter_outliers_row_based(df_loads, ['po'],
                                        lower_bound=0,
                                        upper_bound=RATED_POWER, 
                                        parameter='Mean Active Power',
                                        unit='[kW]', show_plot=False)

fn.plot_scatter('Power vs Rotor Speed', df_loads['ROT'],
                df_loads['po'], 'Active Power', 
                'Rotor Speed [rpm]', 
                'Mean Active Power [kW]', False,
                 )

# make a figure with 2 plots: active power vs wind speed and active power vs rotor speed
fn.power_plot(df_loads, 'Power Curve', 'Wsp_44m',
              'po', 'Wind Speed [m/s]', 'Mean Active Power [kW]', 'Power vs Rotor Speed',
              'ROT', 'po', 'Rotor Speed [rpm]', 'Mean Active Power [kW]', False)

#Rotor speed and pitch angle vs wind speed 
fn.power_plot(df_loads, 'Rotor Speed vs Wind Speed', 'Wsp_44m',
              'ROT', 'Wind Speed [m/s]', 'Rotor Speed [rpm]', 'Pitch vs Wind Speed',
              'Wsp_44m', 'pa', 'Wind Speed [m/s]', 'Pitch angle [deg]', False)

# %% 3. The wind vane data between 2017-03-30 11:10 & 2017-08-08 13:50, is found to be invalid. 
#    Explain why the data from this period needs to be filtered out. 



# plot it from 2017-03-30 11:10 minus 20 days to 2017-08-08 13:50 plus 20 days
print("\n--- Analyzing Invalid Period with Buffer ---")
df_invalid = fn.analyze_wind_vane_period(df_loads, 20170330110, 201708081350, 40, False)

# Analyze additional valid periods for comparison
print("\n--- Analyzing Valid Period 1 (Jan-Mar 2017) ---")
# Note: Data starts 2017-01-01, so this is slightly less than 4 months
df_valid_1 = fn.analyze_wind_vane_period(df_loads, 201701010000, 201703292350, buffer_days=0, show_plot=False)

print("\n--- Analyzing Valid Period 2 (Aug-Dec 2017) ---")
df_valid_2 = fn.analyze_wind_vane_period(df_loads, 201708090000, 201712082350, buffer_days=0, show_plot=False)

print("\n--- Analyzing Valid Period 3 (Dec 2017 - Apr 2018) ---")
df_valid_3 = fn.analyze_wind_vane_period(df_loads, 201712090000, 201804082350, buffer_days=0, show_plot=False)

print("\n--- Analyzing Valid Period 4 (Apr-Jul 2018) ---")
# Note: Data ends 2018-07-31, so this covers the remaining data
df_valid_4 = fn.analyze_wind_vane_period(df_loads, 201804090000, 201807311850, buffer_days=0, show_plot=False)

fn.plot_scatter('Wind_Vane_vs_WS', df_loads['Wsp_44m'], df_loads['Wdir_41m']
                , 'Wind Direction', 'Wind Speed [m/s]', 'Wind Direction [deg]', False, dot_size1 = 50)



# ... after creating df_invalid, df_valid_1, df_valid_2, df_valid_3, df_valid_4 ...

print("\n--- Adding Circular Difference Column to Period DataFrames ---")

# Store the dataframes in a dictionary for easier processing
period_dfs = {
    "Invalid": df_invalid,
    "Valid 1": df_valid_1,
    "Valid 2": df_valid_2,
    "Valid 3": df_valid_3,
    "Valid 4": df_valid_4
}

# Calculate the difference column for each dataframe
for name, df_p in period_dfs.items():
    if not df_p.empty and len(df_p) >= 2:
        # Ensure sorted by time and work on a copy if modifying
        fn.plot_scatter(f'Wind_Vane_vs_WS_{name}', df_p['Wsp_44m'], df_p['Wdir_41m']
                , 'Wind Direction', 'Wind Speed [m/s]', 'Wind Direction [deg]', False, dot_size1 = 50)
        df_p = df_p.sort_values('datetime').copy()
        df_p['Wdir_diff_circ'] = df_p['Wdir_41m'].diff().apply(fn.shortest_angle_diff)
        period_dfs[name] = df_p # Update the dictionary with the modified dataframe
        print(f"Calculated 'Wdir_diff_circ' for {name}")
    elif df_p.empty:
        print(f"Skipping diff calculation for {name}: DataFrame is empty.")
    else:
        print(f"Skipping diff calculation for {name}: Not enough data (rows={len(df_p)}).")

# Now df_invalid, df_valid_1 etc. (accessed via period_dfs dictionary) have the column

# %% CIRCULAR STATISTICS FOR WIND VANE ANALYSIS
# ... (Your existing statistics code can remain here) ...

# %% PLOTTING CIRCULAR DIFFERENCE COMPARISONS

print("\n--- Generating Circular Difference Comparison Plots ---")

period_names = list(period_dfs.keys())

# --- Plot 1: Side-by-side Subplots ---
num_periods = len(period_names)
nrows = 2
ncols = 3
fig_subplots, axes_subplots = plt.subplots(nrows, ncols, figsize=(18, 10), sharey=True)
axes_flat = axes_subplots.flatten()

print("Generating side-by-side difference plots...")
for i, name in enumerate(period_names):
    df_sub_period = period_dfs[name] # Get the dataframe directly

    # Check if the dataframe is not empty and has the difference column
    if not df_sub_period.empty and 'Wdir_diff_circ' in df_sub_period.columns:
        # Get the difference values without NaNs
        diff_values = df_sub_period['Wdir_diff_circ'].dropna()
        # Get the corresponding datetime values using the index of the non-NaN diff values
        datetime_values = df_sub_period.loc[diff_values.index, 'datetime']

        # Plot aligned datetime and difference values
        axes_flat[i].scatter(datetime_values, diff_values,
                             alpha=0.5, s=3) # Adjust alpha and size (s) as needed
        axes_flat[i].set_title(f"Period: {name}")
        axes_flat[i].set_xlabel("Date")
        if i % ncols == 0: # Add Y label only to the first column
             axes_flat[i].set_ylabel("Abs. Circular Diff [deg]")
        axes_flat[i].grid(True, linestyle='--', alpha=0.3)
        # Optional: Set y-limit if needed, e.g., axes_flat[i].set_ylim(-5, 180)
    else:
        axes_flat[i].set_title(f"Period: {name}\n(No Data/Diff)")
        axes_flat[i].axis('off') # Hide axis if no data

# # Hide unused subplots if any
# for j in range(i + 1, nrows * ncols):
#     axes_flat[j].axis('off')

# fig_subplots.suptitle("Circular Wind Direction Difference per Period", fontsize=16)
# fig_subplots.autofmt_xdate()
# plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
# plt.show()

# # --- Plot 2: Combined Plot ---
# fig_combined, ax_combined = plt.subplots(1, 1, figsize=(15, 7))
# colors = plt.cm.viridis(np.linspace(0, 1, num_periods)) # Get distinct colors

# print("Generating combined difference plot...")
# for i, name in enumerate(period_names):
#     df_sub_period = period_dfs[name] # Get the dataframe directly

#     # Check if the dataframe is not empty and has the difference column
#     if not df_sub_period.empty and 'Wdir_diff_circ' in df_sub_period.columns:
#         # Using scatter for combined plot might be clearer
#         ax_combined.scatter(df_sub_period['datetime'], df_sub_period['Wdir_diff_circ'].dropna(), # Drop NaNs from diff
#                            label=name, color=colors[i], alpha=0.5, s=5)
#         # Or use lines (might be messy):
#         # ax_combined.plot(df_sub_period['datetime'], df_sub_period['Wdir_diff_circ'].dropna(),
#         #                  label=name, color=colors[i], alpha=0.7, linewidth=0.8)

# ax_combined.set_title("Combined Circular Wind Direction Difference")
# ax_combined.set_xlabel("Date")
# ax_combined.set_ylabel("Abs. Circular Diff [deg]")
# ax_combined.grid(True, linestyle='.', alpha=0.3)
# ax_combined.legend(markerscale=3) # Increase legend marker size if using scatter
# fig_combined.autofmt_xdate()
# plt.tight_layout()
# plt.show()

# print("Difference comparison plots generated.")

# # ... rest of your code ...


# # %% CIRCULAR STATISTICS FOR WIND VANE ANALYSIS
# from scipy.stats import circstd, circmean
# import numpy as np

# print("\n--- Calculating Circular Statistics for Wind Direction ---")

# # Define the periods (use the same dates as in the visual analysis)
# # ... existing code ...
# periods = {
#     "Invalid": ('201703301110', '201708081350'),
#     "Valid 1 (Jan-Mar 2017)": ('201701010000', '201703292350'),
#     "Valid 2 (Aug-Dec 2017)": ('201708090000', '201712082350'),
#     "Valid 3 (Dec 2017-Apr 2018)": ('201712090000', '201804082350'),
#     "Valid 4 (Apr-Jul 2018)": ('201804090000', '201807311850'),
#     # Convert 'rname' to float before finding min/max
#     "Full Period": (df_loads['rname'].astype(float).min(), df_loads['rname'].astype(float).max())
# }

# results = []
# # ... rest of the code ...

# results = []

# for name, (start_str, end_str) in periods.items():
#     # Convert dates to numeric format for filtering
#     start_num = float(str(start_str).ljust(12, '0'))
#     end_num = float(str(end_str).ljust(12, '0'))

#     # Filter data for the period
#     mask = (df_loads['rname'].astype(float) >= start_num) & \
#            (df_loads['rname'].astype(float) <= end_num)
#     df_period = df_loads[mask]

#     if df_period.empty:
#         print(f"Warning: No data found for period '{name}'. Skipping.")
#         results.append({'Period': name, 'Mean (deg)': np.nan, 'Std Dev (deg)': np.nan, 'Count': 0})
#         continue

#     # Get wind direction data and drop NaNs
#     wind_dir_deg = df_period['Wdir_41m'].dropna()
#     count = len(wind_dir_deg)

#     if count < 2: # Need at least 2 points for std dev
#          print(f"Warning: Not enough data points ({count}) for period '{name}'. Skipping std dev.")
#          results.append({'Period': name, 'Mean (deg)': np.nan, 'Std Dev (deg)': np.nan, 'Count': count})
#          continue

#     # Convert degrees to radians
#     wind_dir_rad = np.deg2rad(wind_dir_deg)

#     # Calculate circular mean (in radians) and convert back to degrees [0, 360)
#     mean_rad = circmean(wind_dir_rad, high=2*np.pi, low=0)
#     mean_deg = np.rad2deg(mean_rad)
#     if mean_deg < 0: # Ensure mean is in [0, 360) range
#         mean_deg += 360

#     # Calculate circular standard deviation (in radians) and convert back to degrees
#     std_rad = circstd(wind_dir_rad, high=2*np.pi, low=0)
#     std_deg = np.rad2deg(std_rad)

#     results.append({'Period': name, 'Mean (deg)': mean_deg, 'Std Dev (deg)': std_deg, 'Count': count})

# # Print results in a table format
# print("\nCircular Statistics Results:")
# print("-" * 60)
# print(f"{'Period':<25} | {'Mean (deg)':>10} | {'Std Dev (deg)':>12} | {'Count':>6}")
# print("-" * 60)
# for res in results:
#     print(f"{res['Period']:<25} | {res['Mean (deg)']:>10.2f} | {res['Std Dev (deg)']:>12.2f} | {res['Count']:>6}")
# print("-" * 60)

# # ... rest of your code ...
# # ... previous circular statistics code ...

# print("\n--- Calculating Mean Absolute Circular Difference for Wind Direction ---")

# diff_results = []

# for name, (start_str, end_str) in periods.items():
#     # Convert dates to numeric format for filtering
#     start_num = float(str(start_str).ljust(12, '0'))
#     end_num = float(str(end_str).ljust(12, '0'))

#     # Filter data for the period
#     mask = (df_loads['rname'].astype(float) >= start_num) & \
#            (df_loads['rname'].astype(float) <= end_num)
#     df_period = df_loads[mask]

#     if df_period.empty or len(df_period) < 2:
#         print(f"Warning: Not enough data for period '{name}' to calculate differences. Skipping.")
#         diff_results.append({'Period': name, 'Mean Abs Diff (deg)': np.nan, 'Count': len(df_period)})
#         continue

#     # Get wind direction data and drop NaNs
#     wind_dir_deg = df_period['Wdir_41m'].dropna()
#     count = len(wind_dir_deg)

#     if count < 2:
#         print(f"Warning: Not enough non-NaN data points ({count}) for period '{name}' to calculate differences. Skipping.")
#         diff_results.append({'Period': name, 'Mean Abs Diff (deg)': np.nan, 'Count': count})
#         continue

#     # Calculate consecutive differences
#     differences = wind_dir_deg.diff().dropna() # Drop the first NaN

#     # Calculate the shortest angle difference (handling wrap-around)
#     # Apply the logic: min(abs(d), 360 - abs(d))
#     shortest_diff = differences.apply(lambda d: min(abs(d), 360 - abs(d)))

#     # Calculate the mean of the absolute shortest differences
#     mean_abs_shortest_diff = shortest_diff.mean()

#     diff_results.append({'Period': name, 'Mean Abs Diff (deg)': mean_abs_shortest_diff, 'Count': count})

# # Print results in a table format
# print("\nMean Absolute Circular Difference Results:")
# print("-" * 55)
# print(f"{'Period':<25} | {'Mean Abs Diff (deg)':>18} | {'Count':>6}")
# print("-" * 55)
# for res in diff_results:
#     print(f"{res['Period']:<25} | {res['Mean Abs Diff (deg)']:>18.4f} | {res['Count']:>6}")
# print("-" * 55)

# # ... rest of your code ...
# # ... other imports ...



# # ... rest of the functions ...

# %% 4. As valid measurement sector, consider the reduced [260°;320°) sector. 
# # Filer out all data outside this sector.
df_loads_full_sector = df_loads.copy()

df_loads = fn.filter_outliers_row_based(df_loads, ['Wdir_41m'],
                                        lower_bound=260.0, 
                                        upper_bound=320.0, 
                                        parameter='Wind Direction',
                                        unit='degrees', show_plot=False)


# 6. State if there is any other necessary filters.  

# %% QUESTION 2: DATA BINNING

# 7. Bin your filtered database into wind speed bins of 1 m/s width 


#  Calculate binned statistics
df_binned = fn.bin_data_by_windspeed(df_loads, ws_col='Wsp_44m', bin_width=1.0)

# Check the results
# if not df_binned_means.empty:
#     print("\nBinned Data (Means):")
#     print(df_binned_means.head())
#     print("\nColumns in binned data:")
#     print(df_binned_means.columns)


# Save results to CSV (optional)
df_binned.to_csv('binned_means_statistics.csv', float_format='%.3f', index=False)
print("\nSaved binned means to 'binned_means_statistics.csv'")
# else:
#     print("\nBinning failed or resulted in an empty DataFrame.")
print(df_binned.columns)

# Instead of creating bins based only on df_binned length:
# ws_bins = np.arange(4, 4 + len(df_binned))


# fn.plot_scatter('Binned Wind Speed vs Wind speed', df_binned['Wsp_44m_mean'], df_binned['ws_bin'],
#                 'Wind Speed', 'Binned Wind Speed [m/s]', 'Wind Speed [m/s]', True, dot_size1 = 50)


# %% QUESTION 3: LOAD VISUALIZATION

# 8. For each structural load channel (all channels except these:
#  ws_bin_intervals', 'count', 'Wsp_44m_mean', 'Wdir_41m_mean','ROT_mean', 
# 'yaw_mean', 'pa_mean', 'po_mean', 'wsn_mean', ), 
# create a 2x1 subplot figure.  
#    a. In the top graph, plot as function of Wsp_44m, the load channel's  
#       i. 10-min maximum values, and its bin averages 
#       ii. 10-min mean, and its bin averages 
#       iii. 10-min minimum, and its bin averages 
#    b. In the bottom graph, plot the 10-min standard deviation values and its bin averages. 
#    In total, ten 2x1 subplot figures are expected.

# ... previous code ...

# %% QUESTION 3: LOAD VISUALIZATION

# 8. For each structural load channel (all channels except these:
#  ws_bin_intervals', 'count', 'Wsp_44m_mean', 'Wdir_41m_mean','ROT_mean',
# 'yaw_mean', 'pa_mean', 'po_mean', 'wsn_mean', ),
# create a 2x1 subplot figure.
#    a. In the top graph, plot as function of Wsp_44m, the load channel's
#       i. 10-min maximum values, and its bin averages
#       ii. 10-min mean, and its bin averages
#       iii. 10-min minimum, and its bin averages
#    b. In the bottom graph, plot the 10-min standard deviation values and its bin averages.
#    In total, ten 2x1 subplot figures are expected.

print("\n--- Generating Load Visualization Plots (Q3) ---")

# Define columns that are NOT structural load channels (based on Q3 instructions)
# These include operational signals and columns added during binning
exclude_cols_binned = [
    'ws_bin_intervals', 'count', 'Wsp_44m_mean', # Binning & WS itself
    'Wdir_41m_mean', 'ROT_mean', 'yaw_mean', 'pa_mean', 'po_mean', 'wsn_mean', # Operational signals (means of)
    'ws_bin' # Manually added columns (adjust if names differ)
]

# Identify base load channel names automatically from df_binned columns
load_channel_bases = set()
stats_suffixes = ['_mean', '_stdev', '_min', '_max'] # Original statistics suffixes

print("Identifying load channels from df_binned columns...")
for binned_col in df_binned.columns:
    # Skip columns explicitly excluded
    if binned_col in exclude_cols_binned:
        continue

    # Check if the column name ends with '_mean' (indicating it's a binned average)
    if binned_col.endswith('_mean'):
        # Remove the trailing '_mean' to get the original statistic column name
        original_stat_col = binned_col[:-len('_mean')]

        # Check if this original stat column name ends with one of the expected suffixes
        for suffix in stats_suffixes:
            if original_stat_col.endswith(suffix):
                # Extract the base name by removing the stat suffix
                base_name = original_stat_col[:-len(suffix)]
                load_channel_bases.add(base_name)
                # print(f"  Found base: {base_name} from {binned_col}") # Debug print
                break # Move to the next binned_col once a base is found

# Convert set to sorted list for consistent order
plot_bases = sorted(list(load_channel_bases))
print(f"Identified load channel bases for plotting: {plot_bases}")
print(f"Total plots expected: {len(plot_bases)}")

# Create directory for saving plots if it doesn't exist
pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures', 'Load_Plots_Q3')
os.makedirs(pictures_dir, exist_ok=True)
print(f"Saving plots to: {pictures_dir}")

# Loop through the identified load channel bases
for base_name in plot_bases:
    print(f"  Generating plot for: {base_name}")

    # Define column names for the original 10-min stats in df_loads
    mean_col = f"{base_name}_mean"
    min_col = f"{base_name}_min"
    max_col = f"{base_name}_max"
    stdev_col = f"{base_name}_stdev"

    # Define corresponding binned column names (mean of the stat) in df_binned
    binned_mean_col = f"{mean_col}_mean"
    binned_min_col = f"{min_col}_mean"
    binned_max_col = f"{max_col}_mean"
    binned_stdev_col = f"{stdev_col}_mean"

    # --- Data Validation ---
    # Check if all required 10-min columns exist in df_loads (filtered data)
    required_load_cols = [mean_col, min_col, max_col, stdev_col]
    if not all(col in df_loads.columns for col in required_load_cols):
        print(f"    Skipping {base_name}: Missing one or more 10-min columns in df_loads: {required_load_cols}")
        continue

    # Check if all required binned columns exist in df_binned
    required_binned_cols = [binned_mean_col, binned_min_col, binned_max_col, binned_stdev_col]
    if not all(col in df_binned.columns for col in required_binned_cols):
        print(f"    Skipping {base_name}: Missing one or more binned columns in df_binned: {required_binned_cols}")
        continue

    # --- ADD THIS CHECK ---
    # Check if df_binned is usable for plotting (not empty and has bin centers)
    if df_binned.empty:
        print(f"    Skipping {base_name}: Binned DataFrame (df_binned) is empty.")
        continue
    if 'ws_bin_center' not in df_binned.columns:
        print(f"    Skipping {base_name}: 'ws_bin_center' column is missing in df_binned. Cannot plot binned averages.")
        # Optionally, you could plot only the 10-min data here, but skipping is safer
        continue
    # --- END OF ADDED CHECK ---

    # Get units (assuming unit is same for mean, min, max, stdev of a channel)
    # Use the original column name (e.g., MyTB_mean) to look up the unit
    unit = column_units.get(mean_col, '') # Get unit from the mean column

    # --- Create Figure ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Load Analysis: {base_name} ({unit}) vs Wind Speed', fontsize=16)

    # --- Top Subplot (Mean, Min, Max) ---
    ax = axes[0]
    # Plot 10-min data (scatter) - use df_loads (filtered data)
    ax.scatter(df_loads['Wsp_44m'], df_loads[max_col], alpha=0.15, s=8, label=f'10-min Max', color='salmon')
    ax.scatter(df_loads['Wsp_44m'], df_loads[mean_col], alpha=0.15, s=8, label=f'10-min Mean', color='lightblue')
    ax.scatter(df_loads['Wsp_44m'], df_loads[min_col], alpha=0.15, s=8, label=f'10-min Min', color='lightgreen')

    # Plot binned averages (lines/markers) - use df_binned
    ax.plot(df_binned['ws_bin_center'], df_binned[binned_max_col], marker='^', linestyle='-', color='red', label='Binned Max Avg', markersize=5)
    ax.plot(df_binned['ws_bin_center'], df_binned[binned_mean_col], marker='o', linestyle='-', color='blue', label='Binned Mean Avg', markersize=5)
    ax.plot(df_binned['ws_bin_center'], df_binned[binned_min_col], marker='v', linestyle='-', color='green', label='Binned Min Avg', markersize=5)

    ax.set_ylabel(f'Load ({unit})')
    ax.set_title('Mean, Min, Max Load')
    ax.legend(fontsize='small', loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Bottom Subplot (Standard Deviation) ---
    ax = axes[1]
    # Plot 10-min data (scatter) - use df_loads
    ax.scatter(df_loads['Wsp_44m'], df_loads[stdev_col], alpha=0.15, s=8, label=f'10-min StDev', color='plum')

    # Plot binned averages (lines/markers) - use df_binned
    ax.plot(df_binned['ws_bin_center'], df_binned[binned_stdev_col], marker='s', linestyle='-', color='purple', label='Binned StDev Avg', markersize=5)

    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel(f'Standard Deviation ({unit})')
    # ax.set_title('Standard Deviation') # Title might be redundant with y-label
    ax.legend(fontsize='small', loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Finalize ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save the figure
    save_path = os.path.join(pictures_dir, f'Load_Plot_{base_name}.png')
    try:
        plt.savefig(save_path)
        # print(f"    Saved plot: {save_path}")
    except Exception as e:
        print(f"    Error saving plot {save_path}: {e}")
    plt.close(fig) # Close the figure to free memory

print("\nFinished generating load plots for Q3.")

# Q5.1.2 Load Signal Interpretation and Calibration Analysis

# --- MyA1 Trend & Calibration Analysis ---

# Check if mean/stdev/min/max of MyA1 show similar trends
# -> Also plot 10-min range (max-min) vs wind speed
# %% QUESTION 5: LOAD INTERPRETATION AND CALIBRATION
print("\n--- Q5.1.2: MyA1 Trend & Calibration Analysis ---")

# Create directory for Q5 plots
q5_pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures', 'Q5_Analysis')
os.makedirs(q5_pictures_dir, exist_ok=True)
print(f"Saving Q5 plots to: {q5_pictures_dir}")

# --- 1. Calculate MyA1 range (max-min) ---
myA1_min_col = 'MyA1_min'
myA1_max_col = 'MyA1_max'
myA1_mean_col = 'MyA1_mean'
myA1_stdev_col = 'MyA1_stdev'

if myA1_min_col in df_loads.columns and myA1_max_col in df_loads.columns:
    # Calculate the range (max-min)
    df_loads['MyA1_range'] = df_loads[myA1_max_col] - df_loads[myA1_min_col]
    print(f"Calculated MyA1 range column")
    
    # --- 2. Plot MyA1 Range vs Wind Speed ---
    plt.figure(figsize=(12, 7))
    plt.scatter(df_loads['Wsp_44m'], df_loads['MyA1_range'], alpha=0.3, s=10, label='10-min Range')
    
    # Add binned averages if available
    # First, we need to make sure the binned data includes the range
    if not df_binned.empty and 'ws_bin_center' in df_binned.columns:
        # Calculate binned statistics for the range
        range_bins = {}
        for bin_center in df_binned['ws_bin_center'].unique():
            # Find ranges for data points in this wind speed bin
            mask = (df_loads['Wsp_44m'] >= bin_center - 0.5) & (df_loads['Wsp_44m'] < bin_center + 0.5)
            bin_ranges = df_loads.loc[mask, 'MyA1_range']
            if not bin_ranges.empty:
                range_bins[bin_center] = bin_ranges.mean()
        
        # Plot the binned ranges
        bin_centers = list(range_bins.keys())
        bin_ranges = list(range_bins.values())
        plt.plot(bin_centers, bin_ranges, 'ro-', linewidth=2, markersize=8, label='Binned Average')
    
    unit = column_units.get(myA1_min_col, '')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel(f'MyA1 Range ({unit})')
    plt.title('MyA1 Range (Max-Min) vs Wind Speed')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(q5_pictures_dir, 'MyA1_Range_vs_WindSpeed.png'))
    plt.close()
    
    # --- 3. Plot MyA1 Mean vs Wind Speed (colored by time) ---
    plt.figure(figsize=(12, 7))
    
    # Convert datetime to a numeric value for color mapping
    time_values = mdates.date2num(df_loads['datetime'])
    
    # Create scatter plot colored by time
    scatter = plt.scatter(df_loads['Wsp_44m'], df_loads[myA1_mean_col], 
                         c=time_values, cmap='jet', alpha=0.6, s=15, vmin=time_values.min(), vmax=time_values.max())
    
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel(f'MyA1 Mean ({unit})')
    plt.title('MyA1 Mean vs Wind Speed (Colored by Date)')
    plt.grid(True, alpha=0.3)
    
    # Add a colorbar with date formatting
    cbar = plt.colorbar(scatter)
    cbar.set_label('Date')
    
    # Format colorbar ticks as dates
    cbar_ticks = cbar.get_ticks()
    cbar.set_ticks(cbar_ticks)
    date_labels = [mdates.num2date(tick).strftime('%Y-%m-%d') for tick in cbar_ticks]
    cbar.set_ticklabels(date_labels)
    
    plt.savefig(os.path.join(q5_pictures_dir, 'MyA1_Mean_vs_WindSpeed_ColorByTime.png'))
    plt.close()
    
    # --- 4. Plot all MyA1 statistics vs Wind Speed (IMPROVED) ---
    plt.figure(figsize=(12, 8))

    # Use more distinct colors and larger markers with higher opacity
    plt.scatter(df_loads['Wsp_44m'], df_loads[myA1_max_col], 
                alpha=0.4, s=15, color='crimson', label='Max')
    plt.scatter(df_loads['Wsp_44m'], df_loads[myA1_mean_col], 
                alpha=0.4, s=15, color='royalblue', label='Mean')
    plt.scatter(df_loads['Wsp_44m'], df_loads[myA1_min_col], 
                alpha=0.4, s=15, color='forestgreen', label='Min')
    plt.scatter(df_loads['Wsp_44m'], df_loads[myA1_stdev_col], 
                alpha=0.4, s=15, color='darkorchid', label='StDev')

    plt.xlabel('Wind Speed (m/s)', fontsize=12)
    plt.ylabel(f'MyA1 Values ({unit})', fontsize=12)
    plt.title('MyA1 Statistics vs Wind Speed', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Create a more visible legend
    legend = plt.legend(fontsize=12, 
                       markerscale=3,        # Make legend markers larger
                       frameon=True,         # Add a frame
                       fancybox=True,        # Round the corners
                       framealpha=1.0,       # No transparency in frame
                       edgecolor='black',    # Black edge around legend
                       loc='upper right')    # Position in upper right

    # Set the legend marker alpha to 1.0 (fully opaque)
    for handle in legend.legend_handles :
        handle.set_alpha(1.0)

    plt.savefig(os.path.join(q5_pictures_dir, 'MyA1_Stats_vs_WindSpeed.png'), dpi=300)
    plt.close()
    
    print("Created MyA1 vs Wind Speed plots")
    
else:
    print(f"Cannot analyze MyA1: Required columns not found in data")

print("\n--- Identifying MyA1 Calibration Shifts ---")
# Create a scatter plot with three distinct time periods instead of color gradient
plt.figure(figsize=(14, 8))

# Define the three time periods based on transition dates
end_date_period1 = pd.to_datetime('2017-05-22')

start_date_period2 = pd.to_datetime('2017-08-09')
end_date_period2 = pd.to_datetime('2018-04-18')

start_date_period3 = pd.to_datetime('2018-06-19')


period1_mask = df_loads['datetime'] < pd.to_datetime(end_date_period1)
period2_mask = (df_loads['datetime'] >= pd.to_datetime(start_date_period2)) & (df_loads['datetime'] < pd.to_datetime(end_date_period2))
period3_mask = df_loads['datetime'] >= pd.to_datetime(start_date_period3)

# Count points in each period for the legend
p1_count = period1_mask.sum()
p2_count = period2_mask.sum()
p3_count = period3_mask.sum()

# Plot each period with a different color and shape
plt.scatter(df_loads.loc[period1_mask, 'Wsp_44m'], df_loads.loc[period1_mask, 'MyA1_mean'], 
            color='crimson', alpha=0.6, s=20, marker='o',
            label=f'Period 1: Jan-Aug 2017 ({p1_count} points)')
plt.scatter(df_loads.loc[period2_mask, 'Wsp_44m'], df_loads.loc[period2_mask, 'MyA1_mean'], 
            color='royalblue', alpha=0.6, s=20, marker='s',
            label=f'Period 2: Aug 2017-Jun 2018 ({p2_count} points)')
plt.scatter(df_loads.loc[period3_mask, 'Wsp_44m'], df_loads.loc[period3_mask, 'MyA1_mean'], 
            color='forestgreen', alpha=0.6, s=20, marker='^',
            label=f'Period 3: Jun-Jul 2018 ({p3_count} points)')

# Add labels, title, grid, and legend
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel(f'MyA1 Mean ({column_units.get("MyA1_mean", "kNm")})', fontsize=12)
plt.title('MyA1 Mean vs Wind Speed by Time Period', fontsize=14)
plt.grid(True, alpha=0.3)

# Create a more visible legend
legend = plt.legend(fontsize=12, 
                   markerscale=2,        # Make legend markers larger
                   frameon=True,         # Add a frame
                   fancybox=True,        # Round the corners
                   framealpha=0.9,       # Slight transparency in frame
                   edgecolor='black',    # Black edge around legend
                   loc='best')           # Let matplotlib choose best location

# Add horizontal reference lines at average MyA1 values for each period 
# to highlight the vertical shift between periods
for mask, color in [(period1_mask, 'crimson'), (period2_mask, 'royalblue'), (period3_mask, 'forestgreen')]:
    if mask.sum() > 0:
        avg = df_loads.loc[mask, 'MyA1_mean'].mean()
        plt.axhline(y=avg, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        plt.text(plt.xlim()[1]*0.95, avg, f'Avg: {avg:.1f} kNm', 
                 color=color, ha='right', va='center', fontweight='bold')

plt.savefig(os.path.join(q5_pictures_dir, 'MyA1_Mean_vs_WindSpeed_ThreePeriods.png'), dpi=300)
plt.close()

# Print statistics for each period
print("\nStatistics for each manually identified period:")
for period_num, mask in [(1, period1_mask), (2, period2_mask), (3, period3_mask)]:
    period_data = df_loads[mask]
    if not period_data.empty:
        print(f"\nPeriod {period_num}:")
        print(f"  Date range: {period_data['datetime'].min().strftime('%Y-%m-%d')} to {period_data['datetime'].max().strftime('%Y-%m-%d')}")
        print(f"  Number of points: {len(period_data)}")
        print(f"  Average MyA1 mean: {period_data['MyA1_mean'].mean():.2f} kNm")
        print(f"  MyA1 mean std dev: {period_data['MyA1_mean'].std():.2f} kNm")


# Calculate calibration adjustments after the statistics section
print("\n--- Calculating Calibration Adjustments ---")

# Original calibration parameters
original_gain = 763.4
original_offset = 312.8

print(f"Original calibration: Gain = {original_gain}, Offset = {original_offset}")

# Use the most recent period (Period 3) as reference
reference_period = 3
reference_value = df_loads.loc[period3_mask, 'MyA1_mean'].mean()

print(f"Using Period {reference_period} as reference (mean = {reference_value:.2f} kNm)")
print("\nSuggested calibration values:")

# Calculate offsets for each period
for period_num, mask in [(1, period1_mask), (2, period2_mask), (3, period3_mask)]:
    period_data = df_loads[mask]
    if not period_data.empty:
        period_mean = period_data['MyA1_mean'].mean()
        
        # Calculate the offset correction needed
        # Since we want to shift earlier periods up to match the reference period:
        offset_adjustment = reference_value - period_mean
        corrected_offset = original_offset - offset_adjustment
        
        print(f"\nPeriod {period_num}:")
        print(f"  Date range: {period_data['datetime'].min().strftime('%Y-%m-%d')} to {period_data['datetime'].max().strftime('%Y-%m-%d')}")
        print(f"  Current average: {period_mean:.2f} kNm")
        print(f"  Adjustment needed: {offset_adjustment:.2f} kNm")
        print(f"  Gain = {original_gain} (unchanged)")
        print(f"  Corrected Offset = {corrected_offset:.1f}")
        
        # Example conversion
        example_ws = 10.0  # Show example for 10 m/s wind speed
        example_data = period_data[(period_data['Wsp_44m'] >= 9.5) & (period_data['Wsp_44m'] <= 10.5)]
        if not example_data.empty:
            example_value = example_data['MyA1_mean'].mean()
            corrected_example = example_value + offset_adjustment
            print(f"  Example at ~{example_ws} m/s:")
            print(f"    Before: {example_value:.2f} kNm")
            print(f"    After: {corrected_example:.2f} kNm")

# --- Cross-checks ---

# Use electrical power (PO) to verify main shaft torque (MzR)


# --- Tower Bottom Moments ---# --- Cross-checks: Verify MzR signal using electrical power (PO) ---
print("\n--- Verifying Main Shaft Torque (MzR) using Electrical Power (PO) ---")

# Create a scatter plot comparing mechanical power vs electrical power
plt.figure(figsize=(12, 8))

# Calculate mechanical power from shaft torque and rotor speed
# Convert ROT from rpm to rad/s by multiplying by 2π/60
df_loads['mech_power_kW'] = df_loads['MzR_mean'] * df_loads['ROT'] * (2 * np.pi / 60) 

# Create the scatter plot
plt.scatter(df_loads['mech_power_kW'], df_loads['po'], alpha=0.5, s=15, c=df_loads['Wsp_44m'], cmap='viridis')
plt.xlabel('Mechanical Power from MzR (kW)', fontsize=12)
plt.ylabel('Electrical Power Output (kW)', fontsize=12)
plt.title('Verification of MzR: Mechanical Power vs Electrical Power', fontsize=14)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Wind Speed (m/s)')

# Add a 1:1 reference line (ideal 100% efficiency)
max_val = max(df_loads['mech_power_kW'].max(), df_loads['po'].max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='1:1 Line (100% Efficiency)')

# Add a trend line using linear regression
from sklearn.linear_model import LinearRegression
import numpy as np

X = df_loads['mech_power_kW'].values.reshape(-1, 1)
y = df_loads['po'].values

# Filter out NaN values
mask = ~np.isnan(X.flatten()) & ~np.isnan(y)
X_clean = X[mask]
y_clean = y[mask]

if len(X_clean) > 0:
    model = LinearRegression(fit_intercept=True)
    model.fit(X_clean, y_clean)
    
    # Get slope (efficiency) and intercept
    efficiency = model.coef_[0]
    intercept = model.intercept_
    
    # Plot the regression line
    x_range = np.linspace(0, max_val, 100)
    plt.plot(x_range, model.predict(x_range.reshape(-1, 1)), 'g-', 
             label=f'Trend Line (Efficiency = {efficiency:.3f})')
    
    plt.text(max_val*0.1, max_val*0.8, 
             f'Efficiency Factor: {efficiency:.3f}\nIntercept: {intercept:.2f} kW', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(q5_pictures_dir, 'MzR_Verification_Power.png'), dpi=300)
plt.close()

# Print the results
if 'efficiency' in locals():
    print(f"Calculated drivetrain efficiency: {efficiency:.3f}")
    print(f"Intercept: {intercept:.2f} kW")
    print(f"This means approximately {efficiency*100:.1f}% of mechanical power is converted to electrical power")
    
    if 0.80 <= efficiency <= 0.98:
        print("This efficiency value is within the expected range for wind turbines (80-98%)")
        print("The MzR signal appears to be properly calibrated.")
    else:
        print(f"Warning: The calculated efficiency of {efficiency:.3f} is outside the expected range (0.80-0.98)")
        print("This suggests a potential calibration issue with the MzR signal.")

# MxTB and MyTB: large scatter
# -> Analyze vs wind direction
# -> Try transforming into new signals with less scatter


# --- Tower Bottom Moments vs Wind Direction Analysis ---
print("\n--- Analyzing Tower Bottom Moments vs Wind Direction ---")

# Temporarily expand the wind direction filter for this analysis
df_expanded = df_loads_full_sector.copy()

# Calculate relative wind direction (wind direction relative to nacelle orientation)
df_expanded['relative_wind_dir'] = (df_expanded['Wdir_41m'] - df_expanded['yaw']) % 360

# Create scatter plots of moments vs wind direction
plt.figure(figsize=(18, 10))

# MxTB vs Wind Direction
plt.subplot(2, 2, 1)
plt.scatter(df_expanded['Wdir_41m'], df_expanded['MxTB_mean'], 
            alpha=0.4, s=10, c=df_expanded['Wsp_44m'], cmap='viridis')
plt.xlabel('Absolute Wind Direction (deg)', fontsize=12)
plt.ylabel('MxTB Mean (kNm)', fontsize=12)
plt.title('MxTB vs Absolute Wind Direction', fontsize=14)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Wind Speed (m/s)')

# MyTB vs Wind Direction
plt.subplot(2, 2, 2)
plt.scatter(df_expanded['Wdir_41m'], df_expanded['MyTB_mean'], 
            alpha=0.4, s=10, c=df_expanded['Wsp_44m'], cmap='viridis')
plt.xlabel('Absolute Wind Direction (deg)', fontsize=12)
plt.ylabel('MyTB Mean (kNm)', fontsize=12)
plt.title('MyTB vs Absolute Wind Direction', fontsize=14)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Wind Speed (m/s)')

# MxTB vs Relative Wind Direction
plt.subplot(2, 2, 3)
plt.scatter(df_expanded['relative_wind_dir'], df_expanded['MxTB_mean'], 
            alpha=0.4, s=10, c=df_expanded['Wsp_44m'], cmap='viridis')
plt.xlabel('Relative Wind Direction (deg)', fontsize=12)
plt.ylabel('MxTB Mean (kNm)', fontsize=12)
plt.title('MxTB vs Relative Wind Direction', fontsize=14)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Wind Speed (m/s)')

# MyTB vs Relative Wind Direction
plt.subplot(2, 2, 4)
plt.scatter(df_expanded['relative_wind_dir'], df_expanded['MyTB_mean'], 
            alpha=0.4, s=10, c=df_expanded['Wsp_44m'], cmap='viridis')
plt.xlabel('Relative Wind Direction (deg)', fontsize=12)
plt.ylabel('MyTB Mean (kNm)', fontsize=12)
plt.title('MyTB vs Relative Wind Direction', fontsize=14)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Wind Speed (m/s)')

plt.tight_layout()
plt.savefig(os.path.join(q5_pictures_dir, 'Tower_Bottom_vs_WindDirection.png'), dpi=300)
plt.close()


# --- Improved Tower Bottom Moment Transformation ---
print("\n--- Improved Tower Bottom Moment Transformation ---")

# For tower base moments, we should transform based on absolute wind direction
# rather than relative wind direction (since the tower doesn't yaw with the nacelle)
df_expanded['wind_dir_abs_rad'] = np.deg2rad(df_expanded['Wdir_41m'])

# Apply coordinate transformation using absolute wind direction
df_expanded['M_fore_aft'] = (df_expanded['MxTB_mean'] * np.cos(df_expanded['wind_dir_abs_rad']) + 
                            df_expanded['MyTB_mean'] * np.sin(df_expanded['wind_dir_abs_rad']))

df_expanded['M_side_side'] = (-df_expanded['MxTB_mean'] * np.sin(df_expanded['wind_dir_abs_rad']) + 
                             df_expanded['MyTB_mean'] * np.cos(df_expanded['wind_dir_abs_rad']))

# Create a improved comparison plot
plt.figure(figsize=(18, 12))

# Original MxTB vs Wind Speed (top left)
plt.subplot(2, 2, 1)
plt.scatter(df_expanded['Wsp_44m'], df_expanded['MxTB_mean'], 
            alpha=0.4, s=10, c=df_expanded['Wdir_41m'], cmap='hsv')
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('MxTB Mean (kNm)', fontsize=12)
plt.title('Original MxTB vs Wind Speed', fontsize=14)
plt.grid(True, alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('Absolute Wind Direction (deg)')

# Original MyTB vs Wind Speed (top right)
plt.subplot(2, 2, 2)
plt.scatter(df_expanded['Wsp_44m'], df_expanded['MyTB_mean'], 
            alpha=0.4, s=10, c=df_expanded['Wdir_41m'], cmap='hsv')
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('MyTB Mean (kNm)', fontsize=12)
plt.title('Original MyTB vs Wind Speed', fontsize=14)
plt.grid(True, alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('Absolute Wind Direction (deg)')

# New Fore-Aft Moment vs Wind Speed (bottom left)
plt.subplot(2, 2, 3)
plt.scatter(df_expanded['Wsp_44m'], df_expanded['M_fore_aft'], 
            alpha=0.4, s=10, c=df_expanded['Wdir_41m'], cmap='hsv')
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('Fore-Aft Moment (kNm)', fontsize=12)
plt.title('Transformed Fore-Aft Moment vs Wind Speed', fontsize=14)
plt.grid(True, alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('Absolute Wind Direction (deg)')

# New Side-Side Moment vs Wind Speed (bottom right)
plt.subplot(2, 2, 4)
plt.scatter(df_expanded['Wsp_44m'], df_expanded['M_side_side'], 
            alpha=0.4, s=10, c=df_expanded['Wdir_41m'], cmap='hsv')
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('Side-Side Moment (kNm)', fontsize=12)
plt.title('Transformed Side-Side Moment vs Wind Speed', fontsize=14)
plt.grid(True, alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('Absolute Wind Direction (deg)')

plt.tight_layout()
plt.savefig(os.path.join(q5_pictures_dir, 'Tower_Bottom_Transformation_Improved.png'), dpi=300)
plt.close()

# --- Quantifying Scatter Reduction in Tower Bottom Moments ---
print("\n--- Analyzing Scatter Reduction in Tower Bottom Moments ---")

# Create wind speed bins for analysis
ws_bins = np.arange(4, 19, 1)  # 4-19 m/s in 1 m/s bins
bin_results = []

# Calculate binned statistics for both original and transformed signals
for bin_start in ws_bins:
    bin_end = bin_start + 1
    bin_mask = (df_expanded['Wsp_44m'] >= bin_start) & (df_expanded['Wsp_44m'] < bin_end)
    bin_data = df_expanded[bin_mask]
    
    if len(bin_data) > 5:  # Only calculate if we have enough data points
        result = {
            'wind_speed_bin': f"{bin_start}-{bin_end}",
            'count': len(bin_data),
            'MxTB_mean': bin_data['MxTB_mean'].mean(),
            'MxTB_std': bin_data['MxTB_mean'].std(),
            'MyTB_mean': bin_data['MyTB_mean'].mean(),
            'MyTB_std': bin_data['MyTB_mean'].std(),
            'M_fore_aft_mean': bin_data['M_fore_aft'].mean(),
            'M_fore_aft_std': bin_data['M_fore_aft'].std(),
            'M_side_side_mean': bin_data['M_side_side'].mean(),
            'M_side_side_std': bin_data['M_side_side'].std()
        }
        bin_results.append(result)

# Convert to DataFrame
df_scatter = pd.DataFrame(bin_results)

# Calculate average reduction in scatter
if not df_scatter.empty:
    avg_MxTB_std = df_scatter['MxTB_std'].mean()
    avg_MyTB_std = df_scatter['MyTB_std'].mean()
    avg_M_fore_aft_std = df_scatter['M_fore_aft_std'].mean()
    avg_M_side_side_std = df_scatter['M_side_side_std'].mean()
    
    # Calculate percentage reduction
    reduction_x = (1 - avg_M_fore_aft_std / avg_MxTB_std) * 100
    reduction_y = (1 - avg_M_side_side_std / avg_MyTB_std) * 100
    
    print(f"Average standard deviation in MxTB: {avg_MxTB_std:.2f} kNm")
    print(f"Average standard deviation in M_fore_aft: {avg_M_fore_aft_std:.2f} kNm")
    print(f"Scatter reduction: {reduction_x:.1f}%")
    
    print(f"Average standard deviation in MyTB: {avg_MyTB_std:.2f} kNm")
    print(f"Average standard deviation in M_side_side: {avg_M_side_side_std:.2f} kNm")
    print(f"Scatter reduction: {reduction_y:.1f}%")

# Create plot comparing standard deviations
plt.figure(figsize=(14, 8))

# Plot standard deviations by wind speed
bins_x = [float(b.split('-')[0]) for b in df_scatter['wind_speed_bin']]

# Original moments
plt.plot(bins_x, df_scatter['MxTB_std'], 'ro-', label='MxTB Std Dev', linewidth=2)
plt.plot(bins_x, df_scatter['MyTB_std'], 'bo-', label='MyTB Std Dev', linewidth=2)

# Transformed moments
plt.plot(bins_x, df_scatter['M_fore_aft_std'], 'rd--', label='M_fore_aft Std Dev', linewidth=2)
plt.plot(bins_x, df_scatter['M_side_side_std'], 'bd--', label='M_side_side Std Dev', linewidth=2)

plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('Standard Deviation (kNm)', fontsize=12)
plt.title('Scatter Comparison: Original vs Transformed Tower Bottom Moments', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(q5_pictures_dir, 'Tower_Bottom_Scatter_Comparison.png'), dpi=300)
plt.close()

# Create a distribution plot to visualize the shape of the distributions
plt.figure(figsize=(16, 8))

# Plot histograms for original and transformed signals
plt.subplot(2, 2, 1)
plt.hist(df_expanded['MxTB_mean'].dropna(), bins=30, alpha=0.7, color='red')
plt.title('MxTB Distribution', fontsize=12)
plt.xlabel('Moment (kNm)', fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.hist(df_expanded['MyTB_mean'].dropna(), bins=30, alpha=0.7, color='blue')
plt.title('MyTB Distribution', fontsize=12)
plt.xlabel('Moment (kNm)', fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.hist(df_expanded['M_fore_aft'].dropna(), bins=30, alpha=0.7, color='red')
plt.title('M_fore_aft Distribution', fontsize=12)
plt.xlabel('Moment (kNm)', fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.hist(df_expanded['M_side_side'].dropna(), bins=30, alpha=0.7, color='blue')
plt.title('M_side_side Distribution', fontsize=12)
plt.xlabel('Moment (kNm)', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(q5_pictures_dir, 'Tower_Bottom_Distributions.png'), dpi=300)
plt.close()


# --- Rotor Tilt Moment ---
# --- Rotor Tilt Moment Analysis ---
# print("\n--- Analyzing Rotor Tilt Moment (Mtilt) ---")

# # Create correlation analysis for Mtilt with other signals
# correlation_signals = ['Wsp_44m', 'MyTB_mean', 'M_fore_aft', 'MxR_mean', 'MyR_mean']
# mtilt_correlations = {}

# for signal in correlation_signals:
#     if signal in df_loads.columns:
#         # Calculate correlation between Mtilt and the signal
#         correlation = df_loads['Mtilt_mean'].corr(df_loads[signal])
#         mtilt_correlations[signal] = correlation

# # Print correlations
# print("Mtilt correlations with other signals:")
# for signal, corr in mtilt_correlations.items():
#     print(f"  {signal}: {corr:.3f}")

# # Low wind speed behavior (zero loading)
# low_ws_mask = df_loads['Wsp_44m'] <= 5.0
# low_ws_mtilt = df_loads.loc[low_ws_mask, 'Mtilt_mean']
# zero_loading = low_ws_mtilt.mean()

# print(f"\nZero loading (average at wind speeds <= 5 m/s): {zero_loading:.2f} kNm")

# # Plot Mtilt vs wind speed
# plt.figure(figsize=(12, 8))
# plt.scatter(df_loads['Wsp_44m'], df_loads['Mtilt_mean'], alpha=0.5, s=15)
# plt.axhline(y=zero_loading, color='red', linestyle='--', label=f'Zero loading: {zero_loading:.2f} kNm')
# plt.xlabel('Wind Speed (m/s)', fontsize=12)
# plt.ylabel('Mtilt Mean (kNm)', fontsize=12)
# plt.title('Rotor Tilt Moment vs Wind Speed', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.savefig(os.path.join(q5_pictures_dir, 'Mtilt_vs_WindSpeed.png'), dpi=300)
# plt.close()

# # Enhanced Rotor Tilt Moment Analysis
# print("\n--- Comprehensive Rotor Tilt Moment (Mtilt) Analysis ---")

# # Create a directory for these specific plots
# mtilt_dir = os.path.join(q5_pictures_dir, 'Mtilt_Analysis')
# os.makedirs(mtilt_dir, exist_ok=True)

# # 1. CORRELATION ANALYSIS - Create correlation matrix with key signals
# correlation_signals = [
#     'Wsp_44m',       # Wind speed
#     'MyTB_mean',     # Tower bottom fore-aft moment
#     'M_fore_aft',    # Transformed tower fore-aft moment
#     'MxR_mean',      # Rotor-aligned moment
#     'MyR_mean',      # Rotor-aligned moment
#     'MyA1_mean',     # Blade 1 root flapwise moment (proxy for edgewise bending moments)
#     'MzR_mean'       # Main shaft torque (related to aerodynamic thrust)
# ]

# # Create a filtered dataframe with just the needed columns 
# # (removing rows with NaN in any of these columns)
# df_mtilt = df_expanded[['Mtilt_mean'] + correlation_signals].dropna()

# # Calculate and display correlation matrix
# correlation_matrix = df_mtilt.corr()
# print("\nMtilt correlation with other signals:")
# for signal in correlation_signals:
#     if signal in correlation_matrix.index:
#         print(f"  {signal}: {correlation_matrix.loc['Mtilt_mean', signal]:.3f}")

# # 2. ZERO LOADING BEHAVIOR ANALYSIS
# # Get more precise zero loading estimate by averaging very low wind speeds
# very_low_ws_mask = df_loads['Wsp_44m'] <= 4.5  # Focus on cut-in region
# low_ws_mtilt = df_loads.loc[very_low_ws_mask, 'Mtilt_mean']
# zero_loading = low_ws_mtilt.mean()

# print(f"\nZero loading (average at wind speeds <= 4.5 m/s): {zero_loading:.2f} kNm")
# print(f"Standard deviation at low wind speeds: {low_ws_mtilt.std():.2f} kNm")

# # 3. DETAILED VISUALIZATIONS

# # 3.1. Mtilt vs Wind Speed with Zero Loading Highlighted
# plt.figure(figsize=(12, 8))
# scatter = plt.scatter(df_loads['Wsp_44m'], df_loads['Mtilt_mean'], 
#                      alpha=0.5, s=15, c=df_loads['datetime'].astype(int), cmap='viridis')
# plt.axhline(y=zero_loading, color='red', linestyle='--', 
#             label=f'Zero loading: {zero_loading:.2f} kNm')

# # Add a shaded region representing the standard deviation at low wind speeds
# plt.axhspan(zero_loading - low_ws_mtilt.std(), 
#             zero_loading + low_ws_mtilt.std(), 
#             color='red', alpha=0.2, 
#             label=f'Standard deviation: ±{low_ws_mtilt.std():.2f} kNm')

# # Add binned average line
# bin_centers = np.arange(4, 19, 1)
# bin_means = []
# for center in bin_centers:
#     mask = (df_loads['Wsp_44m'] >= center-0.5) & (df_loads['Wsp_44m'] < center+0.5)
#     if mask.sum() > 5:  # Only include if we have enough data
#         bin_means.append(df_loads.loc[mask, 'Mtilt_mean'].mean())
#     else:
#         bin_means.append(np.nan)
        
# plt.plot(bin_centers, bin_means, 'ko-', linewidth=2, label='Binned Average')

# plt.xlabel('Wind Speed (m/s)', fontsize=12)
# plt.ylabel('Mtilt Mean (kNm)', fontsize=12)
# plt.title('Rotor Tilt Moment vs Wind Speed with Zero Loading', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=10)
# plt.colorbar(scatter, label='Time (chronological)')
# plt.savefig(os.path.join(mtilt_dir, 'Mtilt_vs_WindSpeed_Enhanced.png'), dpi=300)
# plt.close()

# # 3.2. Correlation Plots - Create a 2x2 grid of key correlations
# plt.figure(figsize=(16, 14))

# # Plot 1: Mtilt vs Fore-Aft Moment (representing tower interaction)
# plt.subplot(2, 2, 1)

# # Create a mask for positive fore-aft moments only
# positive_moments_mask = df_expanded['M_fore_aft'] > 0

# # Filter the data using the mask
# plt.scatter(df_expanded.loc[positive_moments_mask, 'M_fore_aft'], 
#            df_expanded.loc[positive_moments_mask, 'Mtilt_mean'], 
#            alpha=0.5, s=15, 
#            c=df_expanded.loc[positive_moments_mask, 'Wsp_44m'], 
#            cmap='viridis')

# plt.xlabel('Tower Fore-Aft Moment (kNm) - Positive Only', fontsize=12)
# plt.ylabel('Mtilt Mean (kNm)', fontsize=12)
# plt.title('Rotor Tilt Moment vs Tower Fore-Aft Moment (Positive Only)', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.colorbar(label='Wind Speed (m/s)')

# # Recalculate correlation coefficient using only positive values
# filtered_fore_aft = df_expanded.loc[positive_moments_mask, 'M_fore_aft'].dropna()
# filtered_mtilt = df_expanded.loc[positive_moments_mask, 'Mtilt_mean'].dropna()

# # Ensure we have matching indices for correlation calculation
# common_idx = filtered_fore_aft.index.intersection(filtered_mtilt.index)
# if len(common_idx) > 1:  # Need at least 2 points for correlation
#     corr = np.corrcoef(
#         filtered_fore_aft.loc[common_idx], 
#         filtered_mtilt.loc[common_idx]
#     )[0,1]
#     plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', 
#                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# # Plot 2: Mtilt vs edgewise bending moment (MyA1 as proxy for edgewise bending moments)
# plt.subplot(2, 2, 2)
# plt.scatter(df_loads['MyA1_mean'], df_loads['Mtilt_mean'], 
#            alpha=0.5, s=15, c=df_loads['Wsp_44m'], cmap='viridis')
# plt.xlabel('Blade 1 Root Moment (kNm)', fontsize=12)
# plt.ylabel('Mtilt Mean (kNm)', fontsize=12)
# plt.title('Rotor Tilt Moment vs edgewise bending moment', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.colorbar(label='Wind Speed (m/s)')

# # Calculate and display correlation coefficient
# corr = np.corrcoef(df_loads['MyA1_mean'].dropna(), 
#                   df_loads.loc[df_loads['MyA1_mean'].dropna().index, 'Mtilt_mean'])[0,1]
# plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', 
#             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# # Plot 3: Mtilt vs MxR (Rotor moment - related to thrust/aerodynamic effects)
# plt.subplot(2, 2, 3)
# plt.scatter(df_loads['MxR_mean'], df_loads['Mtilt_mean'], 
#            alpha=0.5, s=15, c=df_loads['Wsp_44m'], cmap='viridis')
# plt.xlabel('Rotor Mx Moment (kNm)', fontsize=12)
# plt.ylabel('Mtilt Mean (kNm)', fontsize=12)
# plt.title('Rotor Tilt Moment vs Rotor Mx Moment', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.colorbar(label='Wind Speed (m/s)')

# # Calculate and display correlation coefficient
# corr = np.corrcoef(df_loads['MxR_mean'].dropna(), 
#                   df_loads.loc[df_loads['MxR_mean'].dropna().index, 'Mtilt_mean'])[0,1]
# plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', 
#             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# # Plot 4: Mtilt vs Wind Speed with Power Output as color
# plt.subplot(2, 2, 4)
# scatter = plt.scatter(df_loads['Wsp_44m'], df_loads['Mtilt_mean'], 
#                      alpha=0.6, s=15, c=df_loads['po'], cmap='plasma')
# plt.axhline(y=zero_loading, color='red', linestyle='--', 
#            label=f'Zero loading: {zero_loading:.2f} kNm')
# plt.xlabel('Wind Speed (m/s)', fontsize=12)
# plt.ylabel('Mtilt Mean (kNm)', fontsize=12)
# plt.title('Rotor Tilt Moment vs Wind Speed (colored by Power)', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=10)
# plt.colorbar(scatter, label='Power Output (kW)')

# plt.tight_layout()
# plt.savefig(os.path.join(mtilt_dir, 'Mtilt_Correlation_Matrix.png'), dpi=300)
# plt.close()

# # 3.3. Zero Loading Exploration - Very Low Wind Speed Range
# plt.figure(figsize=(14, 7))
# zero_ws_mask = df_loads['Wsp_44m'] <= 6.0  # Include slightly higher wind speeds for visualization

# # Create a scatter plot with enhanced visual appearance
# plt.scatter(df_loads.loc[zero_ws_mask, 'Wsp_44m'], 
#             df_loads.loc[zero_ws_mask, 'Mtilt_mean'], 
#             alpha=0.6, s=20, c=df_loads.loc[zero_ws_mask, 'datetime'].astype(int), 
#             cmap='viridis', edgecolor='w', linewidth=0.5)

# # Add a horizontal line at the zero loading level
# plt.axhline(y=zero_loading, color='red', linestyle='--', label=f'Zero loading: {zero_loading:.2f} kNm')

# # Add a trend line for the zero loading region
# from scipy.stats import linregress
# x = df_loads.loc[zero_ws_mask, 'Wsp_44m']
# y = df_loads.loc[zero_ws_mask, 'Mtilt_mean']
# slope, intercept, r_value, p_value, std_err = linregress(x, y)
# x_trend = np.linspace(x.min(), x.max(), 100)
# y_trend = slope * x_trend + intercept
# plt.plot(x_trend, y_trend, 'k-', linewidth=2, 
#          label=f'Trend: {slope:.2f}x + {intercept:.2f} (r²={r_value**2:.3f})')

# plt.xlabel('Wind Speed (m/s)', fontsize=12)
# plt.ylabel('Mtilt Mean (kNm)', fontsize=12)
# plt.title('Zero Loading Analysis: Rotor Tilt Moment at Low Wind Speeds', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=10)
# plt.colorbar(label='Time (chronological)')

# # Annotate additional information about zero loading
# plt.text(0.02, 0.95, 
#          f"Zero Loading Statistics:\n" +
#          f"Mean: {zero_loading:.2f} kNm\n" +
#          f"Std Dev: {low_ws_mtilt.std():.2f} kNm\n" +
#          f"Physical Cause: Blade weight moment\n" +
#          f"Expected: Non-zero due to mass imbalance",
#          transform=plt.gca().transAxes,
#          bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
#          fontsize=10, verticalalignment='top')

# plt.tight_layout()
# plt.savefig(os.path.join(mtilt_dir, 'Mtilt_Zero_Loading_Analysis.png'), dpi=300)
# plt.close()

# print(f"Saved Mtilt analysis plots to {mtilt_dir}")


#%%
print('my own analysis of mtilt')

# create mask for wind speeds above cut in, lets say above 6 m/s
tilt_mask = df_expanded['Wsp_44m'] > 6.0
tilt_mask_zero_load = df_expanded['Wsp_44m'] < 6.0

# Create a subdirectory for mtilt analysis plots
mtilt_analysis_dir = os.path.join(q5_pictures_dir, 'mtilt_analysis_plots')
os.makedirs(mtilt_analysis_dir, exist_ok=True)



# plot rotor tilt vs wind speed above cut in
plt.figure(figsize=(12, 8))
plt.scatter(df_expanded.loc[tilt_mask, 'Wsp_44m'], 
            df_expanded.loc[tilt_mask, 'Mtilt_mean'], 
            alpha=0.5, s=15, c=df_expanded.loc[tilt_mask, 'datetime'].astype(int), cmap='viridis')
#add trendline
# Add trendline using linear regression
from scipy.stats import linregress

# Get x and y data (filtered by the tilt_mask)
x = df_expanded.loc[tilt_mask, 'Wsp_44m']
y = df_expanded.loc[tilt_mask, 'Mtilt_mean']

# Filter out NaN values for regression
mask = ~np.isnan(x) & ~np.isnan(y)
if mask.sum() > 1:  # Need at least 2 points for regression
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
    
    # Create trendline data
    x_trend = np.linspace(x.min(), x.max(), 100)
    y_trend = slope * x_trend + intercept
    
    # Plot trendline
    plt.plot(x_trend, y_trend, 'r-', linewidth=2, 
             label=f'Trend: {slope:.2f}x + {intercept:.2f} (r²={r_value**2:.3f})')
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('Mtilt Mean (kNm)', fontsize=12)
plt.title('Rotor Tilt Moment vs Wind Speed (above cut-in)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Time (chronological)')
plt.legend()

# Save using the correctly joined path
plt.savefig(os.path.join(mtilt_analysis_dir, 'Mtilt_vs_WindSpeed_above_cutin.png'), dpi=300)
plt.close()


# plot zero loading behavior at low wind speeds
plt.figure(figsize=(12, 8))
plt.scatter(df_expanded.loc[~tilt_mask, 'Wsp_44m'], 
            df_expanded.loc[~tilt_mask, 'Mtilt_mean'], 
            alpha=0.5, s=15, c=df_expanded.loc[~tilt_mask, 'datetime'].astype(int), cmap='viridis')
# Get x and y data (filtered by the tilt_mask)
x = df_expanded.loc[~tilt_mask, 'Wsp_44m']
y = df_expanded.loc[~tilt_mask, 'Mtilt_mean']

# Filter out NaN values for regression
mask = ~np.isnan(x) & ~np.isnan(y)
if mask.sum() > 1:  # Need at least 2 points for regression
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
    
    # Create trendline data
    x_trend = np.linspace(x.min(), x.max(), 100)
    y_trend = slope * x_trend + intercept
    
    # Plot trendline
    plt.plot(x_trend, y_trend, 'r-', linewidth=2, 
             label=f'Trend: {slope:.2f}x + {intercept:.2f} (r²={r_value**2:.3f})')
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('Mtilt Mean (kNm)', fontsize=12)
plt.title('Rotor Tilt Moment vs Wind Speed (Zero loading at low wind speeds)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Time (chronological)')
plt.legend()

# Save using the correctly joined path
plt.savefig(os.path.join(mtilt_analysis_dir, 'Mtilt_vs_WindSpeed_zero_load.png'), dpi=300)
plt.close()

#plot mtilt against all load channel means with a for loop
load_channels = ['MxTB_mean', 'MyTB_mean', 'MxR_mean', 
                 'MyR_mean', 'MzR_mean', 'MyA1_mean', 
                 'MxA1_mean', 'MyA1_mean', 'Myaw_mean',
                 ]
for channel in load_channels:
    plt.figure(figsize=(12, 8))
    plt.scatter(df_expanded[channel], df_expanded['Mtilt_mean'], 
                alpha=0.5, s=15, c=df_expanded['Wsp_44m'], cmap='viridis')
    
    # Add trendline using linear regression
    x = df_expanded[channel]
    y = df_expanded['Mtilt_mean']

    # Filter out NaN values for regression
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() > 1:  # Need at least 2 points for regression
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
        
        # Create trendline data
        x_trend = np.linspace(x.min(), x.max(), 100)
        y_trend = slope * x_trend + intercept
        
        # Plot trendline
        plt.plot(x_trend, y_trend, 'r-', linewidth=2, 
                 label=f'Trend: {slope:.2f}x + {intercept:.2f} (r²={r_value**2:.3f})')
    
    plt.xlabel(f'{channel} Mean (kNm)', fontsize=12)
    plt.ylabel('Mtilt Mean (kNm)', fontsize=12)
    plt.title(f'Rotor Tilt Moment vs {channel} Mean', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Wind Speed (m/s)')
    plt.legend()

    # Save using the correctly joined path
    plt.savefig(os.path.join(mtilt_analysis_dir, f'Mtilt_vs_{channel}.png'), dpi=300)
    plt.close()




# Check how Mtilt correlates with other loads
# -> Look at zero-loading behavior at low wind speeds

# --- Tower Top Torsion ---

# Check what MzTT correlates with — yaw or tilt moment?
# plot MzTT vs yaw and MzTT vs Mtilt
fn.plot_scatter('Tower Top Torsion vs Yaw (not direction filtered)',
                df_loads_full_sector['MzTT_mean'],
                df_loads_full_sector['Myaw_mean'],
                'Myaw Mean (kNm)', 'MzTT Mean (kNm)', 'Yaw (deg)',
                False)
fn.plot_scatter('Tower Top Torsion vs Mtilt (not direction filtered)',
                df_loads_full_sector['MzTT_mean'],
                df_loads_full_sector['Mtilt_mean'],
                'Mtilt Mean (kNm)', 'MzTT Mean (kNm)', 'Mtilt Mean (kNm)',
                False)

# check with df_loads

fn.plot_scatter('Tower Top Torsion vs Yaw (direction filtered)',
                df_loads['MzTT_mean'],
                df_loads['Myaw_mean'],
                'Myaw Mean (kNm)', 'MzTT Mean (kNm)', 'Yaw (deg)',
                False)

fn.plot_scatter('Tower Top Torsion vs Mtilt (direction filtered)',
                df_loads['MzTT_mean'],
                df_loads['Mtilt_mean'],
                'Mtilt Mean (kNm)', 'MzTT Mean (kNm)', 'Mtilt Mean (kNm)',
                False)
