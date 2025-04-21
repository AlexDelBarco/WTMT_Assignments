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
ws_bins = ws_bins_np = np.arange(4, 19) # 
# print(f'ws_bins: {ws_bins}')
# print(f'length of ws_bins : {len(ws_bins)}')
# print(f'length of df_binned: {len(df_binned)}')
df_binned['ws_bin_center'] = ws_bins


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

# ... rest of the code ...