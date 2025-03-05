# # ASSIGNMENT 2: WIND SPEED MEASUREMENTS
#%%## Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functions as fn
from sklearn.linear_model import LinearRegression
#%% ## Import Data 
df_WindData = pd.read_csv('WindData.csv', parse_dates=True, index_col=0)
df_Original_WindData = df_WindData.copy()
#%% QUESTION 1) PLOT everything before manipulating the dataframe
plot_all_measurements_bool = False
fn.plot_all_measurements(df_WindData,plot_all_measurements_bool)
## Erroneous Data
# For the Cup, Sonic, Vane and Termometer measurements, it will be looked at the data for those periods that the values recorded keeps
#  repeating for more than 5 hours; those periods will be considered as erroneous data and then be converted to NaN
#%%  #Replace repeating values with NaN after a certain threshold (5 hours) of repetitions.
#list of measurements
measured_values_column_names = ['Cup116m_Mean', 'Cup116m_Min', 'Cup116m_Max', 'Cup116m_Stdv','Cup114m_Mean', 'Cup114m_Min', 'Cup114m_Max', 'Cup114m_Stdv','Cup100m_Mean', 'Cup100m_Min', 'Cup100m_Max', 'Cup100m_Stdv','Vane100m_Mean', 'Vane100m_Min', 'Vane100m_Max', 'Vane100m_Stdv','Temp100m_Mean', 'Temp100m_Min', 'Temp100m_Max', 'Temp100m_Stdv','Sonic100m_Scalar_Mean', 'Sonic100m_Scalar_Min', 'Sonic100m_Scalar_Max', 'Sonic100m_Dir', 'Sonic100m_Scalar_Stdv']

df_WindData_cleaned, removed_rows_df = fn.convert_repeating_to_nan(df_WindData, measured_values_column_names)

#check results:
#plot_all_measurements(df_WindData_cleaned)

#%% Filter Outliers

#Filter Temp 0 values
#df_WindData_cleaned_from_zeros = fn.replace_zeros_with_nan(df_WindData_cleaned)
#check results:
#plot_cleaned_from_zeros_bool = False
#fn.plot_all_measurements(df_WindData_cleaned_from_zeros,plot_cleaned_from_zeros_bool)


#%% Remove thermometer, cups and sonic outliers

df_WindData_cleaned_from_outliers = fn.replace_outliers_with_nan(df_WindData_cleaned,
                                                                 ['Temp100m_Mean', 'Temp100m_Max', 'Temp100m_Min', 
                                                                  'Cup100m_Mean', 'Cup100m_Max', 'Cup100m_Min',
                                                                  'Cup114m_Mean','Cup114m_Max', 'Cup114m_Min',
                                                                  'Cup116m_Mean', 'Cup116m_Max', 'Cup116m_Min', 
                                                                  'Sonic100m_Scalar_Mean', 'Sonic100m_Scalar_Min', 'Sonic100m_Scalar_Max'],
                                                                 factor =18)
# check results:
plot_cleaned_from_outliers_bool = False
fn.plot_scatter_and_lines('Temperature',
    df_WindData_cleaned_from_outliers['Temp100m_Mean'],
    df_WindData_cleaned_from_outliers['Temp100m_Max'],
    df_WindData_cleaned_from_outliers['Temp100m_Min'],
    unit='°C', plot_bool=plot_cleaned_from_outliers_bool)
#%% Question 2)
df_WindData = df_WindData_cleaned_from_outliers.copy()
# # Plot the ratio of the cup anemometer and sonic wind speeds at 100m as a function of
#  wind direction. Does the pattern you see agree with your understanding of how the instruments 
# are mounted and their respective boom directions? 


df_WindData['Speed_Ratio'] = df_WindData['Cup100m_Mean'] / df_WindData['Sonic100m_Scalar_Mean']

plot_speed_ratio_sonic = False
fn.plot_scatter('Ratio of Cup Anemometer and Sonic Wind Speeds at 100m',
                df_WindData['Sonic100m_Dir'], df_WindData['Speed_Ratio'], label1 = 'speed ratio', label_y ='Speed Ratio (Cup/Sonic)', 
                label_x = 'Wind Direction (Sonic) (°)',plot_bool = plot_speed_ratio_sonic)

# # We see that the cup is mounted upstream with southwind, and the sonic is mounted downstream, and would expect higher wind speeds measuremtns  from the cup when the wind is southern (180 degrees), and similarly higher ws meas from the sonic when the wind is northern (0/360 degrees), and this is what we see. nice.

#%% Question 3)

# # The coordinates of the mast are (56.440547°N, 8.150868°E). Noting the date of the dataset, 
# examine the site on Google Earth as it was at the time of the measurements. 
# Identify possible obstacles (especially neighbouring wind turbines). 
# A mast should not be considered as an obstacle but what is the justification for this?

# # dataset date: 2015
# # google earth location pictures: 2021
# # 
# # there are 4 turbines directly north of the mast, and there is also three masts located N-W of the mast.
#  otherwise it looks pretty flat. 
# # 
# # A mast is not considered an obstacle in wind measurements because it is a slender structure with minimal
#  impact on the surrounding airflow. Unlike large solid objects (e.g., buildings or dense forests) 
# that create significant turbulence and wake effects, masts cause only localized flow disturbances 
# that are negligible at the scale of wind profile measurements. Additionally, meteorological masts 
# are typically designed to minimize interference, using thin lattice structures and booms to ensure
#  instruments are positioned in undisturbed airflow.
# # 

highest_bound_wt = 346.47 
lowest_bound_wt = 13.24
#filter out the data that is not within the bounds
#fn.plot_directional_check(df_WindData,'before filtering',highest_bound_wt,lowest_bound_wt)


#df_WindData = fn.filter_direction(df_WindData)
# Apply direction filter to get clean sector data
df_filtered = fn.filter_direction(df_WindData, highest_bound_wt, lowest_bound_wt)

# Check how many rows were filtered out
print(f"Original rows: {len(df_WindData)}")
print(f"Filtered rows: {len(df_filtered)}")
print(f"Removed rows: {len(df_WindData) - len(df_filtered)}")

#set working dataframe to the filtered one after confirming that the filtering was successful
df_WindData = df_filtered.copy()



#%% Question 4
#  Make an initial investigation of how the Windcube wind speeds compare to 
# the relevant cup anemometer, using the direction filters you have derived so 
# far (you can also try without the filters). Perform linear regressions and use 
# the gain, offset and R2  as metrics. See how, in addition, using the lidar’s 
# availability as a filter affects the result.  The availability parameter is described 
# as follows: 
 
# The Windcube has an internal Carrier to Noise Ratio (CNR) threshold of -23dB 
# which means that if the CNR is below that level, there’s no valid measurement 
# of the speed and direction from the lidar. The availability parameter (a 
# channel you will find in the data) is related to the CNR values and the CNR 
# threshold. Basically, it shows the percentage of the measurements in a 10min 
# period that had a CNR value above the threshold and are therefore valid.  
 
# Show 3 or 4 different regression plots using different values of the 
# parameters

#First, we need to create scatter plots comparing the Windcube wind speeds (Spd column)
#  with the cup anemometer at 100m (Cup100m_Mean column)


fn.plot_scatter('Wind Speed Comparison',df_filtered['Vane100m_Mean'], df_filtered['Cup100m_Mean'], 'Cup',
                 df_filtered['Vane100m_Mean'], df_filtered['Spd'], 'Windcube', label_x = 'Wind Direction [°]', 
                 label_y = 'Wind Speed [m/s]', plot_bool = False)

# # Create new DataFrame for analysis
df_comparison = df_WindData.copy()

# Function to perform regression analysis and create scatter plot

plot_lidar_vs_cup = False
# Create different comparison plots
# 0. no directional filter
if plot_lidar_vs_cup == True:
        
    fn.analyze_wind_speeds(df_WindData_cleaned_from_outliers, title="Comparison (No Directional Filter)")

    fn.analyze_wind_speeds(df_WindData_cleaned_from_outliers, availability_threshold= 50, title="Comparison (No Directional Filter), Availabilty = 50%")
    # 1. No filters
    fn.analyze_wind_speeds(df_comparison, title="Filtered comparison (No Availability Filter)")

    # 3. With 95% availability threshold
    fn.analyze_wind_speeds(df_comparison, availability_threshold=95, 
                    title="Filtered comparison with 95% Availability Filter")

# %% Question 5
# Filter the wind speed from the cup anemometer to exclude the 
# possibility of ice on the cups. Explain how you have done this and report how 
# many data points have been removed. 


# Apply ice filtering
df_ice_filtered, points_removed = fn.filter_ice_on_cups(df_WindData, ice_threshold=4)

# Optionally plot results
fn.plot_scatter('Cup Speeds Before and After Ice Filtering',
    df_WindData['Temp100m_Mean'], df_WindData['Cup100m_Mean'], 
    label1='Original',
    df2x = df_ice_filtered['Temp100m_Mean'], df2y = df_ice_filtered['Cup100m_Mean'],
    label2='Ice Filtered',
    label_x = 'Temp [°C]', label_y = 'Wind Speed [m/s]',
    plot_bool=True)

     
# A formal lidar calibration should only use wind speeds between 4 and 16 m/s. 
# Explain why this is and perform this filtering on your dataset.


#Filter low wind speeds (3m/s)
fn.plot_check_ws_filter(df_ice_filtered,'before')
df_filtered_ws = fn.filter_high_and_low_ws_out(df_ice_filtered, ['Spd', 'Spd_max', 'Spd_min'])
#check results:
fn.plot_check_ws_filter(df_filtered_ws,'after')

df_Wind_Data = df_filtered_ws.copy()

