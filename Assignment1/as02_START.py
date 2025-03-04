# # ASSIGNMENT 2: WIND SPEED MEASUREMENTS
#%%## Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functions as fn
from sklearn.linear_model import LinearRegression
# ## Import Data 
df_WindData = pd.read_csv('WindData.csv', parse_dates=True, index_col=0)
#%% Question 1) PLOT everything before manipulating the dataframe
plot_all_measurements_bool = False
fn.plot_all_measurements(df_WindData,plot_all_measurements_bool)
# #### Erroneous Data
# By looking at the cups, vane and thermometer plots, it seems the is missing data the around the 2015-11-15. 


# For the Cup, Vane and Termometer measurements, it will be looked at the data for those periods that the values recorded keeps
#  repeating for more than 5 hours; those periods will be considered as erroneous data and then be converted to NaN
#%%  #Replace repeating values with NaN after a certain threshold (5 hours) of repetitions.
#list of measurements
measured_values_column_names = ['Cup116m_Mean', 'Cup116m_Min', 'Cup116m_Max', 'Cup116m_Stdv','Cup114m_Mean', 'Cup114m_Min', 'Cup114m_Max', 'Cup114m_Stdv','Cup100m_Mean', 'Cup100m_Min', 'Cup100m_Max', 'Cup100m_Stdv','Vane100m_Mean', 'Vane100m_Min', 'Vane100m_Max', 'Vane100m_Stdv','Temp100m_Mean', 'Temp100m_Min', 'Temp100m_Max', 'Temp100m_Stdv','Sonic100m_Scalar_Mean', 'Sonic100m_Scalar_Min', 'Sonic100m_Scalar_Max', 'Sonic100m_Dir', 'Sonic100m_Scalar_Stdv']

df_WindData_cleaned, removed_rows_df = fn.convert_repeating_to_nan(df_WindData, measured_values_column_names)

#check results:
#plot_all_measurements(df_WindData_cleaned)
#%% Filter the thermometer outlies, to remove sudden changes, and zero values


df_WindData_cleaned_from_zeros = fn.replace_zeros_with_nan(df_WindData_cleaned)
#check results:
plot_cleaned_from_zeros_bool = False
fn.plot_all_measurements(df_WindData_cleaned_from_zeros,plot_cleaned_from_zeros_bool)

#%% Remove thermometer outliers

df_WindData_cleaned_from_outliers = fn.replace_outliers_with_nan(df_WindData_cleaned_from_zeros,
                                                                 ['Temp100m_Mean', 'Temp100m_Max', 'Temp100m_Min'],
                                                                 factor = 18)
# check results:
plot_cleaned_from_outliers_bool = False
fn.plot_scatter_and_lines('Temperature',
    df_WindData_cleaned_from_outliers['Temp100m_Mean'],
    df_WindData_cleaned_from_outliers['Temp100m_Max'],
    df_WindData_cleaned_from_outliers['Temp100m_Min'],
    unit='°C', plot_bool=plot_cleaned_from_outliers_bool)
#%% Question 2)
df_WindData = df_WindData_cleaned_from_outliers
# # Plot the ratio of the cup anemometer and sonic wind speeds at 100m as a function of
#  wind direction. Does the pattern you see agree with your understanding of how the instruments 
# are mounted and their respective boom directions? 


df_WindData['Speed_Ratio'] = df_WindData['Cup100m_Mean'] / df_WindData['Sonic100m_Scalar_Mean']
plot_speed_ratio_sonic = False
fn.plot_scatter(df_WindData['Sonic100m_Dir'], df_WindData['Speed_Ratio'],'speed ratio',
                 xlabel='Wind Direction (Sonic) (°)',ylabel='Speed Ratio (Cup/Sonic)',
                 title='Ratio of Cup Anemometer and Sonic Wind Speeds at 100m', 
                 plot_bool=plot_speed_ratio_sonic)

plot_speed_ratio_vane = False
fn.plot_scatter(df_WindData['Vane100m_Mean'], df_WindData['Speed_Ratio'],'speed ratio',
                 xlabel='Wind Direction (Vane) (°)',ylabel='Speed Ratio (Cup/Sonic)',
                 title='Ratio of Cup Anemometer and Sonic Wind Speeds at 100m',
                 plot_bool=plot_speed_ratio_vane)

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

#%% Question 4
#First, we need to create scatter plots comparing the Windcube wind speeds (Spd column)
#  with the cup anemometer at 100m (Cup100m_Mean column)
#%% Question 4
# First, create scatter plots comparing Windcube speeds with cup anemometer
fn.plot_scatter(
    df1_x=df_WindData.index, df1_y=df_WindData['Cup100m_Mean'],
    label1='Cup', xlabel='Time', ylabel='Speed (m/s)',
    title='Cup vs Lidar comparison at 100m',
    df2_x=df_WindData.index, df2_y=df_WindData['Spd'],
    label2='Lidar', plot_bool=True
)

#%%
# # Create new DataFrame for analysis
# df_comparison = df_WindData.copy()

# # Function to perform regression analysis and create scatter plot
# def analyze_wind_speeds(df, availability_threshold=None, title="Wind Speed Comparison"):
#     """
#     Perform regression analysis between cup and lidar measurements
    
#     Parameters:
#     df (DataFrame): Input data
#     availability_threshold (float): Minimum availability threshold (0-100)
#     title (str): Plot title
#     """
#     # Apply availability filter if specified
#     if availability_threshold is not None:
#         df = df[df['Available'] >= availability_threshold]
    
#     # Get data without NaN values
#     valid_data = df.dropna(subset=['Cup100m_Mean', 'Spd'])
    
#     # Prepare data for regression
#     X = valid_data['Cup100m_Mean'].values.reshape(-1, 1)
#     y = valid_data['Spd'].values
    
#     # Perform linear regression
#     reg = LinearRegression().fit(X, y)
#     gain = reg.coef_[0]
#     offset = reg.intercept_
#     r2 = reg.score(X, y)
    
#     # Create scatter plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(X, y, alpha=0.5)
#     plt.plot(X, reg.predict(X), color='red', linewidth=2)
    
#     plt.xlabel('Cup Anemometer Speed [m/s]')
#     plt.ylabel('Lidar Speed [m/s]')
#     plt.title(f'{title}\nGain: {gain:.3f}, Offset: {offset:.3f}, R²: {r2:.3f}')
#     plt.grid(True)
#     plt.show()

# # Create different comparison plots
# # 1. No filters
# analyze_wind_speeds(df_comparison, title="No Filters")

# # 2. With 80% availability threshold
# analyze_wind_speeds(df_comparison, availability_threshold=80, 
#                    title="With 80% Availability Filter")

# # 3. With 95% availability threshold
# analyze_wind_speeds(df_comparison, availability_threshold=95, 
#                    title="With 95% Availability Filter")

# # 4. With direction filter from question 3 (you need to add this)
# # Example: good_directions = (df_comparison['Vane100m_Mean'] >= 180) & (df_comparison['Vane100m_Mean'] <= 270)
# # df_filtered = df_comparison[good_directions]
# # analyze_wind_speeds(df_filtered, title="With Direction Filter")
