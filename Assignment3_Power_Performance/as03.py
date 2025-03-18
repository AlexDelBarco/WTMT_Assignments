# # ASSIGNMENT 2: WIND SPEED MEASUREMENTS
#%%## Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import functions as fn
#import functions as fn
#from sklearn.linear_model import LinearRegression
#%% ## Import Data 
#10 minute averages data
file_path = os.path.join(os.path.dirname(__file__), 'windData.csv')

# Read CSV with proper datetime parsing
df_original = pd.read_csv(file_path,na_values=r'\N')
# Convert date_time to datetime and set as index
df_original['date_time'] = pd.to_datetime(df_original['date_time'],
                                           format='%d-%m-%Y %H:%M',
                                           )
df_original.set_index('date_time', inplace=True)
df = df_original.copy()
#%% Create Pictures directory if it doesn't exist
pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures')
if not os.path.exists(pictures_dir):
    os.makedirs(pictures_dir)

#%% Convert numeric columns before calculations
# columns_to_convert = ['AirAbs_70m', 'Press_enc_2m', 'RH_2m', 'ActPow', 'Wsp_44m']
# for col in columns_to_convert:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

#%% Data Cleaning
#plot all the data
fn.plot_all_measurements(df,plot_bool=False)

#columsn to be sorted:
#pitch angle: repeating the same value for a period of time: Ask if this is normal ?
#mean wind speed: way too large values
#TI: values over 100% are not possible, maybe we need to remove TI=0 as well?
#Tempreature: extremely low values
#Pressure: extremely low values
#Humidity: extremely low values

#remove outliers for wind speed
df = fn.remove_outliers_mask(0,30,df,'Wsp_44m','Wind Speed','m/s',show_plot = False)
#print(df['Wsp_44m'].max())
#remove outliers for TI
df = fn.remove_outliers_mask(0.1,100,df,'TI_44m','Turbulence Intensity','%',show_plot = False)
#remove outliers for pressure
df = fn.remove_outliers_mask(900,1100,df,'Press_enc_2m','Pressure','hPa',show_plot = False)
#remove outliers for temperature
df = fn.remove_outliers_mask(-10,30,df,'AirAbs_70m','Temperature','C',show_plot = False)
#remove outliers for humidity
df = fn.remove_outliers_mask(20,100,df,'RH_2m','Humidity','%',show_plot = False)
#%% Wind turbine characteristics
P_rated = 850 #kW
D = 52 #m
hub_height = 44 #m
ws_cutout = 25 #m/s
A = np.pi * (D/2)**2 #m^2

#approximations:
#Wdir @41m = Wdir at hub height
#pressure and humidity @2m = pressure and humidity at hub height

#Other constants
R0 = 287.05 #J/kgK
R_W = 461.5 #J/kgK
rho_0 = 1.225 #kg/m^3 #Reference air density
#%% Q3.2: Determine the filtered and normalized power curve based on data recorded during January - July 2023. 
# A) Perform data normalization and report the mean air density at the site. 
#make a kelvin column (we need temp in kelvin for vapor pressure calculation)
df['Temp_K'] = df['AirAbs_70m'] + 273.15
#calculate vapor pressure
df['Vapor_Pressure'] = fn.vapor_pressure(df['Temp_K'])
#show vapor pressure vs temperature and compare with plots on the internet (It looks correct imo)
fn.plot_scatter('Vapor Pressure',df['AirAbs_70m'],df['Vapor_Pressure'],'Vapor Pressure',label_x='Temperature [C]',label_y='Vapor Pressure [Pa]',
                plot_bool=False)
# Calculate air density
df['rho'] = fn.calculate_rho(df,'Press_enc_2m','Temp_K','RH_2m','Vapor_Pressure')

#for some reason there are some big outliers in the air density, 
# we will remove them, I dont know where they are coming from, lets ask the teachers
#df = fn.remove_outliers_mask(0,4000,df,'rho','Air Density','kg/m^3',show_plot = True)

fn.plot_scatter('Air Density',df.index,df['rho'],'Air Density',label_x='Temperature [degC]',label_y='Air Density [Pa]',
                plot_bool=False)

print(f"Mean air density: {df['rho'].mean():.4f} kg/m³")

#Perform data normalization
df['norm_power'] = fn.normalize_power_stall_regulated(df,'ActPow','rho')

#plot normalized power vs wind speed
fn.plot_scatter('Normalized Power',df['Wsp_44m'],df['norm_power'],'Normalized Power',label_x='Wind Speed [m/s]',label_y='Normalized Power',
                plot_bool=False)

df['norm_ws'] = fn.normalize_wind_active_controlled(df,'Wsp_44m','rho')
#plot normalized wind speed vs wind speed
fn.plot_scatter('Normalized Wind Speed',df.index,df['norm_ws'],'Normalized Wind Speed',label_x='Wind Speed [m/s]',label_y='Normalized Wind Speed',
                plot_bool=False)



#%% 3.2 B) Report the bin-averaged values of mean wind speed, mean power, standard 
#deviation of power, Cp-coefficient, number of observations, as well as the category 
#A, s_i, category B, u_i, and combined, u_ci, uncertainties for each bin i in tables and plots.

#Power curve determination
#use ws bins delta=0.5 m/s centered around 2.0,2.5,3.0... 25 m/s
def calculate_power_curve_bins(df, ws_bins, A):
    """
    Calculate binned statistics and uncertainties for power curve determination.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with normalized wind speed and power
    ws_bins : numpy.ndarray
        Wind speed bin edges
    D : float
        Rotor diameter [m]
    """
    df_copy = df.copy()
    # Create bins in the dataframe
    df_copy['ws_bins'] = pd.cut(df_copy['norm_ws'], bins=ws_bins)
    
    # Calculate statistics for each bin
    df_binned = df_copy.groupby('ws_bins', observed=True).agg({
        'norm_ws': ['mean', 'count'],
        'norm_power': ['mean', 'std'],
        'rho': 'mean'
    }).fillna(0)

    # Flatten column names
    df_binned.columns = ['mean_ws', 'count', 'mean_power', 'std_power','mean_rho']
    df_binned = df_binned.reset_index()

    # Get bin centers from the actual intervals in the data
    df_binned['binned_ws'] = df_binned['ws_bins'].apply(lambda x: (x.left + x.right) / 2)

    # Calculate category A uncertainty (si)
    df_binned['s_i'] = df_binned['std_power'] / np.sqrt(df_binned['count'])
    
    # Calculate category B uncertainty (ui)
    u_V = 0.03  # 3% uncertainty in wind speed
    u_P = 0.02  # 2% uncertainty in power
    df_binned['u_i'] = np.sqrt((u_V * df_binned['mean_ws'])**2 + 
                              (u_P * df_binned['mean_power'])**2)
    
    # Calculate combined uncertainty (uci)
    df_binned['u_c'] = np.sqrt(df_binned['s_i']**2 + df_binned['u_i']**2)

    
    df_binned['Cp'] = df_binned['mean_power'] * 1000 / (0.5 * df_binned['mean_rho'] * A * df_binned['mean_ws']**3)

    # Reorder columns for readability
    columns = ['binned_ws', 'mean_ws', 'mean_power','mean_rho', 'std_power', 
              'count', 'Cp', 's_i', 'u_i', 'u_c']
    df_binned = df_binned[columns]
    
    return df_binned

# Define bin edges for centers at 2.0, 2.5, 3.0, etc.
ws_bins = np.arange(1.75, 20.75, 0.5)

# Calculate binned statistics
df_binned = calculate_power_curve_bins(df, ws_bins, D)
# Save results to CSV
df_binned.to_csv('binned_statistics.csv', float_format='%.3f')


#Scattered plot of power Pi statistics as function of hub height wind speed Vi (What does this sentence mean?)


#Bin-averaged power, Pi, as function of bin-averaged mean wind speed Vi including combined uncertainty as ”errorbar”
fn.plot_errorbar(df_binned,'binned_ws','mean_power', 'Power curve with combined uncertainty',
'Measured Power Curve with Uncertainties','Normalized Wind Speed [m/s]', 'Normalized Power [kW]',showplot=False)

#Bin-averaged Cp as function of bin-averaged mean wind speed Vi.
fn.plot_scatter('Mean Wind Speed vs Cp',df_binned['mean_ws'],df_binned['Cp'],
'Mean Wind Speed vs Cp',label_x='Mean Wind Speed [m/s]',label_y='Cp [-]',
                plot_bool=True)

# table: bin no-i, Vi, Pi, Cp, si, ui & uci
# AEP table: Vave , AEPmeasured , uAEP (absolute), uAEP (relative), AEPextrapolated 







# Print summary statistics
print("\nSite Statistics:")
print(f"Mean air density: {df['rho'].mean():.3f} kg/m³")
print("\nBin Statistics Summary:")
print(df_binned.to_string(float_format=lambda x: '{:.3f}'.format(x)))

# Print summary statistics
print("\nSite Statistics:")
print(f"Mean air density: {df['rho'].mean():.3f} kg/m³")
print("\nBin Statistics Summary:")
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Don't wrap wide tables
pd.set_option('display.precision', 3)  # Set decimal places

# Create a more readable format for the table
print("\nPower Curve Statistics:")
print("=" * 120)  # Separator line
print(df_binned.to_string(
    index=False,
    float_format=lambda x: '{:8.3f}'.format(x),
    col_space=12,  # Minimum column width
    justify='right'
))
print("=" * 120)  # Separator line

#%% Q3.3: Calculate and report the results of AEP-measured and AEP-extrapolated using 
#Rayleigh wind speed distributions with average annual wind speed at hub height 𝑉𝑎𝑣𝑒=4, 5, 
#6, 7 8, 9, 10 and 11 m/s, together with their corresponding (absolute and relative) 
#uncertainties and complete/incomplete labels. The AEP uncertainty should be calculated 
#according to Ref. 1, Annex E, based on parameters listed in Annex 2.

