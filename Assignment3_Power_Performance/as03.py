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

#approximations:
#Wdir @41m = Wdir at hub height
#pressure and humidity @2m = pressure and humidity at hub height

#Other constants
R0 = 287.05 #J/kgK
R_W = 461.5 #J/kgK
#%% Q3.2: Determine the filtered and normalized power curve based on data recorded during January - July 2023. 
#%% A) Perform data normalization and report the mean air density at the site. 
#make a kelvin column (we need temp in kelvin for vapor pressure calculation)
df['Temp_K'] = df['AirAbs_70m'] + 273.15
#calculate vapor pressure
df['Vapor_Pressure'] = fn.vapor_pressure(df['Temp_K'])
#show vapor pressure vs temperature and compare with plots on the internet (It looks correct imo)
fn.plot_scatter('Vapor Pressure',
                df['AirAbs_70m'],
                df['Vapor_Pressure'],
                'Vapor Pressure',
                label_x='Temperature [C]',
                label_y='Vapor Pressure [Pa]',
                plot_bool=True)
# Calculate air density
df['rho'] = 1/df['AirAbs_70m']*(df['Press_enc_2m']/R0-df['RH_2m']*df['Vapor_Pressure']*(1/R0-1/R_W))

#for some reason there are some big outliers in the air density, 
# we will remove them, I dont know where they are coming from, lets ask the teachers
df = fn.remove_outliers_mask(0,4000,df,'rho','Air Density','kg/m^3',show_plot = True)

print(f"Mean air density: {df['rho'].mean():.4f} kg/m³")
#%% B) Report the bin-averaged values of mean wind speed, mean power, standard 
#deviation of power, Cp-coefficient, number of observations, as well as the category 
#A, s_i, category B, u_i, and combined, u_ci, uncertainties for each bin i in tables and plots.

#%% Q3.3: Calculate and report the results of AEP-measured and AEP-extrapolated using 
#Rayleigh wind speed distributions with average annual wind speed at hub height 𝑉𝑎𝑣𝑒=4, 5, 
#6, 7 8, 9, 10 and 11 m/s, together with their corresponding (absolute and relative) 
#uncertainties and complete/incomplete labels. The AEP uncertainty should be calculated 
#according to Ref. 1, Annex E, based on parameters listed in Annex 2.