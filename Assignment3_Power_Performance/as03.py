"""AS03: Power Performance Analysis of a Wind Turbine """
print('\n')
print('################################################################')
print('Initiating Program')
# %% Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
print('Question 1 - Sectors')

RD = 52 #m
HH = 44 #m
co_WS = 25 #m/s

#Distance between WT and WME
L = 124 #m

#Maximum heights for each radius sector to be considered no obstacle 
h_2L = 1/3*(HH-0.5*RD) #m
h_2L4L = 2/3*(HH-0.5*RD) #m
h_4L8L = HH-0.5*RD #m
h_8L16L = 4/3*(HH-0.5*RD) #m

#TURBINE

# 2L

h_2L = 1/3*(HH-0.5*RD) #m

#Risø buildings
Ih_RB = 10 #m
Iw_RB = 200 #m
Le_RB = 80 #m

VS_i_RB = 351 # deg Visual segments
VS_ii_RB = 68 # deg Visual segments
VS_m_RB = VS_ii_RB - (360-VS_i_RB+VS_ii_RB)/2 # deg Middle of the segment

De_RB = fn.ERD(Ih_RB, Iw_RB)
al_RB = fn.alpha(De_RB, Le_RB)

RD_RB = Le_RB/De_RB #m relative distance
print(f'Risø buildings; alpha: {al_RB}, relative distance:{RD_RB}')
print(f'Risø buildings; remove {al_RB} degrees, centerd on {VS_m_RB}, then from {VS_m_RB - al_RB/2} to {VS_m_RB + al_RB/2}')

#Trees
Ih_T = 12 #m
Iw_T = 350 #m
Le_T = 100 #m

VS_i_T = 50 # deg Visual segments
VS_ii_T = 155 # deg Visual segments
VS_m_T = VS_i_T + (VS_ii_T - VS_i_T)/2 # deg Middle of the segment

De_T = fn.ERD(Ih_T, Iw_T)
al_T = fn.alpha(De_T, Le_T)

RD_T = Le_T/De_T #m relative distance
print(f'Trees; alpha: {al_T}, relative distance:{RD_T}')
print(f'Trees; remove {al_T} degrees, centerd on {VS_m_T}, then from {VS_m_T - al_T/2} to {VS_m_T + al_T/2}')



#Turbine V272
Ih_V27 = (31.5 + 27/2) #m
Iw_V27 = 27 #m
Le_V27 = 90 #m

VS_m_V27 = 197 # deg Middle of the segment

De_V27 = 27 
al_V27 = fn.alpha(De_V27, Le_V27)

RD_V27 = Le_V27/De_V27 #m relative distance
print(f'Turbine V27; alpha: {al_V27}, relative distance:{RD_V27}')
print(f'Turbine V27; remove {al_V27} degrees, centerd on {VS_m_V27}, then from {VS_m_V27 - al_V27/2} to {VS_m_V27 + al_V27/2}')



# 4L

h_2L4L = 2/3*(HH-0.5*RD) #m

#House and trees
Ih_HT = 12 #m
Iw_HT = 130 #m
Le_HT = 230 #m

VS_i_HT = 154 # deg Visual segments
VS_ii_HT = 188 # deg Visual segments
VS_m_HT = VS_i_HT + (VS_ii_HT - VS_i_HT)/2 # deg Middle of the segment

De_HT = fn.ERD(Ih_HT, Iw_HT)
al_HT = fn.alpha(De_HT, Le_HT)

RD_HT = Le_HT/De_HT #m relative distance
print(f'Houese and trees; alpha: {al_HT}, relative distance:{RD_HT}')
print(f'Houese and trees; remove {al_HT} degrees, centerd on {VS_m_HT}, then from {VS_m_HT - al_HT/2} to {VS_m_HT + al_HT/2}')
print(f'WT FROM {VS_m_RB - al_RB/2} TO {VS_m_V27 + al_V27/2}')

fn.plot_obstructed_sectors([(VS_m_RB - al_RB/2, VS_m_RB + al_RB/2), (VS_m_T - al_T/2, VS_m_T + al_T/2), 
                            (VS_m_V27 - al_V27/2, VS_m_V27 + al_V27/2), (VS_m_HT - al_HT/2, VS_m_HT + al_HT/2)], ['Risø buildings', 'Trees', 'Turbine V27', 'House and trees'], 'Turbine V52')

#MAST

# 2L

#Risø buildings
Ih_RB2 = 10 #m
Iw_RB2 = 160 #m
Le_RB2 = 170 #m

VS_i_RB2 = 41 # deg Visual segments
VS_ii_RB2 = 92 # deg Visual segments
VS_m_RB2 = VS_i_RB2 + (VS_ii_RB2 - VS_i_RB2)/2 # deg Middle of the segment

De_RB2 = fn.ERD(Ih_RB2, Iw_RB2)
al_RB2 = fn.alpha(De_RB2, Le_RB2)

RD_RB2 = Le_RB2/De_RB2 #m relative distance
print(f'Risø buildings; alpha: {al_RB2}, relative distance:{RD_RB2}')
print(f'Risø buildings; remove {al_RB2} degrees, centerd on {VS_m_RB2}, then from {VS_m_RB2 - al_RB2/2} to {VS_m_RB2 + al_RB2/2}')


# V52
Ih_V522 = 70 #m
Iw_V522 = 52 #m
Le_V522 = 124 #m

VS_i_V522 = 100 # deg Visual segments
VS_ii_V522 = 126 # deg Visual segments
VS_m_V522 = VS_i_V522 + (VS_ii_V522 - VS_i_V522)/2 # deg Middle of the segment

De_V522 = fn.ERD(Ih_V522, Iw_V522)
al_V522 = fn.alpha(De_V522, Le_V522)

RD_V522 = Le_V522/De_V522 #m relative distance
print(f'V52; alpha: {al_V522}, relative distance:{RD_V522}')
print(f'V52; remove {al_V522} degrees, centerd on {VS_m_V522}, then from {VS_m_V522 - al_V522/2} to {VS_m_V522 + al_V522/2}')



# V27
Ih_V272 = (31.5 + 27/2) #m
Iw_V272 = 27 #m
Le_V272 = 160 #m

VS_i_V272 = 140 # deg Visual segments
VS_ii_V272 = 151 # deg Visual segments
VS_m_V272 = VS_i_V272 + (VS_ii_V272 - VS_i_V272)/2 # deg Middle of the segment

De_V272 = fn.ERD(Ih_V272, Iw_V272)
al_V272 = fn.alpha(De_V272, Le_V272)

RD_V272 = Le_V272/De_V272 #m relative distance
print(f'V27; alpha: {al_V272}, relative distance:{RD_V272}')
print(f'V27; remove {al_V272} degrees, centerd on {VS_m_V272}, then from {VS_m_V272 - al_V272/2} to {VS_m_V272 + al_V272/2}')

# 4L

# Trees and house

Ih_TH2 = 12 #m
Iw_TH2 = 60 #m
Le_TH2 = 300 #m

VS_i_TH2 = 90 # deg Visual segments
VS_ii_TH2 = 100 # deg Visual segments
VS_m_TH2 = VS_i_TH2 + (VS_ii_TH2 - VS_i_TH2)/2 # deg Middle of the segment

De_TH2 = fn.ERD(Ih_TH2, Iw_TH2)
al_TH2 = fn.alpha(De_TH2, Le_TH2)

RD_TH2 = Le_TH2/De_TH2 #m relative distance
print(f'Trees and house; alpha: {al_TH2}, relative distance:{RD_TH2}')
print(f'Trees and house; remove {al_TH2} degrees, centerd on {VS_m_TH2}, then from {VS_m_TH2 - al_TH2/2} to {VS_m_TH2 + al_TH2/2}')


# Trees

#Trees
Ih_T2 = 12 #m
Iw_T2 = 100 #m
Le_T2 = 340 #m

VS_i_T2 = 126 # deg Visual segments
VS_ii_T2 = 140 # deg Visual segments
VS_m_T2 = VS_i_T2 + (VS_ii_T2 - VS_i_T2)/2 # deg Middle of the segment

De_T2 = fn.ERD(Ih_T2, Iw_T2)
al_T2 = fn.alpha(De_T2, Le_T2)

RD_T2 = Le_T2/De_T2 #m relative distance
print(f'Trees; alpha: {al_T2}, relative distance:{RD_T2}')
print(f'Trees; remove {al_T2} degrees, centerd on {VS_m_T2}, then from {VS_m_T2 - al_T2/2} to {VS_m_T2 + al_T2/2}')
print(f'WME FROM {VS_m_RB2 - al_RB2/2} TO {VS_m_V272 + al_V272/2}')

fn.plot_obstructed_sectors([(VS_m_RB2 - al_RB2/2, VS_m_RB2 + al_RB2/2), (VS_m_V522 - al_V522/2, VS_m_V522 + al_V522/2), 
                            (VS_m_V272 - al_V272/2, VS_m_V272 + al_V272/2), (VS_m_TH2 - al_TH2/2, VS_m_TH2 + al_TH2/2),
                            (VS_m_T2 - al_T2/2, VS_m_T2 + al_T2/2)],
                            ['Risø buildings', 'Turbine V52', 'Turbine V27', 'House and trees', 'Trees'], 'Met Mast')




print('Imports successful')
# %%   #   #  Import Data
# 10 minute averages data
file_path = os.path.join(os.path.dirname(__file__), 'windData.csv')
# file_path = Path('./windData.csv')
#  Read CSV with proper datetime parsing
df_original = pd.read_csv(file_path, na_values=r'\N')
#  Convert date_time to datetime and set as index
df_original['date_time'] = pd.to_datetime(df_original['date_time'],
                                          format='%d-%m-%Y %H:%M')
df_original.set_index('date_time', inplace=True)
df = df_original.copy()
print('Data imported successfully')
# %% Create Pictures directory if it doesn't exist
pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures')
if not os.path.exists(pictures_dir):
    os.makedirs(pictures_dir)
print('Pictures directory created successfully')

# %% Data Cleaning
# plot all the data
fn.plot_all_measurements(df, plot_bool=False)

# columsn to be sorted:
# pitch angle: repeating the same value for a period of time:
#  Ask if this is normal ?
# mean wind speed: way too large values
# TI: values over 100% are not possible, maybe we need to remove TI=0 as well?
# Tempreature: extremely low values
# Pressure: extremely low values
# Humidity: extremely low values


# remove outliers for wind speed
df = fn.remove_outliers_mask(0, 30, df, 'Wsp_44m', parameter='Wind Speed',
                             unit='m/s', show_plot=False)
print('Wind speed outliers removed')

# remove outliers for TI
df = fn.remove_outliers_mask(0.1, 100, df, 'TI_44m',
                             'Turbulence Intensity', '%', show_plot=False)
print('Turbulence Intensity outliers removed')

# remove outliers for pressure
df = fn.remove_outliers_mask(900, 1100, df, 'Press_enc_2m',
                             'Pressure', 'hPa', show_plot=False)
print('Pressure outliers removed')

# remove outliers for temperature
df = fn.remove_outliers_mask(-15, 30, df, 'AirAbs_70m',
                             'Temperature', 'C', show_plot=False)
print('Temperature outliers removed')

# remove outliers for humidity
df = fn.remove_outliers_mask(20, 100, df, 'RH_2m', 'Humidity',
                             '%', show_plot=False)
print('Humidity outliers removed')

# remove pitch
df = fn.remove_outliers_mask(-3, 20, df, 'Pitch',
                             'Pitch Angle', 'deg', show_plot=False)
print('Pitch outliers removed')

# ROT
df = fn.remove_outliers_mask(14, 100, df, 'ROT', 'Rotor Speed',
                             'rpm', show_plot=False)
print('Rotor Speed outliers removed')

# ActPow filter a bit more advanced, allow low powers until cut in speed,
#   and then sort
#  df = fn.remove_outliers_mask(10,np.inf, df,'ActPow',
#                               'Active Power','kW',show_plot=False)
df = fn.remove_outliers_mask_power(4, 10, df, show_plot=False)
print('Active Power outliers removed')

#  filter for sector
#df = df.mask(
#    ((0 < df["Wdir_41m"]) & (df["Wdir_41m"] < 20)) |
#    ((25.84 < df["Wdir_41m"]) & (df["Wdir_41m"] < 56.16)) |
#    ((58.81 < df["Wdir_41m"]) & (df["Wdir_41m"] < 69.19)) |
#    ((97.76 < df["Wdir_41m"]) & (df["Wdir_41m"] < 119.58)) |
#    ((139.34 < df["Wdir_41m"]) & (df["Wdir_41m"] < 150.66)) |
#    ((186.6 < df["Wdir_41m"]) & (df["Wdir_41m"] < 197.5)) |
#    ((340 < df["Wdir_41m"]) & (df["Wdir_41m"] < 360)))
#print('Data filtered for sector')

df = df.mask(
    ((0 < df["Wdir_41m"]) & (df["Wdir_41m"] < 229)))
print('Data filtered for sector')

# df = fn.remove_outliers_mask()
# %% Wind turbine characteristics
P_RATED = 850   # kW
D = 52   # m
HUB_HEIGHT = 44   # m
WS_CUTIN = 4   # m/s
WS_CUTOUT = 25   # m/s
A = np.pi * (D/2)**2   # m^2

# approximations:
# Wdir @41m = Wdir at hub height
# pressure and humidity @2m = pressure and humidity at hub height

# Other constants
R0 = 287.05   # J/kgK
R_W = 461.5   # J/kgK
RHO_0 = 1.225   # kg/m^3   # Reference air density
Nh = 365*24   # hours in a year

print('Constants defined successfully')
# %% Q3.2: Determine the filtered and normalized power curve based
#  on data recorded during January - July 2023.
#  A) Perform data normalization and report the mean air density at the site.
# make a kelvin column (we need temp in kelvin for vapor pressure calculation)
df['Temp_K'] = df['AirAbs_70m'] + 273.15
# calculate vapor pressure
df['Vapor_Pressure'] = fn.vapor_pressure(df['Temp_K'])
# show vapor pressure vs temperature and compare with
# plots on the internet (It looks correct imo)
fn.plot_scatter('Vapor_Pressure', df['AirAbs_70m'], df['Vapor_Pressure'],
                'Vapor Pressure', label_x='Temperature [C]',
                label_y='Vapor Pressure [Pa]', plot_bool=False)
print('Vapor Pressure calculated successfully')

#  Calculate air density
df['rho'] = fn.calculate_rho(df, 'Press_enc_2m', 'Temp_K',
                             'RH_2m', 'Vapor_Pressure')

# for some reason there are some big outliers in the air density,
#  we will remove them, I dont know where they are coming from,
# # lets ask the teachers

# df = fn.remove_outliers_mask(0,4000,df,'rho','Air Density',
#                              'kg/m^3',show_plot=True)

fn.plot_scatter('Air_Density', df['Temp_K'], df['rho'], 'Air Density',
                label_x='Temperature [degC]', label_y='Air Density [Pa]',
                plot_bool=False)
print('Air Density calculated successfully')

# print(f"Mean air density: {df['rho'].mean():.4f} kg/m³")

# Perform data normalization
df['norm_power'] = fn.normalize_power_stall_regulated(df, 'ActPow', 'rho')

# plot normalized power vs wind speed
fn.plot_scatter('Normalized_Power', df['Wsp_44m'], df['norm_power'],
                'Normalized Power', label_x='Wind Speed [m/s]',
                label_y='Normalized Power', plot_bool=False)

df['norm_ws'] = fn.normalize_wind_active_controlled(df, 'Wsp_44m', 'rho')
# plot normalized wind speed vs wind speed
fn.plot_scatter('Normalized_Wind Speed', df['Wsp_44m'], df['norm_ws'],
                'Normalized Wind Speed', label_x='Wind Speed [m/s]',
                label_y='Normalized Wind Speed', plot_bool=False)


fn.plot_scatter('Normalized_Power_vs_Normalized Wind Speed', df['norm_ws'], df['norm_power'],
                'Normalized Power', label_x='Normalized Wind Speed [m/s]',
                label_y='Normalized Power [kW] ', plot_bool=False)
print('Data normalized successfully')

# %% 3.2 B) Report the bin-averaged values of mean wind speed,
# mean power, standard
# deviation of power, Cp-coefficient, number of observations,
#  as well as the category
# A, s_i, category B, u_i, and combined, u_ci, uncertainties
# for each bin i in tables and plots.

# Power curve determination
# use ws bins delta=0.5 m/s centered around 2.0,2.5,3.0... 25 m/s

#  Define bin edges for centers at 2.0, 2.5, 3.0, etc.
ws_bins = np.arange(1.75, 20.75, 0.5)

#  Calculate binned statistics
df_binned = fn.calculate_power_curve_bins(df, ws_bins, A)
#  Save results to CSV
df_binned.to_csv('binned_statistics.csv', float_format='%.3f')

# Scattered plot of power Pi statistics as function of hub height wind speed Vi
# (What does this sentence mean?)
fn.plot_scatter('Binned_Mean_Active_Power', df_binned['mean_ws'],
                df_binned['mean_power'],
                label1='Binned Mean Normalized Active Power',
                label_x='Wind Speed [m/s]', label_y='Power [kW]',
                plot_bool=False,
                df2x=df_binned['mean_ws'], df2y=df_binned['std_power'],
                label2='Power Std Dev',
                df3x=df_binned['mean_ws'], df3y=df_binned['min_power'],
                label3='Power Min',
                df4x=df_binned['mean_ws'], df4y=df_binned['max_power'],
                label4='Power Max', dot_size1=100)

fn.plot_scatter('Normalized_Mean_Active_Power', df['norm_ws'],
                df['norm_power'], label1='Mean Normalized Active Power',
                label_x='Wind Speed [m/s]', label_y='Power [kW]',
                plot_bool=False,
                df2x=df['norm_ws'], df2y=df['ActPow_stdev'],
                label2='Power Std Dev',
                df3x=df['norm_ws'], df3y=df['ActPow_min'], label3='Power Min',
                df4x=df['norm_ws'], df4y=df['ActPow_max'], label4='Power Max')

# Bin-averaged power, Pi, as function of bin-averaged mean wind speed Vi
#  including combined uncertainty as ”errorbar”
fn.plot_errorbar(df_binned, 'binned_ws', 'mean_power', 'u_c',
                 'Power curve with combined uncertainty',
                 'Measured Power Curve with Uncertainties',
                 'Normalized Wind Speed [m/s]', 'Normalized Power [kW]',
                 showplot=False)

# Bin-averaged Cp as function of bin-averaged mean wind speed Vi.
fn.plot_scatter('CP_vs_Mean_Wind_Speed', df_binned['mean_ws'], df_binned['Cp'],
                'Cp', label_x='Mean Wind Speed [m/s]', label_y='Cp [-]',
                plot_bool=False, draw_line=True)

#  table: bin no-i, Vi, Pi, Cp, si, ui & uci

# Print power curve stats
df_binned_selected = fn.print_power_curve_stats(df_binned)
df_binned_selected.to_csv('binned_statistics_selected.csv', float_format='%.3f')

print('Binned statistics calculated successfully')
# %% Q3.3: Calculate and report the results of AEP-measured and
# AEP-extrapolated using Rayleigh wind speed distributions
#  with average annual wind speed at hub height
# 𝑉𝑎𝑣𝑒=4, 5, 6, 7 8, 9, 10 and 11 m/s, together with their corresponding
#  (absolute and relative) uncertainties and complete/incomplete labels.
#  The AEP uncertainty should be calculated according to Ref. 1, Annex E,
#  based on parameters listed in Annex 2.

#  Create DataFrame with Vave values and calculated AEP data
V_ave = np.arange(4, 12)    # Rayleigh mean wind speeds from 4 to 11 m/s
df_AEP = pd.DataFrame()
df_AEP['V_ave'] = V_ave
# Create df with extrapolated power curve and wind speeds
extrapolated_power = df_binned['mean_power']
last_value = extrapolated_power.iloc[-1]
extrapolated_ws = np.arange(2., 25.5, 0.5)
missing_values = len(extrapolated_ws) - len(extrapolated_power)
for i in range(missing_values):
    extrapolated_power = np.append(extrapolated_power, last_value)

# Create df with extrapolated power curve and wind speeds
df_extrapolated = pd.DataFrame()
df_extrapolated['extrapolated_ws'] = extrapolated_ws
df_extrapolated['extrapolated_power'] = extrapolated_power

# print(df_extrapolated)
#
df_binned['extrapolated_power'] = df_binned['mean_power'].where(
    df_binned['mean_power'] > df_binned['mean_power'].shift(-1), 0)

AEP, uAEP_abs = fn.calculate_AEP(df_binned)
print('AEP calculated successfully')

df_AEP['AEP_measured [MWh]'] = AEP
df_AEP['uAEP_abs [MWh]'] = uAEP_abs
df_AEP['uAEP_rel [%]'] = uAEP_abs / AEP*100
# df_AEP['AEP_extrapolated'] = (df_AEP['AEP_measured [MWh]'])*0.96
df_AEP['AEP_extrapolated'] = fn.calculate_extrapolated_AEP(df_extrapolated,
                                                           Nh=8760)
print('Extrapolated AEP calculated successfully')

df_AEP['quotient'] = (df_AEP['AEP_measured [MWh]']
                      / df_AEP['AEP_extrapolated'])    # calculate quotient
# check if quotient is larger than 95%
df_AEP['check'] = df_AEP['quotient'] > 0.95
#  assign labels based on the check
df_AEP['label'] = np.where(df_AEP['check'], 'complete', 'incomplete')


df_AEP.to_csv('AEP_statistics.csv', float_format='%.3f')
df_extrapolated.to_csv('extrapolated_statistics.csv', float_format='%.3f')
print('AEP statistics saved successfully')

#  Calculate extrapolated AEP for wind speeds outside measurement range
#  This needs to be implemented based on your specific requirements

#  Add completeness labels
#  df_AEP['completeness'] = 'incomplete'    #  Default to incomplete
#  measured_range_mask = (df_aep['Vave'] >= df_binned['mean_ws'].min()) & \
#                       (df_aep['Vave'] <= df_binned['mean_ws'].max())
#  df_aep.loc[measured_range_mask, 'completeness'] = 'complete'

#  Print results table


#  print(f"Annual Energy Production (AEP): {AEP}")
# fn.plot_scatter('Annual Energy Production (AEP) at Rayleigh WS',
#                  V_ave,AEP,'AEP',
#  label_x='Rayleigh mean wind speeds [m/s]',label_y='AEP [MWh]',
#                  plot_bool=False)

# print(f'Lenght Uncertainty AEP : {len(uncertainty_AEP)}')
# print(f'Lenght  AEP : {len(AEP)}')

fn.plot_errorbar(df_AEP, 'V_ave', 'AEP_measured [MWh]', 'uAEP_abs [MWh]',
                 'AEP w. Uncertainty',
                 'AEP at Rayleigh WS with Uncertainty',
                 'Rayleigh Mean Wind Speeds [m/s]', 'AEP [MWh]',
                 showplot=False)


fn.plot_scatter('Extrapolated_AEP_at_Rayleigh_WS', df_AEP['V_ave'],
                df_AEP['AEP_extrapolated'],
                'Extrapolated AEP', label_x='Rayleigh mean wind speeds [m/s]',
                label_y='AEP [MWh]', plot_bool=False)

#  AEP table: Vave , AEPmeasured , uAEP (absolute),
#  uAEP (relative), AEPextrapolated
fn.print_AEP_stats(df_AEP)   # missing extrapolated AEP ??

# plot AEP statistics
fn.plot_AEP(df_AEP, 'V_ave', 'AEP_measured [MWh]', 'AEP_extrapolated',
            'uAEP_abs [MWh]', showplot=False)


print('AEP statistics plotted successfully')
print('\n')
print('program finished successfully')
print('\n')