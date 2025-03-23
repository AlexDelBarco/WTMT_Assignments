# %% Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# %% Question 1 functions

def ERD(Ih, Iw):

    De = (2*Ih*Iw)/(Ih+Iw)

    return De


def alpha(De, Le):

    al = 1.3*np.rad2deg(np.arctan(2.5*((De)/(Le))+0.15))+10

    return al



# %% Plotting functions

def plot_obstructed_sectors(sectors, sector_labels, dev):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')  # Set North as zero degrees
    ax.set_theta_direction(-1)  # Set clockwise direction
    
    # Define the radius (full circle)
    radius = 1

    colors = plt.cm.get_cmap("tab10", len(sectors))
    
    # Plot each obstructed sector
    for i, ((start, end), label) in enumerate(zip(sectors, sector_labels)):
        theta1 = np.deg2rad(start)
        theta2 = np.deg2rad(end)
        ax.bar(x=(theta1 + theta2) / 2, height=radius, width=theta2 - theta1, color=colors(i), alpha=0.5, label=label)
    

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    
    plt.title(f'Disturbed Wind Sectors {dev}')
    #plt.savefig(f'Pictures/Disturbed_Sectors_{dev}.png')
    plt.show()

def plot_scatter_and_lines(measurement, df_mean, df_max=None, df_min=None, height=100,
                            unit ='Wind Speed (m/s)', plot_bool = False ):
    plt.figure(figsize=(50, 10))
    plt.plot(df_mean, label = 'mean', linewidth=1)
    if df_max is not None and df_min is not None:
        plt.plot(df_max, label = 'min', linewidth=1)
        plt.plot(df_min, label = 'max', linewidth=1)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel(unit, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f'{measurement} {height}m 10min Time Series', fontsize=25)
    plt.legend(fontsize=20)
    plt.savefig(f'Pictures/{measurement}_{height}m.png')

    if plot_bool == True:
        plt.show()
    else:
        plt.close()


    plt.figure(figsize=(50, 10))
    plt.scatter(df_mean.index, df_mean, label='mean', s=1)
    if df_max is not None and df_min is not None:
        plt.scatter(df_max.index, df_max, label='min', s=1)
        plt.scatter(df_min.index, df_min, label='max', s=1)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel(unit, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f'{measurement} {height}m 10min Time Series', fontsize=25)
    plt.legend(fontsize=20)
    plt.savefig(f'Pictures/{measurement}_{height}m.png')
    if plot_bool == True:
        plt.show()


def plot_scatter(title, df1x, df1y, label1, label_x='Time [s]', label_y='Wind Speed (m/s)', 
                plot_bool=False, df2x=None, df2y=None, label2=None, df3x=None, df3y=None, 
                label3=None, df4x = None, df4y = None, label4 = None, draw_line=False, dot_size1=5, dot_size2=5):
    """Plot scatter data with optional connecting line through first dataset.
    
    Args:
        title (str): Plot title
        df1x (array-like): X values for first dataset
        df1y (array-like): Y values for first dataset
        label1 (str): Label for first dataset
        label_x (str): X-axis label
        label_y (str): Y-axis label
        plot_bool (bool): Whether to show plot
        df2x (array-like, optional): X values for second dataset
        df2y (array-like, optional): Y values for second dataset
        label2 (str, optional): Label for second dataset
        df3x (array-like, optional): X values for third dataset
        df3y (array-like, optional): Y values for third dataset
        label3 (str, optional): Label for third dataset
        draw_line (bool): Whether to draw a line through first dataset
    """
    
    plt.figure(figsize=(16*2, 9*2))
    
    #  Plot first dataset with optional line
    plt.scatter(df1x, df1y, label=label1, s = dot_size1)
    
    if draw_line:
        plt.plot(df1x, df1y, '-', alpha=0.5)
        
    #  Plot second dataset if provided
    if df2x is not None:
        plt.scatter(df2x, df2y, label=label2, s = dot_size2)
            
    #  Plot third dataset if provided
    if df3x is not None:
        plt.scatter(df3x, df3y, label=label3, s = dot_size1)
    
    if df4x is not None:
        plt.scatter(df4x, df4y, label=label4, s = dot_size1)

    plt.xlabel(label_x, fontsize=20)
    plt.ylabel(label_y, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    #  plt.xlim(x_min, x_max)
    #  plt.ylim(y_min, y_max)
    plt.title(title, fontsize=25)
    # Custom legend markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label1, markerfacecolor='blue', markersize=10),
    ]
    if df2x is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label2, markerfacecolor='orange', markersize=10))
    if df3x is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label3, markerfacecolor='green', markersize=10))
    if df4x is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label4, markerfacecolor='red', markersize=10))
    
    plt.legend(handles=legend_elements, fontsize=20)
    

    
    pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures')
    save_path = os.path.join(pictures_dir, f'{title}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)        
    if plot_bool == True:
        plt.show()
    else:
        plt.close()
def plot_all_measurements(df, plot_bool=False):
    """
    Plot various measurements from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing measurement data.
    plot_bool (bool, optional): If True, plots will be generated and displayed. Defaults to False.
    """

    plot_scatter('Pre-Clean_Active Power',df.index,df['ActPow'],'Active Power',
                    label_x='Date',label_y='Active Power [kW]',plot_bool=plot_bool)
                    # df2x = df.index, df2y = df['ActPow_min'], label2 = 'Active Power Min',
                    # df3x = df.index, df3y = df['ActPow_max'], label3 = 'Active Power Max')

    plot_scatter('Pre-Clean_Mean Rotor Speed',df.index,df['ROT'],'Mean Rotor Speed',
                    label_x='Date',label_y='Mean Rotor Speed [RPM]',plot_bool=plot_bool)
                    # df2x = df.index, df2y = df['ROT_min'], label2 = 'Mean Rotor Speed Min',
                    # df3x = df.index, df3y = df['ROT_max'], label3 = 'Mean Rotor Speed Max')

    plot_scatter('Pre-Clean_Mean Pitch angle',df.index,df['Pitch'],'Pitch angle',
                    label_x='Date',label_y='Pitch angle [deg]',plot_bool=plot_bool)
                    # df2x = df.index, df2y = df['Pitch_min'], label2 = 'Pitch angle Min',
                    # df3x = df.index, df3y = df['Pitch_max'], label3 = 'Pitch angle Max')
    
    plot_scatter('Pre-Clean_Mean Yaw angle',df.index,df['yaw'],'Yaw angle',
                    label_x='Date',label_y='Yaw angle [deg]',plot_bool=plot_bool)
    
    plot_scatter('Pre-Clean_Mean wind speed. Mounted on South boom',df.index,df['Wsp_44m'],'Mean wind speed',
                    label_x='Date',label_y='Mean wind speed [m/s]',plot_bool=plot_bool)
    
    plot_scatter('Pre-Clean_Mean Turbulence Intensity. Mounted on South boom',df.index,df['TI_44m'],' Mean TI',
                    label_x='Date',label_y=' Mean TI [%]',plot_bool=plot_bool)
    
    plot_scatter('Pre-Clean_Mean Wind Direction. Mounted on North boom',df.index,df['Wdir_41m'],' Mean Wind Direction',
                    label_x='Date',label_y=' Mean Wind Direction [deg]',plot_bool=plot_bool)

    plot_scatter('Pre-Clean_Mean temperature. Mounted on South boom',df.index,df['AirAbs_70m'],'Mean Temperature',
                    label_x='Date',label_y='Mean Temperature [degC]',plot_bool=plot_bool)

    plot_scatter('Pre-Clean_Mean Atmospheric Pressure, measured in mast',df.index,df['Press_enc_2m'],'  atm. pressure',
                    label_x='Date',label_y=' Mean  atm. pressure [hPa]',plot_bool=plot_bool)

    plot_scatter('Pre-Clean_Mean Relative Humidity',df.index,df['RH_2m'],'Mean relative humidity',
                     label_x='Date',label_y='  Mean relative humidity [%]',plot_bool=plot_bool)

def plot_check_vane_filter(df,title,lb):
    x_vals = [df.index.min(),df.index.max()]
    lower_y_vals = [lb,lb]
    
    plt.figure(figsize=(50,10))
    plt.scatter(df.index,df['Vane100m_Mean'], label = 'Mean', s = 5)
    plt.scatter(df.index,df['Vane100m_Min'], label = 'Min', s = 5)
    plt.scatter(df.index,df['Vane100m_Max'], label = 'Max', s = 5)
    plt.plot(x_vals, lower_y_vals, label = 'ws filter lower bound', linewidth = 2)
    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel('Direction [°]', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title(f'{measurement} {height}m 10min Time Series', fontsize=25)
    plt.title(f'Vane filter check {title} filtering', fontsize=25)
    plt.legend(fontsize=20)
    plt.savefig(f'Pictures/lidar_ws_filter.png')
    plt.show()

def plot_check_ws_filter(df,plots,title,lb,ub,measurement):
    x_vals = [df.index.min(),df.index.max()]
    lower_y_vals = [lb,lb]
    upper_y_vals = [ub,ub]
    plot1, plot2, plot3 = plots



    plt.figure(figsize=(50,10))
    plt.scatter(df.index,df[plot1], label = 'Mean', s = 5)
    plt.scatter(df.index,df[plot2], label = 'Min', s = 5)
    plt.scatter(df.index,df[plot3], label = 'Max', s = 5)
    plt.plot(x_vals, lower_y_vals, label = 'ws filter lower bound', linewidth = 2)
    plt.plot(x_vals, upper_y_vals, label = 'ws filter upper bound', linewidth = 2)
    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel('Wind Speed (m/s)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title(f'{measurement} {height}m 10min Time Series', fontsize=25)
    plt.title(f'Speed filter check means {title} {measurement} filtering', fontsize=25)
    plt.legend(fontsize=20)
    plt.savefig(f'Pictures/{measurement}_ws_filter.png')
    plt.show()

def plot_directional_check(df,title,highest_bound,lowest_bound, meas):
    direction_filter_lower_bound_list = [lowest_bound,lowest_bound]
    direction_filter_upper_bound_list = [highest_bound,highest_bound]
    y_values_list = [0,30]


    plt.figure(figsize=(50,10))
    plt.scatter(df[meas],df['Cup100m_Mean'], label = 'mean', s = 5)
    plt.axvline(x=lowest_bound, color='r', linestyle='--', label='direction filter lower bound')
    plt.axvline(x=highest_bound, color='g', linestyle='--', label='direction filter upper bound')
    # plt.scatter(direction_filter_lower_bound_list, y_values_list, label = 'direction filter', s = 5)
    # plt.scatter(direction_filter_upper_bound_list, y_values_list, label = 'direction filter', s = 5)
    plt.xlabel('Wind Direction [°]', fontsize=20)
    plt.ylabel('Wind Speed (m/s)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title(f'{measurement} {height}m 10min Time Series', fontsize=25)
    plt.title(f'Directional filter {meas} {title}', fontsize=25)
    plt.legend(fontsize=20)
    plt.savefig(f'Pictures/direction_filter_{meas}.png')
    plt.show()

def plot_errorbar(df_binned,ws,power,uncertainty, label,title, xlabel, ylabel,showplot = False):
    """_summary_
    plot power curve with uncertainties

    Args:

        df_binned (Dataframe): binned dataframe with power and wind speed 
        ws (Str)): column name for wind speed
        power (Str): column name for power
        title (Str): title of the plot
    """
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(df_binned[ws], df_binned[power], 
                yerr=df_binned[uncertainty], fmt='o-', capsize=5,
                label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    #  Ensure the 'Pictures' directory exists
    pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures')
    os.makedirs(pictures_dir, exist_ok=True)

    #  Save the plot
    plt.savefig(os.path.join(pictures_dir, f'{title}.png'))
    if showplot == True:
        plt.show()
    else:
        plt.close()

def plot_AEP(df_AEP, ws, aep, aep_extrapolated, uncertainty, showplot = False):
    """Plot AEP statistics with uncertainties.

    Args:
        df_AEP (DataFrame): DataFrame containing AEP statistics
    """
    #  Create a bar plot with error bars for AEP
    plt.figure(figsize=(10, 6))
    plt.errorbar(df_AEP[ws], df_AEP[aep], 
                yerr=df_AEP[uncertainty], fmt='o-', capsize=5,
                label='AEP w. uncertainty')
    plt.scatter(df_AEP[ws], df_AEP[aep_extrapolated],
                label='AEP extrapolated', color='red')
    plt.xlabel('V_ave [m/s]')
    plt.ylabel('AEP [MWh]')
    plt.title('AEP Statistics')
    plt.grid(True)
    plt.legend()

    #  Ensure the 'Pictures' directory exists
    pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures')
    os.makedirs(pictures_dir, exist_ok=True)

    #  Save the plot
    plt.savefig(os.path.join(pictures_dir, 'AEP_Statistics.png'))
    if showplot == True:
        plt.show()
    else:
        plt.close()  

# %% Print functions
def print_power_curve_stats(df_binned):
    """Print power curve statistics table with selected columns.
    
    Parameters:
    -----------
    df_binned : pandas.DataFrame
        DataFrame containing binned power curve statistics
    """
    #  Create bin numbers starting from 1
    df_selected = pd.DataFrame({
        'Bin': range(1, len(df_binned) + 1),
        'Vi': df_binned['mean_ws'],
        'Pi': df_binned['mean_power'],
        'Cp': df_binned['Cp'],
        'si': df_binned['s_i'],
        'ui': df_binned['u_i'],
        'uci': df_binned['u_c']
    })
    
    print("Power Curve Statistics:")
    print("=" * 80)
    print(df_selected.to_string(
        index=False,
        float_format=lambda x: '{:8.3f}'.format(x),
        col_space=10,
        justify='right'
    ))
    print("=" * 80)
    return df_selected

def print_AEP_stats(df_AEP):
    """Print power curve statistics table with selected columns.
    
    Parameters:
    -----------
    df_binned : pandas.DataFrame
        DataFrame containing binned power curve statistics
    """
    #  Create bin numbers starting from 1
    #  AEP table: Vave , AEPmeasured , uAEP (absolute), uAEP (relative), AEPextrapolated 
    df_selected = pd.DataFrame({
        'V_ave': df_AEP['V_ave'],
        'AEP_measured [MWh]': df_AEP['AEP_measured [MWh]'],
        'uAEP_abs [MWh]': df_AEP['uAEP_abs [MWh]'],
        'uAEP_rel [%]': df_AEP['uAEP_rel [%]'],
        'AEP_extrapolated': df_AEP['AEP_extrapolated'],
        'label': df_AEP['label']
      })
    
    print("AEP Statistics:")
    print("=" * 80)
    print(df_selected.to_string(
        index=False,
        float_format=lambda x: '{:8.3f}'.format(x),
        col_space=10,
        justify='right'
    ))
    print("=" * 80)

def print_AEP_stats_old(df_AEP):
    """Print power curve statistics table with selected columns.
    
    Parameters:
    -----------
    df_binned : pandas.DataFrame
        DataFrame containing binned power curve statistics
    """
    #  Create bin numbers starting from 1
    #  AEP table: Vave , AEPmeasured , uAEP (absolute), uAEP (relative), AEPextrapolated 
    df_selected = pd.DataFrame({
        'V_ave': df_AEP['V_ave'],
        'AEP': df_AEP['AEP'],
        'uncertainty_AEP': df_AEP['uncertainty_AEP']
      })
    
    print("/nAEP Statistics:")
    print("=" * 80)
    print(df_selected.to_string(
        index=False,
        float_format=lambda x: '{:8.3f}'.format(x),
        col_space=10,
        justify='right'
    ))
    print("=" * 80)

def print_uncertainties_and_sensitivity_factors():
    print(f'Cat B power uncertainty: {u_P_i}')
    print(f'Cat B wind speed uncertainty: {u_V_i}')
    print(f'Cat B temperature uncertainty: {u_T_i}')
    print(f'Cat B pressure uncertainty: {u_B_i}')
    print(f'Cat B humidity uncertainty: {u_RH_i}')

    print(f'Cat B power sensitivity factor: {sens_factor_P_i}')
    print(f'Cat B wind speed sensitivity factor: {sens_factor_V_i}')
    print(f'Cat B temperature sensitivity factor: {sens_factor_T_i}')
    print(f'Cat B pressure sensitivity factor: {sens_factor_B_i}')
    print(f'Cat B humidity sensitivity factor: {sens_factor_RH_i}')

# %% Data cleaning functions
def convert_repeating_to_nan(df, columns, threshold_hours=5):
    """
    Replaces repeating values in specified columns of a DataFrame with NaN after a certain threshold of repetitions.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to check for repeating values.
    threshold_hours (int, optional): The threshold in hours for how long a value must repeat before being replaced with NaN. Default is 5 hours.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: The modified DataFrame with repeating values replaced by NaN.
        - pd.DataFrame: A DataFrame containing the rows that were modified.
    """
    threshold = threshold_hours * 6
    removed_rows = pd.DataFrame()
    for column in columns:
        repeating = df[column].eq(df[column].shift())
        count_repeats = repeating.groupby((repeating != repeating.shift()).cumsum()).cumsum()
        periods_to_nan = count_repeats >= threshold
        periods_to_nan = periods_to_nan.groupby((periods_to_nan != periods_to_nan.shift()).cumsum()).transform('any')
        removed_rows = pd.concat([removed_rows, df[periods_to_nan]])
        df.loc[periods_to_nan, column] = np.nan
    removed_rows = removed_rows.drop_duplicates()
    
    return df, removed_rows

def replace_zeros_with_nan(df, columns=None):
    """
    Replace all zero values with NaN in specified columns of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame
    columns (list, optional): List of column names to check. If None, checks all columns.
    
    Returns:
    pd.DataFrame: DataFrame with zeros replaced by NaN
    """
    df_cleaned = df.copy()
    
    #  If no columns specified, use all columns
    if columns is None:
        columns = df.columns
    
    #  Replace zeros with NaN in specified columns
    for column in columns:
        mask = df_cleaned[column] == 0.0
        if mask.any():
            df_cleaned.loc[mask, column] = np.nan
            print(f"Replaced {mask.sum()} zero values with NaN in column: {column}")
    
    return df_cleaned

def filter_high_and_low_ws_out_lidar(df, columns=None, lower_bound=4.0, upper_bound=16.0):
    """
    Replace wind speeds outside the valid range [lower_bound, upper_bound] with NaN.
    For formal lidar calibration, valid range is typically 3-16 m/s.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame
    columns (list, optional): List of column names to check. Must be wind speed columns.
                            If None, checks Cup*_Mean and Sonic*_Mean columns.
    lower_bound (float): Minimum valid wind speed in m/s (default: 3.0)
    upper_bound (float): Maximum valid wind speed in m/s (default: 16.0)
    
    Returns:
    pd.DataFrame: DataFrame with invalid wind speeds replaced by NaN
    """
    df_cleaned = df.copy()
    
    #  If no columns specified, use default wind speed columns
    if columns is None:
        #  Find all lidar columns
        columns = [col for col in df.columns if 
                  ('Spd' in col and 'Mean' in col)]
    
    #  Replace invalid wind speeds with NaN in specified columns
    for column in columns:
        try:
            #  Create mask for invalid wind speeds (too low or too high)
            mask_low = df_cleaned[column] < lower_bound
            mask_high = df_cleaned[column] > upper_bound
            mask_combined = mask_low | mask_high
            
            if mask_combined.any():
                low_count = mask_low.sum()
                high_count = mask_high.sum()
                df_cleaned.loc[mask_combined, column] = np.nan
                print(f"Column {column}:")
                print(f"  - Replaced {low_count} low wind speeds (<{lower_bound} m/s)")
                print(f"  - Replaced {high_count} high wind speeds (>{upper_bound} m/s)")
                print(f"  - Total replaced: {mask_combined.sum()}")
        except Exception as e:
            print(f"Error processing column {column}: {str(e)}")
    
    return df_cleaned

def filter_high_and_low_ws_out_cup(df, columns=None, lower_bound=3.0, upper_bound=100.0):
    """
    Replace wind speeds outside the valid range [lower_bound, upper_bound] with NaN.
    For formal lidar calibration, valid range is typically 3-16 m/s.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame
    columns (list, optional): List of column names to check. Must be wind speed columns.
                            If None, checks Cup*_Mean and Sonic*_Mean columns.
    lower_bound (float): Minimum valid wind speed in m/s (default: 3.0)
    upper_bound (float): Maximum valid wind speed in m/s (default: 16.0)
    
    Returns:
    pd.DataFrame: DataFrame with invalid wind speeds replaced by NaN
    """
    df_cleaned = df.copy()
    
    #  If no columns specified, use default wind speed columns
    if columns is None:
        #  Find all lidar columns
        columns = [col for col in df.columns if 
                  ('Cup' in col and 'Mean' in col)]
    
    #  Replace invalid wind speeds with NaN in specified columns
    for column in columns:
        try:
            #  Create mask for invalid wind speeds (too low or too high)
            mask_low = df_cleaned[column] < lower_bound
            mask_high = df_cleaned[column] > upper_bound
            mask_combined = mask_low | mask_high
            
            if mask_combined.any():
                low_count = mask_low.sum()
                high_count = mask_high.sum()
                df_cleaned.loc[mask_combined, column] = np.nan
                print(f"Column {column}:")
                print(f"  - Replaced {low_count} low wind speeds (<{lower_bound} m/s)")
                print(f"  - Replaced {high_count} high wind speeds (>{upper_bound} m/s)")
                print(f"  - Total replaced: {mask_combined.sum()}")
        except Exception as e:
            print(f"Error processing column {column}: {str(e)}")
    
    return df_cleaned

def filter_vane(df, columns=None, lower_bound=1.5):
    """
    Replace directional data outside the valid range (lower_bound) with NaN.
        
    Parameters:
    df (pd.DataFrame): The input DataFrame
    columns (list, optional): List of column names to check. Must be wind vane columns.
                            If None, checks Vane columns.
    lower_bound (float): Minimum direction wind speed in m/s (default: 1.5)
    
    Returns:
    pd.DataFrame: DataFrame with invalid wind speeds replaced by NaN
    """
    df_cleaned = df.copy()
    
    #  If no columns specified, use default wind speed columns
    if columns is None:
        #  Find all lidar columns
        columns = [col for col in df.columns if 
                  ('Vane' in col and 'Mean' in col)]
    
    #  Replace invalid wind speeds with NaN in specified columns
    for column in columns:
        try:
            #  Create mask for invalid wind speeds (too low)
            mask_low = df_cleaned[column] < lower_bound
                        
            if mask_low.any():
                low_count = mask_low.sum()
            
                df_cleaned.loc[mask_low, column] = np.nan
                print(f"Column {column}:")
                print(f"  - Replaced {low_count} directional data below (<{lower_bound} m/s)")
                
        except Exception as e:
            print(f"Error processing column {column}: {str(e)}")
    
    return df_cleaned

def remove_outliers_mask(lower_bound, upperbound, df, columns=None, parameter=None, unit=None, show_plot=False):
    """
    Remove outliers from specified columns of a DataFrame by replacing them with NaN.

    Parameters:
    lower_bound (float): Minimum acceptable value.
    upperbound (float): Maximum acceptable value.
    df (pd.DataFrame): The input DataFrame.
    columns (str or list, optional): Column name or list of column names to process.
    parameter (str, optional): Name of the parameter for plot titles (used if show_plot=True).
    unit (str, optional): Unit of the parameter for plot labels (used if show_plot=True).
    show_plot (bool, optional): Whether to show scatter plots before and after cleaning.

    Returns:
    pd.DataFrame: The modified DataFrame with outliers replaced by NaN.
    """
    
    df_copy = df.copy()
    # plot_scatter(f'{parameter} Before Cleaning', df.index, df[columns], label1 = parameter,
    #               label_x='Date', label_y=f'{parameter} [{unit}]', plot_bool=show_plot)


    #  Create a mask for outliers based on 'Wsp_44m'
    mask = (df_copy[columns] < lower_bound) | (df_copy[columns] > upperbound)

    #  Apply the mask to all columns in the DataFrame
    df_copy = df_copy.mask(mask, other=np.nan).copy()

    plot_scatter(f'Cleaning_{parameter}', df.index, df[columns], label1 = f'{parameter} Before cleaning',
                  label_x='Date', label_y=f'{parameter} [{unit}] ',
                    plot_bool=show_plot, df2x = df_copy.index, df2y = df_copy[columns],
                      label2 = f'{parameter} [{unit}] After cleaning', dot_size1=100, dot_size2 = 80)

    return df_copy

def remove_outliers_mask_power(cut_in, lower_bound, df, show_plot=False):
    """
    Remove power outliers based on wind speed conditions:
    - Keep all power values when wind speed is below cut-in
    - Replace power values below lower_bound with NaN when wind speed is at or above cut-in
    

    Parameters:
    cut_in (float): Cut-in wind speed of the turbine (m/s)
    lower_bound (float): Minimum acceptable power value after cut-in (typically 10 kW)
    upperbound (float): Maximum acceptable power value
    df (pd.DataFrame): The input DataFrame
    power_column (str): Column name for power values
    wind_column (str): Column name for wind speed values
    parameter (str, optional): Name of the parameter for plot titles
    unit (str, optional): Unit of the parameter for plot labels
    show_plot (bool, optional): Whether to show plots before and after cleaning

    Returns:
    pd.DataFrame: The modified DataFrame with outliers replaced by NaN
    """
    df_copy = df.copy()

    # print(df.columns)
    # plot_scatter('Active Power Before Cleaning', df_copy['Wsp_44m'], df_copy['ActPow'], label1 = 'Active Power',
    #               label_x='Wind Speed [m/s]', label_y='Power [kW]', plot_bool=show_plot)

    #  Create masks for different conditions
    above_cut_in = df_copy['Wsp_44m'] >= cut_in
    below_min_power = df_copy['ActPow'] < lower_bound
    

     #  Only filter out low power values when wind speed is above cut-in
    outlier_mask = (above_cut_in & below_min_power)
    print(f'outliers detected for active power: {len(outlier_mask)}')

    
    #  Apply the mask to all columns in the DataFrame
    df_copy = df_copy.mask(outlier_mask, other=np.nan).copy()

    plot_scatter('Cleaned Active Power', df_copy['Wsp_44m'], df_copy['ActPow'],
                  label1 = 'Active Power After Cleaning',
                  label_x='Wind Speed [m/s]', label_y='Power [kW]',
                    plot_bool=show_plot, df2x = df['Wsp_44m'],
                      df2y = df['ActPow'], label2 = 'Active Power Before Cleaning', dot_size1=100, dot_size2=80)


    return df_copy

def filter_direction(df, highest_bound, lowest_bound, meas):

    """
    Filter the dataframe to only include rows where the wind direction is OUTSIDE 
    the turbine wake sector (346.47° - 13.24°).
    # house south west : 146.6 - 125

    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    #  Handle the wrap-around at 360 degrees properly
    #  Keep data where direction is NOT in the turbine wake sector
    mask = ~((df[meas] >= highest_bound) | (df[meas] <= lowest_bound))
    
    filtered_df = df[mask]
    
    #  Print info about filtered directions
    remaining_directions = filtered_df[meas].dropna()
    print(f"Direction range in filtered data: {remaining_directions.min():.2f}° - {remaining_directions.max():.2f}°")
    
    return filtered_df

def exclude_house_sector(df):
    """
    Filter out data between 125° and 146.6° (house sector).
    
    Parameters:
    df (pd.DataFrame): The input DataFrame
    
    Returns:
    pd.DataFrame: DataFrame with house sector excluded
    """
    mask = ~((df['Vane100m_Mean'] >= 125) & (df['Vane100m_Mean'] <= 146.6))
    filtered_df = df[mask]
    
    remaining_directions = filtered_df['Vane100m_Mean'].dropna()
    print(f"Direction range after excluding house sector: {remaining_directions.min():.2f}° - {remaining_directions.max():.2f}°")
    
    return filtered_df

def filter_ice_on_cups(df, ice_threshold=2):


    """
    Filter the wind speed from the cup anemometer to exclude the possibility of ice on the cups.
    Ice typically forms when temperature is at or below 4°C (default threshold).
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with wind and temperature measurements
    ice_threshold (float): Temperature threshold for ice formation in °C
    
    Returns:
    pd.DataFrame: DataFrame with ice-filtered data
    tuple: (filtered DataFrame, number of points removed)
    """
    #  Create a copy to avoid modifying the original
    df_filtered = df.copy()
    
    cup_columns = ['Cup100m_Mean', 'Cup100m_Min', 'Cup100m_Max', 'Cup100m_Stdv',
                   'Cup114m_Mean', 'Cup114m_Min', 'Cup114m_Max', 'Cup114m_Stdv',
                   'Cup116m_Mean', 'Cup116m_Min', 'Cup116m_Max', 'Cup116m_Stdv']
    
    #  Create mask for potential icing conditions
    ice_mask = df_filtered['Temp100m_Mean'] <= ice_threshold
    
    #  Count original non-NaN values
    original_count = df_filtered[cup_columns].count().sum()
    
    #  Set cup measurements to NaN where temperature indicates possible icing
    for col in cup_columns:
        df_filtered.loc[ice_mask, col] = np.nan
    
    #  Count remaining non-NaN values
    remaining_count = df_filtered[cup_columns].count().sum()
    points_removed = original_count - remaining_count
    
    print(f"Ice filtering results:")
    print(f"Temperature threshold: {ice_threshold}°C")
    print(f"Total points removed: {points_removed}")
    print(f"Percentage of data removed: {(points_removed/original_count)*100:.2f}%")
    
    return df_filtered, points_removed


    """
    Remove outliers from specified columns of a DataFrame by replacing them with NaN.

    Parameters:
    lower_bound (float): Minimum acceptable value.
    upperbound (float): Maximum acceptable value.
    df (pd.DataFrame): The input DataFrame.
    columns (list, optional): List of column names to process. If None, all columns are processed.
    parameter (str, optional): Name of the parameter for plot titles (used if show_plot=True).
    unit (str, optional): Unit of the parameter for plot labels (used if show_plot=True).
    show_plot (bool, optional): Whether to show scatter plots before and after cleaning.

    Returns:
    pd.DataFrame: The modified DataFrame with outliers replaced by NaN.
    """
    #  If no specific columns are provided, process all columns
    if columns is None:
        columns = df.columns

    for column in columns:
        if show_plot:
            plot_scatter(f'{parameter or column} Before Cleaning', df.index, df[column],
                         f'{parameter or column} [{unit or ""}]', label_x='Date', label_y=f'{parameter or column} [{unit or ""}]', plot_bool=show_plot)
        
        #  Create a mask for outliers
        mask = (df[column] < lower_bound) | (df[column] > upperbound)
        
        #  Replace outliers with NaN
        df[column] = df[column].mask(mask).copy()
        
        if show_plot:
            plot_scatter(f'{parameter or column} After Cleaning', df.index, df[column],
                         f'{parameter or column} [{unit or ""}]', label_x='Date', label_y=f'{parameter or column} [{unit or ""}]', plot_bool=show_plot)
    
    return df

# %% Analysis functions
def analyze_wind_speeds(df, availability_threshold=None, title="Wind Speed Comparison", forced=False):
    """
    Perform regression analysis between cup and lidar measurements
    
    Parameters:
    df (DataFrame): Input data
    availability_threshold (float): Minimum availability threshold (0-100)
    title (str): Plot title
    forced (bool): If True, perform forced regression with offset fixed to zero
    
    Returns:
    None
    """
    #  Apply availability filter if specified
    if availability_threshold is not None:
        df = df[df['Available'] >= availability_threshold]
        # print(df['Available'])
    
    #  Get data without NaN values
    valid_data = df.dropna(subset=['Vane100m_Mean', 'Dir'])
    
    #  Prepare data for regression
    X = valid_data['Vane100m_Mean'].values.reshape(-1, 1)
    y = valid_data['Dir'].values
    
    #  Perform linear regression
    if forced:
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        offset = 0
    else:
        reg = LinearRegression().fit(X, y)
        offset = reg.intercept_
    
    gain = reg.coef_[0]
    r2 = reg.score(X, y)
    
    #  Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, reg.predict(X), color='red', linewidth=2)
    
    plt.xlabel('Vane Wind Direction [deg]')
    plt.ylabel('Lidar wind Direction [deg]')
    plt.title(f'{title}\nGain: {gain:.3f}, Offset: {offset:.3f}, R²: {r2:.3f}')
    plt.grid(True)
    plt.savefig(f'Pictures/{title}_lidar_cup_regression_{availability_threshold}.png')
    plt.show()

def analyze_wind_speeds_2(df, availability_threshold=None, title="Wind Speed Comparison", forced=False):
    """
    Perform regression analysis between cup and lidar measurements
    
    Parameters:
    df (DataFrame): Input data
    availability_threshold (float): Minimum availability threshold (0-100)
    title (str): Plot title
    forced (bool): If True, perform forced regression with offset fixed to zero
    
    Returns:
    None
    """
    #  Apply availability filter if specified
    if availability_threshold is not None:
        df = df[df['Available'] >= availability_threshold]
        # print(df['Available'])
    
    #  Get data without NaN values
    valid_data = df.dropna(subset=['Vane100m_Mean', 'Dir'])
    
    #  Filter out data points where the difference between Vane100m_Mean and Dir is greater than 2
    valid_data = valid_data[np.abs(valid_data['Vane100m_Mean'] - valid_data['Dir']) <= 20]
    
    #  Prepare data for regression
    X = valid_data['Vane100m_Mean'].values.reshape(-1, 1)
    y = valid_data['Dir'].values
    
    #  Perform linear regression
    if forced:
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        offset = 0
    else:
        reg = LinearRegression().fit(X, y)
        offset = reg.intercept_
    
    gain = reg.coef_[0]
    r2 = reg.score(X, y)
    
    #  Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, reg.predict(X), color='red', linewidth=2)
    
    plt.xlabel('Vane Wind Direction [deg]')
    plt.ylabel('Lidar wind Direction [deg]')
    plt.title(f'{title}\nGain: {gain:.3f}, Offset: {offset:.3f}, R²: {r2:.3f}')
    plt.grid(True)
    plt.savefig(f'Pictures/{title}_lidar_cup_regression_2_{availability_threshold}.png')
    plt.show()

# %% Physical calculation functions
def vapor_pressure(T_10min):
    """
    Calculate water vapor pressure using an empirical formula
    Args:
        T_10min: Temperature in Kelvin
    Returns:
        Vapor pressure in Pascal
    """
    return 0.0000205 * np.exp(0.0631846 * T_10min)

def calculate_rho(df,pressure, temperature, humidity_rel, vapor_pressure,R0 = 287.05,R_W = 461.5):
    """_summary_
    Calculate air density (rho) based on pressure, temperature, relative humidity, 
    and vapor pressure using the ideal gas law.
    
    Args:
    df (pd.DataFrame): A pandas DataFrame containing the input data.
    pressure (str): Column name in the DataFrame for atmospheric pressure in hPa.
    temperature (str): Column name in the DataFrame for temperature in Kelvin.
    humidity_rel (str): Column name in the DataFrame for relative humidity as a percentage.
    vapor_pressure (str): Column name in the DataFrame for vapor pressure in Pa.
    pd.Series: A pandas Series containing the calculated air density (rho) values.

    Notes:
    - The function assumes that the input DataFrame contains the specified columns.
    - The gas constants used in the calculation are:
        R0: Specific gas constant for dry air (287.05 J/(kg·K)).
        R_W: Specific gas constant for water vapor (461.5 J/(kg·K)).
    - Ensure that temperature is provided in Kelvin, pressure in hPa, and vapor pressure in Pa.
    - Relative humidity is converted from percentage to a fraction for the calculation.

    Returns: pd.Series containing the calculated air density values.

    """
    pressure = df[pressure]*100 # hPa to Pa
    temperature = df[temperature] # K (Use kelvin converted column)
    humidity_rel = df[humidity_rel]/100 # relative humidity to fraction
    vapor_pressure = df[vapor_pressure] # Pa
    # calculate the air density
    rho = 1/temperature*(pressure/R0-humidity_rel*vapor_pressure*(1/R0-1/R_W))
    return rho

# %% Normalization functions
def normalize_power_stall_regulated(df,P_avg,rho_avg,rho_0 = 1.225):
    """_summary_

    Args:
        df (DataFrame): containing data for power and air density
        P_avg (Str): column in data containing power
        rho_avg (Str): column in data containing air density
        
    Returns:
        Pn (Series): normalized power
    """
    Pn = df[P_avg]*(rho_0/df[rho_avg])
    return Pn

def normalize_wind_active_controlled(df,V_avg,rho_avg,rho_0 = 1.225):
    """_summary_

    Args:
        df (DataFrame): containing data for power and air density
        V_avg (Str): column in data containing wind speeds
        rho_avg (Str): column in data containing air density
        
    Returns:
        Pn (Series): normalized power
    """
    Vn = df[V_avg]*(df[rho_avg]/rho_0)**(1/3)
    return Vn


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
    #  Create bins in the dataframe
    df_copy['ws_bins'] = pd.cut(df_copy['norm_ws'], bins=ws_bins)
    
    #  Calculate statistics for each bin
    df_binned = df_copy.groupby('ws_bins', observed=True).agg({
        'norm_ws': ['mean', 'count'],
        'norm_power': ['mean', 'std','min','max'],
        'rho': 'mean',
        'AirAbs_70m': 'mean',
        'Press_enc_2m': 'mean',
        'RH_2m': 'mean'
    }).fillna(0)

    #  Flatten column names
    df_binned.columns = ['mean_ws', 'count', 'mean_power', 'std_power', 'min_power', 'max_power', 'mean_rho','mean_Temp_C', 'mean_pressure','mean_rel_humidity']
    df_binned = df_binned.reset_index()

    #  Get bin centers from the actual intervals in the data
    df_binned['binned_ws'] = df_binned['ws_bins'].apply(lambda x: (x.left + x.right) / 2)

    # Calculate the uncertainties and store them as columns in the dataframe
    df_binned, P_i, V_i, std_P_i, N_i, rho_i = calculate_uncertainties(df_binned) # store the columns of Power, Wind etc as variables for clarity in CP calculation

    # Calculate CP for wind speeds above 4 m/s (cut in)
    # df_binned['Cp'] = 0.0  #  Initialize with zeros
    # mask = df_binned['mean_ws'] > 4.0
    # df_binned.loc[mask, 'Cp'] = 1000*df_binned.loc[mask, 'mean_power']/(0.5 * rho_i * A * df_binned.loc[mask, 'mean_ws']**3)
    RHO_0 = 1.225
    df_binned['Cp'] = 1000*df_binned['mean_power']/(0.5 * RHO_0 * A * df_binned['mean_ws']**3)

    #  Reorder columns for readability
    columns = ['binned_ws', 'mean_ws', 'mean_power','mean_rho', 'mean_Temp_C', 'mean_pressure','mean_rel_humidity', 
               'std_power','min_power', 'max_power','count', 'Cp', 's_i', 'u_i', 'u_c']
    df_binned = df_binned[columns]
    
    return df_binned

# %% Sensitivity and uncertainty functions
def sensitivity_wind_speed(P_i, P_im1, V_i, V_im1):
    """Calculate sensitivity factor for wind speed."""
    return abs(P_i - P_im1) / abs(V_i - V_im1)

def sensitivity_temperature(V_i, c_V_i):
    """Calculate sensitivity factor for air temperature."""
    return c_V_i * V_i / (3 * 288.15)

def sensitivity_pressure(V_i, c_V_i):
    """Calculate sensitivity factor for air pressure."""
    return c_V_i * V_i / (3 * 1013)

def sensitivity_relative_humidity(V_i, c_V_i):
    """Calculate sensitivity factor for relative humidity."""
    return c_V_i * V_i * 0.0018

def uncertainty_power(P_i):
    """Calculate Category B uncertainty in electric power."""
    sens_factor_power = 1
    sensitivity_factor = np.full_like(P_i, sens_factor_power)
    u_Pi =  np.sqrt((0.002 * P_i) ** 2 + 3.7**2 + 0.3**2)
    return u_Pi, sensitivity_factor

def uncertainty_wind_speed_old(V_i, P_i):
    """Calculate Category B uncertainty in wind speed."""
    N = len(V_i)
    
    #  Calculate wind speed uncertainty
    u_Vi = np.sqrt(0.025**2 + (0.038 + 0.0038 * V_i) ** 2 +
                   (0.01 * V_i) ** 2 + (0.02 * V_i) ** 2 + (0.001 * V_i) ** 2)

    #  Initialize sensitivity factor array
    c_V_i = np.empty(N-1)  

    #  Compute sensitivity factors
    for i in range(1, N):
        c_V_i[i-1] = sensitivity_wind_speed(P_i[i], P_i[i-1], V_i[i], V_i[i-1])

    #  Optionally pad `c_V_i` to match `u_Vi` dimensions
    c_V_i = np.insert(c_V_i, 0, c_V_i[0])  #  Duplicates first value

    return u_Vi, c_V_i

def uncertainty_wind_speed(V_i, P_i):
    """Calculate Category B uncertainty in wind speed (vectorized)."""
    
    u_Vi = np.sqrt(0.025**2 + (0.038 + 0.0038 * V_i) ** 2 +
                   (0.01 * V_i) ** 2 + (0.02 * V_i) ** 2 + (0.001 * V_i) ** 2)

    #  Compute sensitivity factors without a loop
    c_V_i = np.abs(np.diff(P_i)) / np.abs(np.diff(V_i))

    #  Pad `c_V_i` to match `u_Vi` length
    c_V_i = np.insert(c_V_i, 0, c_V_i[0])

    return u_Vi, c_V_i

def uncertainty_temperature():
    """Category B uncertainty in air temperature (constant value)."""
    u_Ti = 0.6  #  in Kelvin
    sensitivity_factor = sensitivity_temperature(cV_i, V_i)
    return u_Ti, sensitivity_factor

def uncertainty_pressure():
    """Category B uncertainty in air pressure (constant value)."""
    u_Bi = 2.0  #  in hPa
    sensitivity_factor = sensitivity_pressure(cV_i, V_i)
    return u_Bi, sensitivity_factor

def uncertainty_relative_humidity():

    """Category B uncertainty in relative humidity (constant value)."""
    u_RHi = 0.63  #  in %RH
    sensitivity_factor = sensitivity_relative_humidity(cV_i, V_i)
    return u_RHi, sensitivity_factor

def calculate_uncertainties(df):

    df_binned = df.copy()
    # extract columns for easier uncertainty coding:
    P_i = df_binned['mean_power']
    V_i = df_binned['mean_ws']
    std_P_i = df_binned['std_power']
    N_i = df_binned['count']
    rho_i = df_binned['mean_rho']

    # Uncertainty constants
    uT = 0.6 # Kelvin
    uB = 2.0 # hPa
    uRH = 0.63/100 # RH
    
    #  Calculate category A uncertainty (si)
    df_binned['s_i'] =  std_P_i/ np.sqrt(N_i)
    s_i = df_binned['s_i']
    #  print(f'Category A uncertainty: {s_i}') # list of 35 uncertainties (one for each bin)
    
    #  Calculate category B uncertainty (ui)
    u_P_i, sens_factor_P_i = uncertainty_power(P_i) # uncertainty in power
    u_V_i, sens_factor_V_i = uncertainty_wind_speed(V_i,P_i)  # uncertainty in wind speed
    u_T_i, sens_factor_T_i = np.full_like(V_i, uT), sensitivity_temperature(V_i, sens_factor_V_i) # uncertainty in temperature
    u_B_i, sens_factor_B_i = np.full_like(V_i, uB), sensitivity_pressure(V_i, sens_factor_V_i) # uncertainty in pressure
    u_RH_i, sens_factor_RH_i = np.full_like(V_i, uRH), sensitivity_relative_humidity(V_i, sens_factor_V_i)  # uncertainty in humidity

    # print_uncertainties_and_sensitivity_factors()
    
    # calculate the combined category b uncertainty
    u_i = np.sqrt(sum(x**2 for x in [u_P_i, u_V_i, u_T_i, u_B_i, u_RH_i]))
    #  print(f'Category B uncertainty: {u_i}')

    df_binned['u_i'] = u_i
    
    #  #  Calculate combined uncertainty (uci)
    u_c = np.sqrt(u_i**2+s_i**2)
    df_binned['u_c'] = u_c
    #  print(f'Combined uncertainty: {u_c}')

    return df_binned, P_i, V_i, std_P_i, N_i, rho_i

# %% AEP functions
def Rayleigh_CDF(ws):
    """
    Computes the cumulative distribution function (CDF) of the Rayleigh wind speed distribution.

    Parameters:
    ws (float or array-like): Wind speed(s) for which the CDF is computed.

    Returns:
    np.ndarray: The Rayleigh CDF values for the given wind speed(s).
    """
    V_ave = np.arange(4, 12)  #  Rayleigh mean wind speeds from 4 to 11 m/s

    return 1 - np.exp(-np.pi / 4 * (ws / V_ave) ** 2)

def calculate_AEP(df_binned, Nh=8760):
    """
    Calculates the Annual Energy Production (AEP) using a binned wind speed distribution.

    Parameters:
    df_binned (pd.DataFrame): Dataframe containing binned wind speed (`mean_ws`) and power (`mean_power`).
    Nh (int, optional): Number of hours in a year (default is 8760).

    Returns:
    float: Estimated Annual Energy Production (AEP).
    """
    #  Extract wind speed and power bins
    Vi = df_binned['mean_ws']  #  Normalized and averaged wind speed in bin i
    Pi = df_binned['mean_power']  #  Normalized and averaged power output in bin i
    N = len(df_binned)  #  Number of bins
    s_i = df_binned['s_i']
    u_i = df_binned['u_i']

    sum_AEP = 0  #  Initialize summation for AEP integral
    sum_uncertainty_AEP = 0 #  Initialize summation for AEP uncertainty integral

    #  Compute AEP using numerical integration over wind speed bins
    for i in range(1, N):  #  Start from 1 to avoid index errors with i-1
        delta_F = Rayleigh_CDF(Vi.iloc[i]) - Rayleigh_CDF(Vi.iloc[i-1])  #  Change in Rayleigh CDF
        # print(f'delta_F = {delta_F}')
        avg_P = (Pi.iloc[i] + Pi.iloc[i-1]) / 2  #  Average power output between bins
        # print(f'avg_P : {avg_P}')
        sum_AEP += delta_F * avg_P  #  Contribution to total AEP

        # uncertainty AEP
        sum_uncertainty_AEP += delta_F*s_i.iloc[i]+(delta_F*u_i.iloc[i])**2

    sum_AEP /= 1000 # convert to MW
    AEP = sum_AEP * Nh  #  Scale by total hours in a year to get MWh

    uncertainty_AEP = Nh*np.sqrt(sum_uncertainty_AEP)/1000 # scale uncertainty to MW like AEP

    return AEP, uncertainty_AEP

def calculate_extrapolated_AEP(df_extrapolated, Nh=8760):
    """
    Calculates the Annual Energy Production (AEP) using a binned wind speed distribution.

    Parameters:
    df_binned (pd.DataFrame): Dataframe containing binned wind speed (`mean_ws`) and power (`mean_power`).
    Nh (int, optional): Number of hours in a year (default is 8760).

    Returns:
    float: Estimated Annual Energy Production (AEP).
    """
    #  Extract wind speed and power bins
    Vi = df_extrapolated['extrapolated_ws']  #  Normalized and averaged wind speed in bin i
    Pi = df_extrapolated['extrapolated_power']  #  Normalized and averaged power output in bin i
    N = len(df_extrapolated)  #  Number of bins
    

    sum_AEP = 0  #  Initialize summation for AEP integral
    

    #  Compute AEP using numerical integration over wind speed bins
    for i in range(1, N):  #  Start from 1 to avoid index errors with i-1
        delta_F = Rayleigh_CDF(Vi.iloc[i]) - Rayleigh_CDF(Vi.iloc[i-1])  #  Change in Rayleigh CDF
        # print(f'delta_F = {delta_F}')
        avg_P = (Pi.iloc[i] + Pi.iloc[i-1]) / 2  #  Average power output between bins
        # print(f'avg_P : {avg_P}')
        sum_AEP += delta_F * avg_P  #  Contribution to total AEP

    sum_AEP /= 1000 # convert to MW
    AEP = sum_AEP * Nh  #  Scale by total hours in a year to get MWh

    return AEP

