import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def plot_scatter_and_lines(measurement,df_mean,df_max = None,df_min = None,height = 100, unit ='Wind Speed (m/s)',plot_bool = False ):
    if plot_bool == True:
        plt.figure(figsize=(50,10))
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
        plt.savefig(f'Pictures/{measurement}_{height}m_10min_Time_Series_line.png')
        plt.show()


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
        plt.savefig(f'Pictures/{measurement}_{height}m_10min_Time_Series_scatter.png')
        plt.show()

def plot_scatter(title,df1x, df1y, label1, df2x = None, df2y = None, label2 = None, label_x = 'Time [s]', label_y ='Wind Speed (m/s)',plot_bool = False ):
    """_summary_

    Args:
        title (String): _description_
        df1x (df): _description_
        df1y (df): _description_
        label1 (String): _description_
        df2x (df, optional): _description_. Defaults to None.
        df2y (df, optional): _description_. Defaults to None.
        label2 (String, optional): _description_. Defaults to None.
        label_x (String, optional): _description_. Defaults to 'Time [s]'.
        label_y (str, optional): _description_. Defaults to 'Wind Speed (m/s)'.
        plot_bool (bool, optional): _description_. Defaults to False.
    """
    if plot_bool == True:
        plt.figure(figsize=(50,10))
        plt.scatter(df1x,df1y, label = label1,s=5)
        if df2x is not None:
            plt.scatter(df2x,df2y, label = label2,s=5)
        plt.xlabel(label_x, fontsize=20)
        plt.ylabel(label_y, fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(title, fontsize=25)
        plt.legend(fontsize=20)
        plt.savefig(f'Pictures/{title}_10min_Time_Series_scatter.png')
        plt.show()

def plot_all_measurements(df,plot_bool = False):
    if plot_bool == True:

        # #### Cup 116m
        plot_scatter_and_lines('Cup',df['Cup116m_Mean'],df['Cup116m_Max'],df['Cup116m_Min'],116,plot_bool=True)

        # #### Cup 114m
        plot_scatter_and_lines('Cup',df['Cup114m_Mean'],df['Cup114m_Max'],df['Cup114m_Min'],114,plot_bool=True)

        # #### Cup 100m
        plot_scatter_and_lines('Cup',df['Cup100m_Mean'],df['Cup100m_Max'],df['Cup100m_Min'],plot_bool=True)

        # #### Sonic
        plot_scatter_and_lines('Sonic Wind Speed Scalar',df['Sonic100m_Scalar_Mean'],df['Sonic100m_Scalar_Min'],df['Sonic100m_Scalar_Max'],plot_bool=True)
        plot_scatter_and_lines('Sonic Wind Direction',df['Sonic100m_Dir'], unit = 'Wind Direction [°]',plot_bool=True)

        # #### Termometer
        plot_scatter_and_lines('Thermometer',df['Temp100m_Mean'],df['Temp100m_Max'],df['Temp100m_Min'],unit = '°C',plot_bool=True)

        # #### Vane
        plot_scatter_and_lines('Vane Wind Direction',df['Vane100m_Mean'],df['Vane100m_Max'],df['Vane100m_Min'], unit = 'Wind Direction [°]',plot_bool=True)



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
    
    # If no columns specified, use all columns
    if columns is None:
        columns = df.columns
    
    # Replace zeros with NaN in specified columns
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
    
    # If no columns specified, use default wind speed columns
    if columns is None:
        # Find all lidar columns
        columns = [col for col in df.columns if 
                  ('Spd' in col and 'Mean' in col)]
    
    # Replace invalid wind speeds with NaN in specified columns
    for column in columns:
        try:
            # Create mask for invalid wind speeds (too low or too high)
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
    
    # If no columns specified, use default wind speed columns
    if columns is None:
        # Find all lidar columns
        columns = [col for col in df.columns if 
                  ('Cup' in col and 'Mean' in col)]
    
    # Replace invalid wind speeds with NaN in specified columns
    for column in columns:
        try:
            # Create mask for invalid wind speeds (too low or too high)
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
    
    # If no columns specified, use default wind speed columns
    if columns is None:
        # Find all lidar columns
        columns = [col for col in df.columns if 
                  ('Vane' in col and 'Mean' in col)]
    
    # Replace invalid wind speeds with NaN in specified columns
    for column in columns:
        try:
            # Create mask for invalid wind speeds (too low)
            mask_low = df_cleaned[column] < lower_bound
                        
            if mask_low.any():
                low_count = mask_low.sum()
            
                df_cleaned.loc[mask_low, column] = np.nan
                print(f"Column {column}:")
                print(f"  - Replaced {low_count} directional data below (<{lower_bound} m/s)")
                
        except Exception as e:
            print(f"Error processing column {column}: {str(e)}")
    
    return df_cleaned

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
    #plt.title(f'{measurement} {height}m 10min Time Series', fontsize=25)
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
    #plt.title(f'{measurement} {height}m 10min Time Series', fontsize=25)
    plt.title(f'Speed filter check means {title} {measurement} filtering', fontsize=25)
    plt.legend(fontsize=20)
    plt.savefig(f'Pictures/{measurement}_ws_filter.png')
    plt.show()


def replace_outliers_with_nan(df, columns=None, factor=3,  abs_threshold=None):

    """
    Replace outlier values with NaN in specified columns of a DataFrame.
    Outliers are defined as values whose rate of change exceeds the average
    rate of change by a factor specified by the user.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame
    columns (list, optional): List of column names to check. If None, checks all columns
    factor (float): Multiplier for average gradient to set threshold. Default is 3
    
    Returns:
    pd.DataFrame: DataFrame with outliers replaced by NaN
    """
    df_cleaned = df.copy()
    
    # If no columns specified, use all columns
    if columns is None:
        columns = df.columns
    
    for column in columns:
        # Calculate gradients (rate of change between consecutive values)
        gradients = df_cleaned[column].diff().abs()
        
        # Calculate average gradient (excluding NaN values)
        avg_gradient = gradients.mean()
        
        # Calculate maximum allowed gradient
        max_allowed_gradient = avg_gradient * (1 + factor)
        
        # Create mask for values that exceed the maximum allowed gradient
        mask = gradients > max_allowed_gradient
        
        # Replace outliers with NaN
        if mask.any():
            df_cleaned.loc[mask, column] = np.nan
            print(f"Replaced {mask.sum()} outlier values with NaN in column: {column}")
            print(f"Average gradient: {avg_gradient:.2f}, Max allowed gradient: {max_allowed_gradient:.2f}")

        if abs_threshold is not None:
            abs_outlier_mask = (df_cleaned[column].abs() > abs_threshold)
            df_cleaned.loc[abs_outlier_mask, column] = np.nan
            if abs_outlier_mask.any():
                print(f"Replaced {abs_outlier_mask.sum()} absolute outlier values with NaN in column: {column}")
    
    return df_cleaned


def filter_direction(df, highest_bound, lowest_bound, meas):
    """
    Filter the dataframe to only include rows where the wind direction is OUTSIDE 
    the turbine wake sector (346.47° - 13.24°).
    #house south west : 146.6 - 125

    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    # Handle the wrap-around at 360 degrees properly
    # Keep data where direction is NOT in the turbine wake sector
    mask = ~((df[meas] >= highest_bound) | (df[meas] <= lowest_bound))
    
    filtered_df = df[mask]
    
    # Print info about filtered directions
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

def plot_directional_check(df,title,highest_bound,lowest_bound, meas):
    direction_filter_lower_bound_list = [lowest_bound,lowest_bound]
    direction_filter_upper_bound_list = [highest_bound,highest_bound]
    y_values_list = [0,30]


    plt.figure(figsize=(50,10))
    plt.scatter(df[meas],df['Cup100m_Mean'], label = 'mean', s = 5)
    plt.axvline(x=lowest_bound, color='r', linestyle='--', label='direction filter lower bound')
    plt.axvline(x=highest_bound, color='g', linestyle='--', label='direction filter upper bound')
    #plt.scatter(direction_filter_lower_bound_list, y_values_list, label = 'direction filter', s = 5)
    #plt.scatter(direction_filter_upper_bound_list, y_values_list, label = 'direction filter', s = 5)
    plt.xlabel('Wind Direction [°]', fontsize=20)
    plt.ylabel('Wind Speed (m/s)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title(f'{measurement} {height}m 10min Time Series', fontsize=25)
    plt.title(f'Directional filter {meas} {title}', fontsize=25)
    plt.legend(fontsize=20)
    plt.savefig(f'Pictures/direction_filter_{meas}.png')
    plt.show()

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
    # Apply availability filter if specified
    if availability_threshold is not None:
        df = df[df['Available'] >= availability_threshold]
        #print(df['Available'])
    
    # Get data without NaN values
    valid_data = df.dropna(subset=['Cup100m_Mean', 'Spd'])
    
    # Prepare data for regression
    X = valid_data['Cup100m_Mean'].values.reshape(-1, 1)
    y = valid_data['Spd'].values
    
    # Perform linear regression
    if forced:
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        offset = 0
    else:
        reg = LinearRegression().fit(X, y)
        offset = reg.intercept_
    
    gain = reg.coef_[0]
    r2 = reg.score(X, y)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, reg.predict(X), color='red', linewidth=2)
    
    plt.xlabel('Cup Anemometer Speed [m/s]')
    plt.ylabel('Lidar Speed [m/s]')
    plt.title(f'{title}\nGain: {gain:.3f}, Offset: {offset:.3f}, R²: {r2:.3f}')
    plt.grid(True)
    plt.savefig(f'Pictures/lidar_cup_regression_{availability_threshold}.png')
    plt.show()


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
    # Create a copy to avoid modifying the original
    df_filtered = df.copy()
    
    cup_columns = ['Cup100m_Mean', 'Cup100m_Min', 'Cup100m_Max', 'Cup100m_Stdv',
                   'Cup114m_Mean', 'Cup114m_Min', 'Cup114m_Max', 'Cup114m_Stdv',
                   'Cup116m_Mean', 'Cup116m_Min', 'Cup116m_Max', 'Cup116m_Stdv']
    
    # Create mask for potential icing conditions
    ice_mask = df_filtered['Temp100m_Mean'] <= ice_threshold
    
    # Count original non-NaN values
    original_count = df_filtered[cup_columns].count().sum()
    
    # Set cup measurements to NaN where temperature indicates possible icing
    for col in cup_columns:
        df_filtered.loc[ice_mask, col] = np.nan
    
    # Count remaining non-NaN values
    remaining_count = df_filtered[cup_columns].count().sum()
    points_removed = original_count - remaining_count
    
    print(f"Ice filtering results:")
    print(f"Temperature threshold: {ice_threshold}°C")
    print(f"Total points removed: {points_removed}")
    print(f"Percentage of data removed: {(points_removed/original_count)*100:.2f}%")
    
    return df_filtered, points_removed