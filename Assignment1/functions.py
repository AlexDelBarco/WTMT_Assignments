import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
        plt.show()


        plt.figure(figsize=(50, 10))
        plt.scatter(df_mean.index, df_mean, label='mean', s=10)
        if df_max is not None and df_min is not None:
            plt.scatter(df_max.index, df_max, label='min', s=10)
            plt.scatter(df_min.index, df_min, label='max', s=10)
        plt.xlabel('Time', fontsize=20)
        plt.ylabel(unit, fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(f'{measurement} {height}m 10min Time Series', fontsize=25)
        plt.legend(fontsize=20)
        plt.show()

def plot_all_measurements(df,plot_bool = False):
    if plot_bool == True:

        # #### Cup 116m
        plot_scatter_and_lines('Cup',df['Cup116m_Mean'],df['Cup116m_Max'],df['Cup116m_Min'],116)

        # #### Cup 114m
        plot_scatter_and_lines('Cup',df['Cup114m_Mean'],df['Cup114m_Max'],df['Cup114m_Min'],114)

        # #### Cup 100m
        plot_scatter_and_lines('Cup',df['Cup100m_Mean'],df['Cup100m_Max'],df['Cup100m_Min'])

        # #### Sonic
        plot_scatter_and_lines('Sonic Wind Speed Scalar',df['Sonic100m_Scalar_Mean'],df['Sonic100m_Scalar_Min'],df['Sonic100m_Scalar_Max'])
        plot_scatter_and_lines('Sonic Wind Direction',df['Sonic100m_Dir'], unit = 'Wind Direction [°]')

        # #### Termometer
        plot_scatter_and_lines('Thermometer',df['Temp100m_Mean'],df['Temp100m_Max'],df['Temp100m_Min'],unit = '°C')

        # #### Vane
        plot_scatter_and_lines('Vane Wind Direction',df['Vane100m_Mean'],df['Vane100m_Max'],df['Vane100m_Min'], unit = 'Wind Direction [°]')

def plot_scatter(df1_x, df1_y, label1, xlabel, ylabel, title, df2_x=None, df2_y=None, label2=None, plot_bool=True):
    """
    Create a scatter plot with one or two datasets
    """
    if plot_bool:
        plt.figure(figsize=(50,10))
        plt.scatter(df1_x, df1_y, label=label1, s=10, alpha=0.5)    
        if df2_x is not None:
            plt.scatter(df2_x, df2_y, label=label2, s=10, alpha=0.5)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        plt.show()

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

def replace_outliers_with_nan(df, columns=None, factor=3):
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
    
    return df_cleaned