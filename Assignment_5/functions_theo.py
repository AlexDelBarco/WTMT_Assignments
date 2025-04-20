import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math

def load_csv_with_units(file_path):

    # Read the first two rows to extract column names and units
    temp_df = pd.read_csv(file_path, sep=';', nrows=2, header=None)
    column_names = [
        f"{name}_[{unit}]" if not pd.isna(unit) else name
        for name, unit in zip(temp_df.iloc[0], temp_df.iloc[1])]

    # Load the rest of the file with the merged column names
    dataframe = pd.read_csv(file_path, sep=';', skiprows=2, names=column_names)
    return dataframe


def load_csv_files_to_dict(folder_path):

    dataframes = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df_name = os.path.splitext(file_name)[0]  # Use filename without extension as key
            dataframes[df_name] = pd.read_csv(file_path, sep=';')
    return dataframes


def plot_stats(dataframe, column):

    if column not in dataframe.columns:
        print(f"Error: Column '{column}' not found in the DataFrame.")
        return

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['time'], dataframe[column], marker='o', linestyle='-', label=column)
    plt.xlabel('time')
    plt.ylabel(column)
    plt.title(f"{column} over time")
    plt.legend()
    plt.grid(True)

    safe_column_name = column.replace("[", "").replace("]", "").replace(" ", "_")
    plt.savefig(f'Figures/Plot_load_stats_{safe_column_name}.png')


def filter_outliers_row_based(df, columns, lower_bound, upper_bound, 
                              parameter, unit, show_plot=False):
    """
    Filter outliers in specified columns of a DataFrame by removing entire rows.
    
    Parameters:
    df (DataFrame): Input DataFrame
    columns (list): List of column names to check for outliers
    lower_bound (float): Minimum acceptable value (default: 4.0)
    upper_bound (float): Maximum acceptable value (default: 16.0)
    
    Returns:
    DataFrame: Cleaned DataFrame with entire rows removed if they contain outliers
    """
    df_cleaned = df.copy()
    
    # Create a mask to track rows with any outliers
    master_mask = pd.Series(False, index=df.index)
    
    # Track statistics for reporting
    stats = {}
    
    # Check each column for outliers and update the master mask
    for column in columns:
        try:
            # Create mask for outliers (too low or too high)
            mask_low = df_cleaned[column] < lower_bound
            mask_high = df_cleaned[column] > upper_bound
            mask_combined = mask_low | mask_high
            
            # Update master mask to include any rows with outliers in this column
            master_mask = master_mask | mask_combined
            
            # Store statistics for reporting
            stats[column] = {
                'low_count': mask_low.sum(),
                'high_count': mask_high.sum(),
                'total': mask_combined.sum()
            }
        except Exception as e:
            print(f"Error processing column {column}: {str(e)}")
    
    # Count rows to be removed
    rows_to_remove = master_mask.sum()
    
    # Remove rows with any outliers
    df_cleaned = df_cleaned[~master_mask]
    
    # Print summary
    print(f"Removed {rows_to_remove} entire rows containing outliers:")
    for column, counts in stats.items():
        if counts['total'] > 0:
            print(f"  Column {column}:")
            print(f"    - Found {counts['low_count']} low values (<{lower_bound} m/s)")
            print(f"    - Found {counts['high_count']} high values (>{upper_bound} m/s)")
    
    plot_scatter(f'Cleaning_{parameter}', df.index, df[columns], label1 = f'{parameter} Before cleaning',
                  label_x='Date', label_y=f'{parameter} [{unit}] ',
                    plot_bool=show_plot, df2x = df_cleaned.index, df2y = df_cleaned[columns],
                      label2 = f'{parameter} [{unit}] After cleaning', dot_size1=100, dot_size2 = 80)

    
    return df_cleaned

def filter_outliers_cell_based(lower_bound, upperbound, df, columns=None, parameter=None, unit=None, show_plot=False):
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

def calculate_power_curve_bins(df, ws_bins):
    """
    Calculate binned statistics and uncertainties for power curve determination.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with normalized wind speed and power
    ws_bins : numpy.ndarray
        Wind speed bin edges
    """
    df_copy = df.copy()
    #  Create bins in the dataframe
    df_copy['ws_bin'] = pd.cut(df_copy['Wsp_44m'], bins=ws_bins, include_lowest=True, right=False)

    # Group by the bins and calculate means for every column in the df
    for col in df_copy.columns:
        if col != 'ws_bin':
            if col != 'datetime':
                if col != 'rname':
                    df_copy[col] = df_copy.groupby('ws_bin')[col].transform(lambda x: x.fillna(x.mean()))
    
    df_binned = df_copy

    
    return df_binned

# ... existing code ...

def power_plot(df_loads, 
               title1, x_col1, y_col1, x_label1, y_label1, 
               title2, x_col2, y_col2, x_label2, y_label2, 
               show_plot=False):
     """
     Plot two related scatter plots side-by-side, typically for power curve analysis.
 
     Parameters:
     df_loads (DataFrame): DataFrame containing the data to plot.
     title1 (str): Title for the first subplot.
     x_col1 (str): Column name for the x-axis of the first subplot.
     y_col1 (str): Column name for the y-axis of the first subplot.
     x_label1 (str): X-axis label for the first subplot.
     y_label1 (str): Y-axis label for the first subplot.
     title2 (str): Title for the second subplot.
     x_col2 (str): Column name for the x-axis of the second subplot.
     y_col2 (str): Column name for the y-axis of the second subplot.
     x_label2 (str): X-axis label for the second subplot.
     y_label2 (str): Y-axis label for the second subplot.
     show_plot (bool): Whether to show the plot or save it as an image.
 
     Returns:
     None
     """
     
     # Create a figure with two subplots
     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
     # First subplot
     axes[0].scatter(df_loads[x_col1], df_loads[y_col1], alpha=0.5, s=10)
     axes[0].set_title(title1)
     axes[0].set_xlabel(x_label1)
     axes[0].set_ylabel(y_label1)
     axes[0].grid(True, linestyle='--', alpha=0.7)
 
     # Second subplot
     axes[1].scatter(df_loads[x_col2], df_loads[y_col2], alpha=0.5, s=10)
     axes[1].set_title(title2)
     axes[1].set_xlabel(x_label2)
     axes[1].set_ylabel(y_label2)
     axes[1].grid(True, linestyle='--', alpha=0.7)
 
     plt.tight_layout()
 
     # save the figure 
     # Use a generic title for saving based on the first plot's title
     safe_title = title1.replace(" ", "_").replace("[", "").replace("]", "") 
     pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures')
     save_path = os.path.join(pictures_dir, f'{safe_title}.png')
     os.makedirs(os.path.dirname(save_path), exist_ok=True)
     plt.savefig(save_path)        
 
     if show_plot == True:
         plt.show()
     else:
         plt.close()


def analyze_wind_vane_period(df, start_date_str, end_date_str, buffer_days=20, show_plot=True):
    """
    Analyze and visualize wind vane data and its rate of change for a specified period.
    Includes line plots of direction and a plot of the circular difference over time.
    Returns the filtered and sorted dataframe for the period, including the 'Wdir_diff_circ' column.
    """
    # --- Date Formatting and Filtering (remains the same) ---
    start_date_str = str(start_date_str)
    end_date_str = str(end_date_str)
    if len(start_date_str) < 12:
        start_date_str = start_date_str.ljust(12, '0')
    if len(end_date_str) < 12:
        end_date_str = end_date_str.ljust(12, '0')

    start_datetime_invalid = pd.to_datetime(start_date_str, format='%Y%m%d%H%M')
    end_datetime_invalid = pd.to_datetime(end_date_str, format='%Y%m%d%H%M')
    start_datetime_buffer = start_datetime_invalid - pd.Timedelta(days=buffer_days)
    end_datetime_buffer = end_datetime_invalid + pd.Timedelta(days=buffer_days)
    start_rname_buffer = float(start_datetime_buffer.strftime('%Y%m%d%H%M'))
    end_rname_buffer = float(end_datetime_buffer.strftime('%Y%m%d%H%M'))

    mask = (df['rname'].astype(float) >= start_rname_buffer) & \
           (df['rname'].astype(float) <= end_rname_buffer)
    df_period = df.loc[mask].copy()

    # --- Calculate Difference Column (Moved outside show_plot) ---
    # Ensure the dataframe is not empty before proceeding
    if df_period.empty:
        print(f"Warning: No data found for period {start_date_str} to {end_date_str} with buffer {buffer_days} days. Returning empty DataFrame.")
        # Add the column definition even if empty, to avoid schema issues later
        df_period['Wdir_diff_circ'] = np.nan
        return df_period

    # Sort and calculate difference
    df_period_sorted = df_period.sort_values('datetime').copy()
    if len(df_period_sorted) >= 2:
        df_period_sorted['Wdir_diff_circ'] = df_period_sorted['Wdir_41m'].diff().apply(shortest_angle_diff)
    else:
        # Handle case with only 0 or 1 row
        df_period_sorted['Wdir_diff_circ'] = np.nan


    # --- Plotting Section (Only if show_plot is True) ---
    if show_plot:
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))

        # Sort full dataframe for plotting comparison
        df_sorted = df.sort_values('datetime').copy()
        # Calculate diff for full df plot (only needed if plotting)
        if len(df_sorted) >= 2:
             df_sorted['Wdir_diff_circ'] = df_sorted['Wdir_41m'].diff().apply(shortest_angle_diff)
        else:
             df_sorted['Wdir_diff_circ'] = np.nan


        # --- Subplot 1: Full time period - Wind Direction ---
        axes[0].plot(df_sorted['datetime'], df_sorted['Wdir_41m'],
                    linewidth=0.8, alpha=0.7, color='blue')
        axes[0].axvline(x=start_datetime_invalid, color='r', linestyle='--', label='Start of invalid period')
        axes[0].axvline(x=end_datetime_invalid, color='r', linestyle='--', label='End of invalid period')
        axes[0].set_title('Wind Direction - Full Period')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Wind Direction [degrees]')
        axes[0].grid(True, linestyle='--', alpha=0.3)
        axes[0].legend()

        # --- Subplot 2: Buffer period - Wind Direction ---
        axes[1].plot(df_period_sorted['datetime'], df_period_sorted['Wdir_41m'],
                    linewidth=1.0, alpha=0.7, color='blue')
        axes[1].axvline(x=start_datetime_invalid, color='r', linestyle='--', label='Start of invalid period')
        axes[1].axvline(x=end_datetime_invalid, color='r', linestyle='--', label='End of invalid period')
        axes[1].set_title(f'Wind Direction - Invalid Period ±{buffer_days} days')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Wind Direction [degrees]')
        axes[1].grid(True, linestyle='--', alpha=0.3)
        axes[1].legend()

        # --- Subplot 3: Buffer period - Circular Difference ---
        axes[2].plot(df_period_sorted['datetime'], df_period_sorted['Wdir_diff_circ'],
                    linewidth=1.0, alpha=0.7, color='green', marker=None, linestyle='-') # Line plot
        axes[2].axvline(x=start_datetime_invalid, color='r', linestyle='--', label='Start of invalid period')
        axes[2].axvline(x=end_datetime_invalid, color='r', linestyle='--', label='End of invalid period')
        axes[2].set_title(f'Wind Direction Change (Abs Diff) - Invalid Period ±{buffer_days} days')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Abs. Circular Diff [degrees]')
        axes[2].grid(True, linestyle='--', alpha=0.3)
        axes[2].legend()
        axes[2].set_ylim(bottom=-5) # Start y-axis slightly below 0

        # --- General Figure Formatting ---
        fig.autofmt_xdate()
        plt.tight_layout()

        # Save the figure
        plot_title = f'Wind_Direction_Analysis_and_Diff_{start_date_str}-{end_date_str}'
        pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures')
        # Sanitize title before saving
        safe_plot_title = plot_title.replace(':', '_').replace('/', '_').replace('\\', '_')
        save_path = os.path.join(pictures_dir, f'{safe_plot_title}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig) # Close this figure

        # --- Histogram Section ---
        fig_hist, axes_hist = plt.subplots(1, 2, figsize=(16, 6))
        axes_hist[0].hist(df_sorted['Wdir_41m'].dropna(), bins=36, range=(0, 360),
                         alpha=0.7, color='blue', edgecolor='black')
        axes_hist[0].set_title('Wind Direction Distribution - Full Period')
        axes_hist[0].set_xlabel('Wind Direction [deg]')
        axes_hist[0].set_ylabel('Frequency')
        axes_hist[0].grid(True, linestyle='--', alpha=0.3)

        axes_hist[1].hist(df_period_sorted['Wdir_41m'].dropna(), bins=36, range=(0, 360),
                          alpha=0.7, color='red', edgecolor='black')
        axes_hist[1].set_title(f'Wind Direction Distribution - Suspect Period')
        axes_hist[1].set_xlabel('Wind Direction [deg]')
        axes_hist[1].set_ylabel('Frequency')
        axes_hist[1].grid(True, linestyle='.', alpha=0.3)

        plt.tight_layout()
        hist_plot_title = f'Wind_Direction_Histograms_{start_date_str}-{end_date_str}'
        # Sanitize title before saving
        safe_hist_title = hist_plot_title.replace(':', '_').replace('/', '_').replace('\\', '_')
        hist_save_path = os.path.join(pictures_dir, f'{safe_hist_title}.png')
        plt.savefig(hist_save_path)
        plt.close(fig_hist) # Close the histogram figure

        # Print the date ranges (optional, can be removed if not needed)
        print(f"Full period plotted: {df_sorted['datetime'].min()} to {df_sorted['datetime'].max()}")
        print(f"Invalid period marked: {start_datetime_invalid.strftime('%Y-%m-%d %H:%M')} to {end_datetime_invalid.strftime('%Y-%m-%d %H:%M')}")
        print(f"Buffer period plotted: {df_period_sorted['datetime'].min()} to {df_period_sorted['datetime'].max()}")

    # --- Return Value ---
    # Return the sorted dataframe for the period, which now always includes 'Wdir_diff_circ'
    return df_period_sorted

def shortest_angle_diff(d):
    """Calculates the shortest difference between two angles (0-360)."""
    if pd.isna(d):
        return np.nan
    abs_d = abs(d)
    return min(abs_d, 360 - abs_d)

def power_plot(df_loads, title_1, x1, y1, xlabel_1, ylabel_1, 
               title_2, x2, y2, xlabel_2, ylabel_2, show_plot=False):
    """
    Plot the power curve and rotor speed against wind speed.

    Parameters:
    df_loads (DataFrame): DataFrame containing the data to plot.
    title_1 (str): Title for the plot.
    x1 (str): Column name for the x-axis of the first plot.
    y1 (str): Column name for the y-axis of the first plot.
    xlabel_1 (str): Label for the x-axis of the first plot.
    ylabel_1 (str): Label for the y-axis of the first plot.
    title_2 (str): Title for the second plot.
    x2 (str): Column name for the x-axis of the second plot.
    y2 (str): Column name for the y-axis of the second plot.
    xlabel_2 (str): Label for the x-axis of the second plot.
    ylabel_2 (str): Label for the y-axis of the second plot.
    show_plot (bool): Whether to show the plot or save it as an image.


    Returns:
    None
    """
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # First subplot: Power curve (Active Power vs Wind Speed)
    axes[0].scatter(df_loads[x1], df_loads[y1], alpha=0.5, s=10)
    axes[0].set_title(title_1)
    axes[0].set_xlabel(xlabel_1)
    axes[0].set_ylabel(ylabel_1)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Second subplot: Rotor Speed vs Wind Speed
    axes[1].scatter(df_loads[x2], df_loads[y2], alpha=0.5, s=10)
    axes[1].set_title(title_2)
    axes[1].set_ylabel(ylabel_2)
    axes[1].set_xlabel(xlabel_2)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # save the figure 
    pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures')
    save_path = os.path.join(pictures_dir, f'{title_1}_{title_2}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)        

    if show_plot == True:
        plt.show()
    else:
        plt.close()


def plot_lines(title, df1x, df1y, label1, label_x='Time [s]', label_y='Wind Speed (m/s)', 
               plot_bool=False, df2x=None, df2y=None, label2=None, df3x=None, df3y=None, 
               label3=None, df4x=None, df4y=None, label4=None, linestyle1='-', linestyle2='-',
               linestyle3='-', linestyle4='-', linewidth1=1, linewidth2=1, linewidth3=1, linewidth4=1):
    """Plot line data with multiple datasets.
    
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
        df4x (array-like, optional): X values for fourth dataset
        df4y (array-like, optional): Y values for fourth dataset
        label4 (str, optional): Label for fourth dataset
        linestyle1-4 (str, optional): Line style for each dataset ('-', '--', '-.', ':')
        linewidth1-4 (int, optional): Line width for each dataset
    """
    
    plt.figure(figsize=(16*2, 9*2))
    
    # Plot first dataset
    plt.plot(df1x, df1y, label=label1, linestyle=linestyle1, linewidth=linewidth1)
        
    # Plot second dataset if provided
    if df2x is not None:
        plt.plot(df2x, df2y, label=label2, linestyle=linestyle2, linewidth=linewidth2)
            
    # Plot third dataset if provided
    if df3x is not None:
        plt.plot(df3x, df3y, label=label3, linestyle=linestyle3, linewidth=linewidth3)
    
    # Plot fourth dataset if provided
    if df4x is not None:
        plt.plot(df4x, df4y, label=label4, linestyle=linestyle4, linewidth=linewidth4)

    plt.xlabel(label_x, fontsize=20)
    plt.ylabel(label_y, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    plt.title(title, fontsize=25)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures')
    save_path = os.path.join(pictures_dir, f'{title}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)        
    
    if plot_bool == True:
        plt.show()
    else:
        plt.close()

def bin_data_by_windspeed(df, ws_col='Wsp_44m', bin_width=1.0):
    """
    Bins a DataFrame by a specified wind speed column, centering bins around
    integer values (or multiples of bin_width/2), and calculates the mean
    of all other numeric columns within each bin. Also includes the mean of the
    wind speed column itself for verification.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    ws_col : str, optional
        The name of the column containing wind speed data. Defaults to 'Wsp_44m'.
    bin_width : float, optional
        The desired width of the wind speed bins. Defaults to 1.0.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame containing the binned statistics (mean values per bin).
        Includes bin intervals, bin centers (which should be close to integers
        if bin_width=1.0), count per bin, the mean of the wind speed column
        (ws_col_mean), and mean values for other numeric columns.
        Returns an empty DataFrame if input is empty or ws_col is invalid.
    """
    if df.empty:
        print("Input DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    if ws_col not in df.columns:
        print(f"Error: Wind speed column '{ws_col}' not found in DataFrame. Returning empty DataFrame.")
        return pd.DataFrame()

    if not pd.api.types.is_numeric_dtype(df[ws_col]):
         print(f"Error: Wind speed column '{ws_col}' must be numeric. Returning empty DataFrame.")
         return pd.DataFrame()

    df_copy = df.copy()

    # --- Determine Bin Range (Shifted Edges) ---
    min_val = df_copy[ws_col].min()
    max_val = df_copy[ws_col].max()

    if pd.isna(min_val) or pd.isna(max_val):
        print("Error: Could not determine valid min/max for wind speed column. Check for NaNs. Returning empty DataFrame.")
        return pd.DataFrame()

    # Calculate offset for centering bins
    offset = bin_width / 2.0

    # Determine the first edge (e.g., if min is 4.1 and width is 1, start at 3.5)
    start_edge = math.floor(min_val - offset) + offset
    # Determine the last edge needed (e.g., if max is 18.2 and width is 1, need up to 18.5)
    # The bin centered at ceil(max_val) would be [ceil(max_val)-offset, ceil(max_val)+offset)
    # So the final edge needs to be ceil(max_val) + offset
    end_edge = math.ceil(max_val + offset) - offset # Corrected: ensure last bin covers max_val

    # Generate bin edges using the shifted start and end
    # Add a small epsilon to end_edge in arange to ensure inclusion due to floating point
    ws_bins = np.arange(start_edge, end_edge + bin_width, bin_width)

    if len(ws_bins) < 2:
        print(f"Error: Not enough bins generated with width {bin_width}. Check data range. Returning empty DataFrame.")
        print(f"Calculated start_edge: {start_edge}, end_edge: {end_edge}")
        return pd.DataFrame()

    print(f"Generated bin edges (centered): {ws_bins}")

    # --- Binning ---
    # Create bins based on the wind speed column. right=False means [start, end)
    df_copy['ws_bin_intervals'] = pd.cut(df_copy[ws_col], bins=ws_bins, right=False, include_lowest=True)

    # --- Dynamic Aggregation ---
    # Start with count AND mean for the wind speed column itself
    agg_dict = {
        ws_col: ['count', 'mean'] # Calculate count and mean for the binning column
    }

    # Find other numeric columns to aggregate (calculate their mean)
    other_numeric_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
    # Remove the primary ws_col as it's handled separately
    if ws_col in other_numeric_cols:
        other_numeric_cols.remove(ws_col)

    # Add mean aggregation for these other numeric columns
    for col in other_numeric_cols:
        agg_dict[col] = 'mean'

    # --- Grouping and Aggregating ---
    # Group by the created bins and apply the dynamic aggregations
    try:
        # Use observed=False to include all defined bins initially
        df_binned = df_copy.groupby('ws_bin_intervals', observed=False).agg(agg_dict)
    except Exception as e:
        print(f"Error during aggregation: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

    # --- Formatting Output ---
    # Flatten the multi-index columns
    df_binned.columns = ['_'.join(col).strip('_') for col in df_binned.columns.values]

    # Rename the specific columns for clarity
    ws_col_count_name = f'{ws_col}_count'
    ws_col_mean_name = f'{ws_col}_mean'
    df_binned = df_binned.rename(columns={
        ws_col_count_name: 'count',
        ws_col_mean_name: ws_col_mean_name
    })

    # Filter out bins with zero count AFTER aggregation
    df_binned = df_binned[df_binned['count'] > 0].copy()

    # Reset index to turn 'ws_bin_intervals' into a regular column
    df_binned = df_binned.reset_index()

    # Calculate bin centers from the interval column
    if 'ws_bin_intervals' in df_binned.columns and isinstance(df_binned['ws_bin_intervals'].dtype, pd.IntervalDtype):
         df_binned['ws_bin_center'] = df_binned['ws_bin_intervals'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
         # Reorder columns (optional)
         cols_ordered = ['ws_bin_intervals', 'ws_bin_center', 'count', ws_col_mean_name] + \
                        [col for col in df_binned.columns if col not in ['ws_bin_intervals', 'ws_bin_center', 'count', ws_col_mean_name]]
         df_binned = df_binned[cols_ordered]
    else:
        print("Warning: Could not calculate bin centers. 'ws_bin_intervals' column issue.")


    return df_binned

# ... rest of the functions ...