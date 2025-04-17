import os
import pandas as pd
import matplotlib.pyplot as plt


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

import numpy as np  # Make sure numpy is imported at the top

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
    
    plot_scatter(f'Cleaning {parameter}', df.index, df[columns], label1 = f'{parameter} Before cleaning',
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

def power_plot(df_loads, title, show_plot=False):
    """
    Plot the power curve and rotor speed against wind speed.

    Parameters:
    df_loads (DataFrame): DataFrame containing the data to plot.
    title (str): Title for the plot.
    show_plot (bool): Whether to show the plot or save it as an image.

    Returns:
    None
    """
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # First subplot: Power curve (Active Power vs Wind Speed)
    axes[0].scatter(df_loads['Wsp_44m'], df_loads['po'], alpha=0.5, s=10)
    axes[0].set_title('Power Curve - Normal Operation Assessment')
    axes[0].set_xlabel('Wind Speed at 44m [m/s]')
    axes[0].set_ylabel('Mean Active Power [kW]')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Second subplot: Rotor Speed vs Wind Speed
    axes[1].scatter(df_loads['ROT'], df_loads['po'], alpha=0.5, s=10)
    axes[1].set_title('Power vs Rotor Speed')
    axes[1].set_ylabel('Mean Active Power [kW]')
    axes[1].set_xlabel('Rotor Speed [rpm]')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # save the figure 
    pictures_dir = os.path.join(os.path.dirname(__file__), 'Pictures')
    save_path = os.path.join(pictures_dir, f'{title}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)        

    if show_plot == True:
        plt.show()
    else:
        plt.close()


    
    
