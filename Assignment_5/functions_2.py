import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rainflow import count_cycles
import re


def load_csv_files_to_dict(folder_path):

    dataframes = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df_name = os.path.splitext(file_name)[0]  # Use filename without extension as key
            dataframes[df_name] = pd.read_csv(file_path, sep=';')
    return dataframes


def compute_del(s, f_s, m):
    """
    Computes the 1-Hz Damage Equivalent Load (DEL)
      for a given load signal, frequency and Wohler exponent.

    Args:
        s (array-like): Load signal in time series.
        f_s (float): Sampling frequency of the signal (Hz).
        m (float): Wohler exponent.

    Returns:
        float: The 1-Hz Damage Equivalent Load (DEL).
    """
    # Convert to NumPy array, drop NaN values, verify not an empty series
    signal = np.array(s)
    signal = signal[~np.isnan(signal)]

    if len(signal) == 0:
        print("Error: The signal is empty")
        return None

    # Perform rainflow counting
    cycles = count_cycles(signal)

    # Calculate the damage by cycle
    damage = 0
    for amplitude, count in cycles:
        damage += (amplitude ** m) * count

    # Compute the DEL
    T = len(signal) / f_s  # Total duration of the signal in seconds
    DEL = (damage / T) ** (1 / m)

    return DEL, signal


def load_csv_with_units(file_path):

    # Read the first two rows to extract column names and units
    temp_df = pd.read_csv(file_path, sep=';', nrows=2, header=None)
    column_names = [
        f"{name}_[{unit}]" if not pd.isna(unit) else name
        for name, unit in zip(temp_df.iloc[0], temp_df.iloc[1])]

    # Load the rest of the file with the merged column names
    dataframe = pd.read_csv(file_path, sep=';', skiprows=2, names=column_names)
    return dataframe


def replace_load_stats_with_results(load_stats_tms, results_DEL):
    """
    Replace values in load_stats_tms with corresponding values from results_DEL.

    Parameters:
        load_stats_tms (pd.DataFrame): DataFrame containing load statistics with a 'time' column.
        results_DEL (dict): Dictionary of dictionaries containing DEL values.

    Returns:
        pd.DataFrame: Updated DataFrame with replaced values.
    """
    df = load_stats_tms.copy()  # Create a copy of the DataFrame to avoid modifying the original

    # Iterate over the columns in load_stats_tms that contain "DEL"
    for col in df.columns:
        if "DEL" in col:
            # Extract the signal name (e.g., MyTB, MxTB, etc.) and Wohler exponent (e.g., 3, 6, 9, 12)
            signal_name = col.split("DEL")[0].strip()
            signal_name = signal_name.replace("_", "")  # Remove any spaces
            #print(f"Signal name: {signal_name}")

            if signal_name == "Myaw":
                signal_name = "MYaw"
            
            if signal_name == "Mtilt":
                signal_name = "MTilt"

            # Extract only the numeric part of the Wohler exponent using a regular expression
            wohler_exponent_match = re.search(r'\d+', col.split("DEL")[1])
            print(f"Wohler exponent match: {wohler_exponent_match}")
            if wohler_exponent_match:
                wohler_exponent = int(wohler_exponent_match.group())
            else:
                print(f"Error: No valid Wohler exponent found in column name {col}")
                continue  # Skip if no valid Wohler exponent is found

            # Iterate over the rows in load_stats_tms
            for index, row in df.iterrows():
                # Extract the timestamp key from the current row
                timestamp_key = row["time"]

                # Check if the timestamp_key and signal_name exist in results_DEL
                if timestamp_key in results_DEL and f"{signal_name}_m{wohler_exponent}" in results_DEL[timestamp_key]:
                    # Replace the value in load_stats_tms with the corresponding value from results_DEL
                    df.at[index, col] = results_DEL[timestamp_key][f"{signal_name}_m{wohler_exponent}"]

    return df


def plot_sig(df, x_column, y_columns, title, show_plot=True, x_label=None, y_label=None):
    """
    Plot specified columns from a dataframe against a given x-axis column.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data to plot.
    x_column (str): The name of the column to use for the x-axis.
    y_columns (list or str): A list of column names (or a single column name) to plot on the y-axis.
    title (str): The title of the plot.
    save_path (str, optional): Path to save the plot as an image. If None, the plot is not saved.
    show_plot (bool, optional): Whether to display the plot. Default is True.

    Returns:
    None
    """
    # Ensure y_columns is a list, even if a single column is passed
    if isinstance(y_columns, str):
        y_columns = [y_columns]

    if x_column not in df.columns:
        print(f"Error: Column '{x_column}' not found in the dataframe.")
        return

    for y_column in y_columns:
        if y_column not in df.columns:
            print(f"Error: Column '{y_column}' not found in the dataframe.")
            return

    # Create the plot
    plt.figure(figsize=(10, 6))
    for y_column in y_columns:
        plt.plot(df[x_column], df[y_column], label=y_column, marker='o')

    # Add labels, title, and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(f'Figures/{title}.png', dpi=300, bbox_inches='tight')

    # Show the plot if required
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_sig_scatter(df, x_column, y_columns, title, show_plot=True, x_label=None, y_label=None):
    """
    Plot specified columns from a dataframe against a given x-axis column as a scatter plot.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data to plot.
    x_column (str): The name of the column to use for the x-axis.
    y_columns (list or str): A list of column names (or a single column name) to plot on the y-axis.
    title (str): The title of the plot.
    save_path (str, optional): Path to save the plot as an image. If None, the plot is not saved.
    show_plot (bool, optional): Whether to display the plot. Default is True.

    Returns:
    None
    """
    # Ensure y_columns is a list, even if a single column is passed
    if isinstance(y_columns, str):
        y_columns = [y_columns]

    if x_column not in df.columns:
        print(f"Error: Column '{x_column}' not found in the dataframe.")
        return

    for y_column in y_columns:
        if y_column not in df.columns:
            print(f"Error: Column '{y_column}' not found in the dataframe.")
            return

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    for y_column in y_columns:
        plt.scatter(df[x_column], df[y_column], label=y_column)

    # Add labels, title, and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(f'Figures/{title}_scatter.png', dpi=300, bbox_inches='tight')

    # Show the plot if required
    if show_plot:
        plt.show()
    else:
        plt.close()
