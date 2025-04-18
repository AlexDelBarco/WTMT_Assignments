import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rainflow import count_cycles

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

    return DEL

def load_csv_with_units(file_path):

    # Read the first two rows to extract column names and units
    temp_df = pd.read_csv(file_path, sep=';', nrows=2, header=None)
    column_names = [
        f"{name}_[{unit}]" if not pd.isna(unit) else name
        for name, unit in zip(temp_df.iloc[0], temp_df.iloc[1])]

    # Load the rest of the file with the merged column names
    dataframe = pd.read_csv(file_path, sep=';', skiprows=2, names=column_names)
    return dataframe