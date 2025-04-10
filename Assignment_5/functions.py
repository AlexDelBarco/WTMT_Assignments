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


def plot_load_stats(dataframe, column):

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


