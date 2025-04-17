import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rainflow as rf

def load_csv_files_to_dict(folder_path):

    dataframes = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df_name = os.path.splitext(file_name)[0]  # Use filename without extension as key
            dataframes[df_name] = pd.read_csv(file_path, sep=';')
    return dataframes

def f(s, f_s, m):
    '''
    s is load signal in time series
    f_s is the sampling frequency
    m is the Wohler exponent

    compute rainflow count of the signal s
    DEL : 1-Hz Damage Equivalent Value
    '''

    # take dictionary, each timeseries, each MxA1, MyA1, MxR... count the N using rainflow

