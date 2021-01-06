import pandas as pd
import numpy as np
import csv


#This script loads unprocessed csv's and processes them
#This is done for the Rooflines and for the Heatmaps

def change_column_names(df):
    df.columns = ['NN_Topology', 'HWType', 'Datatype', 'Op mode', 'batch/thread/stream',
       'lat-sys [msec]', 'lat-comp [msec]', 'fps-system [fps]', 'fps-comp [fps]', 'tp-system [TOP/sec]', 'tp-comp [TOP/sec]',
       'top1 [%]', 'top5 [%]', 'Base_Pwr [W]', 'Idle_Pwr [W]', 'Full_Pwr [W]', 'GOPS',
       'PruningFactor', ' level', 'hw_peak_perf [TOP/sec]', 'hw_bandwidth [GBps]',
       'nn_total_operations', 'hw_datatype_prun_net', 'norm-lat-comp',
       'datatype_model', 'tag']
    return df
    
def get_df_by_column(filename, column):
    """
    This function:
        This function:
            -from the csv file gets a dataframe;
            -breaks that datframe into smaller dataframes according to unique values of the column given.
        
    Parameters
    ----------
    filename:  str
        Path to csv file containing the dataframe
    column: str
        The dataframe will be subsetted/broken according to this column's unique values   

    Returns
    -------
    datarames: list of dataframes
        Subsets of the bigger dataframe, subsetted by the column's unique values
    """    
    df = pd.read_csv(filename)
    df = change_column_names(df) #this will change the column names to include the units of the measuments
    unique_column_values = df[column].unique()
    dataframes = []
    for value in unique_column_values:
        dataframe = df[df[column] == value]
        dataframes.append(dataframe)
    
    return dataframes
#---------------------------------------------------------------------------------------------------------------------------------------