import pandas as pd
import re

import numpy as np
import csv

from IPython.display import display, HTML

def save_not_matched_data(df, machine_learning_task):
    """Method that saves the dataframe to a file."""
    df = df.sort_values(by='pairs')
    df.to_csv('data/no_match' + machine_learning_task + '.csv')


def label_point(x, y, val, ax, rot=0):
    """ from https://stackoverflow.com/questions/46027653/adding-labels-in-x-y-scatter-plot-with-seaborn"""
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']), rotation=rot)


def dataframe_contains(input_df: pd.DataFrame, column: str, value: str)->pd.DataFrame:
    """
    Given a dataframe, this function returns a subset of that dataframe by column
    
    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe from which the subset will be taken.       
    column : str
        Column name by which the subsetting will be taken.  
    value : str
        String that contains what we need from the column. 
        Eg.:'dog'
            'banana|apple|peach'

    Returns
    -------
    output_df:  pd.DataFrame
        This dataframe will be a subset from the input dataframe according to the column value given.  
        
    """
    output_df = input_df[input_df[column].str.contains(pat=value, case=False)]
    return output_df

#---------------------------------------------------------------------------------

def replace_data_df(df_: pd.DataFrame(), column:str, list_tuples_data_to_replace: list )-> pd.DataFrame():
    """Method to replace a substring inside a cell inside a dataframe
    Given a dataframe and a specific column, this method replaces a string for another, both from the list of tuples
       
    Parameters
    ----------
     df_: pd.DataFrame()
        Dataframe with data to be replaced.
    column: str
        Column whithin dataframe where all replacements will take place.
    list_tuples_data_to_replace: list
        List with tuples which will contain what to replace by what.
        Eg.:list_tuples_data_to_replace = [(a,b), (c,d), (...) ] -> 'a' will be replaced by 'b', 'c' will be replaced by 'd', and so on.

    Returns
    -------
    df_out: pd.DataFrame()
       Dataframe with all indicated values replaced.
        
    """
    df_out = df_.copy()
    for j, k in list_tuples_data_to_replace:
        df_out[column] = df_out[column].str.replace(j, k)
    return df_out

#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

#next methods are needed in the Raw Measurements ipynb

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
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------this will be needed for the TABLES OVERVIEW OF THE EXPERIMENTS------------------------------------------

def csv_to_array(filename:str)->list:
    """
    Function that reads a csv file and converts it to a list.
    
    Parameters
    ----------
    filename : str
        String with the file path.    

    Returns
    -------
    List with what was inside the csv file.  
        
    """
    list_data= []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            list_data.append(row)
    list_data = np.asarray(list_data)
    return list_data

def csv_to_dataframe_multiindex(filenames: list)->list:
    """
    Function that converts a list of csv's to a list of dataframes with multiindexing.
    
    Parameters
    ----------
    filenames : list of strings
        List of filepaths to the csv's to be converted.       

    Returns
    -------
    List of dataframes. Each dataframe has multiindexing in it.   
        
    """
    dataframes=[]
    for filename in filenames:
        # To read from a csv file into a 2D numpy array
        table = csv_to_array(filename)  
        #To transform to dataframe the first and second row will be header
        dataframe = pd.DataFrame(data=table[2:,:], columns=[table[0,0:], table[1,0:]]) 
        #To remove duplicates from first column
        dataframe.loc[dataframe.duplicated(dataframe.columns[0]) , dataframe.columns[0]] = ''  
        #To save all dataframes in here
        dataframes.append(dataframe)         
    return dataframes

def tableOverviewExperiments(filenames):
    dataframes = csv_to_dataframe_multiindex(filenames)
    for dataframe in dataframes:   
        return display(HTML(dataframe.to_html(index=False)))
    
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
    