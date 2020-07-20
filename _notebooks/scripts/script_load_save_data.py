import pandas as pd
import numpy as np
import csv


#This script loads unprocessed csv's and processes them
#This is done for the Rooflines and for the Heatmaps

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
    unique_column_values = df[column].unique()
    dataframes = []
    for value in unique_column_values:
        dataframe = df[df[column] == value]
        dataframes.append(dataframe)
    
    return dataframes
#---------------------------------------------------------------------------------------------------------------------------------------