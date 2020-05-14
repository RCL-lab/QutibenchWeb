import csv
import pandas as pd
import numpy as np
from IPython.display import display, HTML
#-------------------------------TABLES OVERVIEW OF THE EXPERIMENTS-----------------------------------------

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
    