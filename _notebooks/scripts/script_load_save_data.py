import pandas as pd
import numpy as np
import csv

#This script loads unprocessed csv's and processes them
#This is done for the Rooflines and for the Heatmaps

#---------------------------------------------HEATMAPS--------------------------------------------------------------------------------
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
    output_df = input_df[input_df[column].str.contains(value)]
    return output_df


#-----------------------------------------------------------------


def clean_csv_performance_predictions(path_csv: str):
    """
    This function 
        -reads the csv file;
        -computes the PERFORMANCE PREDICTIONS - HEATMAPS for the input csv file;
        -separates the dataframe into cifar 10, mnist and imagenet dataframes;
        -saves all three dataframes into separate csv files.
    
    Parameters
    ----------
    path_csv : str
        This is a path to the csv file that will be read by this function 

    Returns
    -------

    """
    ## Reading csv file and converting data to (Neural network, Platform, Value)
    df = pd.read_csv(path_csv)

    #----- Creating a dataframe with 3 columns x, y gop_frame
    cleanedList = [x for x in df.platform if x==x] # to take all the nans out
    x, y = np.meshgrid(df.model, cleanedList) 
    gop_frame, _ = np.meshgrid(df.gop_frame, cleanedList)

    #to crate a 1D array from each variable, creating a dataframe with 3 columns
    source = pd.DataFrame({'x': x.ravel(),     
                           'y': y.ravel(),
                           'gop_frame':gop_frame.ravel()}) #auxilary column

    #---Adding a fourth column: top_second  ---- auxilary column
    tops_second= []    #creating a lsit which will contain all top_second columns from the dataframe
    columns = list(df) # creating a list of dataframe columns 

    for i in columns:   
        if 'top_second' in i:
            tops_second.append(df[i])

    source['top_second'] = pd.concat(tops_second,ignore_index=True)

    #------Adding a fith column: values-----------
    source['values'] = source.top_second * 1000 / source.gop_frame

    #---Drop auxilary columns: gop_frame top_scond----
    source = source.drop(columns=['gop_frame','top_second'])
    source = source.round(0)

    #Separate dataframe into: IMAGENET, MNIST, CIFAR10 dataframes
    df_imagenet = dataframe_contains(input_df=source, column='x', value='GoogleNetv|MobileNetv1|ResNet50|EfficientNet')
    df_cifar10 = dataframe_contains(input_df=source, column='x', value='CNV')
    df_mnist = dataframe_contains(input_df=source, column='x', value='MLP')

    #tasks = ['imagenet', 'cifar-10','mnist']
    #path_imagenet = path + '/performance_prediction_' + tasks[0] + '.csv'

    #Saving above dataframes to csv file
    df_imagenet.to_csv('c:/Users/alinav/Documents/GitHub/QutibenchWeb/_notebooks/data/cleaned_csv/performance_prediction_imagenet.csv', index = False)
    df_cifar10.to_csv('c:/Users/alinav/Documents/GitHub/QutibenchWeb/_notebooks/data/cleaned_csv/performance_prediction_cifar10.csv', index = False)
    df_mnist.to_csv('c:/Users/alinav/Documents/GitHub/QutibenchWeb/_notebooks/data/cleaned_csv/performance_prediction_mnist.csv', index = False)
    

#----------------------------------------------------------------------------------------------------------------------------------------    
#--------------------RAW MEASUREMENTS---------------------------------------    

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