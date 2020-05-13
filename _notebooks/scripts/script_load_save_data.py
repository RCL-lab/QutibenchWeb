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
        Inpput dataframe from which the subset will be taken.       
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
    df_imagenet.to_csv('c:/Users/alinav/Documents/GitHub/Qutibench_Web/_notebooks/data/cleaned_csv/performance_prediction_imagenet.csv', index = False)
    df_cifar10.to_csv('c:/Users/alinav/Documents/GitHub/Qutibench_Web/_notebooks/data/cleaned_csv/performance_prediction_cifar10.csv', index = False)
    df_mnist.to_csv('c:/Users/alinav/Documents/GitHub/Qutibench_Web/_notebooks/data/cleaned_csv/performance_prediction_mnist.csv', index = False)
    source.to_csv('c:/Users/alinav/Documents/GitHub/Qutibench_Web/_notebooks/data/cleaned_csv/performance_prediction_imagenet_mnist_cifar10.csv', index = False)
    
#-------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------ROOFLINES--------------------------------------------------------------------------

def clean_csv_rooflines(path_topologies, path_hardware):
    """
    This function:
        -Loads the unprocessed csv's with topologies details and hardware details
        -creates a clean ready to be plotted dataframe for topologies like the following:
                                  Name       arith_intens(x axis)  performance(y axis)
                                 AlexNet          2995                  75.0
                                 AlexNet          2995                 100.0
                                 AlexNet          2995                   0.1
                                 AlexNet          2995                  25.0
                                 CNV                76                  25.0   ...
                               
        -creates a clean, ready to be plotted dataframe for hardware platforms like the following:
                                  Name           arith_intens(x axis)  performance(y axis)
                            Ultra96 DPU INT8            17.1                  0.072846
                            Ultra96 DPU INT8           226.1                  0.960000
                            Ultra96 DPU INT8          160000                  0.960000
                            ZCU104 INT8                  4.1                  0.078720  ...
        -concatenates these two dataframes
        -saves this to a csv
        
    Parameters
    ----------
    path_topologies:  str
        Path to csv file containing topologies.
    path_hardware: str
        Path to csv file containing hardware platforms.  

    Returns
    -------     
    """
    ## Loading Hardware platforms and Neural networks csv
    df_topology = pd.read_csv(path_topologies, sep=',')
    df_hardware = pd.read_csv(path_hardware, sep=',')


    ## Calculate the Arithmetic intensity (x axis) for each NN based on Fwd ops and Total params
    n_bytes=1 
    calc_arith = lambda operations, params, n_bytes: operations/(params*n_bytes)

    for index, row in df_topology.iterrows():             #nditer is a iterator object   
            #calculate the arith intensity with the lambda function
        arith_intens = calc_arith(row['Fwd Ops'], row['Total Params'], n_bytes) 
            #saving it to the dataframe
        df_topology.at[index, 'arith_intens'] = arith_intens              

    #to duplicate the dataframe so each row with (Platform, arith_intens) 
    #will be filled with 100 and then 0s to plot the vertical line later    
    df_topology = pd.concat([df_topology, df_topology])
    df_topology = pd.concat([df_topology, df_topology])
    df_topology = df_topology.drop(columns=['Total Params','Fwd Ops']) #deleting unnecessary columns (Fwd ops and Total params)

    ## Preparing the NNs dataset to be ploted as vertical lines later
    # creating a y list [100,100,100,...75,75,...25,25,25...0.0001,0.0001] to plot a vertical line later
    df_topology['performance'] = [100]*round((len(df_topology.index))/4)   +   [25]*round((len(df_topology.index))/4)  +  [75]*round((len(df_topology.index))/4)  +   [0.1]*round((len(df_topology.index))/4) 


    ## Calculating the rooflines (y axis) for each hardware platform (dataframe = df_topology + df)
    #--------------------------------Calculating the values to plot for the roofline model-----------
    maxX=160000
    x_axis = np.arange(0.1,maxX,1) #to create a list that represents the x axis with numbers between 0 and 1000
    df_hardw_clean = pd.DataFrame(columns=['Name','arith_intens','performance']) 

    for index, row in df_hardware.iterrows():             #nditer is a iterator object 
        FIRST_POINT = True
        for i in np.nditer(x_axis):
            y_point = row['Bandwidth'] * i
            if FIRST_POINT & (y_point > 0.05) :
                df_hardw_clean = df_hardw_clean.append([pd.Series([row['Name'],i, y_point],df_hardw_clean.columns)], ignore_index=True)
                FIRST_POINT=False
            if y_point > row['Peak_Performance']:
                df_hardw_clean = df_hardw_clean.append([pd.Series([row['Name'],i,    row['Peak_Performance']],df_hardw_clean.columns)], ignore_index=True)
                df_hardw_clean = df_hardw_clean.append([pd.Series([row['Name'],maxX, row['Peak_Performance']],df_hardw_clean.columns)], ignore_index=True)
                break


    ## Merging NNs dataset with Hardware Platforms dataset
    df_result = pd.concat([df_hardw_clean,df_topology])

    ##Save
    df_result.to_csv('c:/Users/alinav/Documents/GitHub/Qutibench_Web/_notebooks/data/cleaned_csv/rooflines_hardware_neural_networks.csv', index = False)
#----------------------------------------------------------------------------------------------------------------------------------------    
#--------------------RAW MEASUREMENTS---------------------------------------    

def get_df_by_column(filename, column):
    df_mnist = pd.read_csv(filename)
    unique_column_values = df_mnist[column].unique()
    dataframes = []
    for value in unique_column_values:
        dataframe = df_mnist[df_mnist[column] == value]
        dataframes.append(dataframe)
    
    return dataframes
