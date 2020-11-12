#hide
#hide
import numpy as np
import pandas as pd
import random
import re
import altair as alt
import warnings

W = 600
H = 480

#utils functions------------------------------------
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


def process_csv_for_heatmaps_plot(csv_file: str, machine_learning_task: str)->pd.DataFrame:
    """This method gets the csv for the theoretical predictions file and melts it to make it presentable in the heatmaps plot"""
    
    ## Reading csv file and converting data to (Neural network, Platform, Value)
    df = pd.read_csv(csv_file)

    df_out = pd.DataFrame()
    columns = (df.loc[:, df.columns!='HWType']).columns #select all columns except first
    for column in columns:
        df_=pd.melt(df, id_vars=['HWType'], value_vars=column) #melt df1 into a df1 of 2 columns
        df_out=pd.concat([df_out,df_])
    df_out.columns= ['y','x','values'] #setting new column names
    #replace 0s for NaN values because with 0s the grid doesn't show up
    df_out['values'] = df_out['values'].replace({ 0.0:np.nan})
   
     # Choose the neural networks corresponding to the Machine Learning task asked for.
    if re.search(machine_learning_task, 'imagenet', re.IGNORECASE):
        df_out = dataframe_contains(input_df= df_out, column='x', value='goog|mob|res|effic')

    elif re.search(machine_learning_task, 'mnist', re.IGNORECASE):
        df_out = dataframe_contains(input_df= df_out, column='x', value='mlp')

    elif re.search(machine_learning_task, 'cifar-10', re.IGNORECASE):
        df_out = dataframe_contains(input_df= df_out, column='x', value='cnv')
    
    else: 
        print('The machine learning task was not recognized, please try another one.') 
        return 0
    return df_out
 
def save_not_matched_data(df, machine_learning_task):
    """Method that saves the dataframe to a file."""
    df = df.sort_values(by='pairs')
    df.to_csv('data/no_match_3_' + machine_learning_task + '.csv')

#utils functions------------------------------------

#-------------------------------------------------
def process_theo_top1(csv_theor_accuracies: str) -> pd.DataFrame():
    """
    Method that gets the CNNs and their accuracies table from Theoretical Analysis and melts it into 2 columns 
   
    Parameters
    ----------
    csv_theor_accuracies:str
        Filepath to the CNNs and their accuracy table. 
    
    Returns
    -------
    df_top1_theo: pd.DataFrame()
        Datraframe with 2 columns: |top1 | net_prun_datatype|
        
    """
    # GET THEORETICAL values
    #  get the table above
    df_top1_theo = pd.read_csv(csv_theor_accuracies)
    #  melt it into 2 columns: 
    df_top1_theo = melt_df(df_in= df_top1_theo, cnn_names_col= ' ', new_column_names=['net_prun','datatype','top1'])
    #fix small stuff like deleting rows, merging columns...
    df_top1_theo = fix_small_stuff_df(df= df_top1_theo, col_to_drop=['index','datatype','net_prun'] )
    #  now we have: top1 | net_prun_datatype 
    return df_top1_theo

#------------------------------------------------------

def process_theo_fps(df_top1_theo:pd.DataFrame(), csv_file:str) -> pd.DataFrame():
    """ NOT BEING USED!!
    Method that gets the data from the csv of the Heatmap tables.
    Merges this theoretical df with the given theoretical df (fps+top1) on the 'net_prun_datatype' common column.
    Removes nans from the 'values' column. Changes column order and columns names.
    Replaces things to match.
    
    Notes: Values on the shared column need to be equal for them to be included on the merge. 
            Eg.: 'MLP_100%_INT2' has to match with 'MLP_100%_INT2' otherwise what comes from the performance precitions will be ignored
 
    Parameters
    ----------
    csv_file:str
        Filepath to the CNNs and their fps 
    
    Returns
    -------
    df_top1_theo: pd.DataFrame()
        Datraframe with 2 columns: |top1 | net_prun_datatype|
        
    """
    
    df_fps_theo = process_csv_for_heatmaps_plot(csv_file)
    df_fps_theo.columns=['HWType','net_prun_datatype','fps']
    
    #    remove rows that have 'nan' in the 'values' column
    df_fps_theo = df_fps_theo[df_fps_theo['fps'].notna()]

    #    rename columns
    #   Merge both Theoretical dataframes: fps + top1    
    df_fps_top1_theo = pd.merge(df_top1_theo, df_fps_theo, on='net_prun_datatype', how='outer')
    
    #  change column order
    #order columns
    df_fps_top1_theo = df_fps_top1_theo[['net_prun_datatype', 'HWType', 'top1', 'fps']]
    #  change column names
    df_fps_top1_theo.columns = ['net_prun_datatype', 'hardw_datatype', 'top1', 'fps-comp']
    
    #Note: make sure everything in 'net_prun_datatype' column has NN_Topology + pruning + datatype. If not it will fail

    #  now that we have: net_prun_datatype | hardw_datatype | top1 | fps-comp
    return df_fps_top1_theo

#----------------------------------------------------------------------

def melt_df(df_in: pd.DataFrame(), cnn_names_col: str, new_column_names: list)->pd.DataFrame():
    """Melts a dataframe into 2 columns, the 'cnn_names_col' and the 'value' column. 
    
    Parameters
    ----------
    df_in : pd.DataFrame()
        Dataframe which will be melted.
    cnn_names_col: str
        Column/s which will not be selected to be melted. Eg.:First column ' '.
        
    new_column_names: str
        New column names to give to the dataframe. 
    
    Returns
    -------
    df_out: pd.DataFrame()
        Returns the melted dataframe with the specified column names.
        
        
    """
    df_out = pd.DataFrame()
    #  select all columns except first
    columns = (df_in.loc[:, df_in.columns!=cnn_names_col]).columns 
    for column in columns:
        # melt df1 to have only 2 columns
        df_tmp = pd.melt(df_in, id_vars=[cnn_names_col], value_vars=column) 
        df_out = pd.concat([df_out,df_tmp])
    # setting new columns names
    df_out.columns = new_column_names 
    return df_out

#-----------------------------------------------------

def spot_no_match(list_: list) -> list:
    """
    Method that creates a list of hexadecimal colors. Colors depend on wheteher there is a substring inside each
    list_ item. For 'no match' the color is black, else, the color is created randomly 
   
    Parameters
    ----------
    list_ : list
        List of strings.  
    
    Returns
    -------
    list_of_colors: list
        List with the same size as the input list. Each item is a hexadecimal color. 
               
    """
    sub='no_match'
    list_of_colors=[]
    for index, word in enumerate(list_):
        #if there is no match then appned the black color
        if sub in word:
            list_of_colors.append('#000000')
        else:
            # create random color
            color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
            list_of_colors.append(color[0])
    return list_of_colors
#-----------------------------------------------------

def get_point_chart_enhanced(df: pd.DataFrame, color_groupcol: str,  shape_groupcol: str,  
                    xcol: str,  ycol: str,  shapes: str, title: str, legend_title_groupcol: str)->alt.vegalite.v4.api.Chart: 
    
    """
    Creates an elaborated point chart with the following configurations:
        -different colors
        -different shapes
        -black color to datapoints that don't have a match (theoretical-measured)
        -x axis log scale
        -Text on plot
        -Tooltips
   
    Parameters
    ----------
    df : pd.DataFrame
        
    color_groupcol: str
        Column name which will be what distinguishes colors. 
    shape_groupcol: str
        Column name which will be what distinguishes shapes.
    xcol: str
        Column name which will be the x axis.
    ycol: str
        Column name which will be the y axis.
    shapes: str
        Desired shape range.
    title: str
        Plot title.
    
    legend_title_groupcol:
        Title of the Legend.
    Returns
    -------
    Vega chart: alt.vegalite.v4.api.Chart
        List with the same size as the input list. Each item is a hexadecimal color. 
               
    """
    domain = df[color_groupcol].unique().tolist()
    range_= spot_no_match(list_= domain)
    points= alt.Chart(df).mark_point(size=100, opacity=1, filled =True).properties(
            width= W,
            height= 1.3*H,
            title=title
        ).encode(
            x= alt.X(xcol,  scale=alt.Scale(type="log")),
            y=alt.Y(ycol + ":Q", scale=alt.Scale(zero=False)),
            color=alt.Color(color_groupcol, scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(columns=2, title = legend_title_groupcol)),
            #tooltip=["HWType", "Precision", "PruningFactor", "batch/thread/stream", ycol, xcol],
            shape=alt.Shape(shape_groupcol, scale=alt.Scale(range=shapes), legend=alt.Legend(title = 'Datapoint Type')),
            tooltip=['hardw_datatype_net_prun',color_groupcol, shape_groupcol, xcol, ycol],

        )
    text = points.mark_text(
        angle=325,
        align='left',
        baseline='middle',
        dx=7
    ).encode(
        text='HWType'
    )
    return (points + text).interactive()
#----------------------------------------------------

def get_line_chart(df: pd.DataFrame, groupcol: str, xcol: str, ycol:str, color:str) ->alt.vegalite.v4.api.Chart:
    """
    Creates simple line chart. With tooltips and log scale on the x axis
   
    Parameters
    ----------
     df: pd.DataFrame()
        Contains the data for the chart.
     groupcol: str
        Column name which can be what distinguishes colors. Not used atm. 
     xcol: str
        Column name which will be the x axis.
     ycol: str
        Column name which will be the y axis.
     color: str
        Line color.
    Returns
    -------
    chart: alt.vegalite.v4.api.Chart
         Returns a simple altair line chart based on the inputs.
        
    """
    return alt.Chart(df).interactive().mark_line(point=True).encode(
        x=alt.X(xcol, scale=alt.Scale(type='log')),
        y=alt.Y(ycol + ":Q", scale=alt.Scale(zero=False)),
        color=alt.value(color),
        tooltip=[groupcol, xcol, ycol],
    )
#---------------------------------------------------

def get_pareto_df(df: pd.DataFrame(), groupcol: str, xcol: str, ycol: str) -> pd.DataFrame():
    """
    Creates a pareto dataframe. This method doesn't take into account when the lines go up instead of being constant. 
   
    Parameters
    ----------
     df: pd.DataFrame()
        Dataframe from which the pareto dataframe will be created. 
     groupcol: str
         Column name which will be used to determine the groups for the groupby.
     xcol: str
          Column name which will be used for the x axis later. Used in te groupby.    
     ycol: str
         Column name which will be the used for the groupby to create the y axis.
    Returns
    -------
    pareto_line_df: pd.DataFrame()
        Dataframe with the pareto information.
        
    """
    pareto_line_df = df.groupby(groupcol)[xcol].max().to_frame("x")
    pareto_line_df['y'] = df.groupby(groupcol)[ycol].agg(lambda x: x.value_counts().index[0])
    pareto_line_df.sort_values('y', ascending=False, inplace=True)
    pareto_line_df['x'] = pareto_line_df.x.cummax()
    pareto_line_df.drop_duplicates('x', keep='first', inplace=True)
    pareto_line_df = pareto_line_df.sort_values('x', ascending=False).drop_duplicates('y').sort_index()
    pareto_line_df['group'] = pareto_line_df.index
    return pareto_line_df

#-----------------------------------------------------


def get_several_paretos_df(list_df: pd.DataFrame, groupcol: str, xcol: str, ycol:str, colors: list)->pd.DataFrame():
    """Method that:
        -Receives several dataframes as input inside a list. For each one of them:
            -Gets the pareto dataframe;
            -Creates a line chart from the above mentiioned pareto dataframe;
        -creates a df with all charts inside a column;
   
    Parameters
    ----------
     list_df: pd.DataFrame()
        Contains all dataframes from which the line charts will be generated and put inside the output dataframe (df_out_charts).
     groupcol: str
         Column name which will be used to determine the groups for the groupby.
     xcol:str
         Column name which will be used for the x axis later. Used in te groupby.
     ycol: str
         Column name which will be the used for the groupby to create the y axis.
     colors:list
         List with the colors for each line plot, for each dataframe inside the input list_df.
         
    Returns
    -------
    df_out_charts: pd.DataFrame()
       Dataframe with all output charts.
        
    """
    df_out_charts = pd.DataFrame(columns=['charts'])
    for i, df in enumerate(list_df):
        pareto_df = get_pareto_df(df= df , groupcol= groupcol, xcol= xcol, ycol= ycol)
        chart = get_line_chart(df= pareto_df, groupcol= 'group', xcol= 'x', ycol= 'y', color = colors[i]) 
        df_out_charts = df_out_charts.append(pd.DataFrame([[chart]], columns=['charts']))
    return df_out_charts

#---------------------------------------------


def process_measured_df(df_theoret: pd.DataFrame(), csv_measured: str )-> pd.DataFrame():
    """ NOT BEING USED!!
    Method that gets the measured dataframe from the csv file and fixes small stuff inside it, concatenates with the theoretical df.
   
    Parameters
    ----------
     df_theoret: pd.DataFrame()
        Datafrmae which will be concatenated with the measured df.
        
     csv_measured: str
         Path to the csv file in which small stuff will be fixed inside it. 
    Returns
    -------
    df_out: pd.DataFrame()
       Processed dataframe which is the combination of theoretical with measured.
        
    """
    #   get the measured dataframe
    
    df_measured= pd.read_csv(csv_measured)
    #   fix samll stuff in the measured dataframe so things match
    df_measured = replace_data_df(df_=df_measured, column='hardw_datatype_net_prun', list_tuples_data_to_replace=[("RN50", "ResNet50"),("MNv1", "MobileNetv1"),('GNv1','GoogLeNetv1'),('100.0','100')])
    df_measured = replace_data_df(df_=df_measured, column='NN_Topology', list_tuples_data_to_replace=[("RN50", "ResNet50"),("MNv1", "MobileNetv1"),('GNv1','GoogLeNetv1')])
    #  concatenate both measured with theoretical
    df_out = pd.concat([df_theoret, df_measured])
    
    return df_out
#-------------------------------------------------

def create_delete_columns_theoretical(df_theo: pd.DataFrame()) -> pd.DataFrame():
    """Method that processes the dataframe to make it look like the measured dataframe so they can be matched together later.
    Parameters
    ----------
    df_theo: pd.DataFrame()
        Dataframe with the data upon which these alterations will be done
    Returns
    -------
    df_theo: pd.DataFrame()
        Processed df. 
        
    """
    #   given that we have on theoretical df:  net_prun_datatype | hardw_datatype | top1 | fps-comp
    #   and that we have on the measured df:   hardw_datatype_net_prun | batch/thread/stream  | hardw | NN_Topology | fps-comp | top1 | type
    # We need to:
    #   1. Create 'NN_Topology', 'type', 'hardware' and 'hardw_datatype_net_prun'
    df_theo['NN_Topology'] = df_theo['net_prun_datatype'].str.split('_').str[0]
    df_theo['type'] = 'predicted'
    df_theo['HWType'] = df_theo['hardw_datatype'].str.split('_').str[0]
    #the following operation has in consideration this df, note the mismatch in the A53, which is okay
    # top1    net_prun_datatype    hardw_datatype              fps-comp
    # 96.85    MLP_12.5_INT2       ZCU104-FINN_INT2         229,709.0352000
    # 96.85    MLP_12.5_INT2       ZCU104-BISMO_INT2        229,709.0352000
    # 96.85    MLP_12.5_INT2       U96-Quadcore-A53_INT8    50,966.6921900
    df_theo['hardw_datatype_net_prun'] = df_theo['hardw_datatype'].str.split('_').str[0] +'_'+ df_theo['net_prun_datatype'].str.split('_').str[2]+'_'+ df_theo['net_prun_datatype'].str.split('_').str[0]+'_'+ df_theo['net_prun_datatype'].str.split('_').str[1]
        
    #   delet unnecessary columns
    df_theo = df_theo.drop(columns = ['net_prun_datatype','hardw_datatype'])

    return df_theo
#-------------------------------------------------------------

def fix_small_stuff_df(df: pd.DataFrame(), col_to_drop: list, ) -> pd.DataFrame():
    """Method that fixes small stuff in a dataframe. Things like:
        -remove rows with 'nm'
        - drop unnecessary columns
        -...
   
    Parameters
    ----------
     df: pd.DataFrame()
        Dataframe which will endure all there alterations.
     col_to_drop: list
         List of columns to be dropped.
    Returns
    -------
    df_out: pd.DataFrame()
       Processed dataframe.
        
    """
    df_out = df.copy()
    #   delete all rows that have 'top1 (top5) [%]' inside
    df_out = df_out[df_out['top1'] !='top1 (top5) [%]']
    #    delete all rows with 'nm'
    df_out = df_out[df_out.top1!='nm'] 
    df_out = df_out.reset_index()
    #   merge 'net_prun' with 'datatype' column into 'net_prun_datatype'
    df_out['net_prun_datatype'] = df_out.net_prun + ' ' + df_out.datatype
    df_out = df_out.drop(columns = col_to_drop)

    #    Some cells have [top1 (top5)] accuracies, create col only with top1 acc
    df_out['top1'] = df_out['top1'].str.split(' ').str[0] #take top5 acc out
    #    separate by underscore instead of space
    df_out = replace_data_df(df_=df_out, column='net_prun_datatype', list_tuples_data_to_replace=[(' ','_')])
    return df_out

#--------------------------------------------------------
def process_measured_data(csv_filepath:str)->pd.DataFrame():
    """ This is to create a df to be joined with the theoretical df in 'Theoretical Analysis' to create the overlapped paretos

    Steps
    ------
    1. Create subset from imagenet that doesn't have the ResNet50 v15 measurements because it does not have accuracy measures
    2. Create new hardware column that has hardware and operation mode, beware with NaNs
    3. Create new 'hardw_datatype_net_prun' with hardware + datatype + netwrok + pruning
    4. Create a suset of the dataframe with the above mentioned column and the corresponding ones
    5. With groupby for col 'hardw_datatype_net_prun', for each unique value get the rows with biggest batch 
    6. Add 'type column', reset the index from 'hardw_datatype_net_prun' to ints and save it
    
    Parameters
    ----------
     csv_filepath: str
        Contains  the file path to the file with all measurements which will be read and prepared to be later joined with the theoretical predictions
    
      Returns
    -------
    pd.DataFrame()
        Processed dataframe to match the theoretical predictions dataframe.
    
    """
    df = pd.read_csv(csv_filepath)
    # ResNet50 v15 does not have accuracy measurements yet, so it needs to be taken out
    df = df[df.NN_Topology != 'ResNet-50v15']
    # create hardw column to include: hardware + op_mode - already done in mnist ipynb
    #df['hardw'] = df['HWType'] + ('-' + df['Op mode']).fillna('')
    #create hardw_datatype_net_prun col with all those columns merged
    df['hardw_datatype_net_prun'] = df.apply(lambda r: "_".join([r.HWType, r.Datatype, r.NN_Topology, str(r.PruningFactor)]), axis=1)
    
    #create a subset of the dataframe with only those columns
    df = df[['hardw_datatype_net_prun','HWType', 'NN_Topology' ,'fps-comp', 'top1','batch/thread/stream','hw_peak_perf', 'nn_total_operations']]
    #Only get the points corresponding to the biggest batch
    df = df.groupby('hardw_datatype_net_prun')[['batch/thread/stream','HWType', 'NN_Topology','fps-comp', 'top1', 'hw_peak_perf', 'nn_total_operations']].max()
    
    #add and delete columns 
    df['type'] = 'measured'
    # reset index to start being numeric 
    df = df.reset_index()
    df = df.drop(columns=['batch/thread/stream'])

    #   fix samll stuff in the df so things match with the other side
    df['hardw_datatype_net_prun'] = df['hardw_datatype_net_prun'].str.replace(pat='.0', repl='', regex=False)

    return df
#---------------------------------------------------------------------------------------------------------------
def identify_pairs_nonpairs(df: pd.DataFrame, column: str) -> pd.DataFrame():
    """This method identifies equal values in the column and signals them, and creates another column with labels for each case 

    Parameters
    ----------
     df: pd.DataFrame()
        Dataframe which will be processed.
     column: str
         Column which has: hardware platform, datatype, NN_Topology and pruning factor. It has duplicated values.
    Returns
    -------
    df: pd.DataFrame()
       Processed dataframe.
        
    """
    # IDENTIFY ALL PAIRS AND CREATE A SPECIAL COLUMN FOR THEM
    #get all pair and then get unique names out of those pairs
    df['pairs'] = df[column].duplicated(keep=False)
    unique_names = df.loc[df.pairs ==True, column].unique()
    #set a color for each one of them
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(unique_names))]
    #put it into a dict
    names_with_colors = {key:color for key,color in zip(unique_names,color)}
    #assign it to the dataframe color column. Only fill up rows (with the same color) that have pairs
    df['color'] = df[column].apply(lambda x: x if x in names_with_colors else '')
    #fill up the rest of the rows that do not have a pair
    df['color'] = df.apply(lambda row: 'predicted_no_match' if row.type=='predicted' and row.color=='' else 
                                                     ('measured_no_match' if row.type=='measured' and row.color=='' else (row.color)), axis=1)
    #df = df.drop(columns=['pairs'])
    return df

#-------------------------------------------------------

#hide
def plot_it_now(df: pd.DataFrame, xcol: str, ycol: str, groupcol: str, title: str) -> alt.vegalite.v4.api.Chart:
    """This method creates all plots for the overlapped pareto and layers them all together. 
    These are: 2 pareto lines and the points plot.
    All points plot are binded to checkboxes
   
    Parameters
    ----------
     df: pd.DataFrame()
        Contains data to be plotted.
     xcol: str
         Dataframe column which has the information for the x axis.
     ycol: str
         Dataframe column which has the information for the y axis.
     groupcol: str
          Dataframe column which has the information for the color.
     title:str
         Title to give to the plot.
    
    Returns
    -------
    Layered chart: -> alt.vegalite.v4.api.Chart
       Layered chart, both theoretical pareto curves and the points chart
        
    """
    #get the pareto data to built the pareto lines
    df_theo =df.loc[df.type=='predicted',:]
    df_exper = df.loc[df.type=='measured',:]
    df_charts = get_several_paretos_df(list_df = [df_theo, df_exper], groupcol= groupcol, xcol= xcol , ycol= ycol, colors=['#FFA500', '#0066CC'])
    
    #this is to be used in the color field to set a different color for each field, and to set to black all that doesn't have a match
    domain = df[groupcol].unique().tolist()
    range_= spot_no_match(list_= domain)
    
    #Select data from the dataframe to bind to each checkbox
    FINN_data= df.loc[df[groupcol].str.contains("finn")]
    BISMO_data= df.loc[df[groupcol].str.contains("bismo")]
    A53_data= df.loc[df[groupcol].str.contains("a53")]
    TX2_data= df.loc[df[groupcol].str.contains("tx2")]
    NCS_data= df.loc[df[groupcol].str.contains("ncs")]
    TPU_data= df.loc[df[groupcol].str.contains("tpu")]
    DPU_data= df.loc[df[groupcol].str.contains("dpu")]
   
    if FINN_data.shape[0] + BISMO_data.shape[0] + A53_data.shape[0] + TX2_data.shape[0] + NCS_data.shape[0] + TPU_data.shape[0] + DPU_data.shape[0] != len(df.index):
        print('There are datapoints missing in the plot')
    
    #The type of binding will be a checkbox
    filter_checkbox = alt.binding_checkbox()
    
    #Create all checkboxes
    #measu_no_match_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="Measured_Without_Match") 
    #predicted_no_match_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="Predicted_Without_Match") 
    FINN_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="ZCU104_FINN") 
    BISMO_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="ZCU104_BISMO")
    A53_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="U96_Quadcore_A53")
    TX2_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="TX2")
    NCS_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="NCS")
    TPU_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="TPU")
    DPU_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="DPU")
    
    legend_title_groupcol ='Hardw_Datatype_Net_Prun'
    #Color Conditions for each plot
    FINN_cond    = alt.condition(FINN_select, alt.Color(groupcol+':N', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(columns=2, title = legend_title_groupcol, symbolLimit=0)),alt.value(None))
    BISMO_cond   = alt.condition(BISMO_select, alt.Color(groupcol+':N', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(columns=2, title = legend_title_groupcol, symbolLimit=0)),alt.value(None))
    A53_cond     = alt.condition(A53_select, alt.Color(groupcol+':N', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(columns=2, title = legend_title_groupcol, symbolLimit=0)),alt.value(None))
    TX2_cond     = alt.condition(TX2_select, alt.Color(groupcol+':N', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(columns=2, title = legend_title_groupcol, symbolLimit=0)),alt.value(None))
    NCS_cond     = alt.condition(NCS_select, alt.Color(groupcol+':N', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(columns=2, title = legend_title_groupcol, symbolLimit=0)),alt.value(None))
    TPU_cond     = alt.condition(TPU_select, alt.Color(groupcol+':N', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(columns=2, title = legend_title_groupcol, symbolLimit=0)),alt.value(None))
    DPU_cond     = alt.condition(DPU_select, alt.Color(groupcol+':N', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(columns=2, title = legend_title_groupcol, symbolLimit=0)),alt.value(None))
    
    #Create the charts
    FINN_chart=get_point_chart_selection(df= FINN_data, condition=FINN_cond, selection=FINN_select, color_groupcol= 'color', shape_groupcol= 'type',shapes=['cross', 'circle'], xcol= xcol, ycol= ycol, title=title, legend_title_groupcol="Hardw_Datatype_Net_Prun" )
    BISMO_chart=get_point_chart_selection(df= BISMO_data, condition=BISMO_cond, selection=BISMO_select, color_groupcol= 'color', shape_groupcol= 'type',shapes=['cross', 'circle'], xcol= xcol, ycol= ycol, title=title, legend_title_groupcol="Hardw_Datatype_Net_Prun" )
    A53_chart=get_point_chart_selection(df= A53_data, condition=A53_cond, selection=A53_select, color_groupcol= 'color', shape_groupcol= 'type',shapes=['cross', 'circle'], xcol= xcol, ycol= ycol, title=title, legend_title_groupcol="Hardw_Datatype_Net_Prun" )
    TX2_chart=get_point_chart_selection(df= TX2_data, condition=TX2_cond, selection=TX2_select, color_groupcol= 'color', shape_groupcol= 'type',shapes=['cross', 'circle'], xcol= xcol, ycol= ycol, title=title, legend_title_groupcol="Hardw_Datatype_Net_Prun" )
    NCS_chart=get_point_chart_selection(df= NCS_data, condition=NCS_cond, selection=NCS_select, color_groupcol= 'color', shape_groupcol= 'type',shapes=['cross', 'circle'], xcol= xcol, ycol= ycol, title=title, legend_title_groupcol="Hardw_Datatype_Net_Prun" )
    TPU_chart=get_point_chart_selection(df= TPU_data, condition=TPU_cond, selection=TPU_select, color_groupcol= 'color', shape_groupcol= 'type',shapes=['cross', 'circle'], xcol= xcol, ycol= ycol, title=title, legend_title_groupcol="Hardw_Datatype_Net_Prun" )
    DPU_chart=get_point_chart_selection(df= DPU_data, condition=DPU_cond, selection=DPU_select, color_groupcol= 'color', shape_groupcol= 'type',shapes=['cross', 'circle'], xcol= xcol, ycol= ycol, title=title, legend_title_groupcol="Hardw_Datatype_Net_Prun" )
    warnings.filterwarnings("ignore")
    #sum the pareto lines
    chart = df_charts.charts.sum(numeric_only = False)
    #layer the pareto lines with the points chart with checkboxes
    charts = alt.layer(FINN_chart + BISMO_chart + A53_chart+ TX2_chart+ NCS_chart +TPU_chart + DPU_chart +chart
    ).resolve_scale(color='independent',shape='independent').properties(title=title).configure_legend(symbolOpacity=0.1)
    return charts


#-----------------------------------------------------


def get_df_theo_fps_top1(machine_learning_task: str) -> pd.DataFrame():
    """This method get the Theoretical top1 accuracies (of NNs) and performance (fps) of the NNs + hardware and merges these both to get the theoretical performance.
    Reads this file: data/performance_predictions_imagenet_mnist_cifar.csv
   
    Parameters
    ----------
     machine_learning_task: str
        Desired machine learning task to choose from the performance predictions.
    
    Returns
    -------
    pd.DataFrame()
    Eg.: Outputs a dataframe like the following:
         top1      net_prun_datatype       hardw_datatype          fps-comp
    16   69.24     GoogLeNetv1_100_INT8    Ultra96-DPU_INT8        306.709265
    17   69.24     GoogLeNetv1_100_INT8    ZCU104-DPU_INT8         1,469.6485    
    """

    # 1. Get Theoretical TOP1 Accuracies table (Theoretical_Analysis/CNNs and their accuracy...)
    df_top1_theo = process_theo_top1(csv_theor_accuracies ='data/cnn_topologies_accuracy.csv')
    
    # 2. Get Theoretical FPS to match with that Theoretical TOP1 - from Performance Predictions
    df_fps_theo = process_csv_for_heatmaps_plot("data/performance_predictions_imagenet_mnist_cifar.csv", machine_learning_task)
    df_fps_theo.columns=['hardw_datatype','net_prun_datatype','fps-comp']
    
    # 3. Merge Theoretical top1 + Theoretical fps
    df_theo_fps_top1 = pd.merge(df_top1_theo, df_fps_theo, on='net_prun_datatype', how='outer')
    #take nans out- the ones that don't have correspondence- probably because they are from different machine learning task
    df_theo_fps_top1 = df_theo_fps_top1.dropna()

    return df_theo_fps_top1
      
def get_overlapped_pareto(machine_learning_task: str, title:str):
    """
    Main method to get the overlapped pareto plots.
    What it does: Get top1 acc. -> Get fps correpsonding to previous acc. -> Get measured pareto -> join them -> identify pairs -> plot it
        1.
   
    Parameters
    ----------
    net_keyword: str
        This string should contain the Classification type needed by user. 
        It is not case sensistive.
        Eg.: imagenet, mnist or cifar-10
    Returns
    -------
    Heatmap Chart: altair.vegalite.v4.api.Chart
        This is an Altair/Vega-Lite Heatmap chart. 
        It returns the overlapped pareto plot (theoretical + measured + 2 pareto lines(theoretical+measured)).       
    """
    # 1.2.3. Get the theoretical df with accuracies and fps-comp
    df_theo_fps_top1 = get_df_theo_fps_top1(machine_learning_task)
    
    # 4. Get Measured df for the desired Classification Task (experimental_data_m_l_task)
    if re.search(machine_learning_task, 'imagenet', re.IGNORECASE):      
        df_measured = process_measured_data(csv_filepath= 'data/cleaned_csv/experimental_data_imagenet.csv')
    elif re.search(machine_learning_task, 'mnist', re.IGNORECASE):
        df_measured = process_measured_data(csv_filepath= 'data/cleaned_csv/experimental_data_mnist.csv')
    elif re.search(machine_learning_task, 'cifar-10', re.IGNORECASE):
        df_measured = process_measured_data(csv_filepath= 'data/cleaned_csv/experimental_data_cifar.csv')

    # 5. Improve theoretical to match with Measured df after this,also creates columns based on another ones,deals w/ datatypes and pruning
    df_theo_fps_top1 = create_delete_columns_theoretical(df_theo= df_theo_fps_top1)

    # 6. Merge Measured + Theoretical = Overlapped Pareto df
    overlapped_pareto = pd.concat([df_theo_fps_top1, df_measured])
   
    # 7. Small changes like: apply lowercase, organizing alphabetically by column
    overlapped_pareto.hardw_datatype_net_prun = overlapped_pareto.hardw_datatype_net_prun.str.casefold() 
    overlapped_pareto= overlapped_pareto.sort_values(by='hardw_datatype_net_prun')

    # 8. Identify Pairs and create a special column for them 
    overlapped_pareto = identify_pairs_nonpairs(df=overlapped_pareto, column='hardw_datatype_net_prun')
    
    overlapped_pareto= overlapped_pareto.sort_values(by='pairs')
    save_not_matched_data(overlapped_pareto, machine_learning_task)
    
    # 9. Delete all with no match
    overlapped_pareto = overlapped_pareto.loc[(overlapped_pareto.color !='measured_no_match') &(overlapped_pareto.color !='predicted_no_match')  ]
    
    #return overlapped_pareto
    return plot_it_now(df= overlapped_pareto, xcol= 'fps-comp', ycol= 'top1', groupcol= 'color', title=title)    

#----------------------------------------------------

def get_point_chart(df: pd.DataFrame, groupcol: str, xcol: str, ycol:str, title: str) ->alt.vegalite.v4.api.Chart:
    """Creates simple point chart
    
    Parameters
    ----------
     df: pd.DataFrame()
        Contains data to be plotted.
     xcol: str
         Dataframe column which has the information for the x axis.
     ycol: str
         Dataframe column which has the information for the y axis.
     groupcol: str
          Dataframe column which has the information for the color.
     title:str
         Title to give to the plot.
    
    Returns
    -------
    Chart: alt.vegalite.v4.api.Chart
       Simple point chart.
        
    """
    points = alt.Chart(df).mark_point(filled=True).properties(
            width= W,
            height= 1.3*H,
            title=title
        ).encode(
        x= alt.X(xcol, scale=alt.Scale(type='log')),
        y= alt.Y(ycol + ":Q", scale=alt.Scale(zero=False)),
        color= alt.Color(groupcol),
        tooltip= [groupcol, xcol, ycol],
    )
    text = points.mark_text(
        angle=325,
        align='left',
        baseline='middle',
        dx=7
    ).encode(
        text='HWType'
    )
    return (points+text).interactive()
#----------------------------------------------------
def get_point_chart_selection(df: pd.DataFrame, color_groupcol: str, 
                              condition: dict,
                              selection: alt.vegalite.v4.api.Selection,
                              shape_groupcol: str,  
                    xcol: str,  ycol: str,  shapes: str, title: str, legend_title_groupcol: str)->alt.vegalite.v4.api.Chart: 
    
    """
    Creates an elaborated point chart with the following configurations:
        -different colors
        -different shapes
        -black color to datapoints that don't have a match (theoretical-measured)
        -x axis log scale
        -Text on plot
        -Tooltips
   
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with data ot be plotted.
    condition: dict
        Condition for the color.
        Eg.: {'condition': {'selection': 'FPGAs  Ultra96  DPU  ZCU  ', 'type': 'nominal', 'field': 'Name'}, 'value': 'lightgray'}  
    selection: alt.vegalite.v4.api.Selection
        Selection object to select what information the selection is tied to.
    color_groupcol: str
        Column name which will be what distinguishes colors. 
    shape_groupcol: str
        Column name which will be what distinguishes shapes.
    xcol: str
        Column name which will be the x axis.
    ycol: str
        Column name which will be the y axis.
    shapes: str
        Desired shape range.
    title: str
        Plot title.
    
    legend_title_groupcol:
        Title of the Legend.
    Returns
    -------
    Vega chart: alt.vegalite.v4.api.Chart
        List with the same size as the input list. Each item is a hexadecimal color. 
               
    """
    domain = df[color_groupcol].unique().tolist()
    range_= spot_no_match(list_= domain)
    points= alt.Chart(df).mark_point(size=100, opacity=1, filled =True, color='black').properties(
            width= W,
            height= 1.3*H,
            title=title
        ).encode(
            x= alt.X(xcol,  scale=alt.Scale(type="log")),
            y=alt.Y(ycol + ":Q", scale=alt.Scale(zero=False)),
            color=condition,
            shape=alt.Shape(shape_groupcol, scale=alt.Scale(range=shapes), legend=alt.Legend(title = 'Datapoint Type')),
            tooltip=['hardw_datatype_net_prun',color_groupcol, shape_groupcol, xcol, ycol],

        )
    text = points.mark_text(
        angle=325,
        align='left',
        baseline='middle',
        dx=7
    ).encode(
        text='HWType'
    )
    layered_chart = alt.layer(points, text).interactive().add_selection(selection)
    return layered_chart

#---------------------------------------------------------------

def theor_pareto(machine_learning_task, title: str) ->alt.vegalite.v4.api.Chart:
    """Creates a Theoretical Pareto Plot.
   
   
    Parameters
    ----------
    net_keyword : pd.DataFrame()
        Desired classification task. Eg.: MNIST, CIFAR, ImageNet. 
    title: str
        Plot title.
        
    Returns
    -------
    Chart: alt.vegalite.v4.api.Chart
       
        
    """
    xcol='fps-comp'
    ycol='top1'
    groupcol='hardw_datatype_net_prun'
    colors=['#FFA500']
    
    # 1. Get Theoretical TOP1 Accuracies table (Theoretical_Analysis/CNNs and their accuracy...)
    df_top1_theo = process_theo_top1(csv_theor_accuracies ='data/cnn_topologies_accuracy.csv')
    
    # 2. Get Theoretical FPS to match with that Theoretical TOP1 - from Performance Predictions
    df_fps_theo = process_csv_for_heatmaps_plot("data/performance_predictions_imagenet_mnist_cifar.csv", machine_learning_task)
    df_fps_theo.columns=['hardw_datatype','net_prun_datatype','fps-comp']
    
    # 3. Merge Theoretical top1 + Theoretical fps
    df_theo_fps_top1 = pd.merge(df_top1_theo, df_fps_theo, on='net_prun_datatype', how='outer')
    #take nans out- the ones that don't have correspondence- probably because they are from different machine learning task
    df_theo_fps_top1 = df_theo_fps_top1.dropna()
    
    # 5. Improve Theoretical to match with Measured df after this
    df_theo_fps_top1 = create_delete_columns_theoretical(df_theo= df_theo_fps_top1)
    
    df_charts = get_several_paretos_df(list_df = [df_theo_fps_top1], xcol= xcol, ycol= ycol, groupcol= groupcol, colors=colors)
    points = get_point_chart(df= df_theo_fps_top1, groupcol= groupcol, xcol= xcol, ycol=ycol, title=title) 
    
    return (points + df_charts.iloc[0,0])
#----------------------------------------------------

def get_percentage_colum(df: pd.DataFrame, col_elements: str, newcol: str ) -> pd.DataFrame:
    """
    Method that:
        1.Creates a new empty column
        2.Fills in that column with percentages based on another column.
        3.percentage= (fps-comp (measured) / fps-comp (theoretical)) * 100 for each unique element from col_elements
        4.Resulting df: col_elements  fps-comp    type           percentage
                            A             x       measured         (x/y)*100
                            A             y       theoretical
                            B ...  
    Parameters
    ----------
     df: pd.DataFrame()
        Contains all needed data 
     col_elements: str
             Column with doubled elements. 
             Each element in this column will correspond to measured and theoretical values in type column.
     new_col: str
         New column name which will be created.
    Returns
    -------
    df_: pd.DataFrame()
       Dataframe with the new column
        
    """
    df_ = df.copy()
    #Create new column
    df_[newcol] = ''

    pairs= df_[col_elements].unique()
    percentage= []
    for par in pairs:
        df_pair = df_.loc[df_[col_elements] == par]
        theoret = df_pair.loc[df_pair.type == 'predicted','fps-comp']
        measured = df_pair.loc[df_pair.type == 'measured','fps-comp']
        percentage.append(str(round((measured.values[0]/theoret.values[0])*100,1)) + '%')
    dict_= {key:value for key,value in zip(pairs,percentage)}
    df_['percentage'] = df_.apply(lambda row: dict_[row[col_elements]]  if row[col_elements] in dict_ and row.type=='measured' else '', axis=1)
    return df_

#----------------------------------------------------

def fill_values(col_value: str, dict_: dict)->int:
    """Method that given a dict.:
        -iterates through every key, splitting it,
        -tries to find each part of each key in the string (col_value). 
        -Criteria: if all parts of any key are found inside the string then that key is returned.
        -Note: The first key to meet this criteria is returned.
    
    Parameters
    ----------
     col_value: str
         String which will be searched for the key parts
     dict_: dict
         ditionary with all keys to be iterated
    Returns
    -------
     dict_:dict
          if all parts of any key are found inside the string the key is returned.
     0: int
         if all parts of any key are not found inside the string then 0 is returned.
    """
    for key in dict_: 
        substrings = key.split(' ')
        found=True
        for substring in substrings:
            if substring.lower() not in col_value.lower():
                found=False
        if found==True: 
            return dict_[key]
    return 0

#----------------------------------------------------


def get_peak_perf_gops_df(df_: pd.DataFrame ) ->pd.DataFrame:
        # NOT BEING USED BECAUSE ADDED 3 COLUMNS TO THE BACKUP CSV
    """-Selects a subset of the given df (the 'hardw_datatype_net_prun' column)
        -Gets Peak Compute Performance from the csv file (peakPerfBandHardPlatf) and 
        fills the given df with that value for each hardware in a separate column called 'peak_compute'
        -Gets GOPs for each CNN from csv file (cnn_topologies_compute_memory_requirements) and
        fills another column called 'gops' in the given df
        -Multiplies both of these new columns to get a new column which corresponds to the
        Theoretical Peak Performance, however this is put in a column called 'fps-comp' to be able to, later,
        merge this df with another df and also to plot this on a y-axis scale which has fps-comp.
        
    Parameters
    ----------
     df: pd.DataFrame()
         Classifiation dataframe (MNIST, ImageNet, CIFAR-10..) which will be used for the subsetting and will be filled with 
        Theoretical Peak Performance values.
    
    Returns
    -------
     efficiency_df: pd.DataFrame()
        Dataframe like the following one:
          ----------------------------------------------------------------------------
          | hardw_datatype_net_prun   |         type                 |      fps-comp |
          ----------------------------------------------------------------------------
          |    NCS_FP16_MLP_100       |   Theoret. Peak Compute      |          x    |
          |    NCS_FP16_MLP_12.5      |   Theoret. Peak Compute      |          y    |
          |      ...                  |           ""                 |         ...   |
          ----------------------------------------------------------------------------
    """
    # Create a subset form
    efficiency_df = df_.loc[:, ['hw_datatype_prun_net']]
    efficiency_df['type'] = 'Theoret. Peak Compute'
    #Get Peak Compute for all hardware
    peak_compute_hardw_df=pd.read_csv('data/peakPerfBandHardPlatf.csv')
    peak_compute_hardw_dict=pd.Series(peak_compute_hardw_df.Peak_Performance.values,index=peak_compute_hardw_df.Name).to_dict()
    efficiency_df['peak_compute'] = ''
    efficiency_df['peak_compute']=efficiency_df.apply(lambda row: fill_values(row.hw_datatype_prun_net, peak_compute_hardw_dict), axis=1)
    #Get GOPS for all CNNs
    gops_df = pd.read_csv('data/cnn_topologies_compute_memory_requirements.csv')
    #get first two columns
    gops_df = gops_df.loc[:,[' ','Total OPs']]
    #rename column
    gops_df.columns=['NN_Topology', 'gops']
    #remove first row with double names
    gops_df= gops_df.iloc[1:,:]
    # remove %, correct ResNet-50 to ResNet50...
    #comentei a linha abaixo agora
    #gops_df = replace_data_df(df_=gops_df, column= 'NN_Topology', list_tuples_data_to_replace= [('%',''),('ResNet-50','ResNet50'),('EfficientNet Edge L','EfficientNetL'),('EfficientNet Edge S','EfficientNetS'),('EfficientNet Edge M','EfficientNetM')])
    #create dictionary out of the df
    gops_dict=pd.Series(gops_df.gops.values,index=gops_df.NN_Topology).to_dict()
    #create a new column with the GOPs values
    efficiency_df['gops'] = efficiency_df.apply(lambda row: fill_values(row.hw_datatype_prun_net, gops_dict), axis=1)
    #now that all is done lets create the theoretical performance column which will be called 'fps-comp' and calculate the numbers
    try:
        efficiency_df['fps-comp'] = efficiency_df.apply(lambda row: float(row.peak_compute)*1000/float(row.gops), axis=1)
    except ZeroDivisionError as err:
        print('Zero Division Error. An error has occurred. Possibly GOPs CNN value was 0. This means there was a mismatch between names in the data coming from cnn_topologies_compute_memory_requirements.csv file and the dataframe given as input.', err)
    
    efficiency_df = efficiency_df.drop(columns=['gops','peak_compute'])
   
    #remove all rows that have fps-comp = 0
    efficiency_df= efficiency_df.loc[efficiency_df['fps-comp'] != 0, :]
    
    return efficiency_df
#----------------------------------------------------

def faceted_bar_chart(df: pd.DataFrame, xcol: str, ycol:str, colorcol: str, textcol: str, columncol: str, title:str) -> alt.vegalite.v4.api.Chart:
    """
    Creates simple faceted bar chart.   
       
    Parameters
    ----------
     df: pd.DataFrame()
        Data to plot. 
     xcol: str
         DataFrame column name which will be used for the x axis.
     ycol: str
         DataFrame column name which will be used for the y axis.
     colorcol: str
         DataFrame column name which will be used for the color separation.
     textcol: str
         DataFrame column name which will be used for the text on plot.
     columncol: str
         DataFrame column name which will be used for the faceted bar chart (what separates into several mini-barcharts)
     title: str
         Plot title.
    Returns
    -------
    Bar chart: alt.vegalite.v4.api.Chart:
       Simple Faceted Bar chart with text on top of the columns and text inside the plot (percentage)
        
    """
    bars = alt.Chart().mark_bar().encode(
        x=alt.X(xcol +':N', sort=['predicted','measured'], title=''),
        y=alt.Y(ycol +':Q', scale= alt.Scale(type='log', domain = (0.01,100000000))),
        color=alt.Color(colorcol +':N', title='Datapoint Type'),
    )
    text = bars.mark_text(
        angle=270,
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text= alt.Text(textcol)
    )
    return alt.layer(bars, text, data=df).facet(
        column=alt.Column(columncol+':N', header=alt.Header(labelAngle=-85, labelAlign='right'), title=title)
    ).interactive()

#----------------------------------------------------

def efficiency_plot(machine_learning_task: str, title: str) -> alt.vegalite.v4.api.Chart:
    """
    Method that creates a faceted bar chart.
    In the y axis we have fps-compute and in the x axis we have several combinations of hardware platorms and neural networks. 
       
    Parameters
    ----------
     net_keyword: str
        Desired Machine Learning task.
     title: str
         Plot title.
    Returns
    -------
     Bar chart: alt.vegalite.v4.api.Chart
        Simple Faceted Bar chart with text on top of the columns and text inside the plot (percentage) 
    """
    
    # 1.2.3. Get the theoretical df with accuracies and fps-comp
    df_theo_fps_top1 = get_df_theo_fps_top1(machine_learning_task) 
    df_theo_fps_top1.rename({'fps-comp': 'predicted'}, axis=1, inplace=True)
    
    # 4. Get Measured df for the desired Classification Task (experimental_data_m_l_task)
    if re.search(machine_learning_task, 'imagenet', re.IGNORECASE):      
        df_measured = process_measured_data(csv_filepath= 'data/cleaned_csv/experimental_data_imagenet.csv')
    elif re.search(machine_learning_task, 'mnist', re.IGNORECASE):
        df_measured = process_measured_data(csv_filepath= 'data/cleaned_csv/experimental_data_mnist.csv')
    elif re.search(machine_learning_task, 'cifar-10', re.IGNORECASE):
        df_measured = process_measured_data(csv_filepath= 'data/cleaned_csv/experimental_data_cifar.csv')
    
    # 5. Improve theoretical to match with Measured df after this
    df_theo_fps_top1 = create_delete_columns_theoretical(df_theo= df_theo_fps_top1)
    
    df_measured.rename({'fps-comp': 'measured'}, axis=1, inplace=True)
    
    # 6. Merge Measured + Theoretical = Overlapped Pareto df
    overlapped_pareto = pd.concat([df_theo_fps_top1, df_measured])
    
    overlapped_pareto['Theoret. Peak Compute'] = overlapped_pareto.apply(lambda row: float(row.hw_peak_perf)*1000 / float(row.nn_total_operations),  axis=1)
   
    overlapped_pareto = pd.melt(overlapped_pareto, id_vars=['hardw_datatype_net_prun','NN_Topology','HWType'], value_vars=['predicted','measured','Theoret. Peak Compute'],var_name='type', value_name='fps-comp' )

    overlapped_pareto = overlapped_pareto.dropna()
    
    # 7. Small changes like: apply lowercase, organizing alphabetically by column
    overlapped_pareto.hardw_datatype_net_prun = overlapped_pareto.hardw_datatype_net_prun.str.casefold() 
    overlapped_pareto= overlapped_pareto.sort_values(by='hardw_datatype_net_prun')
   
    # 8. Identify Pairs and create a special column for them 
    overlapped_pareto = identify_pairs_nonpairs(df=overlapped_pareto, column='hardw_datatype_net_prun')

    # 9. Remove the ones with no match
    overlapped_pareto = overlapped_pareto.loc[(overlapped_pareto.color!='measured_no_match') & (overlapped_pareto.color!='predicted_no_match')]
    
     # 10. Count how many times each 'hardw_datatype_net_prun' combination is repeated
    df_tmp =overlapped_pareto['hardw_datatype_net_prun'].value_counts().to_frame().rename(columns={'hardw_datatype_net_prun':'count'})  
    #get the ones that are only repeated less than 3 times
    list_to_remove =df_tmp.loc[df_tmp['count'] < 3, :].index
    #remove them 
    overlapped_pareto = overlapped_pareto[~overlapped_pareto.hardw_datatype_net_prun.isin(list_to_remove)]
       
    #return overlapped_pareto
    #create a percentage column
    overlapped_pareto = get_percentage_colum(df=overlapped_pareto, col_elements='hardw_datatype_net_prun', newcol='percentage')   
    
    
    #split in case there are too many rows, because the faceted bar chart will be too full
    if overlapped_pareto.hardw_datatype_net_prun.unique().size < 8:
        return faceted_bar_chart(df=overlapped_pareto , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title ) 
    
    usb_devices_df= overlapped_pareto.loc[overlapped_pareto.HWType.str.lower().str.contains('dge|ncs|a53')]
    fpga_df= overlapped_pareto.loc[overlapped_pareto.HWType.str.lower().str.contains('zcu|ultra')]
    gpu_df= overlapped_pareto.loc[overlapped_pareto.HWType.str.lower().str.contains('tx2')]

    # Plot it - Faceted Bar chart
    a=faceted_bar_chart(df=usb_devices_df , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title ) 
    b=faceted_bar_chart(df=fpga_df , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title ) 
    c=faceted_bar_chart(df=gpu_df , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title) 
    
    return (a.display() or b.display() or c.display())
    #return  overlapped_pareto
   
    #------------------------------------------------------------------
    
