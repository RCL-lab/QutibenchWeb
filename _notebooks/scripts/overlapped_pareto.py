#hide
import numpy as np
import pandas as pd
import random
import re
import altair as alt

W = 600
H = 480


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

def process_theo_fps(df_top1_theo:pd.DataFrame(),csv_files: list) -> pd.DataFrame():
    """
    Method that gets the data from the csv of the Heatmap tables.
    Merges this theoretical df with the given theoretical df (fps+top1) on the 'net_prun_datatype' common column.
    Removes nans from the 'values' column. Changes column order and columns names.
    Replaces things to match.
    
    Notes: Values on the shared column need to be equal for them to be included on the merge. 
            Eg.: 'MLP_100%_INT2' has to match with 'MLP_100%_INT2' otherwise what comes from the performance precitions will be ignored
 
    Parameters
    ----------
    csv_theor_accuracies:str
        Filepath to the CNNs and their accuracy table. 
    
    Returns
    -------
    df_top1_theo: pd.DataFrame()
        Datraframe with 2 columns: |top1 | net_prun_datatype|
        
    """
    df_fps_theo = pd.DataFrame()
    for csv_file in csv_files:
        df_tmp = pd.read_csv(csv_file)
        df_fps_theo = pd.concat([df_fps_theo, df_tmp])
    df_fps_theo['x']= df_fps_theo['x'].str.replace('-','_')
    #    remove rows that have 'nan' in the 'values' column
    df_fps_theo = df_fps_theo[df_fps_theo['values'].notna()]
    #    rename columns
    df_fps_theo.columns=['hardw','net_prun_datatype','fps']

    #   Merge both Theoretical dataframes: fps + top1 
    df_fps_top1_theo = pd.merge(df_top1_theo, df_fps_theo, on='net_prun_datatype', how='outer')
    #  change column order
    df_fps_top1_theo = df_fps_top1_theo[['net_prun_datatype', 'hardw', 'top1', 'fps']]
    #  change column names
    df_fps_top1_theo.columns = ['net_prun_datatype', 'hardw_datatype', 'top1', 'fps-comp']

    #Notes: 1. make sure everything in 'net_prun_datatype' column has network + prunning + datatype. If not it will fail
    df_fps_top1_theo = replace_data_df(df_=df_fps_top1_theo, column= 'net_prun_datatype', list_tuples_data_to_replace= [('GoogLeNetv1','GoogLeNetv1_100%'),('MobileNetv1','MobileNetv1_100%'),('GoogleNetv1','GoogleNetv1_100%'), ('EfficientNet_S','EfficientNet-S_100%'), ('EfficientNet_M','EfficientNet-M_100%'), ('EfficientNet_L','EfficientNet-L_100%'), ('%','')])
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
        text='hardw'
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
#-------------------------------------------------

def process_measured_df(df_theoret: pd.DataFrame(), csv_measured:str ):
    """ Method that gets the measured dataframe and fixes small stuff inside it, concatenates with the  
   
    Parameters
    ----------
     df_theoret: pd.DataFrame()
        Contains bla bla 
    
    Returns
    -------
    : pd.DataFrame()
       
        
    """
    #   get the measured dataframe
    df_measured = pd.read_csv(csv_measured)
    #   fix samll stuff in the measured dataframe so things match
    df_measured = replace_data_df(df_=df_measured, column='hardw_datatype_net_prun', list_tuples_data_to_replace=[("RN50", "ResNet50"),("MNv1", "MobileNetv1"),('GNv1','GoogLeNetv1'),('100.0','100')])
    df_measured = replace_data_df(df_=df_measured, column='network', list_tuples_data_to_replace=[("RN50", "ResNet50"),("MNv1", "MobileNetv1"),('GNv1','GoogLeNetv1')])
    #  concatenate both measured with theoretical
    df_out = pd.concat([df_theoret, df_measured])
    
    return df_out
#-------------------------------------------------

def select_cnn_match_theo_for_measured(df_theo: pd.DataFrame(), net_prun_datatype: str) -> pd.DataFrame():
    """
    Method that processes the dataframe to make it look like the measured dataframe.
    Eliminates all NaNs and replaces elements to make dfs look alike. 
   
    Parameters
    ----------
    df_theo: pd.DataFrame()
        Dataframe with the data upon which these alterations will be done
         
    Returns
    -------
    df_theo: pd.DataFrame()
        Processed df. 
        
    """
    # create a subset from the given dataframe
    #     there is another way to do this 
    #df_theo = df_superset[df_superset.apply(lambda row: row[net_prun_datatype].split('_')[0] == cnn_keyword, axis=1)]
    #    the line below is not needed because there is only 1 classification
    #df_theo = df_superset.loc[df_superset[net_prun_datatype].str.contains(cnn_keyword, na=False)]
    df_theo = df_theo[df_theo['top1'].notna()]
    df_theo = df_theo[df_theo['fps-comp'].notna()]
    
    #   given that we have on theoretical df:  net_prun_datatype | hardw_datatype | top1 | fps-comp
    #   and that we have on the measured df:   hardw_datatype_net_prun | batch/thread/stream  | hardw | network | fps-comp | top1 | type
    #We need to:
    #   1. Create 'network', 'type', 'hardware' and 'hardw_datatype_net_prun'
    df_theo['network'] = df_theo['net_prun_datatype'].str.split('_').str[0]
    df_theo['type'] = 'predicted'
    #replace elemnts out of hardw column - take datatypes out of hardw_datatype column
    df_theo = replace_data_df(df_=df_theo, column= 'hardw_datatype', list_tuples_data_to_replace=[("-INT2", ""), ("-INT4", ""), ("-INT8", ""), ("-FP16", ""), ("-FP32", "")])      
    # 'hardw_datatype' column only has the hardware now
    df_theo['hardw_datatype_net_prun'] = df_theo['hardw_datatype']+'_'+df_theo['net_prun_datatype'].str.split('_').str[2] +'_'+ df_theo['network']+'_'+df_theo['net_prun_datatype'].str.split('_').str[1]
        
    #   delet unnecessary columns
    df_theo = df_theo.drop(columns = ['net_prun_datatype'])
    #  change column order
    df_theo= df_theo[['hardw_datatype_net_prun', 'hardw_datatype','network', 'fps-comp', 'top1', 'type']]

    #   rename columns
    df_theo.columns=['hardw_datatype_net_prun','hardw','network', 'fps-comp', 'top1', 'type']
    return df_theo
#-------------------------------------------------------------

def fix_small_stuff_df(df: pd.DataFrame(), col_to_drop: list, ) -> pd.DataFrame():
    """Method that: from the input dfs creates a pareto df & creates a plot from the pareto df, does this for all input dfs
    Method that  
   
    Parameters
    ----------
     : pd.DataFrame()
        Contains bla bla 
    
    Returns
    -------
    : pd.DataFrame()
       
        
    """
    #   delete all rows that have 'top1 (top5) [%]' inside
    df = df[df['top1'] !='top1 (top5) [%]']
    #    delete all rows with 'nm'
    df = df[df.top1!='nm'] 
    df = df.reset_index()
    #   merge 'net_prun' with 'datatype' column into 'net_prun_datatype'
    df['net_prun_datatype'] = df.net_prun + ' ' + df.datatype
    df = df.drop(columns = col_to_drop)

    #    Some cells have [top1 (top5)] accuracies, create col only with top1 acc
    df['top1'] = df['top1'].str.split(' ').str[0] #take top5 acc out
    #    separate by underscore instead of space
    df = replace_data_df(df_=df, column='net_prun_datatype', list_tuples_data_to_replace=[(' ','_')])
    return df

#--------------------------------------------------------

def identify_pairs_nonpairs(df: pd.DataFrame, column: str) -> pd.DataFrame():
    """This method identifies equal values in the column and signals them, and creates another column with labels for each case """
    """Method that: from the input dfs creates a pareto df & creates a plot from the pareto df, does this for all input dfs
    Method that  
   
    Parameters
    ----------
     : pd.DataFrame()
        Contains bla bla 
    
    Returns
    -------
    : pd.DataFrame()
       
        
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
    df['color'] = df.apply(lambda row: 'theoretical_no_match' if row.type=='predicted' and row.color=='' else 
                                                     ('measured_no_match' if row.type=='measured' and row.color=='' else (row.color)), axis=1)
    #df = df.drop(columns=['pairs'])
    return df

#-------------------------------------------------------

def plot_it_now(df: pd.DataFrame, xcol: str, ycol: str, groupcol: str, title: str) -> alt.vegalite.v4.api.Chart:
    """This method gets makes the overlapped pareto plots. With the 2 pareto lines and the points plot"""
    """Method that: from the input dfs creates a pareto df & creates a plot from the pareto df, does this for all input dfs
    Method that  
   
    Parameters
    ----------
     : pd.DataFrame()
        Contains bla bla 
    
    Returns
    -------
    : pd.DataFrame()
       
        
    """
    df_theo =df.loc[df.type=='predicted',:]
    df_exper = df.loc[df.type=='measured',:]
    df_charts = get_several_paretos_df(list_df = [df_theo, df_exper], groupcol= groupcol, xcol= xcol , ycol= ycol, colors=['#FFA500', '#0066CC'])
    chart1 = get_point_chart_enhanced(df= df, color_groupcol= 'color', shape_groupcol= 'type',shapes=['cross', 'circle'], xcol= xcol, ycol= ycol, title=title, legend_title_groupcol="Hardw_Datatype_Net_Prun" )
    chart = df_charts.charts.sum(numeric_only = False)
    charts = alt.layer(
        chart1,
        chart
    ).resolve_scale(color='independent',shape='independent').properties(title=title)
    return charts

#-----------------------------------------------------

def get_overlapped_pareto(net_keyword: str):
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
    # 1. Get the CNNs Accuracies table (Theoretical_Analysis/CNNs and their accuracy...) that only has the top1 accuracy and process it.
    #   theoretical top1
    df_top1_theo = process_theo_top1(csv_theor_accuracies ='data/cnn_topologies_accuracy.csv')
    #now we have: |top1 | net_prun_datatype| 
   
    # 2. Now we need Theoretical FPS-COMP to match with that Theoretical TOP1
    # 3. We need to get the above mentioned Theoretical FPS-COMP from the Heatmaps- Performance Predictions and merge them
    # depending on the user input this is retrieved for the desired Classification Task
    if re.search(net_keyword, 'imagenet', re.IGNORECASE):
        df_fps_top1_theo = process_theo_fps(df_top1_theo= df_top1_theo, csv_files=["data/cleaned_csv/performance_prediction_imagenet.csv"])
        df_measured = pd.read_csv('data/cleaned_csv/pareto_data_imagenet.csv')
    elif re.search(net_keyword, 'mnist', re.IGNORECASE):
        df_fps_top1_theo = process_theo_fps(df_top1_theo= df_top1_theo, csv_files=["data/cleaned_csv/performance_prediction_mnist.csv"])
        df_measured = pd.read_csv('data/cleaned_csv/pareto_data_mnist.csv')
    elif re.search(net_keyword, 'cifar-10', re.IGNORECASE):
        df_fps_top1_theo = process_theo_fps(df_top1_theo= df_top1_theo, csv_files=["data/cleaned_csv/performance_prediction_cifar10.csv"])
        df_measured = pd.read_csv('data/cleaned_csv/pareto_data_cifar.csv')
    
    df_fps_top1_theo = select_cnn_match_theo_for_measured(df_theo= df_fps_top1_theo, net_prun_datatype = 'net_prun_datatype')
    # now we have: |hardw_datatype_net_prun | hardw | network | fps-comp | top1 | type|

    #  concatenate both measured with theoretical to get the overlapped pareto
    overlapped_pareto = pd.concat([df_fps_top1_theo, df_measured])
    # now we have everything together and matched

    overlapped_pareto.sort_values(by='hardw_datatype_net_prun')
     
    # identify all pairs and create a special column for them 
    overlapped_pareto = identify_pairs_nonpairs(df=overlapped_pareto, column='hardw_datatype_net_prun')
    # now we have: |hardw_datatype_net_prun | hardw | network | fps-comp | top1 | type | color|
    
    #plot it
    return plot_it_now(df= overlapped_pareto, xcol= 'fps-comp', ycol= 'top1', groupcol= 'hardw_datatype_net_prun', title='Overlapped Pareto Plots Theoretical + Measured for' + ' ' + net_keyword.upper())    

#----------------------------------------------------

def get_point_chart(df: pd.DataFrame, groupcol: str, xcol: str, ycol:str, title: str) ->alt.vegalite.v4.api.Chart:
    """Creates simple line chart"""
    """Method that: from the input dfs creates a pareto df & creates a plot from the pareto df, does this for all input dfs
    Method that  
   
    Parameters
    ----------
     : pd.DataFrame()
        Contains bla bla 
    
    Returns
    -------
    : pd.DataFrame()
       
        
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
        text='hardw'
    )
    return (points+text).interactive()
#----------------------------------------------------

def theor_pareto(net_keyword, title: str) ->pd.DataFrame:
    """Creates a Theoretical Pareto Plot."""
    """Method that: from the input dfs creates a pareto df & creates a plot from the pareto df, does this for all input dfs
    Method that  
   
    Parameters
    ----------
     : pd.DataFrame()
        Contains bla bla 
    
    Returns
    -------
    : pd.DataFrame()
       
        
    """
    xcol='fps-comp'
    ycol='top1'
    groupcol='hardw_datatype_net_prun'
    colors=['#FFA500']
    # 1. Get the CNNs Accuracies table (Theoretical_Analysis/CNNs and their accuracy...) that only has the top1 accuracy and process it.
    #   theoretical top1  
    df_top1_theo = process_theo_top1(csv_theor_accuracies ='data/cnn_topologies_accuracy.csv')
    
    if re.search(net_keyword, 'imagenet', re.IGNORECASE):           
        df_fps_top1_theo = process_theo_fps(df_top1_theo= df_top1_theo, csv_files=["data/cleaned_csv/performance_prediction_imagenet.csv"])
    elif re.search(net_keyword, 'mnist', re.IGNORECASE):   
        df_fps_top1_theo = process_theo_fps(df_top1_theo= df_top1_theo, csv_files=["data/cleaned_csv/performance_prediction_mnist.csv"])
    elif re.search(net_keyword, 'cifar-10', re.IGNORECASE):   
        df_fps_top1_theo = process_theo_fps(df_top1_theo= df_top1_theo, csv_files=["data/cleaned_csv/performance_prediction_cifar10.csv"])     
    
    df_fps_top1_theo = select_cnn_match_theo_for_measured(df_theo= df_fps_top1_theo, net_prun_datatype = 'net_prun_datatype')
    # now we have: |hardw_datatype_net_prun | hardw | network | fps-comp | top1 | type|
    
    
    df_charts = get_several_paretos_df(list_df = [df_fps_top1_theo], xcol= xcol, ycol= ycol, groupcol= groupcol, colors=colors)
    points = get_point_chart(df= df_fps_top1_theo, groupcol= groupcol, xcol= xcol, ycol=ycol, title=title) 
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
    efficiency_df = df_.loc[:, ['hardw_datatype_net_prun']]
    efficiency_df['type'] = 'Theoret. Peak Compute'
    #G  et Peak Compute for all hardware
    peak_compute_hardw_df=pd.read_csv('data/peakPerfBandHardPlatf.csv')
    peak_compute_hardw_dict=pd.Series(peak_compute_hardw_df.Peak_Performance.values,index=peak_compute_hardw_df.Name).to_dict()
    efficiency_df['peak_compute'] = ''
    efficiency_df['peak_compute']=efficiency_df.apply(lambda row: fill_values(row.hardw_datatype_net_prun, peak_compute_hardw_dict), axis=1)

    #Get GOPS for all CNNs
    gops_df = pd.read_csv('data/cnn_topologies_compute_memory_requirements.csv')
    #get first two columns
    gops_df = gops_df.loc[:,[' ','Total OPs']]
    #rename column
    gops_df.columns=['network', 'gops']
    #remove first row with double names
    gops_df= gops_df.iloc[1:,:]
    # remove %, correct ResNet-50 to ResNet50...
    gops_df = replace_data_df(df_=gops_df, column= 'network', list_tuples_data_to_replace= [('%',''),('ResNet-50','ResNet50'),('EfficientNet Edge L','EfficientNetL'),('EfficientNet Edge S','EfficientNetS'),('EfficientNet Edge M','EfficientNetM')])
    #create dictionary out of the df
    gops_dict=pd.Series(gops_df.gops.values,index=gops_df.network).to_dict()
    #create a new column with the GOPs values
    efficiency_df['gops'] = efficiency_df.apply(lambda row: fill_values(row.hardw_datatype_net_prun, gops_dict), axis=1)
    #now that all is done lets create the theoretical performance column which will be called 'fps-comp' and calculate the numbers
    try:
        efficiency_df['fps-comp'] = efficiency_df.apply(lambda row: float(row.peak_compute)*1000/float(row.gops), axis=1)
    except ZeroDivisionError as err:
        print('An error has occurred. Possibly GOPs CNN value was 0. This means there was a mismatch between names in the data coming from cnn_topologies_compute_memory_requirements.csv file and the dataframe given as input.', err)
    
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
    )

#----------------------------------------------------

def efficiency_plot(net_keyword: str, df_theo_peak_compute: pd.DataFrame, title: str) -> alt.vegalite.v4.api.Chart:
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
    
    # 1. Get the CNNs Accuracies table (Theoretical_Analysis/CNNs and their accuracy...) that only has the top1 accuracy and process it.
    df_top1_theo = process_theo_top1(csv_theor_accuracies ='data/cnn_topologies_accuracy.csv')
    #now we have: |top1 | net_prun_datatype| 

    # 2. Now we need Theoretical FPS-COMP to match with that Theoretical TOP1
    # 3. We need to get the above mentioned Theoretical FPS-COMP from the Heatmaps- Performance Predictions and merge them
    # depending on the user input this is retrieved for the desired Classification Task
    df_fps_top1_theo = process_theo_fps(df_top1_theo= df_top1_theo, csv_files=["data/cleaned_csv/performance_prediction_imagenet.csv"])
    df_measured = pd.read_csv('data/cleaned_csv/pareto_data_imagenet.csv')

    #see which classification type is required
    if re.search(net_keyword, 'imagenet', re.IGNORECASE):           
        df_fps_top1_theo = process_theo_fps(df_top1_theo= df_top1_theo, csv_files=["data/cleaned_csv/performance_prediction_imagenet.csv"])
        df_measured = pd.read_csv('data/cleaned_csv/pareto_data_imagenet.csv')
    elif re.search(net_keyword, 'mnist', re.IGNORECASE):   
        df_fps_top1_theo = process_theo_fps(df_top1_theo= df_top1_theo, csv_files=["data/cleaned_csv/performance_prediction_mnist.csv"])
        df_measured = pd.read_csv('data/cleaned_csv/pareto_data_mnist.csv')
    elif re.search(net_keyword, 'cifar-10', re.IGNORECASE):   
        df_fps_top1_theo = process_theo_fps(df_top1_theo= df_top1_theo, csv_files=["data/cleaned_csv/performance_prediction_cifar10.csv"])     
        df_measured = pd.read_csv('data/cleaned_csv/pareto_data_cifar.csv')
    
    df_fps_top1_theo = select_cnn_match_theo_for_measured(df_theo= df_fps_top1_theo, net_prun_datatype = 'net_prun_datatype')
    # now we have: |hardw_datatype_net_prun | hardw | network | fps-comp | top1 | type|

    #  concatenate both measured with theoretical to get the overlapped
    overlapped_pareto = pd.concat([df_fps_top1_theo, df_measured])
    # now we have everything together and matched

    
    # identify all pairs and create a special column for them 
    overlapped_pareto = identify_pairs_nonpairs(df=overlapped_pareto, column='hardw_datatype_net_prun')
    # now we have: |hardw_datatype_net_prun | hardw | network | fps-comp | top1 | type | color|
    
    #remove the ones that don't have a match
    overlapped_pareto = overlapped_pareto.loc[(overlapped_pareto.color!='measured_no_match') & (overlapped_pareto.color!='theoretical_no_match')]
    
    #create a percentage column
    overlapped_pareto = get_percentage_colum(df=overlapped_pareto, col_elements='hardw_datatype_net_prun', newcol='percentage')
    
    #merge with peak compute df which has the data for te 3rd bar 
    overlapped_pareto= pd.concat([overlapped_pareto,df_theo_peak_compute])
    overlapped_pareto= overlapped_pareto.fillna('')
    overlapped_pareto['hardw'] = overlapped_pareto['hardw_datatype_net_prun'].str.split('_').str[0]

    overlapped_pareto =overlapped_pareto.sort_values(by='hardw_datatype_net_prun')

    #count how many times each 'hardw_datatype_net_prun' combination is repeated
    df_tmp =overlapped_pareto['hardw_datatype_net_prun'].value_counts().to_frame().rename(columns={'hardw_datatype_net_prun':'count'})
    #get the ones that are only repeated once
    list_to_remove =df_tmp.loc[df_tmp['count'] < 3, :].index
    #remove them 
    overlapped_pareto = overlapped_pareto[~overlapped_pareto.hardw_datatype_net_prun.isin(list_to_remove)]
    
    #split in case there are too many rows, because the faceted bar chart will be too full
    if overlapped_pareto.hardw_datatype_net_prun.unique().size < 8:
        return faceted_bar_chart(df=overlapped_pareto , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title ) 
    
    
    usb_devices_df= overlapped_pareto.loc[overlapped_pareto.hardw.str.contains('TPU|NCS|A53')]
    fpga_df= overlapped_pareto.loc[overlapped_pareto.hardw.str.contains('ZCU|Ultra')]
    gpu_df= overlapped_pareto.loc[overlapped_pareto.hardw.str.contains('TX2')]

    # Plot it - Faceted Bar chart
    #return faceted_bar_chart(df=usb_devices_df , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title ) | faceted_bar_chart(df=fpga_df , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title ) | faceted_bar_chart(df=gpu_df , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title) 
    a=faceted_bar_chart(df=usb_devices_df , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title ) 
    b=faceted_bar_chart(df=fpga_df , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title ) 
    c=faceted_bar_chart(df=gpu_df , xcol='type', ycol='fps-comp', colorcol='type', textcol='percentage', columncol='hardw_datatype_net_prun', title=title) 
    return (a.display() or b.display() or c.display())
    #return  overlapped_pareto
    #----------------------------------------------------

