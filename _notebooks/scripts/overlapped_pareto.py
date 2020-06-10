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
    """Melts a dataframe into 2 columns, the 'cnn_names_col' and the 'value' column. Return the melted dataframe"""
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

def spot_no_match(list_: list):
    """Creates a list of hexadecimal colors. The colors depend wheteher there is a substring inside each
    list item. For 'no match' the color is black, else, the color is created randomly
    
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

def get_point_chart_simple(df: pd.DataFrame, 
                    color_groupcol: str, 
                    shape_groupcol: str,  
                    xcol: str, 
                    ycol: str, 
                    shapes: str,
                    title: str,
                    legend_title_groupcol: str)->alt.vegalite.v4.api.Chart: 
    """Creates an elaborated point chart with all these configurations"""
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
    """Creates simple line chart"""
    return alt.Chart(df).interactive().mark_line(point=True).encode(
        x=alt.X(xcol, scale=alt.Scale(type='log')),
        y=alt.Y(ycol + ":Q", scale=alt.Scale(zero=False)),
        color=alt.value(color),
        tooltip=[groupcol, xcol, ycol],
    )
#---------------------------------------------------

def get_pareto_df(df: pd.DataFrame(), groupcol: str, xcol: str, ycol: str) -> pd.DataFrame():
    """Creates a pareto line from the dataframe. This function doesn't correctly correspond x to y datapoints"""
    pareto_line_df = df.groupby(groupcol)[xcol].max().to_frame("x")
    pareto_line_df['y'] = df.groupby(groupcol)[ycol].agg(lambda x: x.value_counts().index[0])
    pareto_line_df.sort_values('y', ascending=False, inplace=True)
    pareto_line_df['x'] = pareto_line_df.x.cummax()
    pareto_line_df.drop_duplicates('x', keep='first', inplace=True)
    pareto_line_df['group'] = pareto_line_df.index
    return pareto_line_df

#-----------------------------------------------------


def get_several_paretos_df(list_df: pd.DataFrame, groupcol: str, xcol: str, ycol:str, colors: list)->list:
    """Method that: from the input dfs creates a pareto df & creates a plot from the pareto df, does this for all input dfs"""
    list_df_out_charts = pd.DataFrame(columns=['charts'])
    for i, df in enumerate(list_df):
        pareto_df = get_pareto_df(df= df , groupcol= groupcol, xcol= xcol, ycol= ycol)
        chart = get_line_chart(df= pareto_df, groupcol= 'group', xcol= 'x', ycol= 'y', color = colors[i]) 
        list_df_out_charts = list_df_out_charts.append(pd.DataFrame([[chart]], columns=['charts']))
    return list_df_out_charts

#---------------------------------------------

def replace_data_df(df_: pd.DataFrame(), column:str, list_tuples_data_to_replace: list )-> pd.DataFrame():
    """Method to replace a substring inside a cell inside a dataframe"""
    df = df_.copy()
    for j, k in list_tuples_data_to_replace:
        df[column] = df[column].str.replace(j, k)
    return df
#-------------------------------------------------

def process_measured_df(df_theoret: pd.DataFrame(), csv_measured:str ):
    #   get the measured dataframe
    df_measured = pd.read_csv(csv_measured)
    #   fix samll stuff in the measured dataframe so things match
    df_measured = replace_data_df(df_=df_measured, column='hardw_datatype_net_prun', list_tuples_data_to_replace=[("RN50", "ResNet50"),("MNv1", "MobileNetv1"),('GNv1','GoogLeNetv1'),('100.0','100')])
    df_measured = replace_data_df(df_=df_measured, column='network', list_tuples_data_to_replace=[("RN50", "ResNet50"),("MNv1", "MobileNetv1"),('GNv1','GoogLeNetv1')])
    #  concatenate both measured with theoretical
    df = pd.concat([df_theoret, df_measured])

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
    df_theo['type'] = 'theoretical'
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
    df['color'] = df.apply(lambda row: 'theoretical_no_match' if row.type=='theoretical' and row.color=='' else 
                                                     ('measured_no_match' if row.type=='measured' and row.color=='' else (row.color)), axis=1)
    df = df.drop(columns=['pairs'])
    return df

#-------------------------------------------------------

def plot_it_now(df: pd.DataFrame, xcol: str, ycol: str, groupcol: str, title: str) -> alt.vegalite.v4.api.Chart:
    """This method gets makes the overlapped pareto plots. With the 2 pareto lines and the points plot"""
    df_theo =df.loc[df.type=='theoretical',:]
    df_exper = df.loc[df.type=='measured',:]
    df_charts = get_several_paretos_df(list_df = [df_theo, df_exper], groupcol= groupcol, xcol= xcol , ycol= ycol, colors=['#FFA500', '#0066CC'])
    chart1 = get_point_chart_simple(df= df, color_groupcol= 'color', shape_groupcol= 'type',shapes=['cross', 'circle'], xcol= xcol, ycol= ycol, title=title, legend_title_groupcol="Hardw_Datatype_Net_Prun" )
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
