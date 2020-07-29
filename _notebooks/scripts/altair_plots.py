#---------------------Imports-------------------------
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', 80)

import altair as alt
import csv
import re
#from overlapped_pareto import *



#--------------------
#utils functions
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

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------HEATMAPS-------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
def heatmap_rect(df: pd.DataFrame, 
                 title: str, 
                 mouseover_selection: alt.vegalite.v4.api.Selection, 
                 color_selection: alt.vegalite.v4.schema.channels.Color)->alt.vegalite.v4.api.Chart:
    """
    Function that creates and returns a Heatmap with the arguments provided. The Heatmap does not have text in it.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe whose walues will be plotted with a heatmap.       
    title : str
        Title to give to the plot.        
    mouseover_selection: altair.vegalite.v4.api.Selection
        The selection object that will be used in chart creation. 
        The type of the selection is one of: ["interval", "single", or "multi"].        
    color_selection: altair.vegalite.v4.schema.channels.Color 
        A FieldDef with Condition :raw-html:`<ValueDef>`. 
        It can define the folowing:
            shorthand=Undefined,
            aggregate=Undefined,
            bin=Undefined,
            condition=Undefined,
            field=Undefined,
            legend=Undefined,
            scale=Undefined,
            sort=Undefined,
            timeUnit=Undefined,
            title=Undefined,
            type=Undefined
    Returns
    -------
    Heatmap Chart: altair.vegalite.v4.api.Chart
        This is an Altair/Vega-Lite Heatmap chart.    
        
    """
    return alt.Chart(df, width=700, height=350).mark_rect(
        stroke='black', 
        strokeWidth=1, 
        invalid = None).add_selection(mouseover_selection).properties(title=title).encode(
        alt.X('x:O', title = 'Models'),
        alt.Y('y:O', title = 'Hardware Platforms'),
        color = color_selection, #alt.condition(mouseover_selection, alt.value(mouseover_color), color_selection), #this is because of color staying in before hoovering with mouse
        tooltip = [alt.Tooltip('values:Q', title = 'Input/sec'),
                   alt.Tooltip('x:N', title = 'Model'),
                   alt.Tooltip('y:N', title = 'Hardware Platform'),
                  ]
     
        )


def heatmap_text(df: pd.DataFrame, color_condition: dict)->alt.vegalite.v4.api.Chart:
    """
    Function that creates and returns a Text for the Heatmap. This text incorporates all numbers across the chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe whose walues will be plotted with a heatmap.       
    color_condition: dict
        {'condition': {'test': condition, 'value if true': value1}, 'value if false': value2}
    Returns
    -------
    Text Heatmap Chart: altair.vegalite.v4.api.Chart
        This is an Altair/Vega-Lite Text Heatmap chart.    
        
    """
    return alt.Chart(df).mark_text(color = 'white').encode(
    alt.X('x:O',  title = 'Models'),
    alt.Y('y:O',  title = 'Hardware Platforms' ),
    text = alt.Text('values:Q', format = '.0f'),
    color= color_condition,
    tooltip = [
               alt.Tooltip('values:Q', format = '.0f', title = 'Input/sec'),
               alt.Tooltip('x:N', title = 'Model'),
               alt.Tooltip('y:N', title = 'Hardware Platform'),
              ]
        )




def heatmap(csv_file: str, machine_learning_task:str, title: str)->alt.vegalite.v4.api.Chart:
    """
    Function that creates and returns a Heatmap + Text.
    
    Parameters
    ----------
    csv_file: str
        File path to the file with the Theoretical predictions       
    machine_learning_task: str
        Desired machine learning task to be plotted on the Heatmaps
    title: str
        Title to give to the plot
    Returns
    -------
    Heatmap + Text Heatmap chart: altair.vegalite.v4.api.Chart
        This is an Altair/Vega-Lite Text Heatmap chart.    
        
    """
   
    # First process the raw csv file to make it able to be plotted
    df = process_csv_for_heatmaps_plot(csv_file= csv_file, machine_learning_task=machine_learning_task)

    mouseover_selection = alt.selection_single(on='mouseover', nearest=True)
    color_selection = alt.Color('values:Q', title= 'Input/second', scale=alt.Scale(type='log', scheme='lightmulti'))
    color_condition = alt.condition(alt.datum.values > 1, alt.value('black'), alt.value('white'))

    heatmap_rect_ = heatmap_rect(df, title, mouseover_selection, color_selection)
    heatmap_text_ = heatmap_text(df, color_condition) 
    
    Heatmap = heatmap_rect_ + heatmap_text_
    return Heatmap

#---------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------ROOFLINES--------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
def clean_csv_rooflines(path_topologies, path_hardware):
    """
    Preprocesses the csv files to create the Rooflines for Hardware Platforms and for topologies.
    More precisely:
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
    
    df_topology.columns=['Name','Total OPs','Total Model Size','INT2','INT4','INT8','FP16','FP32'] 
   
    ## Calculate the Arithmetic intensity (x axis) for each NN based on Fwd ops and Total params
    df_topology = df_topology.drop(0)
    df_topology = pd.melt(df_topology, id_vars=['Name'], value_vars=['INT2','INT4','INT8','FP16','FP32'],value_name='arith_intens', var_name='datatype')
    df_topology.Name = df_topology.Name + ' ' + df_topology.datatype
    df_topology = df_topology.drop(columns=['datatype'])
    
    #to quadruplicate the dataframe so each row with (Platform, arith_intens) 
    #will be filled with 100 and then 0s to plot the vertical line later    
    df_topology = pd.concat([df_topology, df_topology, df_topology, df_topology])
    

    ## Preparing the NNs dataset to be ploted as vertical lines later
    # creating a y list [100,100,100,...75,75,...25,25,1...0.0001,0.0001] to plot a vertical line later
    df_topology['performance'] = [30]*round((len(df_topology.index))/4)   +   [1]*round((len(df_topology.index))/4)  +  [10]*round((len(df_topology.index))/4)  +   [0.1]*round((len(df_topology.index))/4) 

    ## Calculating the rooflines (y axis) for each hardware platform
    #--------------------------------Calculating the values to plot for the roofline model-----------
    maxX=160000
    #to create a list that represents the x axis with numbers between 0 and 1000
    x_axis = np.arange(0.1,maxX,1) 
    df_hardw_clean = pd.DataFrame(columns=['Name','arith_intens','performance']) 
    #Create hardware dataframe (df_hardw_clean) based on df_hardware
    #Each hardware platform will have 3 coordinates (x,y), initial point, turning point and final point
    for index, row in df_hardware.iterrows():             
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

    return df_topology, df_hardw_clean
#----------------------------------------------------------------------------------------------------

def line_chart_w_checkbox(data: pd.DataFrame, condition: dict, selection: alt.vegalite.v4.api.Selection)->alt.vegalite.v4.api.Chart:
    """
    This function creates an Altair line chart with checkboxes.
    
    Parameters
    ----------
        data: pd.DataFrame
            Dataframe from which the plot will be created.       
        condition: dict
            Condition for the color.
            Eg.: {'condition': {'selection': 'FPGAs  Ultra96  DPU  ZCU  ', 'type': 'nominal', 'field': 'Name'}, 'value': 'lightgray'}      
        selection: Selection
            Selection object to select what information the checkbox is tied to.
            Eg.: Selection('FPGAs  Ultra96  DPU  ZCU  ', SelectionDef({
                             bind: BindCheckbox({ input: 'checkbox' }),
                             fields: ['Hide'],
                             type: 'single'
                          }))
    Returns
    -------
        Line Chart with checkboxes.          
    """
    maxX=160000
    width =700 
    height = 500
    chart = alt.Chart(data, width=width,height=height).properties(title='Comparing Hardware Platforms Rooflines and Neural Networks Arithmetic Intensity').mark_line(clip=True).encode(
        alt.X('arith_intens:Q', 
              title = 'ARITHMETIC INTENSITY (OPS/BYTE)', 
              scale = alt.Scale(type='log', domain = (0.1,maxX) )
             ),
        alt.Y('performance:Q', 
              title = 'PERFORMANCE (TOPS/S)', 
              scale=alt.Scale(type='log', domain = (0.2,40) )
             ),    
        color=condition,
        order='arith_intens:Q',
    ).add_selection(selection)
    return chart



def rooflines(neural_network: str)->alt.vegalite.v4.api.Chart:
    """
    This function creates an Altair line chart with checkboxes. Creates a lot of them and then sums them up.
    
    Parameters
    ----------
        data: pd.DataFrame
            Dataframe from which the plot will be created.       
        neural_network:str
            neural network that will also be plotted besides all hardware platforms.
            Eg.:'imagenet|mnist|cifar'
                'imagenet'
     Returns
    -------
        Line Chart with checkboxes, all charts are summed up.          
    """
    maxX=160000
    width =700 
    height = 500
    
    nn_df, hw_df = clean_csv_rooflines(path_topologies='data/cnn_topologies_compute_memory_requirements.csv',
                        path_hardware='data/peakPerfBandHardPlatf.csv')

    
    #to select data to be plotted according to user input
    if neural_network in 'imagenet':
        nn_df   = nn_df[nn_df['Name'].str.contains("GoogLeNetv1|MobileNetv1|ResNet|EfficientNet")]
    elif neural_network in 'cifar':
        nn_df   = nn_df[nn_df['Name'].str.contains("CNV")]
    elif neural_network in 'mnist':
        nn_df   = nn_df[nn_df['Name'].str.contains("MLP")]
    elif neural_network != 'imagenet|mnist|cifar':
         return 'There were no results for the neural network asked. Please insert another network'

        
    #regex expression replace for all MLPs because they overlapp in the plot because they have the same value for MLP100%,MLP50%..
    #nn_df['Name']=nn_df['Name'].replace(r'.*MLP.*', 'MLP', regex=True)
    nn_df = replace_data_df(nn_df, 'Name',[('MLP 100%','MLP*'),('MLP 50%','MLP*'),('MLP 25%','MLP*'), ('MLP 12.5%','MLP*')])
    nn_df= nn_df.drop_duplicates()
    
    #This part is to create all plots binded to checkboes-------------
    #Selecting data for each checkbox, from dataset. Each checkbox will be tied to each one of these data        
    FPGA_data   = hw_df[hw_df['Name'].str.contains("Ultra96 DPU|ZCU")]
    NVIDIA_data = hw_df[hw_df['Name'].str.contains("TX2")]
    GOOGLE_data = hw_df[hw_df['Name'].str.contains("EdgeTPU")]
    INTEL_data  = hw_df[hw_df['Name'].str.contains("NCS")]

    INT2_data = nn_df[nn_df['Name'].str.contains("INT2")]
    INT4_data    = nn_df[nn_df['Name'].str.contains("INT4")]
    INT8_data    = nn_df[nn_df['Name'].str.contains("INT8")]
    FP16_data = nn_df[nn_df['Name'].str.contains("FP16")]
    FP32_data     = nn_df[nn_df['Name'].str.contains("FP32")]
    
    
    #To say that the binding type will be a checkbox
    #BindCheckbox({ input: 'checkbox'})
    filter_checkbox = alt.binding_checkbox()

    #To create all checkboxes with the specifications info for each set
    #Selection('FPGAs:', SelectionDef({ bind: BindCheckbox({ input: 'checkbox' }), fields: ['Ultra96 DPU,ZCU104,ZCU102,ZCU104 FINN,ZCU104 BISMO'], type: 'single' }))
    FPGA_select   = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="FPGAs  Ultra96  DPU  ZCU  ")                 
    NVIDIA_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="INVIDIA  TX2  maxn, maxp, maxq  ")
    GOOGLE_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="GOOGLE  EdgeTPU, fast, slow  ")
    INTEL_select  = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="INTEL  NCS  ")

    INT2_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="INT2")
    INT4_select    = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="INT4")   
    INT8_select    = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="INT8")   
    FP16_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="FP16")
    FP32_select     = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="FP32")

    #Color Condiotions for each plot
    #{'condition': {'selection': 'FPGAs:', 'type': 'nominal', 'field': 'Name'}, 'value': 'lightgray'}
    FPGA_cond     = alt.condition(FPGA_select, alt.Color("Name:N"), alt.value("lightgray"))
    NVIDIA_cond   = alt.condition(NVIDIA_select, alt.Color("Name:N"), alt.value("lightgray"))
    GOOGLE_cond   = alt.condition(GOOGLE_select, alt.Color("Name:N"), alt.value("lightgray"))
    INTEL_cond    = alt.condition(INTEL_select, alt.Color("Name:N"), alt.value("lightgray"))

    INT2_cond = alt.condition(INT2_select, alt.Color("Name:N"), alt.value("lightgray"))
    INT4_cond    = alt.condition(INT4_select, alt.Color("Name:N"), alt.value("lightgray"))
    INT8_cond    = alt.condition(INT8_select, alt.Color("Name:N"), alt.value("lightgray"))
    FP16_cond = alt.condition(FP16_select, alt.Color("Name:N"), alt.value("lightgray"))
    FP32_cond     = alt.condition(FP32_select, alt.Color("Name:N"), alt.value("lightgray"))

    #Creating all plots 
    
    FPGA_chart     = line_chart_w_checkbox(FPGA_data,     FPGA_cond,    FPGA_select)
    NVIDIA_chart   = line_chart_w_checkbox(NVIDIA_data,   NVIDIA_cond,  NVIDIA_select)
    GOOGLE_chart   = line_chart_w_checkbox(GOOGLE_data,   GOOGLE_cond,  GOOGLE_select)                         
    INTEL_chart    = line_chart_w_checkbox(INTEL_data,    INTEL_cond,   INTEL_select)

    INT2_chart =    line_chart_w_checkbox(INT2_data, INT2_cond, INT2_select)
    INT4_chart    = line_chart_w_checkbox(INT4_data,    INT4_cond,    INT4_select)
    INT8_chart    = line_chart_w_checkbox(INT8_data,    INT8_cond,    INT8_select)
    FP16_chart =    line_chart_w_checkbox(FP16_data, FP16_cond, FP16_select)
    FP32_chart     = line_chart_w_checkbox(FP32_data,     FP32_cond,     FP32_select)

   
    #--------------------------------------------------------------------------------------------------
    # Create line plot
    line_chart = alt.Chart().mark_line(clip=True).interactive().encode(
            alt.X('arith_intens:Q'), 
            alt.Y('performance:Q'),
            alt.Color('Name:N', legend=alt.Legend(columns=2))
    )
    
    
        #Create the selection which chooses nearest point on mouse hoover
    selection = alt.selection(type='single', nearest=True, on='mouseover', fields=['arith_intens']) #to leave suggestions on, just replace arith_intens wiith anything else
   
        #Create text plot to show the text values on mouse hoovering
    text = (line_chart).mark_text(align='left', dx=3, dy=-3,clip=True).encode(  text=alt.condition(selection, 'Name:N', alt.value(' ')))

    #Creates the points plot for the NNs. The points will be invisible
    selectors = alt.Chart().mark_point(clip=True).encode(
                alt.X('arith_intens:Q'), 
                alt.Y('performance:Q'),
                opacity=alt.value(0),
    ).add_selection(selection)
    

    
    chart_all = (pd.Series([INT2_chart, INT4_chart, INT8_chart, FP16_chart, FP32_chart], name="charts")).to_frame()
    
    #Chart = alt.layer(FPGA_chart + NVIDIA_chart + GOOGLE_chart + INTEL_chart + INT2_chart + INT4_chart + INT8_chart + FP16_chart+ FP32_chart
    #Chart = alt.layer(chart_filtered.squeeze() + FPGA_chart + NVIDIA_chart + GOOGLE_chart + INTEL_chart, selectors, text, data=dataframe, width=700, height=500)
    Chart = alt.layer(chart_all.charts.sum(numeric_only = False) + FPGA_chart + NVIDIA_chart + GOOGLE_chart + INTEL_chart, 
                      selectors, text, data= pd.concat([nn_df,hw_df]), width=700, height=500)
    
    #return nn_df
    return Chart
#----------------------------------------------------------------------------------------------------------------------------------
    # PROCESSING FOR PERFORMANCE PLOTS (LINE PLOT, BOXPLOT, PARETO GRAPH)
    
def norm_by_group(df: pd.DataFrame(), column:str, group_col:str)->pd.DataFrame():
    """ Normalizes pandas series by group """
    df["norm-"+column] = df.groupby(group_col)[column].apply(lambda x: (x / x.max()))
    return df

def select_color(sel: alt.vegalite.v4.api.Selection, column: str) -> dict:
    """ Easy way to set colors based on selection for altair plots
    """
    return alt.condition(sel, 
                      alt.Color(column),
                      alt.value('lightgray'))

def boxplot(df:pd.DataFrame(), xaxis:str, yaxis: str, color_col: str, facet_column: str, title: str)-> alt.vegalite.v4.api.Chart:
    """ Creates a boxplot based on the df, yaxis and title """
    return alt.Chart(df).mark_boxplot().encode(
        x=alt.X(xaxis + ':O'),
        y=alt.Y(yaxis, scale=alt.Scale(type="log"), title=yaxis),
        color=alt.Color(color_col + ':O', title='Pruning Factor'),
    ).facet(column=facet_column).properties(
        title = title,
    ).interactive()


def get_pareto_df(df: pd.DataFrame(), groupcol: str, xcol: str, ycol: str) -> pd.DataFrame():
    """Creates a pareto line from the dataframe. This function doesn't correctly correspond x to y datapoints"""

    pareto_line_df = df.groupby(groupcol)[xcol].max().to_frame("x")
    pareto_line_df['y'] = df.groupby(groupcol)[ycol].agg(lambda x: x.value_counts().index[0])
    pareto_line_df.sort_values('y', ascending=False, inplace=True)
    pareto_line_df['x'] = pareto_line_df.x.cummax()
    pareto_line_df.drop_duplicates('x', keep='first', inplace=True)
    pareto_line_df['group'] = pareto_line_df.index
    return pareto_line_df

def get_pareto_df_improved(df: pd.DataFrame(), groupcol: str, xcol: str, ycol: str) -> pd.DataFrame():
    """
    Creates a dataframe with the datapoints for a pareto line.
    Improved version, it deals with lines that go up, this function correctly corresponds x to y datapoints 
    
    Parameters
    ----------
        df: pd.DataFrame
            Dataframe from which the pareto line will be created       
        groupcol: st
           the dataframe column which has all hardware platforms 
        xcol: str
            the dataframe column which has the x axis information. Typically the fps-comp 
        ycol: str
           the dataframe columnwhich has the y axis information. Typically the top1 accuracy in % 
            
     Returns
    -------
        pareto_line_df: pd.DataFrame()
           dataframe with datapoints for a pareto line          
    """
    df_ = df.loc[:,[groupcol,ycol,xcol]]

    pareto_line_df = df.groupby(groupcol)[xcol].max().to_frame(xcol)

    pareto_line_df = pd.merge(pareto_line_df, df_, left_on=xcol, right_on=xcol, how='left')
    pareto_line_df = pareto_line_df.set_index(groupcol)
    pareto_line_df.columns = ['x','y']

    pareto_line_df.sort_values('y', ascending=False, inplace=True)
    pareto_line_df['x'] = pareto_line_df.x.cummax()
    pareto_line_df.drop_duplicates('x', keep='first', inplace=True)
    pareto_line_df['group'] = pareto_line_df.index
    pareto_line_df.head()
    return pareto_line_df


def pareto_graph(df: pd.DataFrame(), groupcol: str , xcol: str, ycol: str, W: int, H: int, title: str ) -> alt.vegalite.v4.api.Chart:
    """
    Creates a pareto graph with the inputs given.
    
    Parameters
    ----------
        data: pd.DataFrame
            Dataframe from which the pareto graph will be created       
        groupcol: st
           the dataframe column which has all hardware platforms 
        xcol: str
            the dataframe column which has the x axis information. Typically the fps-comp 
        ycol: str
           the dataframe column which has the y axis information. Typically the top1 accuracy in % 
        W: int
           Plot width           
        H: int
           Plot height 
        title: str
           Title to give to the plot
            
     Returns
    -------
        Line chart + Pareto chart          
    """
    df_pareto = get_pareto_df_improved(df, groupcol, xcol, ycol)

    df_lines = alt.Chart(df).mark_line(point=True).encode(
        x=xcol,
        y=alt.Y(ycol + ":Q", scale=alt.Scale(zero=False)),
        color=alt.Color(groupcol, legend=alt.Legend(columns=1, title = "Hardware_Quantization_Pruning")),
        #tooltip=["HWType", "Precision", "PruningFactor", "batch/thread/stream", ycol, xcol],
        tooltip=[groupcol, ycol, xcol],
    )
    df_pareto_plot = alt.Chart(df_pareto).mark_line().encode(
        x="x",
        y=alt.Y("y", scale=alt.Scale(zero=False)),
    )
    return (df_lines+df_pareto_plot).interactive().properties(
        width=W,
        height=H,
        title=title
    )

def pareto_graph_points(df: pd.DataFrame(), groupcol: str , xcol: str, ycol: str, W: int, H: int, title: str ) -> alt.vegalite.v4.api.Chart:
    """
    Creates a pareto graph with the inputs given.
    
    Parameters
    ----------
        data: pd.DataFrame
            Dataframe from which the pareto graph will be created       
        groupcol: st
           the dataframe column which has all hardware platforms 
        xcol: str
            the dataframe column which has the x axis information. Typically the fps-comp 
        ycol: str
           the dataframe column which has the y axis information. Typically the top1 accuracy in % 
        W: int
           Plot width           
        H: int
           Plot height 
        title: str
           Title to give to the plot
            
     Returns
    -------
        Line chart + Pareto chart          
    """
    df_pareto = get_pareto_df(df, groupcol, xcol, ycol)

    df_points = alt.Chart(df).mark_circle(size=200, opacity=1).encode(
        x=xcol,
        y=alt.Y(ycol + ":Q", scale=alt.Scale(zero=False)),
        color=alt.Color(groupcol, legend=alt.Legend(columns=2, title = "Hardware_Quantization_Pruning")),
        #tooltip=["HWType", "Precision", "PruningFactor", "batch/thread/stream", ycol, xcol],
        tooltip=[groupcol, xcol, ycol],
    )
    df_pareto_plot = alt.Chart(df_pareto).mark_line().encode(
        x="x",
        y=alt.Y("y", scale=alt.Scale(zero=False)),
    )
    return (df_points).interactive().properties(
        width=W,
        height=H,
        title=title
    )

def delete_unique_values(df: pd.DataFrame(), col_a: str, col_b:str)->pd.DataFrame():
    """ Delets DataFrame rows based on column values.
    Delets rows whose values in column A has only 1 unique value in column B.
    eg.:       Input:                    Desired output:
         col_a       col_b               col_a       col_b 
         NCS         3                    NCS         3
         NCS         8                    NCS         8
         EdgeTPU         14.5                
         EdgeTPU         14.5    
             
    Parameters
    ----------
        data: pd.DataFrame
            Dataframe with needed data.
        col_hardw: str
            dataframe column.
        col_data: str
            dataframe column.
    Returns
    -------
        df: pd.DataFrame()
            Processed DataFrame.
    """
    #drop all duplicated values of colB
    df = df[~df.duplicated(col_b)]
    #create new column that has the number of diferent values that column A has
    df['count'] = df.groupby(col_a)[col_a].transform('count')
    #delete rows that only have 1 data value ( eg.: only 1 power value)
    df = df.query('count!= 1')
    del df['count']
    return df


#--------------------------------------BAR CHARTS---------------------------------------------------------------

def simple_bar_chart(df:pd.DataFrame(), xaxis: str, yaxis: str, coloraxis: str, xaxis_title: str, yaxis_title: str, plot_title:str ) -> alt.vegalite.v4.api.Chart:
    """
    Method that returns a simple colored bar chart.
   
    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe with data to be plotted.
    xaxis: str
        Dataframe column which will be plotted on the x axis.
    yaxis: str
        Dataframe column which will be plotted on the y axis.
    coloraxis: str
        Dataframe column which will be used to set the bars colors.
    xaxis_title: str 
        X axis title.
    yaxis_title: str
        Y axis title.
    plot_title: str
        Plot title.
    
    Returns
    -------
    alt.vegalite.v4.api.Chart
       Simple bar chart
        
    """

     
    bars=alt.Chart(df).mark_bar().encode(
    x=alt.X(xaxis + ':N', 
            title= xaxis_title),
    y=alt.Y(yaxis + ':Q',
            title= yaxis_title,
            scale=alt.Scale(type='symlog')),
    color=alt.Color(coloraxis + ':N'),
)
    text = bars.mark_text(
   
    dy=-5  # Nudges text upwards so it doesn't appear on top of the bar
).encode(
    text= yaxis+':Q'
)
    return (bars + text).properties(width=350, height=350, title=plot_title).interactive()

#---------------------------------------------------------------------------------------

def get_compute_memory_cnn_chart(csv_file: str)-> alt.vegalite.v4.api.Chart:
    """
    Method that creates 2 simple bar charts based on two different columns('Total OPs' and 'Total Model Size') 
    from the csv file provided. 
   
    Parameters
    ----------
     csv_file: str
         File from wich data will be used for the bar charts.
    
    Returns
    -------
    alt.vegalite.v4.api.Chart:
       2 simple bar charts with colored bars.
        
    """

    df = pd.read_csv('data/cnn_topologies_compute_memory_requirements.csv')
    pd.options.display.float_format = '{:20,.2f}'.format
    df=df.drop(0)
    df['network'] = df[' '].str.split(' ').str[0]
    gops_chart= simple_bar_chart(df=df,
                                xaxis=' ',
                                yaxis='Total OPs',
                                coloraxis='network',
                                xaxis_title='All convolutional neural networks',
                                yaxis_title= 'Number of Operations [GOPs]',
                                plot_title= 'Compute and Memory Requirements for All CNNs in Number of Operations')

    model_size_chart= simple_bar_chart(df=df,
                                xaxis=' ',
                            yaxis='Total Model Size',
                            coloraxis='network',
                            xaxis_title='All convolutional neural networks',
                            yaxis_title= 'Total Model Size in Millions of Elements [ME]',
                            plot_title= 'Compute and Memory Requirements for All CNNs in Model Size')

    return gops_chart | model_size_chart

#---------------------------------------------------------------------------------------------

def get_peak_perf_bar_chart(csv_file)->alt.vegalite.v4.api.Chart:
    """
    Method that creates a simple grouped bar chart from the csv file:
   
    Parameters
    ----------
    csv_file: str
        csv file from which the bar chart will be created.
    
    Returns
    -------
    alt.vegalite.v4.api.Chart
        Simple, grouped bar chart.
        
    """

    df= pd.read_csv(csv_file)
    df=df.drop(0)
    df = pd.melt(df, id_vars=['Hardware Platforms'], value_vars=['INT2','INT4','INT8','FP16','FP32','Memory Bandwidth'], var_name='Datatypes and MB')
    df=df.dropna()

    bars= alt.Chart().mark_bar().encode(
        x=alt.X('Datatypes and MB:O', title=''),
        y=alt.Y('value:Q',scale=alt.Scale(type='log'), title='Peak Performance [TOP/sec] and MB [GBps]'),
        color='Datatypes and MB:N',
    )
    text = bars.mark_text(
    dy=-5  # Nudges text upwards so it doesn't appear on top of the bar
).encode(
    text= 'value:Q'
)
    return alt.layer(bars, text, data=df).facet(columns=5, facet=alt.Facet('Hardware Platforms:N', title='Hardware Platforms')).properties(title='Peak Performance and Memory Bandwidth for All Hardware Platforms')

#-------------------------------------------------------------------------------------------------------