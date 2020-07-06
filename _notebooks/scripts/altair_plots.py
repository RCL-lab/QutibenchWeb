#---------------------Imports-------------------------
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', 80)

import altair as alt
import csv


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------HEATMAPS-------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
def heatmap_rect(df: pd.DataFrame, 
                 title: str, 
                 mouseover_color: str, 
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
    mouseover_color: str
        Color that will be displayed when hoovering with the mouse over the plot       
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

def heatmap(dataframe: pd.DataFrame, mouseover_color: str, title: str)->alt.vegalite.v4.api.Chart:
    """
    Function that creates and returns a Heatmap + Text.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe whose walues will be plotted with a heatmap.       
    mouseover_color: str
        Color that will be displayed when hoovering with the mouse over the plot 
    title: str
        Title to give to the plot
    Returns
    -------
    Heatmap + Text Heatmap chart: altair.vegalite.v4.api.Chart
        This is an Altair/Vega-Lite Text Heatmap chart.    
        
    """
    mouseover_selection = alt.selection_single(on='mouseover', nearest=True)
    color_selection = alt.Color('values:Q', title= 'Input/second', scale=alt.Scale(type='log', scheme='lightmulti'))
    color_condition = alt.condition(alt.datum.values > 1, alt.value('black'), alt.value('white'))

    heatmap_rect_ = heatmap_rect(dataframe, title, mouseover_color, mouseover_selection, color_selection)
    heatmap_text_ = heatmap_text(dataframe, color_condition) 
    
    Heatmap = heatmap_rect_ + heatmap_text_
    return Heatmap

#---------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------ROOFLINES--------------------------------------
#-------------------------------------------------------------------------------------------------------------------------

# Checkboxes with on-plot tooltips
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
        color=condition
    ).add_selection(selection)
    return chart



def rooflines(dataframe: pd.DataFrame, neural_network: str)->alt.vegalite.v4.api.Chart:
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
    #hide_input
    maxX=160000
    width =700 
    height = 500
    data=dataframe
    
    #to select data to be plotted according to user input
    if neural_network in 'imagenet':
        nn_df   = dataframe[dataframe['Name'].str.contains("GoogLeNetv1|MobileNetv1|ResNet|EfficientNet")]
    elif neural_network in 'cifar':
        nn_df   = dataframe[dataframe['Name'].str.contains("CNV")]
    elif neural_network in 'mnist':
        nn_df   = dataframe[dataframe['Name'].str.contains("MLP")]
    elif neural_network in 'imagenet|mnist|cifar':
        nn_df   = dataframe[dataframe['Name'].str.contains("GoogLeNetv1|MobileNetv1|ResNet|EfficientNet|CNV|MLP")]
    else:
         return 'There were no results for the neural network asked. Please insert another network'
    
    #This part is to create all plots binded to checkboes-------------
    #Selecting data for each checkbox, from dataset. Each checkbox will be tied to each one of these data        
    FPGA_data   = dataframe[dataframe['Name'].str.contains("Ultra96 DPU|ZCU")]
    NVIDIA_data = dataframe[dataframe['Name'].str.contains("TX2")]
    GOOGLE_data = dataframe[dataframe['Name'].str.contains("EdgeTPU")]
    INTEL_data  = dataframe[dataframe['Name'].str.contains("NCS")]

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


    #Create a selector which will tell us the name of the Platform / Neural network
    selectors = alt.Chart().mark_point(clip=True).encode(
                alt.X('arith_intens:Q'), 
                alt.Y('performance:Q'),
                opacity=alt.value(0),
    ).add_selection(selection)
    
    chart_all = (pd.Series([INT2_chart, INT4_chart, INT8_chart, FP16_chart, FP32_chart], name="charts")).to_frame()
   
    #Chart = alt.layer(FPGA_chart + NVIDIA_chart + GOOGLE_chart + INTEL_chart + INT2_chart + INT4_chart + INT8_chart + FP16_chart+ FP32_chart
    #Chart = alt.layer(chart_filtered.squeeze() + FPGA_chart + NVIDIA_chart + GOOGLE_chart + INTEL_chart, selectors, text, data=dataframe, width=700, height=500)
    Chart = alt.layer(chart_all.charts.sum(numeric_only = False) + FPGA_chart + NVIDIA_chart + GOOGLE_chart + INTEL_chart, selectors, text, data=dataframe, width=700, height=500)

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
    df_pareto = get_pareto_df(df, groupcol, xcol, ycol)

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
                                plot_title= 'Compute and Memory Requirements for all CNNs in number of operations')

    model_size_chart= simple_bar_chart(df=df,
                                xaxis=' ',
                            yaxis='Total Model Size',
                            coloraxis='network',
                            xaxis_title='All convolutional neural networks',
                            yaxis_title= 'Total Model Size in Millions of Elements [ME]',
                            plot_title= 'Compute and Memory Requirements for all CNNs in Model size')

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
    return alt.layer(bars, text, data=df).facet(columns=5, facet=alt.Facet('Hardware Platforms:N', title='Hardware Platforms')).properties(title='Peak Performance and Memory Bandwidth for all Hardware Platforms')

#-------------------------------------------------------------------------------------------------------