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
def line_chart_w_checkbox(data: pd.DataFrame, condition: dict, selection: Selection)->alt.vegalite.v4.api.Chart:
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


def line_chart_no_checkbox(data: pd.DataFrame, condition: dict, selection)->alt.vegalite.v4.api.Chart:
    """
    This function creates an Altair line chart with no checkboxes.
    
    Parameters
    ----------
        data: pd.DataFrame
            Dataframe from which the plot will be created.       
        condition: dict
            Condition for the color.
            Eg.: {'condition': {'selection': 'FPGAs  Ultra96  DPU  ZCU  ', 'type': 'nominal', 'field': 'Name'}, 'value': 'lightgray'}      
    Returns
    -------
        Line Chart with no checkboxes.          
    """
    maxX=160000
    width =600 
    height = 400
    chart = alt.Chart(data, width=width,height=height).properties(title='Comparing Hardware Platforms Rooflines and Neural Networks Arithmetic Intensity with checkboxes').mark_line(clip=True).encode(
        alt.X('arith_intens:Q', 
              title = 'ARITHMETIC INTENSITY (OPS/BYTE)', 
              scale = alt.Scale(type='log', domain = (0.1,maxX) )
             ),
        alt.Y('performance:Q', 
              title = 'PERFORMANCE (TOPS/S)', 
              scale=alt.Scale(type='log', domain = (0.2,40) )
             ),    
        color=alt.Color("Name:N")
    )
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
    
    #This part is to create all plots binded to checkboes-------------
    #Selecting data for each checkbox, from dataset. Each checkbox will be tied to each one of these data
    FPGA_data   = dataframe[dataframe['Name'].str.contains("Ultra96 DPU|ZCU")]
    NVIDIA_data = dataframe[dataframe['Name'].str.contains("TX2")]
    GOOGLE_data = dataframe[dataframe['Name'].str.contains("TPU")]
    INTEL_data  = dataframe[dataframe['Name'].str.contains("NCS")]

    IMAGENET_data = dataframe[dataframe['Name'].str.contains("ResNet|GoogLeNet|MobileNet|VGG|AlexNet")]
    MNIST_data    = dataframe[dataframe['Name'].str.contains("MLP")]
    CIFAR_data    = dataframe[dataframe['Name'].str.contains("CNV")]
    MASKRCNN_data = dataframe[dataframe['Name'].str.contains("MaskRCNN")]
    GNMT_data     = dataframe[dataframe['Name'].str.contains("GNMT")]

    #To say that the binding type will be a checkbox
    #BindCheckbox({ input: 'checkbox'})
    filter_checkbox = alt.binding_checkbox()

    #To create all checkboxes with the specifications info for each set
    #Selection('FPGAs:', SelectionDef({ bind: BindCheckbox({ input: 'checkbox' }), fields: ['Ultra96 DPU,ZCU104,ZCU102,ZCU104 FINN,ZCU104 BISMO'], type: 'single' }))
    FPGA_select   = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="FPGAs  Ultra96  DPU  ZCU  ")                 
    NVIDIA_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="INVIDIA  TX2  maxn, maxp, maxq  ")
    GOOGLE_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="GOOGLE  TPU, fast, slow  ")
    INTEL_select  = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="INTEL  NCS  ")

    IMAGENET_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="IMAGENET  ResNet  GoogLeNet  MobileNet  VGG  AlexNet  ")
    MNIST_select    = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="MNIST  MLP  ")   
    CIFAR_select    = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="CIFAR-10  CNV  ")   
    MASKRCNN_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="MASKRCNN  ")
    GNMT_select     = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="GNMT  ")

    #Color Condiotions for each plot
    #{'condition': {'selection': 'FPGAs:', 'type': 'nominal', 'field': 'Name'}, 'value': 'lightgray'}
    FPGA_cond     = alt.condition(FPGA_select, alt.Color("Name:N"), alt.value("lightgray"))
    NVIDIA_cond   = alt.condition(NVIDIA_select, alt.Color("Name:N"), alt.value("lightgray"))
    GOOGLE_cond   = alt.condition(GOOGLE_select, alt.Color("Name:N"), alt.value("lightgray"))
    INTEL_cond    = alt.condition(INTEL_select, alt.Color("Name:N"), alt.value("lightgray"))

    IMAGENET_cond = alt.condition(IMAGENET_select, alt.Color("Name:N"), alt.value("lightgray"))
    MNIST_cond    = alt.condition(MNIST_select, alt.Color("Name:N"), alt.value("lightgray"))
    CIFAR_cond    = alt.condition(CIFAR_select, alt.Color("Name:N"), alt.value("lightgray"))
    MASKRCNN_cond = alt.condition(MASKRCNN_select, alt.Color("Name:N"), alt.value("lightgray"))
    GNMT_cond     = alt.condition(GNMT_select, alt.Color("Name:N"), alt.value("lightgray"))

    #Creating all plots 
    FPGA_chart     = line_chart_w_checkbox(FPGA_data,     FPGA_cond,    FPGA_select)
    NVIDIA_chart   = line_chart_w_checkbox(NVIDIA_data,   NVIDIA_cond,  NVIDIA_select)
    GOOGLE_chart   = line_chart_w_checkbox(GOOGLE_data,   GOOGLE_cond,  GOOGLE_select)                         
    INTEL_chart    = line_chart_w_checkbox(INTEL_data,    INTEL_cond,   INTEL_select)

    IMAGENET_chart = line_chart_w_checkbox(IMAGENET_data, IMAGENET_cond, IMAGENET_select)
    MNIST_chart    = line_chart_w_checkbox(MNIST_data,    MNIST_cond,    MNIST_select)
    CIFAR_chart    = line_chart_w_checkbox(CIFAR_data,    CIFAR_cond,    CIFAR_select)
    MASKRCNN_chart = line_chart_w_checkbox(MASKRCNN_data, MASKRCNN_cond, MASKRCNN_select)
    GNMT_chart     = line_chart_w_checkbox(GNMT_data,     GNMT_cond,     GNMT_select)

    #--------------------------------------------------------------------------------------------------
    #The following part was adapted from https://stackoverflow.com/questions/53287928/tooltips-in-altair-line-charts
    #This part will add information to the plot
    # Step 1: create the lines
    lines = alt.Chart().mark_line(clip=True).interactive().encode(
            alt.X('arith_intens:Q'), 
            alt.Y('performance:Q'),
            alt.Color('Name:N', legend=alt.Legend(columns=2))
    )

    # Step 2: Selection that chooses nearest point based on value on x-axis
    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['arith_intens']) #to leave suggestions on, just replace arith_intens wiith anything else


    # Step 3: Transparent selectors across the chart. This is what tells us the name of the Platform / Neural network
    selectors = alt.Chart().mark_point(clip=True).encode(
                alt.X('arith_intens:Q'), 
                alt.Y('performance:Q'),
                opacity=alt.value(0),
    ).add_selection(nearest)

    # Step 4: Add text, show values about platforms when it's the nearest point to 
    # mouseover, else show blank
    text = (lines).mark_text(align='left', dx=3, dy=-3,clip=True).encode(  text=alt.condition(nearest, 'Name:N', alt.value(' ')))
    
    chart_all = (pd.Series([IMAGENET_chart, MNIST_chart, CIFAR_chart, MASKRCNN_chart, 
                               GNMT_chart], name="charts")).to_frame()
    chart_all.index = ['imagenet', 'mnist', 'cifar10', 'maskrcnn','gnmt']
    chart_all.index.name = 'index'
    
    chart_filtered = chart_all[chart_all.index.astype(str).str.contains(neural_network)] 
    if len(chart_filtered.index) == 0:
        return 'There were no results for the neural network asked. Please insert another network'
    
    #Chart = alt.layer(FPGA_chart + NVIDIA_chart + GOOGLE_chart + INTEL_chart + IMAGENET_chart + MNIST_chart + CIFAR_chart + MASKRCNN_chart+ GNMT_chart
    #Chart = alt.layer(chart_filtered.squeeze() + FPGA_chart + NVIDIA_chart + GOOGLE_chart + INTEL_chart, selectors, text, data=dataframe, width=700, height=500)
    Chart = alt.layer(chart_filtered.charts.sum(numeric_only = False) + FPGA_chart + NVIDIA_chart + GOOGLE_chart + INTEL_chart, selectors, text, data=dataframe, width=700, height=500)

    return Chart

#----------------------------------------------------------------------------------------------------------------------------------
    # PROCESSING FOR PERFORMANCE PLOTS (LINE PLOT, BOXPLOT, PARETO GRAPH)
    
def norm_by_group(df: pd.DataFrame(), column:str, group_col:str)->pd.DataFrame():
    """ Normalizes pandas series by group """
    df["norm-"+column] = df.groupby(group_col)[column].apply(lambda x: (x / x.max()))
    return df

def select_color(sel: Selection, column:str) -> dict:
    """ Easy way to set colors based on selection for altair plots
    """
    return alt.condition(sel, 
                      alt.Color(column),
                      alt.value('lightgray'))

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

def label_point(x, y, val, ax, rot=0):
    """ from https://stackoverflow.com/questions/46027653/adding-labels-in-x-y-scatter-plot-with-seaborn"""
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']), rotation=rot)


def boxplot(df:pd.DataFrame(), yaxis: str, title: str)-> alt.vegalite.v4.api.Chart:
    """ Creates a boxplot based on the df, yaxis and title """
    return alt.Chart(df).mark_boxplot().encode(
    x=alt.X('PruningFactor:O'),
    y=alt.Y(yaxis, scale=alt.Scale(type="log"), title=yaxis),
    color=alt.Color('PruningFactor:O', title='Pruning Factor'),
    ).facet(column="quant_model").properties(
    title = title,
    ).interactive()

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
           the dataframe columnwhich has the y axis information. Typically the top1 accuracy in % 
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
        tooltip=["HWType", "Precision", "PruningFactor", "batch/thread/stream", ycol, xcol],
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

