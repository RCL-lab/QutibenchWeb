#---------------------Imports-------------------------
import numpy as np
import pandas as pd

import altair as alt
import csv
from IPython.display import display, HTML
#---------------------HEATMAPS----------------------

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
        alt.Y('y:O', title = 'Hardware Platfroms'),
        color = alt.condition(mouseover_selection, alt.value(mouseover_color), color_selection),
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
    alt.Y('y:O',  title = 'Hardware Platfroms' ),
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
#-------------------------------ROOFLINES--------------------------------------
#-------------------------------------------------------------------------------------------------------------------------

#hide
# Checkboxes with on-plot tooltips
def line_chart_w_checkbox(data, condition, selection):
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


#hide
def line_chart_no_checkbox(data, condition, selection):
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

def rooflines(dataframe, neural_network: str):
    #hide_input
    maxX=160000
    width =700 
    height = 500
    data=dataframe

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
    FPGA_select   = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="FPGAs Ultra96 DPU ZCU")                 
    NVIDIA_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="HNVIDIA TX2 maxn,maxp,maxq")
    GOOGLE_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="GOOGLE TPU,fast,slow")
    INTEL_select  = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="INTEL NCS")

    IMAGENET_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="IMAGENET ResNet GoogLeNet MobileNet VGG AlexNet")    
    MNIST_select    = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="MNIST MLP")   
    CIFAR_select    = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="CIFAR10 CNV")   
    MASKRCNN_select = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="MASKRCNN")
    GNMT_select     = alt.selection_single( fields=["Hide"], bind=filter_checkbox, name="GNMT")

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
    #Adapted from https://stackoverflow.com/questions/53287928/tooltips-in-altair-line-charts
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

    # Layer them all together
    Chart = alt.layer(FPGA_chart + NVIDIA_chart + GOOGLE_chart + INTEL_chart + IMAGENET_chart + MNIST_chart + CIFAR_chart + MASKRCNN_chart + GNMT_chart, selectors, text, data=dataframe, width=700, height=500)
    return Chart


#-------------------------------TABLES OVERVIEW OF THE EXPERIMENTS-----------------------------------------

## Just some needed functions
#Function to read from a csv file and return a numpy 2D array
def read_from_csv(filename):  
    array= []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            array.append(row)
    array = np.asarray(array)
    return array

def load_and_display(filenames):
    dataframes=[]
    for filename in filenames:
        table = read_from_csv(filename)  # To read from a csv file into a 2D numpy array
        dataframe = pd.DataFrame(data=table[2:,:], columns=[table[0,0:], table[1,0:]])  #To transform to dataframe the first and second row will be header
        dataframe.loc[dataframe.duplicated(dataframe.columns[0]) , dataframe.columns[0]] = ''  #To remove duplicates from first column
        dataframes.append(dataframe)     #To save all dataframes in here
    pd.set_option('display.width', 2000)
    return dataframes

def tableOverviewExperiments(filenames):
    pd.set_option('display.width', 2000)
    dataframes = load_and_display(filenames)
    for dataframe in dataframes:    
        return display(HTML(dataframe.to_html(index=False)))
#-----------------------------------------------------------------------------------------------------------
    