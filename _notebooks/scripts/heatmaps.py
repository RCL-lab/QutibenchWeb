#---------------------Imports-------------------------
import numpy as np
import pandas as pd

import altair as alt

#---------------------Functions----------------------

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
    
    
    
    
    