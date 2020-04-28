
import altair as alt
from altair import datum

import numpy as np
import pandas as pd

import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import csv

from tabulate import tabulate
from pandas.plotting import table 

from labellines import labelLine, labelLines
from IPython.display import display, HTML


#Function to create a heatmap
def heatmap_rect(df, title, color, mouseover_selection, color_selection):
    return alt.Chart(df, width=700, height=350).mark_rect(
        stroke='black', 
        strokeWidth=1, 
        invalid = None).add_selection(mouseover_selection).properties(title=title).encode(
        alt.X('x:O', title = 'Models'),
        alt.Y('y:O', title = 'Hardware Platfroms'),
        color = alt.condition(mouseover_selection, alt.value(color), color_selection),
        tooltip = [alt.Tooltip('values:Q', title = 'Input/sec'),
                   alt.Tooltip('x:N', title = 'Model'),
                   alt.Tooltip('y:N', title = 'Hardware Platform'),
                  ]
     
        )


#Function to create a text to sum with heatmap
def heatmap_text(df, color_condition):
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

def heatmap(df_cifar10,mouseover_color):
    mouseover_selection = alt.selection_single(on='mouseover', nearest=True)
    color_selection = alt.Color('values:Q', title= 'Input/second',scale=alt.Scale(type='log', scheme='lightmulti'))
    color_condition = alt.condition(alt.datum.values > 1, alt.value('black'), alt.value('white'))

    Cifar10Heatmap = heatmap_rect(df_cifar10, 'Performance predictions for CIFAR 10', mouseover_color, mouseover_selection,color_selection)
    text_c = heatmap_text(df_cifar10, color_condition) 
    
    Cifar10Heatmap = Cifar10Heatmap + text_c
    return Cifar10Heatmap
    
    
    
    
    