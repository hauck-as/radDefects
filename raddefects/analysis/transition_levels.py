"""Analysis for charge transition levels within the pydefect format."""
import logging
from typing import TYPE_CHECKING, Optional

import os
import glob
from pathlib import Path
import json
from monty.serialization import dumpfn, loadfn
from monty.json import MSONable, MontyEncoder, MontyDecoder

import re
import pprint

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from pydefect.analyzer.unitcell import Unitcell
from pydefect.analyzer.band_edge_states import BandEdgeOrbitalInfos, BandEdgeState, PerfectBandEdgeState
from pydefect.analyzer.defect_energy import DefectEnergyInfo, ChargeEnergies, SingleChargeEnergies, CrossPoints
from pydefect.analyzer.transition_levels import TransitionLevels, make_transition_levels

# Change Plotly default template to simple white and modify for 
pl_paper_theme = pio.templates['simple_white']
pl_paper_theme.layout.xaxis.ticks = 'inside'
pl_paper_theme.layout.yaxis.ticks = 'inside'
pl_paper_theme.layout.xaxis.mirror = 'ticks'  # True | "ticks" | False | "all" | "allticks"
pl_paper_theme.layout.yaxis.mirror = 'ticks'  # True | "ticks" | False | "all" | "allticks"
pl_paper_theme.layout.font.size = 32
# pl_paper_theme.layout.xaxis.title.standoff = 20
pl_paper_theme.layout.xaxis.title.font.size = 44
pl_paper_theme.layout.xaxis.tickfont.size = 36
pl_paper_theme.layout.yaxis.title.standoff = 24
pl_paper_theme.layout.yaxis.title.font.size = 28
#pl_paper_theme.layout.coloraxis.colorbar.title.standoff = 20
pio.templates.default = pl_paper_theme


def generate_transition_level_diagram(transition_levels, im_write=False, im_name='transition_levels.png'):
    """
    Given a TransitionLevels object from pydefect, plots a charge transition level diagram
    showing the Fermi energy levels and which charges for each defect calculated.
    """
    # add capability to remove defects without transition levels or that are not of interest
    # ex. transition_levels.transition_levels.pop(3)
    defects_list = [defect.name for defect in transition_levels.transition_levels]
    
    fig = go.Figure()
    
    for i in range(len(defects_list)):
        if len(transition_levels.transition_levels[i].fermi_levels) > 0:
            for j in range(len(transition_levels.transition_levels[i].fermi_levels)):
                defect_fermi_levels = transition_levels.transition_levels[i].fermi_levels
                defect_fermi_levels.sort()
                fig.add_trace(go.Scatter(x=[transition_levels.transition_levels[i].name for j in range(len(defect_fermi_levels))],
                                         y=[defect_fermi_levels[j] for j in range(len(defect_fermi_levels))],
                                         text=[f'{transition_levels.transition_levels[i].charges[j][0]:+}/{transition_levels.transition_levels[i].charges[j][1]:+}' for j in range(len(defect_fermi_levels))],
                                         mode='markers+text',
                                         marker=dict(symbol='line-ew-open',
                                                     size=28,
                                                     color=px.colors.qualitative.Plotly[i],
                                                     line=dict(width=6)
                                                    )
                                        ))
        else:
            fig.add_trace(go.Scatter(x=[transition_levels.transition_levels[i].name],
                                     y=[0.],
                                     marker=dict(size=0, opacity=0)
                                    ))

    fig.update_traces(textposition='top center')

    # VBM
    fig.add_hline(y=transition_levels.supercell_vbm, line=dict(color='black', width=3, dash='dash'), annotation_text='VBM', annotation_position='bottom left')
    fig.add_hrect(y0=transition_levels.supercell_vbm-0.2, y1=transition_levels.supercell_vbm, line_width=0, fillcolor='red', opacity=0.2)
    
    # CBM
    fig.add_hline(y=transition_levels.supercell_cbm, line=dict(color='black', width=3, dash='dash'), annotation_text='CBM', annotation_position='top left')
    fig.add_hrect(y0=transition_levels.supercell_cbm, y1=transition_levels.supercell_cbm+0.2, line_width=0, fillcolor='blue', opacity=0.2)

    fig.update_layout(xaxis_title=r'Defects',
                      yaxis_title=r'$\Huge{E_{F} \; [eV]}$',
                      xaxis_type='category',
                      yaxis_showticklabels=False,
                      showlegend=False,
                      margin=dict(l=20, r=20, t=20, b=20),
                      autosize=False, width = 850, height = 650
                     )

    fig.show()
    
    if im_write == True:
        fig.write_image(im_name, scale=1)
    elif im_write == False:
        pass
    
    return

