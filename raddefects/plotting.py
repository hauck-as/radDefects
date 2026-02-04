#!/usr/bin/env python3
"""Python module used to setup and analyze carrier capture calculations using VASP."""
from pathlib import Path
from typing import Any, Union
import io
from string import ascii_lowercase as alc
import importlib.resources as ilr

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.colors import color_parser, convert_colors_to_same_type

from pypdf import PdfWriter, PdfReader, Transformation
from pypdf.annotations import FreeText
from contextlib import contextmanager
from fpdf import FPDF, get_scale_factor

import random as rand
import numpy as np
import pandas as pd
from pymatgen.util.typing import PathLike

from raddefects.carriercapture import calc_all_cap_rates

# interpolation and plotting functions
txt_positions_reg = ['top center', 'bottom center']
txt_positions_extra = ['top center', 'bottom center', 'top right', 'top left', 'bottom right', 'bottom left']


def improve_text_position(
    x: list[Any],
    txt_positions: list[str] = txt_positions_reg
) -> list[str]:
    """
    Function to improve the text position for Plotly scatter plots. More efficient if the x values are sorted.

    Args
    ---------
        x (list[Any]):
            List of any type of data to be used for annotating on a plot.
        txt_positions (list(str)):
            List of strings corresponding to possible text position options. Defaults to txt_positions_reg,
            defined prior to function definition as ['top center', 'bottom center']. Another option,
            txt_positions_extra, is included: ['top center', 'bottom center', 'top right', 'top left',
            'bottom right', 'bottom left']. Another option may be specified.

    Returns
    ---------
        List of text positions associated for each x value.
    """
    return [txt_positions[(i % len(txt_positions)-(rand.randint(1,2)))] for i in range(len(x))]


# Change Plotly default template to simple white and modify for 
pl_paper_theme = pio.templates['simple_white']
pl_paper_theme.layout.xaxis.ticks = 'inside'
pl_paper_theme.layout.yaxis.ticks = 'inside'
pl_paper_theme.layout.xaxis.minor.ticks = 'inside'
pl_paper_theme.layout.yaxis.minor.ticks = 'inside'
pl_paper_theme.layout.xaxis.mirror = 'ticks'  # True | "ticks" | False | "all" | "allticks"
pl_paper_theme.layout.yaxis.mirror = 'ticks'  # True | "ticks" | False | "all" | "allticks"
pl_paper_theme.layout.font.size = 24
# pl_paper_theme.layout.xaxis.title.standoff = 20
pl_paper_theme.layout.xaxis.title.font.size = 28
pl_paper_theme.layout.xaxis.tickfont.size = 24
# pl_paper_theme.layout.yaxis.title.standoff = 24
pl_paper_theme.layout.yaxis.title.font.size = 28
pl_paper_theme.layout.yaxis.tickfont.size = 24
# pl_paper_theme.layout.coloraxis.colorbar.title.standoff = 20
pio.templates.default = pl_paper_theme


def annotate_subfigs(subfigs_path, fig_path, num_figs=2, layout='horizontal', font_size=26, sub_list=alc):
    """
    """
    subfigs_pdf = PdfWriter(subfigs_path)
    subfigs_page = subfigs_pdf.pages[0]
    page_width, page_height = subfigs_page.mediabox.right/num_figs, subfigs_page.mediabox.top/num_figs
    
    
    @contextmanager
    def add_to_page(reader_page, unit='pt'):
        """
        Use function from fpdf2+pypdf tutorial to add annotations (change font
        size and style).
        https://py-pdf.github.io/fpdf2/CombineWithPypdf.html#combine-with-pypdf
        """
        k = get_scale_factor(unit)
        fpdf_format = (reader_page.mediabox[2] / k, reader_page.mediabox[3] / k)
        pdf = FPDF(format=fpdf_format, unit=unit)
        pdf.add_page()
        yield pdf
        page_overlay = PdfReader(io.BytesIO(pdf.output())).pages[0]
        reader_page.merge_page(page2=page_overlay)
    
    
    with add_to_page(subfigs_page) as pdf:
        pdf.set_font('times', style='B', size=font_size)
        if layout.lower()[0] == 'h':
            for i in range(num_figs):
                pdf.text((i*page_width)+10, 100, f'{alc[i]})')
        elif layout.lower()[0] == 'v':
            for i in range(num_figs):
                pdf.text(10, (i*page_height)+100, f'{alc[i]})')
        else:
            print('Please choose a valid layout (horizontal/vertical).')
    
    if fig_path is not None:
        subfigs_pdf.write(fig_path)
    
    return subfigs_pdf


def combine_figures(fig_list, fig_path, layout='horizontal', font_size=26, sub_list=alc):
    """
    Combine pdf figures into a single figure with alphabetical corner annotations.
    Assumes all figures are the same size and shape.
    """
    num_figs = len(fig_list)
    fig_dict = {pdf_path: PdfReader(pdf_path) for pdf_path in fig_list}
    combined_pdf = PdfWriter()
    combined_page = fig_dict[fig_list[0]].pages[0]
    page_width, page_height = combined_page.mediabox.right, combined_page.mediabox.top

    if layout.lower()[0] == 'h':
        combined_page.mediabox = combined_page.mediabox.scale(sx=num_figs, sy=1.)
        page_width, page_height = combined_page.mediabox.right/num_figs, combined_page.mediabox.top/num_figs
    elif layout.lower()[0] == 'v':
        combined_page.mediabox = combined_page.mediabox.scale(sx=1., sy=num_figs)
        page_width, page_height = combined_page.mediabox.right/num_figs, combined_page.mediabox.top/num_figs
        combined_page.add_transformation(Transformation().translate(tx=0, ty=(num_figs-1)*page_height))
    else:
        print('Please choose a valid layout (horizontal/vertical).')

    if layout.lower()[0] == 'h':
        for i in range(num_figs-1):
            combined_page.merge_translated_page(
                fig_dict[fig_list[i+1]].pages[0],
                tx=(i+1)*page_width,
                ty=0.,
                expand=False,
                over=False
            )
    elif layout.lower()[0] == 'v':
        for i in range(num_figs-1):
            combined_page.merge_translated_page(
                fig_dict[fig_list[i+1]].pages[0],
                tx=0.,
                ty=(num_figs-i-2)*page_height,
                expand=False,
                over=False
            )
    else:
        print('Please choose a valid layout (horizontal/vertical).')

    
    @contextmanager
    def add_to_page(reader_page, unit='pt'):
        """
        Use function from fpdf2+pypdf tutorial to add annotations (change font
        size and style).
        https://py-pdf.github.io/fpdf2/CombineWithPypdf.html#combine-with-pypdf
        """
        k = get_scale_factor(unit)
        fpdf_format = (reader_page.mediabox[2] / k, reader_page.mediabox[3] / k)
        pdf = FPDF(format=fpdf_format, unit=unit)
        pdf.add_page()
        yield pdf
        page_overlay = PdfReader(io.BytesIO(pdf.output())).pages[0]
        reader_page.merge_page(page2=page_overlay)
    

    with add_to_page(combined_page) as pdf:
        pdf.set_font('times', style='B', size=font_size)
        if layout.lower()[0] == 'h':
            for i in range(num_figs):
                pdf.text((i*page_width)+2, 25, f'{sub_list[i]})')
        elif layout.lower()[0] == 'v':
            for i in range(num_figs):
                pdf.text(2, (i*page_height)+25, f'{sub_list[i]})')
        else:
            print('Please choose a valid layout (horizontal/vertical).')
    
    combined_pdf.add_page(combined_page)
    
    if fig_path is not None:
        combined_pdf.write(fig_path)
    
    return combined_pdf


def combine_ccd_cap_coeff_figs(defect_name, q_initial, q_final, suffix='', cc_path=Path.cwd(), layout='h', font_size=26):
    """
    Combine CCD and scaled capture coefficient figures for a carrier capture calculation.
    """
    defect_name_charges = '_'.join([defect_name, str(q_initial), str(q_final)])
    if suffix != '':
        defect_name_charges = '_'.join([defect_name_charges, suffix])
    capture_calc_path = cc_path / defect_name_charges

    # CCD and capture coefficient figures
    ccd_path, cap_coeff_path = capture_calc_path / 'config-coord-dia.pdf', capture_calc_path / 'scaled_cap_coeff.pdf'

    # combine figures horizontally
    combine_figures(
        [ccd_path, cap_coeff_path],
        capture_calc_path / f'{defect_name_charges.lower()}.pdf',
        layout=layout,
        font_size=font_size
    )
    
    return None


def combine_all_ccd_cap_coeff_figs(cc_path=Path.cwd(), layout='h', font_size=26):
    """
    Combine CCD and scaled capture coefficient figures for all carrier capture calculation.
    """
    ccd_paths = list(cc_path.glob('*/config-coord-dia.pdf'))
    cap_coeff_paths = list(cc_path.glob('*/scaled_cap_coeff.pdf'))
    for i in range(len(ccd_paths)):
        defect_name_charges = ccd_paths[i].parent.name
        if defect_name_charges != cap_coeff_paths[i].parent.name:
            continue
            
        defect_info_list = defect_name_charges.split('_', 5)
        if len(defect_info_list) == 4:
            defect_atom, defect_site, q_i, q_f = defect_info_list
            suffix = ''
        elif len(defect_info_list) > 4:
            defect_atom, defect_site, q_i, q_f, suffix = defect_info_list
        
        defect_name = '_'.join([defect_atom, defect_site])
    
        combine_ccd_cap_coeff_figs(
            defect_name,
            q_i,
            q_f,
            suffix=suffix,
            cc_path=cc_path,
            layout=layout,
            font_size=font_size
        )
    
    return None


def combine_hole_ele_eff_barrier_figs(defect_name, q_initial, q_final, suffix='', cc_path=Path.cwd(), layout='h', font_size=26):
    """
    Combine effective hole and electron capture fit figures for a carrier capture calculation.
    """
    defect_name_charges = '_'.join([defect_name, str(q_initial), str(q_final)])
    if suffix != '':
        defect_name_charges = '_'.join([defect_name_charges, suffix])
    capture_calc_path = cc_path / defect_name_charges

    # hole and electron coefficient figures
    hole_barrier_path = capture_calc_path / 'fit_hole_cap_coeff_eff_barrier.pdf'
    ele_barrier_path = capture_calc_path / 'fit_ele_cap_coeff_eff_barrier.pdf'

    # combine figures horizontally
    combine_figures(
        [hole_barrier_path, ele_barrier_path],
        capture_calc_path / f'{defect_name_charges.lower()}_eff_barriers.pdf',
        layout=layout,
        font_size=font_size
    )
    
    return None

    
def combine_all_eff_barrier_figs(cc_path=Path.cwd(), layout='h', font_size=26):
    """
    Combine effective hole and electron capture fit figures for all carrier capture calculations.
    """
    hole_barrier_paths = list(cc_path.glob('*/fit_hole_cap_coeff_eff_barrier.pdf'))
    ele_barrier_paths = list(cc_path.glob('*/fit_ele_cap_coeff_eff_barrier.pdf'))
    for i in range(len(hole_barrier_paths)):
        defect_name_charges = hole_barrier_paths[i].parent.name
        if defect_name_charges != ele_barrier_paths[i].parent.name:
            continue
            
        defect_info_list = defect_name_charges.split('_', 5)
        if len(defect_info_list) == 4:
            defect_atom, defect_site, q_i, q_f = defect_info_list
            suffix = ''
        elif len(defect_info_list) > 4:
            defect_atom, defect_site, q_i, q_f, suffix = defect_info_list
        
        defect_name = '_'.join([defect_atom, defect_site])
    
        combine_hole_ele_eff_barrier_figs(
            defect_name,
            q_i,
            q_f,
            suffix=suffix,
            cc_path=cc_path,
            layout=layout,
            font_size=font_size
        )
    
    return None


def plot_plnr_avg_only(plot_data, title=None, ax=None, style_file=None):
    """
    Plots exclusively the planar-averaged electrostatic potential.

    Original code templated from the original PyCDT and new
    pymatgen.analysis.defects implementations as plot_FNV in doped.
    Functionality for plotting only the planar-averaged electrostatic
    potential is kept.

    Args:
         plot_data (dict):
            Dictionary of Freysoldt correction metadata to plot
            (i.e. defect_entry.corrections_metadata["plot_data"][axis] where
            axis is one of [0, 1, 2] specifying which axis to plot along (a, b, c)).
         title (str): Title for the plot. Default is no title.
         ax (matplotlib.axes.Axes): Axes object to plot on. If None, makes new figure.
         style_file (str):
            Path to a mplstyle file to use for the plot. If None (default), uses
            the default doped style (from doped/utils/doped.mplstyle).
    """

    x = plot_data["x"]
    dft_diff = plot_data["dft_diff"]

    style_file = style_file or f"{ilr.files('doped')}/utils/doped.mplstyle"
    plt.style.use(style_file)  # enforce style, as style.context currently doesn't work with jupyter
    with plt.style.context(style_file):
        if ax is None:
            plt.close("all")  # close any previous figures
            fig, ax = plt.subplots()
        (line1,) = ax.plot(x, dft_diff, c="red", label=r"$\Delta$(Locpot)")
        leg1 = ax.legend(handles=[line1], loc=9)  # middle top legend
        ax.add_artist(leg1)  # so isn't overwritten with later legend call

        ax.set_xlim(round(x[0]), round(x[-1]))
        ymin = min(*dft_diff)
        ymax = max(*dft_diff)
        ax.set_ylim(-0.2 + ymin, 0.2 + ymax)
        ax.set_xlabel(r"Distance along axis ($\AA$)")
        ax.set_ylabel("Potential (V)")
        ax.axhline(y=0, linewidth=0.2, color="black")
        if title is not None:
            ax.set_title(str(title))
        ax.set_xlim(0, max(x))

        return ax


def plot_plnr_avg_abc(plot_data, title=None, ax=None, style_file=None):
    axis_label_dict = {0: r"$a$-axis", 1: r"$b$-axis", 2: r"$c$-axis"}
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 3.5), dpi=600)
    for direction in range(3):
        plot_plnr_avg_only(
            plot_data['plot_data'][direction]['pot_plot_data'],
            ax=axs[direction],
            title=axis_label_dict[direction]
        )
    return


def plot_cap_rates(cap_coeff_path, conc_path, color_dict, temps=[300, 800], im_name=None):
    """
    Calculates nonradiative capture rates and plots them as a function of temperature.
    """
    fig = go.Figure()

    hole_cap_dict = {'Defect': [], 'Temperature (K)': [], 'Hole Capture Rate (cm^-3 s^-1)': [], 'Color': []}
    electron_cap_dict = {'Defect': [], 'Temperature (K)': [], 'Electron Capture Rate (cm^-3 s^-1)': [], 'Color': []}
    
    for i, T_i in enumerate(temps):
        cap_rate_df_i = calc_all_cap_rates((cap_coeff_path / f'capture_coeffs_{T_i}K.csv'), (conc_path / 'defect_concentration_temp.csv'), temp=T_i)

        defect_names = cap_rate_df_i.index
        q_pos, q_neg = cap_rate_df_i[['q_i', 'q_f']].max(axis=1), cap_rate_df_i[['q_i', 'q_f']].min(axis=1)
        defect_states_hole_cap = [f'{m.split('_')[0]}<sub>{m.split('_')[1]}</sub><sup>{n}</sup>' for m, n in zip(defect_names, q_neg)]
        defect_states_electron_cap = [f'{m.split('_')[0]}<sub>{m.split('_')[1]}</sub><sup>{n}</sup>' for m, n in zip(defect_names, q_pos)]
        defect_color_list = [color_dict[k] for k in defect_names]
        
        hole_cap_dict['Defect'].extend(defect_states_hole_cap)
        hole_cap_dict['Temperature (K)'].extend([T_i for j in defect_names])
        hole_cap_dict['Hole Capture Rate (cm^-3 s^-1)'].extend(cap_rate_df_i['R_p'])
        hole_cap_dict['Color'].extend(defect_color_list)
        
        electron_cap_dict['Defect'].extend(defect_states_electron_cap)
        electron_cap_dict['Temperature (K)'].extend([T_i for j in defect_names])
        electron_cap_dict['Electron Capture Rate (cm^-3 s^-1)'].extend(cap_rate_df_i['R_n'])
        electron_cap_dict['Color'].extend(defect_color_list)

    hole_cap_df, electron_cap_df = pd.DataFrame(hole_cap_dict), pd.DataFrame(electron_cap_dict)
    
    fig.add_trace(go.Scatter(
        x=hole_cap_df['Temperature (K)'], y=hole_cap_df['Hole Capture Rate (cm^-3 s^-1)'],
        text=hole_cap_df['Defect'], textposition=improve_text_position(hole_cap_df['Defect'], txt_positions=txt_positions_reg),
        name=r'$\Large{R_{p}}$',
        mode='markers+text',
        marker=dict(
            symbol='circle',
            color=hole_cap_df['Color'],
        )
    ))
    
    fig.add_trace(go.Scatter(
        x=electron_cap_df['Temperature (K)'], y=electron_cap_df['Electron Capture Rate (cm^-3 s^-1)'],
        text=electron_cap_df['Defect'], textposition=improve_text_position(electron_cap_df['Defect'], txt_positions=txt_positions_reg),
        name=r'$\Large{R_{n}}$',
        mode='markers+text',
        marker=dict(
            symbol='square',
            color=electron_cap_df['Color'],
        )
    ))
    
    fig.update_layout(
        xaxis=dict(
            title=r'$\Large{T \; \text{(K)}}$',
            range=[0, 1000]
        ),
        yaxis=dict(
            title=r'$\Large{R \; \text{(cm}^{-3} \text{s}^{-1} \text{)}}$',
            type='log'
        ),
        margin=dict(l=20, r=20, t=20, b=40),
        autosize=False, width = 1600, height = 900
    )

    if im_name is not None:
        fig.write_image(im_name, scale=2)
    
    return fig