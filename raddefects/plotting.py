"""Python module used to setup and analyze carrier capture calculations using VASP."""
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, Union
from numpy.typing import ArrayLike
from pymatgen.util.typing import PathLike
import io
import re
from string import ascii_lowercase as alc
import importlib.resources as ilr

import random as rand
import numpy as np
import pandas as pd

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

from pydefect.analyzer.transition_levels import TransitionLevels
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
pl_paper_theme.layout.font.size = 32
# pl_paper_theme.layout.xaxis.title.standoff = 20
pl_paper_theme.layout.xaxis.title.font.size = 44
pl_paper_theme.layout.xaxis.tickfont.size = 36
# pl_paper_theme.layout.yaxis.title.standoff = 24
pl_paper_theme.layout.yaxis.title.font.size = 44
pl_paper_theme.layout.yaxis.tickfont.size = 36
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


def generate_transition_level_diagram(
    transition_levels: TransitionLevels,
    skip_defects: list = [],
    charge_style: str = 'mid',
    q_prox_tol: float | int = 0.05,
    fig_name: PathLike | None = 'transition_levels.pdf'
) -> go.Figure():
    """
    Given a TransitionLevels object from pydefect, plots a charge transition level diagram
    showing the Fermi energy levels and which charges for each defect calculated.

    Args
    ---------
        transition_levels (TransitionLevels):
            `pydefect` TransitionLevels object to get CTL values.
        skip_defects (list):
            List of defect names to skip from the transition_levels file. Defaults
            to an empty list.
        charge_style (str):
            String representing style for how to print the charge states. Defaults
            to 'mid', the midpoint between the nearest CTLs.
        q_prox_tol (float or int):
            Tolerance value for two CTLs or CTLs and band edges, below which
            the charge annotation is not printed. Defaults to 0.05 eV.
        fig_name (PathLike or None):
            Name for figure to save or None to not save a figure. Defaults to
            'transition_levels.pdf'.

    Returns
    ---------
        Figure showing CTL values in the bandgap.
    """
    for idx, tl in enumerate(transition_levels.transition_levels):
        if tl.name in skip_defects:
            transition_levels.transition_levels.pop(idx)
        
    defects_list = [defect.name for defect in transition_levels.transition_levels]

    # format defect names for plotting
    formatted_defects = list(map(lambda s: re.sub(r'\d+', '', s), defects_list))  # remove subscript numbers
    formatted_defects = list(map(lambda s: re.sub('Va', 'V', s), formatted_defects))  # vacancy Va->V
    for m, defect_m in enumerate(formatted_defects):
        if '–' in defect_m:  # check for defect complex with en dashes
            c_split = defect_m.split('–')
            complex_list = []
            for c, defect_c in enumerate(c_split):
                m_split = defect_c.split('_')
                if m_split[0] == 'V':  # vacancy
                    complex_list.append(fr'\LARGE{{{m_split[0]}}}_{{\LARGE{{\text{{{m_split[1]}}}}}}}')
                elif m_split[1] == 'i':  # interstitial
                    complex_list.append(fr'\LARGE{{\text{{{m_split[0]}}}}}_{{\LARGE{{{m_split[1]}}}}}')
                else:  # other
                    complex_list.append(fr'\LARGE{{\text{{{m_split[0]}}}}}_{{\LARGE{{\text{{{m_split[1]}}}}}}}')
            formatted_defects[m] = '$'+'–'.join(complex_list)+'$'
        else:  # not a defect complex
            m_split = defect_m.split('_')
            if m_split[0] == 'V':  # single vacancy
                formatted_defects[m] = fr'$\LARGE{{{m_split[0]}}}_{{\LARGE{{\text{{{m_split[1]}}}}}}}$'
            elif re.match('(V)+', m_split[0]) and len(m_split[0]) > 1:  # check for vacancy cluster
                vac_sites = list(filter(None, re.split(r'(?=[A-Z])', m_split[1])))
                vac_in_cluster = [fr'\LARGE{{V}}_{{\LARGE{{\text{{{vac}}}}}}}' for vac in vac_sites]
                formatted_defects[m] = '$'+'–'.join(vac_in_cluster)+'$'
            elif m_split[1] == 'i':  # interstitial
                formatted_defects[m] = fr'$\LARGE{{\text{{{m_split[0]}}}}}_{{\LARGE{{{m_split[1]}}}}}$'
            else:  # other
                formatted_defects[m] = fr'$\LARGE{{\text{{{m_split[0]}}}}}_{{\LARGE{{\text{{{m_split[1]}}}}}}}$'
    
    fig = go.Figure()
    
    for i in range(len(defects_list)):
        if len(transition_levels.transition_levels[i].fermi_levels) > 0:
            defect_fermi_levels = transition_levels.transition_levels[i].fermi_levels
            defect_fermi_levels.sort()

            defect_names = [transition_levels.transition_levels[i].name for j in range(len(defect_fermi_levels))]
            defect_ctls = [defect_fermi_levels[j] for j in range(len(defect_fermi_levels))]

            q_curr = [f'{transition_levels.transition_levels[i].charges[j][0]:+}' for j in range(len(defect_fermi_levels))]
            q_next = [f'{transition_levels.transition_levels[i].charges[j][1]:+}' for j in range(len(defect_fermi_levels))]
            q_slash = [f'{x}/{y}' for x, y in zip(q_curr, q_next)]

            fig.add_trace(go.Scatter(x=defect_names,
                                     y=defect_ctls,
                                     text=q_slash if charge_style.lower()[0] == 's' else None,
                                     mode='markers+text',
                                     marker=dict(symbol='line-ew-open',
                                                 size=24,
                                                 color=px.colors.qualitative.Dark24[i],
                                                 line=dict(width=6)
                                                )
                                    ))

            if charge_style.lower()[0] == 'm':
                for k in range(len(defect_names)):
                    if k == 0:
                        if defect_ctls[k] >= q_prox_tol:
                            fig.add_annotation(
                                x=defect_names[k], y=defect_ctls[k]/2,
                                text=q_curr[k],
                                showarrow=False,
                                # yshift=10
                            )
                        else:
                            print(f'{defect_names[k]} charge state {q_curr[k]} too close to VBM for plotting')
                    elif k == len(defect_names)-1:
                        if defect_ctls[k] - defect_ctls[k-1] >= q_prox_tol:
                            fig.add_annotation(
                                x=defect_names[k], y=(defect_ctls[k-1]+defect_ctls[k])/2,
                                text=q_curr[k],
                                showarrow=False,
                                # yshift=10
                            )
                        else:
                            print(f'{defect_names[k]} charge state {q_curr[k]} too close to {q_next[k]} for plotting')

                        if transition_levels.cbm - defect_ctls[k] >= q_prox_tol:
                            fig.add_annotation(
                                x=defect_names[k], y=(defect_ctls[k]+transition_levels.cbm)/2,
                                text=q_next[k],
                                showarrow=False,
                                # yshift=10
                            )
                        else:
                            print(f'{defect_names[k]} charge state {q_next[k]} too close to CBM for plotting')
                    else:
                        if defect_ctls[k] - defect_ctls[k-1] >= q_prox_tol:
                            fig.add_annotation(
                                x=defect_names[k], y=(defect_ctls[k-1]+defect_ctls[k])/2,
                                text=q_curr[k],
                                showarrow=False,
                                # yshift=10
                            )
                        else:
                            print(f'{defect_names[k]} charge state {q_curr[k]} too close to {q_next[k]} for plotting')
        else:
            fig.add_trace(go.Scatter(x=[transition_levels.transition_levels[i].name],
                                     y=[0.],
                                     marker=dict(size=0, opacity=0)
                                    ))

    fig.update_traces(textposition='top center')

    # VBM
    fig.add_hline(y=0., line=dict(color='black', width=3, dash='dash'), annotation_text='VBM', annotation_position='bottom left')
    fig.add_hrect(y0=-0.2, y1=0., line_width=0, fillcolor='red', opacity=0.2)
    
    # CBM
    fig.add_hline(y=transition_levels.cbm, line=dict(color='black', width=3, dash='dash'), annotation_text='CBM', annotation_position='top left')
    fig.add_hrect(y0=transition_levels.cbm, y1=transition_levels.cbm+0.25, line_width=0, fillcolor='blue', opacity=0.2)

    fig.update_layout(
        xaxis=dict(
            title=r'Defects',
            type='category',
            tickmode='array',
            ticktext=formatted_defects,
            tickvals=defects_list,
        ),
        yaxis=dict(
            title=r'$\Huge{E_{F} \; \text{(eV)}}$',
            showticklabels=True,
            tickformat='.1f'
        ),
        showlegend=False,
        margin=dict(l=60, r=20, t=20, b=20),
        autosize=False, width = 1200, height = 1200
    )

    if fig_name is not None:
        fig.write_image(fig_name, scale=2)
    
    return fig


def generate_carrier_capture_ctl(
    transition_levels: TransitionLevels,
    coeffs_df: pd.DataFrame,
    skip_defects: list = [],
    charge_style: str = 'mid',
    coeff_namelist: list = ['C_p', 'C_n'],
    q_prox_tol: float | int = 0.05,
    cscale: list = px.colors.sequential.Plasma[:-1]+px.colors.sequential.Aggrnyl_r[1:-1],
    fig_name: PathLike | None = 'capture_coeff_ctls.pdf'
) -> go.Figure():
    """
    Generates the CTL diagram with hole and electorn capture coefficients on a
    relative colorscale.

    Args
    ---------
        transition_levels (TransitionLevels):
            `pydefect` TransitionLevels object to get CTL values.
        coeffs_df (DataFrame):
            DataFrame containing capture coefficient values for each CTL.
        skip_defects (list):
            List of defect names to skip from the transition_levels file. Defaults
            to an empty list.
        charge_style (str):
            String representing style for how to print the charge states. Defaults
            to 'mid', the midpoint between the nearest CTLs.
        coeff_namelist (list):
            List of strings corresponding to the headers for coeffs_df to use.
            Defaults to ['C_p', 'C_n'] for hole and electron capture.
        q_prox_tol (float or int):
            Tolerance value for two CTLs or CTLs and band edges, below which
            the charge annotation is not printed. Defaults to 0.05 eV.
        cscale (list):
            Colorscale to use for heatmap. Defaults to a combination of
            px.colors.sequential.Plasma and Aggrnyl_r.
        fig_name (PathLike or None):
            Name for figure to save or None to not save a figure. Defaults to
            'capture_coeff_ctls.pdf'.

    Returns
    ---------
        Figure showing capture coefficient values on a colorscale on CTL values in
        the bandgap.
    """
    coeff_min, coeff_max = coeffs_df[coeff_namelist].min(axis=None), coeffs_df[coeff_namelist].max(axis=None)
    cmax, cmin = np.log10(coeff_min)/np.log10(coeff_min), np.log10(coeff_max)/np.log10(coeff_min)
    cscale_edit = ['rgb(0, 0, 0)']
    for i in cscale:
        cscale_edit.append(i)
    
    ctl_fig = generate_transition_level_diagram(transition_levels, skip_defects=skip_defects, charge_style='s')
    data, layout = ctl_fig.data, ctl_fig.layout
    MARKER_SYMBOL, MARKER_SIZE, MARKER_LINE_WIDTH = data[0].marker.symbol, data[0].marker.size, data[0].marker.line.width
    MARKER_SYMBOL_EMPTY, MARKER_SIZE_EMPTY, MARKER_LINE_WIDTH_EMPTY = 'square-open', 12, 3
    
    fig = make_subplots(rows=1, cols=len(coeff_namelist), shared_yaxes=True, subplot_titles=('Hole Capture (300 K)', 'Electron Capture (300 K)'))
    fig.update_annotations(font_size=36)
    subplot_titles = fig.layout.annotations
    vbm_cbm_text = layout.annotations
    layout.update({'annotations': subplot_titles+vbm_cbm_text})
    fig.update_layout(layout)
    fig.update_xaxes(
        title_text=layout.xaxis.title.text,
        tickmode=layout.xaxis.tickmode,
        ticktext=layout.xaxis.ticktext,
        tickvals=layout.xaxis.tickvals,
        row=1, col=2
    )
    fig.update_yaxes(showticklabels=True, tickformat='.1f')

    for shape in layout.shapes:
        fig.add_shape(shape, row=1, col=2)
    for text in layout.annotations:
        fig.add_annotation(text, row=1, col=2)

    fig.update_layout(
        height=1200,
        width=2400,
        margin=dict(t=80, l=60)
    )

    turn_colorbar_on = True
    color_arrs = {key: {} for key in coeff_namelist}
    for f in range(len(coeff_namelist)):
        q_cnt = {}
        for defect in data:
            defect_name = defect.x[0]
            defect_ctls = defect.y
            ctl_cnt = len(defect_ctls)
            for ctl in defect_ctls:
                if defect_name not in q_cnt:
                    # print(defect_name)
                    q_cnt.update({defect_name: 0})
                    color_arrs[coeff_namelist[f]].update({defect_name: np.zeros((ctl_cnt))})
                    marker_symbol_list = [MARKER_SYMBOL]*ctl_cnt
                    marker_size_list = [MARKER_SIZE]*ctl_cnt
                    marker_line_width_list = [MARKER_LINE_WIDTH]*ctl_cnt
                if defect.text is not None:
                    q1q2 = defect.text[q_cnt[defect_name]]
                    q_list = list(map(lambda x: int(x), q1q2.split('/')))
                    qi, qf = q_list[np.argmax(np.abs(q_list))], q_list[np.argmin(np.abs(q_list))]
                    
                    coeff = coeffs_df[coeffs_df.defect == defect_name][coeffs_df.q_i == qi][coeffs_df.q_f == qf][coeff_namelist[f]]
                    
                    if coeff.size == 0:
                        print(defect_name, 'CTL present, no coeff value found')
                        color_arrs[coeff_namelist[f]][defect_name][q_cnt[defect_name]] = 0.  # np.array(cmin)
                        marker_symbol_list[q_cnt[defect_name]] = MARKER_SYMBOL_EMPTY
                        marker_size_list[q_cnt[defect_name]] = MARKER_SIZE_EMPTY
                        marker_line_width_list[q_cnt[defect_name]] = MARKER_LINE_WIDTH_EMPTY
                    else:
                        coeff = coeff.iloc[0]
                        print(defect_name, 'CTL present, coeff value found')
                        defect.marker.colorscale = cscale_edit
                        defect.marker.cauto = False
                        defect.marker.cmin = 0.
                        defect.marker.cmax = cmax
                        defect.marker.showscale = turn_colorbar_on
                        marker_symbol_list[q_cnt[defect_name]] = MARKER_SYMBOL
    
                        # lever rule
                        cval = (np.log10(coeff) - np.log10(coeff_min))/(np.log10(coeff_max) - np.log10(coeff_min))
                        color_arrs[coeff_namelist[f]][defect_name][q_cnt[defect_name]] = cval
    
                else:
                    print(defect_name, 'no CTL')
                    color_arrs[coeff_namelist[f]][defect_name][q_cnt[defect_name]] = 0.
                
                defect.marker.color = color_arrs[coeff_namelist[f]][defect_name]
                defect.marker.symbol = marker_symbol_list
                defect.marker.size = marker_size_list
                defect.marker.line.width = marker_line_width_list
                q_cnt[defect_name] += 1
            fig.add_trace(defect, row=1, col=f+1)

    print(fig.data)
    
    if charge_style.lower()[0] == 'm':
        for d in fig.data:
            for f in range(len(coeff_namelist)):
                defect_names = d.x
                defect_ctls = d.y
                q_slash = d.text
                q_curr, q_next = [], []
    
                if q_slash != None:
                    for qsq in q_slash:
                        q_temp = qsq.split('/')
                        q_curr.append(q_temp[0])
                        q_next.append(q_temp[-1])
                else:
                    continue

                # print(defect_names, defect_ctls, q_slash, q_curr, q_next)
    
                for k in range(len(defect_names)):
                    if k == 0:
                        if defect_ctls[k] >= q_prox_tol:
                            fig.add_annotation(
                                x=defect_names[k], y=defect_ctls[k]/2,
                                text=q_curr[k],
                                showarrow=False,
                                row=1, col=f+1,
                                # yshift=10
                            )
                        else:
                            print(f'{defect_names[k]} charge state {q_curr[k]} too close to VBM for plotting')
                    elif k == len(defect_names)-1:
                        if defect_ctls[k] - defect_ctls[k-1] >= q_prox_tol:
                            fig.add_annotation(
                                x=defect_names[k], y=(defect_ctls[k-1]+defect_ctls[k])/2,
                                text=q_curr[k],
                                showarrow=False,
                                row=1, col=f+1,
                                # yshift=10
                            )
                        else:
                            print(f'{defect_names[k]} charge state {q_curr[k]} too close to {q_next[k]} for plotting')

                        if transition_levels.cbm - defect_ctls[k] >= q_prox_tol:
                            fig.add_annotation(
                                x=defect_names[k], y=(defect_ctls[k]+transition_levels.cbm)/2,
                                text=q_next[k],
                                showarrow=False,
                                row=1, col=f+1,
                                # yshift=10
                            )
                        else:
                            print(f'{defect_names[k]} charge state {q_next[k]} too close to CBM for plotting')
                    else:
                        if defect_ctls[k] - defect_ctls[k-1] >= q_prox_tol:
                            fig.add_annotation(
                                x=defect_names[k], y=(defect_ctls[k-1]+defect_ctls[k])/2,
                                text=q_curr[k],
                                showarrow=False,
                                row=1, col=f+1,
                                # yshift=10
                            )
                        else:
                            print(f'{defect_names[k]} charge state {q_curr[k]} too close to {q_next[k]} for plotting')

        for d in fig.data:
            d.text = None

    if fig_name is not None:
        fig.write_image(fig_name, scale=2)
    
    return fig


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