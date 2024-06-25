"""Core functions and classes for analyzing rad-induced defects from VASP."""
import logging
from typing import TYPE_CHECKING, Optional
import importlib.resources as ilr

import os
import glob
from pathlib import Path
import shutil
import json
from monty.serialization import dumpfn, loadfn
from monty.json import MSONable, MontyEncoder, MontyDecoder

import subprocess
import re
import pprint

import math as m
from fractions import Fraction
import random as rand

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from pymatgen.io.vasp.sets import VaspInputSet, Poscar
from pymatgen.io.vasp.outputs import Locpot, Outcar, Vasprun
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.symmetry.analyzer import *
from pymatgen.analysis.defects.core import DefectComplex, Substitution, Vacancy
from pymatgen.analysis.defects.finder import DefectSiteFinder
from mp_api.client import MPRester

from pydefect.analyzer.unitcell import Unitcell
from pydefect.analyzer.band_edge_states import BandEdgeOrbitalInfos, BandEdgeState, PerfectBandEdgeState
from pydefect.analyzer.calc_results import CalcResults
from pydefect.analyzer.defect_energy import DefectEnergyInfo
from pydefect.input_maker.defect_entry import DefectEntry
from pydefect.corrections.efnv_correction import ExtendedFnvCorrection

from doped.analysis import DefectParser, DefectsParser

# pydefect file structure
base_path = os.getcwd()
cpd_path, defects_path, unitcell_path, band_path, diele_path, dos_path, struc_opt_path = os.path.relpath('cpd'), os.path.relpath('defect'), os.path.relpath('unitcell'), os.path.relpath('unitcell/band'), os.path.relpath('unitcell/dielectric'), os.path.relpath('unitcell/dos'), os.path.relpath('unitcell/structure_opt')
pydefect_paths = [cpd_path, defects_path, unitcell_path, band_path, diele_path, dos_path, struc_opt_path]


# create pydefect file structure
def create_pydefect_file_structure():
    os.makedirs([os.path.join(base_path, i) for i in pydefect_paths])
    return


# maybe define class for defective supercells
# get defect and perfect supercells on definition
def get_mpid_from_supercell(perfect_poscar_path):
    """
    Given a path for a perfect supercell POSCAR, determine the
    characteristics of the unitcell and fetches from the
    Materials Project.
    """
    # read POSCAR files with pymatgen and determine unitcell from perfect supercell
    perfect_pos = Poscar.from_file(perfect_poscar_path)
    unitcell_strc = perfect_pos.structure.get_primitive_structure()
    unitcell_spg = SpacegroupAnalyzer(unitcell_strc)
    
    # query MPD to get material ID matching perfect supercell structure
    # API key is parsed via .pmgrc.yaml
    with MPRester() as mpr:
        docs = mpr.materials.summary.search(
            elements=map(str, unitcell_strc.elements),
            num_elements=len(unitcell_strc.elements),
            num_sites=len(unitcell_strc.sites),
            spacegroup_symbol=unitcell_spg.get_space_group_symbol(),
            fields=['material_id', 'composition', 'symmetry', 'band_gap', 'structure']
        )
        mpid_dict = {doc.material_id: [doc.composition, doc.symmetry, doc.band_gap, doc.structure] for doc in docs}
        if len(mpid_dict.keys()) == 1:
            unitcell_mpid = list(mpid_dict.keys())[0]
            unitcell_info = {
                'mpid': unitcell_mpid,
                'composition': mpid_dict[unitcell_mpid][0],
                'symmetry': mpid_dict[unitcell_mpid][1],
                'band_gap': mpid_dict[unitcell_mpid][2],
                'structure': mpid_dict[unitcell_mpid][3]
            }
        else:
            raise UserWarning('Multiple matching structures were found.')
    return unitcell_info


# band edge orbital infos gives fermi level, states below should be occupied and above should be unoccupied
# interested in occupation of states between VBM & CBM: gap states
def find_gap_states(pbes_path, beois_path):
    """
    Given a path to the perfect band edge state and band edge orbital
    info JSON files, determine VBM & CBM energies as well as orbital
    energies and occupations from pydefect.
    """
    pbes, beois = PerfectBandEdgeState.from_dict(json.load(open(pbes_path, 'r'))), BandEdgeOrbitalInfos.from_dict(json.load(open(beois_path, 'r')))
    e_vbm, e_cbm = pbes.vbm_info.orbital_info.energy, pbes.cbm_info.orbital_info.energy

    for s in range(len(beois.orbital_infos)):    # spin up/down channels
        for k in range(len(beois.orbital_infos[s])):    # kpt idx
            for b in range(len(beois.orbital_infos[s][k])):    # band idx
                beois_ens, beois_occs = beois.orbital_infos[s][k][b].energy, beois.orbital_infos[s][k][b].occupation

# maybe make a GapState class that incorporates the Orbital Infos plus the s/k/b for the given gap state
# gap state defined as energy between e_vbm and e_cbm
# gap state occupancy is of interest
# use gap state to determine charge localization?


def get_plnr_avg_only(
    defect_locpot: Locpot,
    bulk_locpot: Locpot,
    lattice: Optional[Lattice] = None
):
    """Calculates the planar average electrostatic potential.
    Based on FNV correction function from pymatgen.analysis.defects.

    Args:
        defect_locpot:
            Locpot of defect
        bulk_locpot:
            Locpot of bulk
        lattice:
            Lattice of the defect supercell. If None, then uses the lattice of the
            defect_locpot.

    Returns:
        Plotting data for the planar average electrostatic potential.
            ```
            plot_plnr_avg(result.metadata[0], title="Lattice Direction 1")
            plot_plnr_avg_only(plnr_avg_data['plot_data'][direction]['pot_plot_data'], ax=axs[direction], title=axis_label_dict[direction])
            ```
    """

    if isinstance(defect_locpot, Locpot):
        list_axis_grid = [*map(defect_locpot.get_axis_grid, [0, 1, 2])]
        list_defect_plnr_avg_esp = [
            *map(defect_locpot.get_average_along_axis, [0, 1, 2])
        ]
        lattice_ = defect_locpot.structure.lattice.copy()
        if lattice is not None and lattice != lattice_:
            raise ValueError(
                "Lattice of defect_locpot and user provided lattice do not match."
            )
        lattice = lattice_
    elif isinstance(defect_locpot, dict):
        defect_locpot_ = {int(k): v for k, v in defect_locpot.items()}
        list_defect_plnr_avg_esp = [defect_locpot_[i] for i in range(3)]
        list_axis_grid = [
            *map(
                np.linspace,
                [0, 0, 0],
                lattice.abc,
                [len(i) for i in list_defect_plnr_avg_esp],
            )
        ]
    else:
        raise ValueError("defect_locpot must be of type Locpot or dict")

    # TODO this can be done with regridding later
    if isinstance(bulk_locpot, Locpot):
        list_bulk_plnr_avg_esp = [*map(bulk_locpot.get_average_along_axis, [0, 1, 2])]
    elif isinstance(bulk_locpot, dict):
        bulk_locpot_ = {int(k): v for k, v in bulk_locpot.items()}
        list_bulk_plnr_avg_esp = [bulk_locpot_[i] for i in range(3)]
    else:
        raise ValueError("bulk_locpot must be of type Locpot or dict")

    plot_data = dict()

    for x, pureavg, defavg, axis in zip(
        list_axis_grid, list_bulk_plnr_avg_esp, list_defect_plnr_avg_esp, [0, 1, 2]
    ):
        # log plotting data:
        metadata = dict()
        metadata["pot_plot_data"] = {
            "x": x,
            "dft_diff": np.array(defavg) - np.array(pureavg)
        }
        
        plot_data[axis] = metadata

    corr_metadata = {"plot_data": plot_data}

    return corr_metadata


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

