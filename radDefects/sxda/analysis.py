"""Analysis for sxdefectalign calculations using pydefect, doped, and radDefects tools."""
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

from pydefect.analyzer.unitcell import Unitcell
from pydefect.analyzer.band_edge_states import BandEdgeOrbitalInfos, BandEdgeState, PerfectBandEdgeState
from pydefect.analyzer.defect_energy import DefectEnergyInfo, ChargeEnergies, SingleChargeEnergies, CrossPoints
from pydefect.analyzer.transition_levels import TransitionLevels, make_transition_levels

from raddefects.sxda.io import parse_sxda_vatoms


def sxda_transition_levels(base_path=os.getcwd(), chg_offset=0):
    """
    Using sxdefectalign calculations within the pydefect structure, calculate
    charge transition levels.
    """
    # use unitcell.yaml and perfect_band_edge_state.json files to get CBM/VBM info
    unitcell_yaml = os.path.join(base_path, 'unitcell', 'unitcell.yaml')
    defect_path=os.path.join(base_path, 'defect')
    perfect_band_edge_state_json = os.path.join(defect_path, 'perfect', 'perfect_band_edge_state.json')
    uc_info, pbes = Unitcell.from_yaml(unitcell_yaml), loadfn(perfect_band_edge_state_json)
    cbm, supercell_vbm, supercell_cbm = uc_info.cbm, pbes.vbm_info.energy, pbes.cbm_info.energy
    cbm -= supercell_vbm
    supercell_cbm -= supercell_vbm
    supercell_vbm -= supercell_vbm
    
    # gather pydefect defect directories
    defect_paths = glob.glob(os.path.join(defect_path, '*_*/'), recursive=True)
    
    # empty dictionaries for sxda charge energies and sxda cross points
    sxda_single_charge_energies = {}
    sxda_charge_energies = {}
    sxda_cross_point_dicts = {}
    
    for defect_dir in defect_paths:
        # get defect type and charge from defect directory
        defect = os.path.basename(os.path.dirname(defect_dir))
        defect_type, defect_chg = '_'.join(defect.split('_')[0:2]), int(defect.split('_')[2])
        
        # add defect type to single charge energies dictionary
        if defect_type not in sxda_single_charge_energies:
            sxda_single_charge_energies.update({defect_type: []})
        
        # get defect.sxda filename
        if 'FP' in defect_type:
            if defect_chg > 0:
                vac_chg, int_chg = chg_offset, defect_chg+chg_offset
            elif defect_chg < 0:
                vac_chg, int_chg = defect_chg-chg_offset, chg_offset
            elif defect_chg == 0:
                if float(chg_offset) == 0.:
                    vac_chg, int_chg = 0., 0.
                else:
                    vac_chg, int_chg = -1*(chg_offset/2.), chg_offset/2.
            else:
                raise ValueError('Defect charge must be a valid number.')

            sxda_outfile = os.path.join(defect_dir, f'{defect}_qv{vac_chg:.2f}qi{int_chg:.2f}.sxda')
        else:
            sxda_outfile = os.path.join(defect_dir, f'{defect}.sxda')
        
        # compare first line of defect.sxda with defect type and charge
        with open(sxda_outfile) as f:
            sxda_lines = f.readlines()
        title_line, correction_line = sxda_lines[0].strip('\n'), sxda_lines[-1]

        # check if defect type is correct
        sxda_defect_type, sxda_defect_chg = title_line.split(',')[0], int(title_line.split(',')[1].split(' = ')[1])
        if (sxda_defect_type != defect_type) or (sxda_defect_chg != defect_chg):
            raise ValueError('Defect type/charge does not match between pydefect and sxdefectalign')
        
        # ensure correction line contains the correction and extract correction value
        if 'correction' in correction_line:
            sxda_corr = float(re.findall(r'\d[.,]?\d*', correction_line)[0])
        else:
            continue
        
        bes = loadfn(os.path.join(defect_dir, 'band_edge_orbital_infos.json'))
        fermi_level = bes.fermi_level - supercell_vbm
        
        dei = DefectEnergyInfo.from_yaml(os.path.join(defect_dir, 'defect_energy_info.yaml'))
        formation_en = dei.defect_energy.energy(with_correction=False)
        formation_en_sxda = formation_en + sxda_corr
        
        # add charge and formation energies to single charge energies dictionary
        sxda_single_charge_energies[defect_type] += [(int(defect_chg), formation_en_sxda)]
        
    for d_type in sxda_single_charge_energies:
        sxda_charge_energies.update({d_type: SingleChargeEnergies(sxda_single_charge_energies[d_type])})
    sxda_energies = ChargeEnergies(sxda_charge_energies, e_min=0., e_max=cbm)
    
    sxda_cross_point_dicts = sxda_energies.cross_point_dicts

    # create TransitionLevels object
    sxda_tls = make_transition_levels(
        cross_point_dicts=sxda_cross_point_dicts,
        cbm=cbm,
        supercell_vbm=supercell_vbm,
        supercell_cbm=supercell_cbm
    )
    
    return sxda_tls


def atomic_potential_convergence(base_path=os.getcwd()):
    """
    Calculates the convergence for atomic site potential data from sxdefectalign
    calculations in terms of averages and standard deviations within the sampling
    region.
    """
    # defect directory
    defect_path=os.path.join(base_path, 'defect')
    
    # gather pydefect defect directories
    defect_paths = glob.glob(os.path.join(defect_path, '*_*/'), recursive=True)
    
    # empty dictionary for alignment averages and standard deviations
    alignment_dict = {}
    
    for defect_dir in defect_paths:
        # get defect type and charge from defect directory
        defect = os.path.basename(os.path.dirname(defect_dir))
        defect_type, defect_chg = '_'.join(defect.split('_')[0:2]), int(defect.split('_')[2])
        
        # use parse_sxda_vatoms on directory
        vatoms = parse_sxda_vatoms(defect_dir)
    
        # use lattice parameters for defect region radius or use pydefect correction.json file
        if os.path.isfile(os.path.join(defect_dir, 'correction.json')):
            correction_json = loadfn(os.path.join(defect_dir, 'correction.json'))
            defect_region_radius = correction_json.defect_region_radius
        else:
            contcar_filepath = os.path.join(filepath, 'CONTCAR')
            contcar = Poscar.from_file(contcar_filepath)
            abc = np.array(contcar.structure.lattice.abc)
            abc /= 2.
            defect_region_radius = np.min(abc)
    
        # alignment stats (mean, stdev) from atomic site potential dataframe with r>defect_region_radius
        vatoms_total = pd.concat((vatoms[i] for i in vatoms.keys()))
        vatoms_sample_region = vatoms_total.where(vatoms_total['r']>defect_region_radius)
        alignment_average = vatoms_sample_region.loc[:, 'V_defect-V_ref-V_lr'].mean(skipna=True)
        alignment_stdev = vatoms_sample_region.loc[:, 'V_defect-V_ref-V_lr'].std(skipna=True)
        
        alignment_dict.update({defect: (alignment_average, alignment_stdev)})
    
    return alignment_dict

