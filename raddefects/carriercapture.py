#!/usr/bin/env python3
"""Python module used to setup and analyze carrier capture calculations using VASP."""
import os
import sys
from pathlib import Path
import shutil
import json
import yaml
from monty.serialization import dumpfn, loadfn
from itertools import zip_longest
from math import isclose, ceil, floor
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar, Kpoints, Incar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Eigenval, Oszicar
from pymatgen.electronic_structure.core import Spin

import pydefect
from pydefect.analyzer.unitcell import Unitcell
from pydefect.analyzer.transition_levels import TransitionLevels
from nonrad.ccd import get_dQ, get_Q_from_struct, get_cc_structures
from nonrad.elphon import get_Wif_from_WSWQ
from nonrad.scaling import sommerfeld_parameter, charged_supercell_scaling_VASP
import sumo.electronic_structure.effective_mass as sem


def create_carrier_capture_disp_vasp_inputs(poscar_structure, charge, disp_factor, potcar=None, incar_settings={}):
    """
    Create a set of VASP inputs for carrier capture displacement calculations.
    """
    # POSCAR file from displacement structure
    poscar = Poscar(poscar_structure)
    
    # POTCAR file defaulting to standard PAW PBEv54 potentials without suffixes
    potcar = Potcar(poscar.site_symbols, 'PBE_54') if potcar == None else potcar
    
    # get electron count from POSCAR/POTCAR
    nelect = 0
    for i, ele in enumerate(poscar.site_symbols):
        for j in potcar:
            if ele == j.element:
                nelect += int(j.nelectrons)*poscar.natoms[i]
    nelect -= charge
    
    # single special kpoint
    kpt_spec = Kpoints.gamma_automatic(shift=(0.5, 0.5, 0.5))
    
    # INCAR file for single-energy HSE calculation
    incar_dict = {
        'NCORE': 12,
        'LREAL': 'Auto',
        'EDIFF': 1e-05,
        'ALGO': 'Normal',
        'ENCUT': 400.0,
        'PREC': 'Accurate',
        'LHFCALC': True,
        'GGA': 'PE',
        'HFSCREEN': 0.2,
        'AEXX': 0.31,
        'ISMEAR': 0,
        'SIGMA': 0.05,
        'SYMPREC': 0.001,
        'ISIF': 2,
        'IBRION': -1,
        'NELM': 100,
        'NSW': 0,
        'NELECT': nelect,
        'NUPDOWN': int(nelect%2),
        'ISPIN': 2,
        'MAGMOM': f'{poscar.structure.num_sites}*0.6',
        'LASPH': True,
        'LMAXMIX': 4,
        'LORBIT': 10,
        'LCHARG': False,
        'LWAVE': True if disp_factor <= 0.2 and disp_factor >= -0.2 else False,
        'LVTOT': False,
        'LVHAR': False
    }
    incar_dict.update(incar_settings)
    
    incar = Incar(incar_dict)

    disp_inps = VaspInput(poscar=poscar, potcar=potcar, kpoints=kpt_spec, incar=incar)
    
    return disp_inps


def carriercapturejl_interpolate(struct_i, struct_f, disp_range=np.linspace(-1, 1, 11)):
    """
    Interpolates pymatgen structures using the method defined in the CarrierCapture.jl code gen_cc_struct.py script.
    Modified to match I/O of nonrad.
    """
    interpolated_structs = []
    
    # model code on CarrierCapture.jl gen_cc_struct.py script
    # A. Alkauskas, Q. Yan, and C. G. Van de Walle, Physical Review B 90, 27 (2014)
    delta_R = struct_f.frac_coords - struct_i.frac_coords
    delta_R = (delta_R + 0.5) % 1 - 0.5
    
    lattice = struct_i.lattice.matrix #[None,:,:]
    delta_R = np.dot(delta_R, lattice)
    
    masses = np.array([spc.atomic_mass for spc in struct_i.species])
    delta_Q2 = masses[:,None] * delta_R ** 2
    
    for frac in disp_range:
        disp = frac * delta_R
        struct = Structure(struct_i.lattice, struct_i.species, \
                           struct_i.cart_coords + disp, \
                           coords_are_cartesian=True)
        interpolated_structs.append(struct)
    
    return interpolated_structs

    
def setup_carrier_capture_from_pydefect(defect_name, q_initial, q_final, displacements=None, struct_gen_type='CarrierCapture.jl', base_path=Path.cwd()):
    """
    Setup carrier capture calculations from previous pydefect calculations. 
    """
    defect_path, capture_path = base_path / 'defect', base_path / 'carrier_capture'
    defect_initial_path, defect_final_path = defect_path / '_'.join([defect_name, str(q_initial)]), defect_path / '_'.join([defect_name, str(q_final)])
    capture_calc_path = capture_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    capture_calc_path.mkdir(parents=True, exist_ok=True)

    charge_diff = q_final-q_initial
    capture_initial_path, capture_final_path = capture_calc_path / 'i_q', capture_calc_path / f'f_q{charge_diff:+}'
    capture_initial_path.mkdir(parents=True, exist_ok=True)
    capture_final_path.mkdir(exist_ok=True)
    capture_disp_path = capture_calc_path / '_'.join([defect_name, 'DISPLACEMENT'])
    capture_disp_path.mkdir(exist_ok=True)

    shutil.copyfile(defect_initial_path / 'CONTCAR', capture_disp_path / 'POSCAR_i')
    shutil.copyfile(defect_final_path / 'CONTCAR', capture_disp_path / 'POSCAR_f')

    excited_poscar, ground_poscar = Poscar.from_file(capture_disp_path / 'POSCAR_i'), Poscar.from_file(capture_disp_path / 'POSCAR_f')
    excited_struct, ground_struct = excited_poscar.structure, ground_poscar.structure

    displacements = np.array([-1.0, -0.6, -0.4, -0.2, -0.1, 0., 0.1, 0.2, 0.4, 0.6, 1.0]) if displacements is None else displacements
    if struct_gen_type.lower() == 'carriercapture.jl':
        ground = carriercapturejl_interpolate(ground_struct, excited_struct, disp_range=displacements)
        excited = carriercapturejl_interpolate(excited_struct, ground_struct, disp_range=displacements)
    elif struct_gen_type.lower() == 'nonrad':
        ground, excited = get_cc_structures(ground_struct, excited_struct, displacements, remove_zero=False)
    else:
        raise ValueError('Please choose a valid method for generating displacement structures: CarrierCapture.jl or Nonrad')

    disp_dir_i_path, disp_dir_f_path = capture_disp_path / 'disp_dir_i', capture_disp_path / 'disp_dir_f'
    disp_dir_f_path.mkdir(exist_ok=True)
    disp_dir_i_path.mkdir(exist_ok=True)
    
    for i, struct in enumerate(excited):
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'
        struct.to(filename=str(disp_dir_i_path / f'POSCAR_{disp_suffix}'), fmt='poscar')
        disp_calc_path = capture_initial_path / f'DISP_{disp_suffix}'
        vasp_disp_inputs = create_carrier_capture_disp_vasp_inputs(struct, q_initial, displacements[i])
        vasp_disp_inputs.write_input(disp_calc_path, make_dir_if_not_present=True)
    
    for i, struct in enumerate(ground):
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'
        struct.to(filename=str(disp_dir_f_path / f'POSCAR_{disp_suffix}'), fmt='poscar')
        disp_calc_path = capture_final_path / f'DISP_{disp_suffix}'
        vasp_disp_inputs = create_carrier_capture_disp_vasp_inputs(struct, q_final, displacements[i])
        vasp_disp_inputs.write_input(disp_calc_path, make_dir_if_not_present=True)
    
    return capture_calc_path


def create_carrier_capture_wav_vasp_inputs(disp_dir, wav_dir, charge=0, potcar=None, incar_settings={}):
    """
    Read in VASP inputs from carrier capture displacement calculations and create VASP inputs for WAVECAR calculations.
    """
    # input files from displacement structure
    disp_inps = VaspInput.from_directory(disp_dir)

    # update POTCAR if specified
    potcar = Potcar(disp_inps.poscar.site_symbols, 'PBE_54') if potcar == None else potcar

    # get electron count from POSCAR/POTCAR
    nelect = 0
    for i, ele in enumerate(disp_inps.poscar.site_symbols):
        for j in potcar:
            if ele == j.element:
                nelect += int(j.nelectrons)*disp_inps.poscar.natoms[i]
    nelect -= charge
    
    # use gamma point
    kpt_gam = Kpoints.gamma_automatic()
    
    # update INCAR if specified, change NELECT based on POTCAR
    incar_dict = disp_inps.incar
    incar_dict.update({
        'NELECT': nelect,
        'NUPDOWN': int(nelect%2)
    })
    incar_dict.update(incar_settings)
    incar = Incar(incar_dict)

    # create updated VASP input set with modified input files
    wav_inps = VaspInput(poscar=disp_inps.poscar, potcar=potcar, kpoints=kpt_gam, incar=incar)
    
    return wav_inps


def setup_carrier_capture_wav(defect_name, q_initial, q_final, cc_path=Path.cwd(), displacements=None, potcar=None, incar_settings={}):
    """
    Setup carrier capture WAVECAR calculations from displacement calculations.
    """
    capture_calc_path = cc_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    charge_diff = q_final-q_initial
    capture_initial_path, capture_final_path = capture_calc_path / 'i_q', capture_calc_path / f'f_q{charge_diff:+}'

    displacements = np.array([-0.2, -0.1, 0., 0.1, 0.2]) if displacements is None else displacements

    for i in range(displacements.shape[0]):
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'

        # setup WAVECAR calcs for excited structures
        disp_calc_path_i, wav_calc_path_i = capture_initial_path / f'DISP_{disp_suffix}', capture_initial_path / f'WAV_{disp_suffix}'
        vasp_wav_inputs = create_carrier_capture_wav_vasp_inputs(disp_calc_path_i, wav_calc_path_i, charge=q_initial, potcar=potcar, incar_settings=incar_settings)
        vasp_wav_inputs.write_input(wav_calc_path_i, make_dir_if_not_present=True)

        # setup WAVECAR calcs for ground structures
        disp_calc_path_f, wav_calc_path_f = capture_final_path / f'DISP_{disp_suffix}', capture_final_path / f'WAV_{disp_suffix}'
        vasp_wav_inputs = create_carrier_capture_wav_vasp_inputs(disp_calc_path_f, wav_calc_path_f, charge=q_final, potcar=potcar, incar_settings=incar_settings)
        vasp_wav_inputs.write_input(wav_calc_path_f, make_dir_if_not_present=True)
        
    return


def create_carrier_capture_wswq_vasp_inputs(wav_dir, wswq_dir, charge=0, potcar=None, incar_settings={}):
    """
    Read in VASP inputs from carrier capture WAVECAR calculations and create VASP inputs for WSWQ calculations.
    """
    # input files from displacement structure
    wav_inps = VaspInput.from_directory(wav_dir)

    # update POTCAR if specified
    potcar = wav_inps.potcar if potcar == None else potcar

    # get electron count from POSCAR/POTCAR
    nelect = 0
    for i, ele in enumerate(wav_inps.poscar.site_symbols):
        for j in potcar:
            if ele == j.element:
                nelect += int(j.nelectrons)*wav_inps.poscar.natoms[i]
    nelect -= charge
    
    # update INCAR if specified, change NELECT based on POTCAR
    incar_dict = wav_inps.incar
    incar_dict.update({
        'ALGO': None,
        'LWSWQ': True,
        'NELECT': nelect,
        'NUPDOWN': int(nelect%2)
    })
    incar_dict.update(incar_settings)
    incar = Incar(incar_dict)

    # create updated VASP input set with modified input files
    wswq_inps = VaspInput(poscar=wav_inps.poscar, potcar=potcar, kpoints=wav_inps.kpoints, incar=incar)
    
    return wswq_inps


def setup_carrier_capture_wswq(defect_name, q_initial, q_final, ref_wavecar_path, cc_path=Path.cwd(), displacements=None, potcar=None, incar_settings={}):
    """
    Setup carrier capture WSWQ calculations from WAVECAR calculations.
    """
    capture_calc_path = cc_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    charge_diff = q_final-q_initial
    capture_initial_path, capture_final_path = capture_calc_path / 'i_q', capture_calc_path / f'f_q{charge_diff:+}'

    displacements = np.array([-0.2, -0.1, 0., 0.1, 0.2]) if displacements is None else displacements

    for i in range(displacements.shape[0]):
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'

        # setup WSWQ calcs for excited structures
        wav_calc_path_i, wswq_calc_path_i = capture_initial_path / f'WAV_{disp_suffix}', capture_initial_path / f'WSWQ_{disp_suffix}'
        vasp_wswq_inputs = create_carrier_capture_wswq_vasp_inputs(wav_calc_path_i, wswq_calc_path_i, charge=q_initial, potcar=potcar, incar_settings=incar_settings)
        vasp_wswq_inputs.write_input(wswq_calc_path_i, make_dir_if_not_present=True, files_to_transfer={'WAVECAR.qqq': wav_calc_path_i / 'WAVECAR'})
        shutil.copyfile(ref_wavecar_path, wswq_calc_path_i / 'WAVECAR')

        # setup WSWQ calcs for ground structures
        wav_calc_path_f, wswq_calc_path_f = capture_final_path / f'WAV_{disp_suffix}', capture_final_path / f'WSWQ_{disp_suffix}'
        vasp_wswq_inputs = create_carrier_capture_wswq_vasp_inputs(wav_calc_path_f, wswq_calc_path_f, charge=q_final, potcar=potcar, incar_settings=incar_settings)
        vasp_wswq_inputs.write_input(wswq_calc_path_f, make_dir_if_not_present=True, files_to_transfer={'WAVECAR.qqq': wav_calc_path_f / 'WAVECAR'})
        shutil.copyfile(ref_wavecar_path, wswq_calc_path_f / 'WAVECAR')
        
    return


def setup_carrier_capture_perfect_ref(base_path=Path.cwd(), wswq_mid_path=Path('WSWQ_000'), potcar=None, incar_settings={}):
    """
    Setup carrier capture perfect eigenvalue reference calculation.
    """
    perfect_path = base_path / 'defect' / 'perfect'
    eig_ref_path = perfect_path / 'WAV_ref'

    perfect_poscar = Poscar.from_file(perfect_path / 'CONTCAR')
    wswq_potcar = Potcar.from_file(wswq_mid_path / 'POTCAR')

    # match POTCAR from defect calculation, ensuring only elements present in the perfect POSCAR are included
    potcar_spec = []
    for i, ele in enumerate(wswq_potcar.symbols):
        ele_symbol = ele.split('_')[0]
        if ele_symbol in perfect_poscar.site_symbols:
            potcar_spec.append(ele)
    eig_ref_potcar = Potcar(potcar_spec, 'PBE_54')
    
    wswq_kpoints = Kpoints.from_file(wswq_mid_path / 'KPOINTS')

    incar_dict = Incar.from_file(perfect_path / 'INCAR')
    incar_dict.update({
        'IBRION': -1,
        'NSW': 0
    })
    incar_dict.update(incar_settings)
    eig_ref_incar = Incar(incar_dict)
    
    eig_ref_inputs = VaspInput(poscar=perfect_poscar, potcar=eig_ref_potcar, kpoints=wswq_kpoints, incar=eig_ref_incar)
    eig_ref_inputs.write_input(eig_ref_path, make_dir_if_not_present=True)
        
    return


def gather_qe_data(defect_name, q_initial, q_final, cc_path=Path.cwd(), displacements=None, qe_filename='potential.csv'):
    """
    Gather Q vs. E data for configuration coordinate diagram generation assuming 1D potential energy surfaces.
    """
    capture_calc_path = cc_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    charge_diff = q_final-q_initial
    capture_initial_path, capture_final_path = capture_calc_path / 'i_q', capture_calc_path / f'f_q{charge_diff:+}'
    
    # set default for displacements array corresponding to DISP calculations
    displacements = np.array([-1.0, -0.6, -0.4, -0.2, -0.1, 0., 0.1, 0.2, 0.4, 0.6, 1.0]) if displacements is None else displacements

    Qi, Qf, Ei, Ef = [], [], [], []    
    for i in range(displacements.shape[0]):
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'
        
        # get excited & ground state structures
        excited_poscar, ground_poscar = Poscar.from_file(capture_initial_path / 'DISP_000' / 'CONTCAR'), Poscar.from_file(capture_final_path / 'DISP_000' / 'CONTCAR')
        # get intermediate structures and energy files
        inter_poscar_i, inter_poscar_f = Poscar.from_file(capture_initial_path / f'DISP_{disp_suffix}' / 'CONTCAR'), Poscar.from_file(capture_final_path / f'DISP_{disp_suffix}' / 'CONTCAR')
        oszi_i, oszi_f = Oszicar(capture_initial_path / f'DISP_{disp_suffix}' / 'OSZICAR'), Oszicar(capture_final_path / f'DISP_{disp_suffix}' / 'OSZICAR')
        
        Qi.append(get_Q_from_struct(ground_poscar.structure, excited_poscar.structure, inter_poscar_i.structure))
        Qf.append(get_Q_from_struct(ground_poscar.structure, excited_poscar.structure, inter_poscar_f.structure))
        Ei.append(oszi_i.final_energy)
        Ef.append(oszi_f.final_energy)

    # create dataframe for QE data
    qe_i_df, qe_f_df = pd.DataFrame({'Q': Qi, 'E': Ei}), pd.DataFrame({'Q': Qf, 'E': Ef})
    qe_i_df.to_csv(capture_initial_path / qe_filename, index=False)
    qe_f_df.to_csv(capture_final_path / qe_filename, index=False)

    return qe_i_df, qe_f_df


def parse_carrier_capture_info(defect_name, q_initial, q_final, coupling_state='final', g=1, m_e=0.2, m_h=0.8, dielectric_const=10., kpt_idx=0, spin=0, base_path=Path.cwd(), displacements=None, savefig=None):
    """
    Analyze defect calculations to gather parameters for carrier capture calculations.
    """
    # initialize yaml with given information
    cc_yaml_filename = 'carrier_capture_params.yaml'
    cc_dict = {
        'defect': str(defect_name),
        'q_initial': int(q_initial),
        'q_final': int(q_final),
        'coupling state': str(coupling_state)
    }

    # defect & carrier capture calculation paths
    defect_path, capture_path = base_path / 'defect', base_path / 'carrier_capture'
    defect_initial_path, defect_final_path = defect_path / '_'.join([defect_name, str(q_initial)]), defect_path / '_'.join([defect_name, str(q_final)])
    capture_calc_path = capture_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    
    charge_diff = q_final-q_initial
    capture_initial_path, capture_final_path = capture_calc_path / 'i_q', capture_calc_path / f'f_q{charge_diff:+}'

    # add charge transition level between q_i/q_f & VBM/CBM if transition_levels.json exists
    try:
        transition_levels = loadfn(defect_path / 'transition_levels.json')
        for tl in transition_levels.transition_levels:
            tl.fermi_levels.sort()
            for name, charge, energy, fermi in zip_longest([tl.name], tl.charges, tl.energies, tl.fermi_levels, fillvalue=tl.name):
                if name in defect_name and q_initial in charge and q_final in charge:
                    cc_dict.update({'formation energy': float(energy), 'ctl': float(fermi)})
        cc_dict.update({'vbm': float(0.), 'cbm': float(transition_levels.cbm)})
    except:
        cc_dict.update({'formation energy': None, 'ctl': None, 'vbm': None, 'cbm': None})
        print('No transition_levels.json file found in defect directory.')

    # get VBM/CBM from EIGENVAL if not determined from transition_levels.json
    # get band indices for degenerate valence band states
    perfect_eigenvals = Eigenval(defect_path / 'perfect' / 'WAV_ref' / 'EIGENVAL', separate_spins=True)
    # band properties are tuple of (band gap, cbm, vbm, is_band_gap_direct), each tuples of 2, with index 0 = spin-up & index 1 = spin-down
    band_props = perfect_eigenvals.eigenvalue_band_properties
    if cc_dict['vbm'] is None or cc_dict['cbm'] is None:
        cc_dict.update({'vbm': float(band_props[2][0]), 'cbm': float(band_props[1][0]), 'Eg': float(band_props[0][0])})
    else:
        cc_dict.update({'Eg': float(band_props[0][0])})
    valence_indices, conduction_indices, defect_indices = [], [], []
    # eigenvalues are dict of {(spin): NDArray(shape=(nkpt, nbands, 2))}, kpoint index is 0-based
    eig_occ_up, eig_occ_down = perfect_eigenvals.eigenvalues[Spin.up][kpt_idx], perfect_eigenvals.eigenvalues[Spin.down][kpt_idx]
    vbm_bandidx_up, vbm_bandidx_down = np.where(np.isclose(eig_occ_up[:, 0], band_props[2][0], atol=0.0001)), np.where(np.isclose(eig_occ_down[:, 0], band_props[2][1], atol=0.0001))
    if isclose(vbm_bandidx_up[0][0], vbm_bandidx_down[0][0], abs_tol=0.1):
        valence_indices.append(vbm_bandidx_up[0][0])
        for band_idx in range(vbm_bandidx_up[0][0]):
            if isclose(eig_occ_up[:, 0][vbm_bandidx_up[0][0]], eig_occ_up[:, 0][band_idx], abs_tol=0.2):
                valence_indices.append(band_idx)
            elif isclose(eig_occ_down[:, 0][vbm_bandidx_down[0][0]], eig_occ_down[:, 0][band_idx], abs_tol=0.2):
                valence_indices.append(band_idx)
    valence_indices.sort()

    cbm_bandidx_up, cbm_bandidx_down = np.where(np.isclose(eig_occ_up[:, 0], band_props[1][0], atol=0.0001)), np.where(np.isclose(eig_occ_down[:, 0], band_props[1][1], atol=0.0001))
    if isclose(cbm_bandidx_up[0][0], cbm_bandidx_down[0][0], abs_tol=0.1):
        conduction_indices.append(cbm_bandidx_up[0][0])
        for band_idx in range(cbm_bandidx_up[0][0]):
            if isclose(eig_occ_up[:, 0][cbm_bandidx_up[0][0]], eig_occ_up[:, 0][band_idx], abs_tol=0.2):
                conduction_indices.append(band_idx)
            elif isclose(eig_occ_down[:, 0][cbm_bandidx_down[0][0]], eig_occ_down[:, 0][band_idx], abs_tol=0.2):
                conduction_indices.append(band_idx)
    conduction_indices.sort()
    
    # determine dQ from relaxed defect calculation structures
    poscar_initial, poscar_final = Poscar.from_file(defect_initial_path / 'CONTCAR'), Poscar.from_file(defect_final_path / 'CONTCAR')
    dQ = get_dQ(poscar_initial.structure, poscar_final.structure)
    
    Qmin, Qmax = 0., 0.
    for i in capture_calc_path.glob('*_q*/'):
        for j in i.glob('DISP_*'):
            Qmin = min(dQ*float(j.name.split('_')[-1])/10., Qmin)
            Qmax = max(dQ*float(j.name.split('_')[-1])/10., Qmax)

    # set default for displacements array corresponding to WSWQ calculations
    displacements = np.array([-0.2, -0.1, 0., 0.1, 0.2]) if displacements is None else displacements
    Q0 = displacements[displacements.shape[0]//2]
    
    # calculate electron-phonon coupling matrix elements from WSWQ calculations
    i_struc, f_struc = Structure.from_file(defect_initial_path / 'CONTCAR'), Structure.from_file(defect_final_path / 'CONTCAR')
    i_WSWQs, f_WSWQs = [], []
    fig_h, fig_e = plt.figure(figsize=(12, 5)), plt.figure(figsize=(12, 5))
    if coupling_state.lower() == 'final' or coupling_state.lower() == 'f' or coupling_state.lower() == 'ground':
        # path to initial vasprun
        ground_vr_path = capture_final_path / f'WAV_{str(Q0).replace('.', ''):0>3}' / 'vasprun.xml'
        # adjust valence/conduction band indices by electron difference from perfect reference to coupling reference
        defect_eigenvals = Eigenval(capture_final_path / f'WAV_{str(Q0).replace('.', ''):0>3}' / 'EIGENVAL', separate_spins=True)
        nelect_diff = perfect_eigenvals.nelect - defect_eigenvals.nelect
        valence_indices = [i - floor(nelect_diff/2) + 1 for i in valence_indices]
        conduction_indices = [i - floor(nelect_diff/2) + 2 for i in conduction_indices]
        for d in capture_final_path.glob('WSWQ_*'):
            if str(d) in [f'{str(capture_final_path)}/WSWQ_{str(i).replace('.', ''):0>3}' for i in displacements]:
                Q_struc = Structure.from_file(d / 'CONTCAR')
                Q = get_Q_from_struct(f_struc, i_struc, Q_struc)
                f_WSWQs.append((Q, d / 'WSWQ'))
        # wavefunction indexing for get_Wif_from_WSWQ is 1-based indexing, spin (0 - up, 1 - down), & kpoint defaults to first kpoint
        f_Wifs_h = get_Wif_from_WSWQ(f_WSWQs, str(ground_vr_path), int(max(valence_indices)+1), valence_indices, spin=spin, kpoint=kpt_idx+1, fig=fig_h)
        f_Wifs_e = get_Wif_from_WSWQ(f_WSWQs, str(ground_vr_path), int(min(conduction_indices)-1), conduction_indices, spin=spin, kpoint=kpt_idx+1, fig=fig_e)
        Wif_h, Wif_e = np.sqrt(np.mean([x[1]**2 for x in f_Wifs_h])), np.sqrt(np.mean([x[1]**2 for x in f_Wifs_e]))
    elif coupling_state.lower() == 'initial' or coupling_state.lower() == 'i' or coupling_state.lower() == 'excited':
        # path to initial vasprun
        ground_vr_path = capture_initial_path / f'WAV_{str(Q0).replace('.', ''):0>3}' / 'vasprun.xml'
        # adjust valence/conduction band indices by electron difference from perfect reference to coupling reference
        defect_eigenvals = Eigenval(capture_final_path / f'WAV_{str(Q0).replace('.', ''):0>3}' / 'EIGENVAL', separate_spins=True)
        nelect_diff = perfect_eigenvals.nelect - defect_eigenvals.nelect
        valence_indices = [i - floor(nelect_diff/2) for i in valence_indices]
        conduction_indices = [i - floor(nelect_diff/2) + 1 for i in conduction_indices]
        for d in capture_initial_path.glob('WSWQ_*'):
            if str(d) in [f'{str(capture_initial_path)}/WSWQ_{str(i).replace('.', ''):0>3}' for i in displacements]:
                Q_struc = Structure.from_file(d / 'CONTCAR')
                Q = get_Q_from_struct(f_struc, i_struc, Q_struc)
                i_WSWQs.append((Q, d / 'WSWQ'))
        # wavefunction indexing for get_Wif_from_WSWQ is 1-based indexing, spin (0 - up, 1 - down), & kpoint defaults to first kpoint
        i_Wifs_h = get_Wif_from_WSWQ(i_WSWQs, str(ground_vr_path), int(max(valence_indices)+1), valence_indices, spin=spin, kpoint=kpt_idx+1, fig=fig_h)
        i_Wifs_e = get_Wif_from_WSWQ(i_WSWQs, str(ground_vr_path), int(min(conduction_indices)-1), conduction_indices, spin=spin, kpoint=kpt_idx+1, fig=fig_e)
        Wif_h, Wif_e = np.sqrt(np.mean([x[1]**2 for x in i_Wifs_h])), np.sqrt(np.mean([x[1]**2 for x in i_Wifs_e]))
    else:
        raise ValueError('Please choose either the final/initial state for use in calculating the electron-phonon coupling matrix element.')

    # add band indices for defects considered in electron-phonon coupling matrix element calculations
    defect_indices.append(int(max(valence_indices)+1))
    if int(min(conduction_indices)-1) not in defect_indices:
        defect_indices.append(int(min(conduction_indices)-1))
    defect_indices.sort()
    
    plt.tight_layout()
    plt.ylim(-0.35,0.35)
    if savefig == None:
        plt.show()
    elif type(savefig) == str:
        plt.savefig(capture_calc_path / savefig, dpi=300)
    else:
        raise ValueError('Please choose a valid image name.')

    # calculate effective mass of the charge carrier
    m_h, m_e = m_h, m_e

    # calculate configurational degeneracy of the defect
    g = g

    # add info to yaml dict
    cc_dict.update({
        'volume': float(poscar_initial.structure.volume),
        'dQ': float(dQ),
        'Qmin': float(Qmin),
        'Qmax': float(Qmax),
        'Q0': float(Q0),
        'g': int(g),
        'valence bands': str(valence_indices),
        'conduction bands': str(conduction_indices),
        'defect bands': str(defect_indices),
        'Wif_h': float(Wif_h),
        'Wif_e': float(Wif_e),
        'm_h': float(m_h),
        'm_e': float(m_e),
        'dielectric': float(dielectric_const)
    })

    with open(capture_calc_path / cc_yaml_filename, 'w') as yaml_file:
        yaml.dump(cc_dict, yaml_file)
    
    return cc_dict