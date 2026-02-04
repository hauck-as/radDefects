#!/usr/bin/env python3
"""Python module used to setup and analyze carrier capture calculations using VASP."""
import os
from pathlib import Path
import shutil
import re
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
# import sumo.electronic_structure.effective_mass as sem


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
        'ALGO': 'All',
        'ENCUT': 400.0,
        'PREC': 'Accurate',
        'LHFCALC': True,
        'GGA': 'PE',
        'HFSCREEN': 0.2,
        'AEXX': 0.305,
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
    if struct_gen_type.lower()[0] == 'c':
        ground = carriercapturejl_interpolate(ground_struct, excited_struct, disp_range=displacements)
        excited = carriercapturejl_interpolate(excited_struct, ground_struct, disp_range=displacements)
    elif struct_gen_type.lower()[0] == 'n':
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


def create_carrier_capture_wav_vasp_inputs(disp_dir, charge=0, potcar=None, incar_settings={}):
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
        vasp_wav_inputs = create_carrier_capture_wav_vasp_inputs(disp_calc_path_i, charge=q_initial, potcar=potcar, incar_settings=incar_settings)
        vasp_wav_inputs.write_input(wav_calc_path_i, make_dir_if_not_present=True)

        # setup WAVECAR calcs for ground structures
        disp_calc_path_f, wav_calc_path_f = capture_final_path / f'DISP_{disp_suffix}', capture_final_path / f'WAV_{disp_suffix}'
        vasp_wav_inputs = create_carrier_capture_wav_vasp_inputs(disp_calc_path_f, charge=q_final, potcar=potcar, incar_settings=incar_settings)
        vasp_wav_inputs.write_input(wav_calc_path_f, make_dir_if_not_present=True)
        
    return None


def create_carrier_capture_wswq_vasp_inputs(wav_dir, charge=0, potcar=None, incar_settings={}):
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
        vasp_wswq_inputs = create_carrier_capture_wswq_vasp_inputs(wav_calc_path_i, charge=q_initial, potcar=potcar, incar_settings=incar_settings)
        vasp_wswq_inputs.write_input(wswq_calc_path_i, make_dir_if_not_present=True, files_to_transfer={'WAVECAR.qqq': wav_calc_path_i / 'WAVECAR'})
        shutil.copyfile(ref_wavecar_path, wswq_calc_path_i / 'WAVECAR')

        # setup WSWQ calcs for ground structures
        wav_calc_path_f, wswq_calc_path_f = capture_final_path / f'WAV_{disp_suffix}', capture_final_path / f'WSWQ_{disp_suffix}'
        vasp_wswq_inputs = create_carrier_capture_wswq_vasp_inputs(wav_calc_path_f, charge=q_final, potcar=potcar, incar_settings=incar_settings)
        vasp_wswq_inputs.write_input(wswq_calc_path_f, make_dir_if_not_present=True, files_to_transfer={'WAVECAR.qqq': wav_calc_path_f / 'WAVECAR'})
        shutil.copyfile(ref_wavecar_path, wswq_calc_path_f / 'WAVECAR')
        
    return None


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
        
    return None


def gather_qe_data(excited_poscar_path, ground_poscar_path, disp_path, displacements=None):
    """
    Gather Q vs. E data for configuration coordinate diagram generation assuming 1D potential energy surfaces.
    """
    # set default for displacements array corresponding to DISP calculations
    displacements = np.array([-1.0, -0.6, -0.4, -0.2, -0.1, 0., 0.1, 0.2, 0.4, 0.6, 1.0]) if displacements is None else displacements

    Q, E = [], []    
    for i in range(displacements.shape[0]):
        # assume displacements are organized as DISP_XXX (e.g., DISP_001 or DISP_-01)
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'
        
        # get excited & ground state structures
        excited_poscar = Poscar.from_file(excited_poscar_path)
        ground_poscar = Poscar.from_file(ground_poscar_path)
        
        # get intermediate structures and energy files
        inter_poscar = Poscar.from_file(disp_path / f'DISP_{disp_suffix}' / 'CONTCAR')
        oszi = Oszicar(disp_path / f'DISP_{disp_suffix}' / 'OSZICAR')
        
        Q.append(get_Q_from_struct(ground_poscar.structure, excited_poscar.structure, inter_poscar.structure))
        E.append(oszi.final_energy)

    # create dataframe for QE data
    qe_df = pd.DataFrame({'Q': Q, 'E': E})

    return qe_df


def gather_qe_carrier_capture(defect_name, q_initial, q_final, cc_path=Path.cwd(), displacements=None, qe_filename='potential.csv'):
    """
    Gather Q vs. E data for carrier capture CCD.
    """
    capture_calc_path = cc_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    charge_diff = q_final-q_initial
    capture_initial_path, capture_final_path = capture_calc_path / 'i_q', capture_calc_path / f'f_q{charge_diff:+}'

    # gather Q vs. E data for initial and final states for carrier capture
    excited_poscar_path = capture_initial_path / 'DISP_000' / 'CONTCAR'
    ground_poscar_path = capture_final_path / 'DISP_000' / 'CONTCAR'
    qe_i_df = gather_qe_data(excited_poscar_path, ground_poscar_path, capture_initial_path, displacements=displacements)
    qe_f_df = gather_qe_data(excited_poscar_path, ground_poscar_path, capture_final_path, displacements=displacements)
    
    qe_i_df.to_csv(capture_initial_path / qe_filename, index=False)
    qe_f_df.to_csv(capture_final_path / qe_filename, index=False)

    return qe_i_df, qe_f_df


def gather_qe_migration(defect_name, q_ext, q_initial, q_final, mig_path=Path.cwd(), displacements=None, qe_filename='potential.csv'):
    """
    Gather Q vs. E data for carrier-capture-enhanced athermal migration.
    """
    mig_calc_path = mig_path / '_'.join([defect_name, str(q_ext), f'{q_initial}to{q_final}'])
    capture_initial_path = mig_path / '_'.join([defect_name, str(q_initial), f'{q_initial}to{q_final}'])
    capture_final_path = mig_path / '_'.join([defect_name, str(q_final), f'{q_initial}to{q_final}'])

    # gather Q vs. E data for migration charge state
    excited_poscar_path = capture_initial_path / 'DISP_000' / 'CONTCAR'
    ground_poscar_path = capture_final_path / 'DISP_000' / 'CONTCAR'
    qe_mig_df = gather_qe_data(excited_poscar_path, ground_poscar_path, mig_calc_path, displacements=displacements)
    
    qe_mig_df.to_csv(mig_calc_path / qe_filename, index=False)

    return qe_mig_df


def calculate_g_capture(degen_initial, degen_final):
    """
    Calculate capture pathway degeneracy from initial and final degeneracy dictionaries.
    Dictionaries include orientational and spin degeneracies as well as multiplicity
    of the defect site in the bulk cell.
    """
    g_initial = float(degen_initial['g_Orient'])
    g_final = float(degen_final['g_Orient'])
    g_cap = round(max((g_final/g_initial), 1.), 0)
    g_cap *= float(degen_final['g_Spin'])
    return g_cap


def calculate_g_capture_e_and_h(q_1, q_2, degen_1, degen_2):
    """
    Calculate capture pathway degeneracies for hole and electron capture.
    """
    g_capture_dict = {'g_h': 1., 'g_e': 1.}
    g_12 = calculate_g_capture(degen_1, degen_2)
    g_21 = calculate_g_capture(degen_2, degen_1)

    if q_1 > q_2:
        g_capture_dict.update({'g_h': g_21, 'g_e': g_12})
    elif q_1 < q_2:
        g_capture_dict.update({'g_h': g_12, 'g_e': g_21})
    else:
        print('Charge states must be different from each other.')
    
    return g_capture_dict


def parse_carrier_capture_info(defect_name, q_initial, q_final, coupling_state='final', g_e=1, g_h=1, m_e=0.2, m_h=0.8, dielectric_const=10., kpt_idx=0, spin=0, base_path=Path.cwd(), displacements=None, savefig=None):
    """
    Analyze defect calculations to gather parameters for carrier capture calculations.
    """
    # initialize yaml with given information
    cc_yaml_filename = 'carrier_capture_params.yaml'
    cc_dict = {
        'defect': str(defect_name),
        'q_initial': int(q_initial),
        'q_final': int(q_final),
        'coupling state': str(coupling_state),
        'kpoint': int(kpt_idx + 1),
        'spin': int(spin)
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
    fig_vbm, fig_cbm = plt.figure(figsize=(12, 5)), plt.figure(figsize=(12, 5))
    if coupling_state.lower() == 'final' or coupling_state.lower() == 'f' or coupling_state.lower() == 'ground':
        # path to initial vasprun
        ground_vr_path = capture_final_path / f'WAV_{str(Q0).replace('.', ''):0>3}' / 'vasprun.xml'
        # adjust valence/conduction band indices by electron difference from perfect reference to coupling reference
        defect_eigenvals = Eigenval(capture_final_path / f'WAV_{str(Q0).replace('.', ''):0>3}' / 'EIGENVAL', separate_spins=True)
        nelect_diff = perfect_eigenvals.nelect - defect_eigenvals.nelect
        valence_indices = [i - floor(nelect_diff/2) for i in valence_indices]
        conduction_indices = [i - floor(nelect_diff/2) + 1 for i in conduction_indices]
        for d in capture_final_path.glob('WSWQ_*'):
            if str(d) in [f'{str(capture_final_path)}/WSWQ_{str(i).replace('.', ''):0>3}' for i in displacements]:
                Q_struc = Structure.from_file(d / 'CONTCAR')
                Q = get_Q_from_struct(f_struc, i_struc, Q_struc)
                f_WSWQs.append((Q, d / 'WSWQ'))
        # wavefunction indexing for get_Wif_from_WSWQ is 1-based indexing, spin (0 - up, 1 - down), & kpoint defaults to first kpoint
        f_Wifs_vbm = get_Wif_from_WSWQ(f_WSWQs, str(ground_vr_path), int(max(valence_indices)+1), valence_indices, spin=spin, kpoint=kpt_idx+1, fig=fig_vbm)
        f_Wifs_cbm = get_Wif_from_WSWQ(f_WSWQs, str(ground_vr_path), int(min(conduction_indices)-1), conduction_indices, spin=spin, kpoint=kpt_idx+1, fig=fig_cbm)
        Wif_vbm, Wif_cbm = np.sqrt(np.mean([x[1]**2 for x in f_Wifs_vbm])), np.sqrt(np.mean([x[1]**2 for x in f_Wifs_cbm]))
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
        i_Wifs_vbm = get_Wif_from_WSWQ(i_WSWQs, str(ground_vr_path), int(max(valence_indices)+1), valence_indices, spin=spin, kpoint=kpt_idx+1, fig=fig_vbm)
        i_Wifs_cbm = get_Wif_from_WSWQ(i_WSWQs, str(ground_vr_path), int(min(conduction_indices)-1), conduction_indices, spin=spin, kpoint=kpt_idx+1, fig=fig_cbm)
        Wif_vbm, Wif_cbm = np.sqrt(np.mean([x[1]**2 for x in i_Wifs_vbm])), np.sqrt(np.mean([x[1]**2 for x in i_Wifs_cbm]))
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

    # calculate capture degeneracies of the defect
    if g_e is None or g_h is None:
        try:
            degen_df = pd.read_csv(defect_path / 'degeneracies.csv')
            degen_initial = degen_df.loc[(degen_df['Defect'] == defect_name) & (degen_df['q'] == q_initial)]
            degen_final = degen_df.loc[(degen_df['Defect'] == defect_name) & (degen_df['q'] == q_final)]
        
            try:
                degen_initial_dict = degen_initial.to_dict(orient='records')[0]
                degen_final_dict = degen_final.to_dict(orient='records')[0]
                g_cap_dict = calculate_g_capture_e_and_h(q_initial, q_final, degen_initial_dict, degen_final_dict)
                g_e, g_h = g_cap_dict['g_e'], g_cap_dict['g_h']
            except IndexError:
                cc_dict.update({'g_e': None, 'g_h': None})
                print('Defect transition not present in degeneracies.csv')
            
            if g_e.is_integer() == False:
                print('Electron capture degeneracy not an integer.')
            if g_h.is_integer() == False:
                print('Hole capture degeneracy not an integer.')
                
            cc_dict.update({'g_e': int(g_e), 'g_h': int(g_h)})
        except FileNotFoundError:
            cc_dict.update({'g_e': None, 'g_h': None})
            print('No degeneracies.csv file found in defect directory.')
    else:
        cc_dict.update({'g_e': int(g_e), 'g_h': int(g_h)})

    # add info to yaml dict
    cc_dict.update({
        'volume': float(poscar_initial.structure.volume),
        'dQ': float(dQ),
        'Qmin': float(Qmin),
        'Qmax': float(Qmax),
        'Q0': float(Q0),
        'valence bands': str(valence_indices),
        'conduction bands': str(conduction_indices),
        'defect bands': str(defect_indices),
        'Wif_vbm': float(Wif_vbm),
        'Wif_cbm': float(Wif_cbm),
        'm_h': float(m_h),
        'm_e': float(m_e),
        'dielectric': float(dielectric_const)
    })

    with open(capture_calc_path / cc_yaml_filename, 'w') as yaml_file:
        yaml.dump(cc_dict, yaml_file)
    
    return cc_dict


def capture_rate(
    cap_coeff: float,
    defect_conc: float,
    carrier_conc: float
) -> float:
    """
    Calculates the nonradiative capture rate for a given temperature from the nonradiative
    capture coefficient, defect concentration, and carrier concentration.
    
    Args
    ---------
        cap_coeff (float):
            Capture coefficient in units of (cm^3 s^-1).
        defect_conc (float):
            Defect concentration in units of (cm^-3).
        carrier_conc (float):
            Carrier density/concentration in units of (cm^-3).

    Returns
    ---------
        Capture rate with units of (cm^-3 s^-1).
    """
    cap_rate = cap_coeff*defect_conc*carrier_conc
    return cap_rate


def partial_capture_rate(
    cap_coeff: float,
    carrier_conc: float
) -> float:
    """
    Calculates the nonradiative capture rate for a given temperature from the nonradiative
    capture coefficient and carrier concentration.
    
    Args
    ---------
        cap_coeff (float):
            Capture coefficient in units of (cm^3 s^-1).
        carrier_conc (float):
            Carrier density/concentration in units of (cm^-3).

    Returns
    ---------
        Partial capture rate with units of (s^-1).
    """
    partial_cap_rate = cap_coeff*carrier_conc
    return partial_cap_rate


def effective_band_dos(
    T: float,
    eff_mass: float = 1.,
    mc: int = 1
) -> float:
    """
    Calculates the temperature-dependent effective density of states for the valence/conduction band.
    
    Args
    ---------
        T (float):
            Temperature in units of (K).
        eff_mass (float):
            Effective mass of the carrier in the associated band in units of rest electron mass.
            Defaults to 1 (equal to electron rest mass).
        mc (int):
            Number of equivalent energy minima for the conduction band. Defaults to 1, should only
            be specified for conduction band effective DOS.

    Returns
    ---------
        Effective band density of states with units of (cm^-3).
    """
    PLANCK_EV = 4.135667696e-15  # eV s
    BOLTZMANN_EV = 8.617333262e-5  # eV / K
    ELECTRON_MASS_EV = 0.51099895069e6  # eV
    LIGHT_SPEED_CM = 2.99792458e10  # cm / s

    eff_dos = 2*mc*((2*np.pi*eff_mass*ELECTRON_MASS_EV*BOLTZMANN_EV*T)/((PLANCK_EV**2)*(LIGHT_SPEED_CM**2)))**(3/2)
    
    return eff_dos


def emission_coeff(
    T: float,
    cap_coeff: float,
    dE: float,
    eff_mass: float = 1.,
    mc: int = 1
) -> float:
    """
    Calculates the carrier thermal emission coefficient as a function of temperature using the
    capture coefficient and effective density of states for the valence/conduction band.
    
    Args
    ---------
        T (float):
            Temperature in units of (K).
        cap_coeff (float):
            Capture coefficient in units of (cm^3 s^-1).
        dE (float):
            Thermodynamic transition level in units of (eV).
        eff_mass (float):
            Effective mass of the carrier in the associated band in units of rest electron mass.
            Defaults to 1 (equal to electron rest mass).
        mc (int):
            Number of equivalent energy minima for the conduction band. Defaults to 1, should only
            be specified for conduction band effective DOS.

    Returns
    ---------
        Emission coefficient with units of (s^-1).
    """
    BOLTZMANN_EV = 8.617333262e-5  # eV / K

    eff_dos = effective_band_dos(T, eff_mass=eff_mass, mc=mc)
    emission_coeff = cap_coeff*eff_dos*np.exp(-dE/(BOLTZMANN_EV*T))
    
    return emission_coeff


def emission_rate(
    emission_coeff: float,
    defect_conc: float
) -> float:
    """
    Calculates the emission rate for a given temperature from the emission coefficient
    and defect concentration.
    
    Args
    ---------
        emission_coeff (float):
            Emission coefficient in units of (s^-1).
        defect_conc (float):
            Defect concentration in units of (cm^-3).

    Returns
    ---------
        Emission rate with units of (cm^-3 s^-1).
    """
    emission_rate = emission_coeff*defect_conc
    return emission_rate


def emission_rate_factor(
    T: float,
    dE: float,
    defect_conc: float,
    eff_mass: float = 1.,
    mc: int = 1
) -> float:
    """
    Calculates the carrier thermal emission exponential factor times the defect concentration as a
    function of temperature using the effective density of states for the valence/conduction band.
    
    Args
    ---------
        T (float):
            Temperature in units of (K).
        defect_conc (float):
            Defect concentration in units of (cm^-3).
        dE (float):
            Thermodynamic transition level in units of (eV).
        eff_mass (float):
            Effective mass of the carrier in the associated band in units of rest electron mass.
            Defaults to 1 (equal to electron rest mass).
        mc (int):
            Number of equivalent energy minima for the conduction band. Defaults to 1, should only
            be specified for conduction band effective DOS.

    Returns
    ---------
        Emission exponential factor times the defect concentration with units of (cm^-6).
    """
    BOLTZMANN_EV = 8.617333262e-5  # eV / K

    eff_dos = effective_band_dos(T, eff_mass=eff_mass, mc=mc)
    emission_exp_factor = eff_dos*np.exp(-dE/(BOLTZMANN_EV*T))
    emission_rate_sans_coeff = emission_exp_factor*defect_conc
    
    return emission_rate_sans_coeff


def calc_all_cap_rates(cap_coeff_csv, conc_csv, temp=300):
    """
    Calculates all nonradiative capture rates for a given temperature using DataFrames for
    nonradiative capture coefficients and defect/carrier concentrations. DataFrames are made from
    the paths to the CSV files for each type of data. Capture coefficients are made from radDefects
    and defect/carrier concentrations are calculated and formatted using doped.
    """
    cap_coeff_df = pd.read_csv(cap_coeff_csv, index_col=0)
    conc_df = pd.read_csv(conc_csv, index_col=0)
    
    cap_rate_df = cap_coeff_df.copy(deep=True)
    cap_rate_df.rename(columns={'C_p': 'R_p', 'C_n': 'R_n'}, inplace=True)
    
    # select concentrations for specified temperature
    conc_temp_df = conc_df.query(f'`Temperature (K)` == {temp}')

    # remove defects that don't have both capture coefficient values and concentration values
    doped_defects, coeff_defects = conc_temp_df.index, cap_coeff_df.index
    simplified_defect_names = list(map(lambda x: re.sub(r'\d+', '', x), doped_defects))
    coeff_doped_cross_dict = {}
    
    for i, defect in enumerate(simplified_defect_names):
        if defect in coeff_defects:
            coeff_doped_cross_dict.update({defect: doped_defects[i]})
        else:
            # need to change if defect matches more than one alt_defect (i.e., two site types for one defect)
            for j, alt_defect in enumerate(coeff_defects):
                if defect in alt_defect:
                    coeff_doped_cross_dict.update({alt_defect: doped_defects[i]})

    cap_rate_df = cap_rate_df.loc[coeff_doped_cross_dict.keys()]
    hole_cap_rates, electron_cap_rates = cap_rate_df['R_p'].copy(deep=False), cap_rate_df['R_n'].copy(deep=False)

    defect_tracking = []
    for capture in cap_rate_df.iterrows():
        capture_defect, capture_info = capture
        if capture_defect not in defect_tracking:
            if type(cap_rate_df.loc[capture_defect]['R_p']) == pd.Series:
                hole_cap_rates.loc[capture_defect] = cap_rate_df.loc[capture_defect]['R_p'].apply(
                    lambda x: capture_rate(
                        x,
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]]['Concentration (cm^-3)'],
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]]['Holes (cm^-3)']
                    )
                )
                electron_cap_rates.loc[capture_defect] = cap_rate_df.loc[capture_defect]['R_n'].apply(
                    lambda x: capture_rate(
                        x,
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]]['Concentration (cm^-3)'],
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]]['Electrons (cm^-3)']
                    )
                )
            elif type(cap_rate_df.loc[capture_defect]['R_p']) == np.float64:
                hole_cap_rates.loc[capture_defect] = pd.Series(cap_rate_df.loc[capture_defect]['R_p'], index=[capture_defect]).apply(
                    lambda x: capture_rate(
                        x,
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]]['Concentration (cm^-3)'],
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]]['Holes (cm^-3)']
                    )
                )
                electron_cap_rates.loc[capture_defect] = pd.Series(cap_rate_df.loc[capture_defect]['R_n'], index=[capture_defect]).apply(
                    lambda x: capture_rate(
                        x,
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]]['Concentration (cm^-3)'],
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]]['Electrons (cm^-3)']
                    )
                )
            defect_tracking.append(capture_defect)
    
    return cap_rate_df


def estimate_transition_time(cap_coeff_path, q_i, q_f, defect_name, temp=300, p_conc=1e18, n_conc=1e18):
    """
    Use to estimate time it takes to go from q_i to q_f, assuming single charge transitions
    """
    cap_coeff_df = pd.read_csv(cap_coeff_path, index_col=0)
    time = 0
    
    if q_i < q_f:  # hole capture
        cap_type = 'p'
        q_all = np.arange(q_i, q_f+1, 1)
    elif q_i > q_f:  # electron capture
        cap_type = 'n'
        q_all = np.arange(q_i, q_f-1, -1)
    else:
        print('Neither hole or electron capture, q_i and q_f may be same.')

    for i in range(q_all.shape[0]-1):
        try:
            cap_coeff = cap_coeff_df.loc[defect_name].query(f'q_i=={q_all[i]} & q_f=={q_all[i+1]}')[f'C_{cap_type}'].iloc[0]
        except IndexError:
            cap_coeff = cap_coeff_df.loc[defect_name].query(f'q_f=={q_all[i]} & q_i=={q_all[i+1]}')[f'C_{cap_type}'].iloc[0]
        print(f'Capture coefficient ({cap_type}) for ({q_all[i]}/{q_all[i+1]}): {cap_coeff} cm^3/s')

        if cap_type == 'p':
            t_i = partial_capture_rate(cap_coeff, p_conc)**-1
            print(f'Time for ({q_all[i]}/{q_all[i+1]}) transition @ p={p_conc}: {t_i} s')
        elif cap_type == 'n':
            t_i = partial_capture_rate(cap_coeff, n_conc)**-1
            print(f'Time for ({q_all[i]}/{q_all[i+1]}) transition @ n={n_conc}: {t_i} s')

        time += t_i
    
    return time
