"""Python module to setup/analyze VASP carrier capture calculations."""
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from numpy.typing import ArrayLike
from pymatgen.util.typing import PathLike

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
from numpy.typing import ArrayLike
from pymatgen.util.typing import PathLike

import pydefect
from pydefect.analyzer.unitcell import Unitcell
from pydefect.analyzer.transition_levels import TransitionLevels
from nonrad.ccd import get_dQ, get_Q_from_struct, get_cc_structures
from nonrad.elphon import get_Wif_from_WSWQ
from nonrad.scaling import sommerfeld_parameter, charged_supercell_scaling_VASP
# import sumo.electronic_structure.effective_mass as sem


def create_carrier_capture_disp_vasp_inputs(
    poscar_structure: Structure,
    charge: int,
    disp_factor: float,
    potcar: Potcar | None = None,
    kpt: Kpoints | None = None,
    incar_settings: dict = {}
) -> VaspInput:
    """
    Create a set of VASP inputs for carrier capture displacement
    calculations.

    Args
    ---------
        poscar_structure (Structure):
            `pymatgen` Structure object corresponding to the displaced
            structure.
        charge (int):
            Integer charge state for the defect.
        disp_factor (float):
            Fractional configurational displacement based on
            interpolation between equilibrium structures.
        potcar (Potcar):
            `pymatgen` Potcar object. Defaults to None (Potcar made from
            standard PBEv54 potentials).
        kpt (Kpoints):
            `pymatgen` Kpoints object. Defaults to None (Special k-point
            at (0.5, 0.5, 0.5)).
        incar_settings (dict):
            Dictionary of INCAR settings to to use for the single-energy
            displacement calculations. IBRION, NSW, NELECT, NUPDOWN,
            MAGMOM, and LWAVE are set within the function. Defaults to
            no additional settings.

    Returns
    ---------
        pymatgen.io.vasp.inputs.VaspInput set for displacement calcs.
    """
    # POSCAR file from displacement structure
    poscar = Poscar(poscar_structure)
    
    # POTCAR file of standard PAW PBEv54 potentials without suffixes
    potcar = Potcar(poscar.site_symbols, 'PBE_54') if potcar is None else potcar
    
    # get electron count from POSCAR/POTCAR
    nelect = 0
    for i, ele in enumerate(poscar.site_symbols):
        for j in potcar:
            if ele == j.element:
                nelect += int(j.nelectrons)*poscar.natoms[i]
    nelect -= charge
    
    # single special k-point to improve energies over Gamma-point
    kpt = Kpoints.gamma_automatic(shift=(0.5, 0.5, 0.5)) if kpt is None else kpt
    
    # INCAR file for single-point energy calculation with charged defect
    incar_dict = {
        # no update to atomic positions
        'IBRION': -1,
        'NSW': 0,
        # electron counts from POSCAR/POTCAR
        'NELECT': nelect,
        'NUPDOWN': int(nelect%2),
        # MAGMOM initial guess of 0.6 for all atoms
        'MAGMOM': f'{poscar.structure.num_sites}*0.6',
        # write WAVECAR around equilibrium configuration
        'LWAVE': True if disp_factor <= 0.2 and disp_factor >= -0.2 else False,
    }
    incar_dict.update(incar_settings)
    
    incar = Incar(incar_dict)

    disp_inps = VaspInput(
        poscar=poscar,
        potcar=potcar,
        kpoints=kpt,
        incar=incar
    )
    
    return disp_inps


def carriercapturejl_interpolate(
    struct_i: Structure,
    struct_f: Structure,
    disp_range: ArrayLike = np.linspace(-1, 1, 11)
) -> list[Structure]:
    """
    Interpolates pymatgen structures using the method defined in the
    CarrierCapture.jl code gen_cc_struct.py script. Modified to match
    I/O of nonrad.

    Args
    ---------
        struct_i (Structure):
            `pymatgen` Structure object corresponding to the initial
            structure.
        struct_f (Structure):
            `pymatgen` Structure object corresponding to the final
            structure.
        disp_range (ArrayLike):
            Array of fractional displacements between the initial and
            final structures. 0 corresponds to the equilibrium structure
            of the initial state and 1 corresponds to the equilibrium
            structure of the final state. Defaults to -1.0 to 1.0 in
            0.2 increments.

    Returns
    ---------
        List of interpolated/extrapolated `pymatgen` Structure objects.
    """
    interpolated_structs = []
    
    # model code on CarrierCapture.jl gen_cc_struct.py script
    # A. Alkauskas, Q. Yan, and C. G. Van de Walle, Phys. Rev. B 90, 27 (2014)
    delta_R = struct_f.frac_coords - struct_i.frac_coords
    delta_R = (delta_R + 0.5) % 1 - 0.5
    
    lattice = struct_i.lattice.matrix  #[None,:,:]
    delta_R = np.dot(delta_R, lattice)
    
    masses = np.array([spc.atomic_mass for spc in struct_i.species])
    delta_Q2 = masses[:,None] * delta_R ** 2
    
    for frac in disp_range:
        disp = frac * delta_R
        struct = Structure(
            struct_i.lattice,
            struct_i.species,
            struct_i.cart_coords + disp,
            coords_are_cartesian=True
        )
        interpolated_structs.append(struct)
    
    return interpolated_structs

    
def setup_carrier_capture_from_pydefect(
    defect_name: str,
    q_initial: int,
    q_final: int,
    displacements: ArrayLike = np.array([
        -1.0, -0.6, -0.4, -0.2, -0.1, 0., 0.1, 0.2, 0.4, 0.6, 1.0
    ]),
    struct_gen_type: str = 'CarrierCapture.jl',
    base_path: Path = Path.cwd(),
    potcar: Potcar | None = None,
    kpt: Kpoints | None = None,
    incar_settings: dict = {
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
        'NELM': 100,
        'ISPIN': 2,
        'LASPH': True,
        'LMAXMIX': 4,
        'LORBIT': 10,
        'LCHARG': False,
        'LVTOT': False,
        'LVHAR': False
    }
) -> Path:
    """
    Setup carrier capture calculations from previous `pydefect`
    calculations.

    Args
    ---------
        defect_name (str):
            Name of defect in defect_site# format (e.g., Va_N1).
        q_initial (int):
            Initial charge state of defect.
        q_final (int):
            Final charge state of defect.
        displacements (ArrayLike):
            Array of fractional displacements between the initial and
            final structures. 0 corresponds to the equilibrium structure
            of the final state and 1 corresponds to the equilibrium
            structure of the initial state. Defaults to -1.0 to 1.0 in
            varying increments (finer near equilibrium).
        struct_gen_type (str):
            Method for generating displaced structures. Implemented
            options are 'CarrierCapture.jl' and 'nonrad' for the
            corresponding methods. Defaults to 'CarrierCapture.jl'.
        base_path (Path):
            Base path to be used for setting up the carrier capture
            calculations. Should be the base directory for `pydefect`
            subdirectories and contain the defect and carrier_capture
            subdirectories. Defaults to Path.cwd().
        potcar (Potcar):
            `pymatgen` Potcar object. Defaults to None (Potcar made from
            standard PBEv54 potentials).
        kpt (Kpoints):
            `pymatgen` Kpoints object. Defaults to None (Special k-point
            at (0.5, 0.5, 0.5)).
        incar_settings (dict):
            Dictionary of INCAR settings to to use for the single-energy
            displacement calculations. IBRION, NSW, NELECT, NUPDOWN,
            MAGMOM, and LWAVE are set within the input creation
            function. Defaults to additional settings used for an HSE
            calc for GaN with spin-polarization.

    Returns
    ---------
        Path object of the directory containing the displacement
        calculations.
    """
    defect_path = base_path / 'defect'
    capture_path = base_path / 'carrier_capture'
    defect_initial_path = defect_path / '_'.join([defect_name, str(q_initial)])
    defect_final_path = defect_path / '_'.join([defect_name, str(q_final)])
    capture_calc_path = capture_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    capture_calc_path.mkdir(parents=True, exist_ok=True)

    charge_diff = q_final - q_initial
    capture_initial_path = capture_calc_path / 'i_q'
    capture_final_path = capture_calc_path / f'f_q{charge_diff:+}'
    capture_initial_path.mkdir(parents=True, exist_ok=True)
    capture_final_path.mkdir(exist_ok=True)
    capture_disp_path = capture_calc_path / '_'.join([defect_name, 'DISPLACEMENT'])
    capture_disp_path.mkdir(exist_ok=True)

    shutil.copyfile(defect_initial_path / 'CONTCAR', capture_disp_path / 'POSCAR_i')
    shutil.copyfile(defect_final_path / 'CONTCAR', capture_disp_path / 'POSCAR_f')

    excited_poscar = Poscar.from_file(capture_disp_path / 'POSCAR_i')
    ground_poscar = Poscar.from_file(capture_disp_path / 'POSCAR_f')
    excited_struct, ground_struct = excited_poscar.structure, ground_poscar.structure

    if struct_gen_type.lower()[0] == 'c':
        ground = carriercapturejl_interpolate(
            ground_struct,
            excited_struct,
            disp_range=displacements
        )
        excited = carriercapturejl_interpolate(
            excited_struct,
            ground_struct,
            disp_range=displacements
        )
    elif struct_gen_type.lower()[0] == 'n':
        ground, excited = get_cc_structures(
            ground_struct,
            excited_struct,
            displacements,
            remove_zero=False
        )
    else:
        raise ValueError('Please choose a valid method for generating displacement ' +
                         'structures: CarrierCapture.jl or Nonrad')

    disp_dir_i_path = capture_disp_path / 'disp_dir_i'
    disp_dir_f_path = capture_disp_path / 'disp_dir_f'
    disp_dir_f_path.mkdir(exist_ok=True)
    disp_dir_i_path.mkdir(exist_ok=True)
    
    for i, struct in enumerate(excited):
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'
        struct.to(filename=str(disp_dir_i_path / f'POSCAR_{disp_suffix}'), fmt='poscar')
        disp_calc_path = capture_initial_path / f'DISP_{disp_suffix}'
        vasp_disp_inputs = create_carrier_capture_disp_vasp_inputs(
            struct,
            q_initial,
            displacements[i],
            potcar=potcar,
            kpt=kpt,
            incar_settings=incar_settings
        )
        vasp_disp_inputs.write_input(disp_calc_path, make_dir_if_not_present=True)
    
    for i, struct in enumerate(ground):
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'
        struct.to(filename=str(disp_dir_f_path / f'POSCAR_{disp_suffix}'), fmt='poscar')
        disp_calc_path = capture_final_path / f'DISP_{disp_suffix}'
        vasp_disp_inputs = create_carrier_capture_disp_vasp_inputs(
            struct,
            q_final,
            displacements[i],
            potcar=potcar,
            kpt=kpt,
            incar_settings=incar_settings
        )
        vasp_disp_inputs.write_input(disp_calc_path, make_dir_if_not_present=True)
    
    return capture_calc_path


def create_carrier_capture_wav_vasp_inputs(
    disp_dir: PathLike,
    charge: int = 0,
    potcar: Potcar | None = None,
    kpt: Kpoints | None = None,
    incar_settings: dict = {}
) -> VaspInput:
    """
    Read in VASP inputs from carrier capture displacement calculations
    and create VASP inputs for WAVECAR calculations.

    Args
    ---------
        disp_dir (PathLike):
            Path to displacement calculation directory.
        charge (int):
            Integer charge state for the defect. Defaults to 0 (neutral).
        potcar (Potcar):
            `pymatgen` Potcar object. Defaults to None (Potcar made from
            standard PBEv54 potentials).
        kpt (Kpoints):
            `pymatgen` Kpoints object. Defaults to None (Gamma-point
            only).
        incar_settings (dict):
            Dictionary of INCAR settings to to use for the single-energy
            displacement calculations. NELECT and NUPDOWN are set within
            the function and everything else is taken from the previous
            displacement calc. Defaults to no additional settings.

    Returns
    ---------
        pymatgen.io.vasp.inputs.VaspInput set for wavefunction calcs.
    """
    # input files from displacement structure
    disp_inps = VaspInput.from_directory(disp_dir)
    poscar_disp = disp_inps.poscar

    # update POTCAR if specified
    potcar = Potcar(poscar_disp.site_symbols, 'PBE_54') if potcar == None else potcar

    # get electron count from POSCAR/POTCAR
    nelect = 0
    for i, ele in enumerate(poscar_disp.site_symbols):
        for j in potcar:
            if ele == j.element:
                nelect += int(j.nelectrons)*poscar_disp.natoms[i]
    nelect -= charge
    
    # use gamma point (assume direct bandgap)
    kpt = Kpoints.gamma_automatic() if kpt is None else kpt
    
    # update INCAR if specified, change NELECT based on POTCAR
    incar_dict = disp_inps.incar
    incar_dict.update({
        'NELECT': nelect,
        'NUPDOWN': int(nelect%2)
    })
    incar_dict.update(incar_settings)
    incar = Incar(incar_dict)

    # create updated VASP input set with modified input files
    wav_inps = VaspInput(
        poscar=poscar_disp,
        potcar=potcar,
        kpoints=kpt,
        incar=incar
    )
    
    return wav_inps


def setup_carrier_capture_wav(
    defect_name: str,
    q_initial: int,
    q_final: int,
    displacements: ArrayLike = np.array([
        -0.2, -0.1, 0., 0.1, 0.2
    ]),
    cc_path: Path = Path.cwd(),
    potcar: Potcar | None = None,
    kpt: Kpoints | None = None,
    incar_settings: dict = {}
) -> None:
    """
    Setup carrier capture WAVECAR calculations from displacement
    calculations.
    
    Args
    ---------
        defect_name (str):
            Name of defect in defect_site# format (e.g., Va_N1).
        q_initial (int):
            Initial charge state of defect.
        q_final (int):
            Final charge state of defect.
        displacements (ArrayLike):
            Array of fractional displacements between the initial and
            final structures. 0 corresponds to the equilibrium structure
            of the final state and 1 corresponds to the equilibrium
            structure of the initial state. Defaults to -0.2 to 0.2 in
            0.1 increments (around final state equilibrium).
        cc_path (Path):
            Path to carrier_capture subdirectory. Defaults to Path.cwd().
        potcar (Potcar):
            `pymatgen` Potcar object. Defaults to None (Potcar made from
            standard PBEv54 potentials).
        kpt (Kpoints):
            `pymatgen` Kpoints object. Defaults to None (Gamma-point
            only).
        incar_settings (dict):
            Dictionary of INCAR settings to to use for the single-energy
            displacement calculations. NELECT and NUPDOWN are set within
            the function and everything else is taken from the previous
            displacement calc. Defaults to no additional settings.

    Returns
    ---------
        Nothing.
    """
    capture_calc_path = cc_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    charge_diff = q_final - q_initial
    capture_initial_path = capture_calc_path / 'i_q'
    capture_final_path = capture_calc_path / f'f_q{charge_diff:+}'

    for i in range(displacements.shape[0]):
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'

        # setup WAVECAR calcs for excited structures
        disp_calc_path_i = capture_initial_path / f'DISP_{disp_suffix}'
        wav_calc_path_i = capture_initial_path / f'WAV_{disp_suffix}'
        vasp_wav_inputs = create_carrier_capture_wav_vasp_inputs(
            disp_calc_path_i,
            charge=q_initial,
            potcar=potcar,
            kpt=kpt,
            incar_settings=incar_settings
        )
        vasp_wav_inputs.write_input(wav_calc_path_i, make_dir_if_not_present=True)

        # setup WAVECAR calcs for ground structures
        disp_calc_path_f = capture_final_path / f'DISP_{disp_suffix}'
        wav_calc_path_f = capture_final_path / f'WAV_{disp_suffix}'
        vasp_wav_inputs = create_carrier_capture_wav_vasp_inputs(
            disp_calc_path_f,
            charge=q_final,
            potcar=potcar,
            kpt=kpt,
            incar_settings=incar_settings
        )
        vasp_wav_inputs.write_input(wav_calc_path_f, make_dir_if_not_present=True)
        
    return None


def create_carrier_capture_wswq_vasp_inputs(
    wav_dir: PathLike,
    charge: int = 0,
    potcar: Potcar | None = None,
    incar_settings: dict = {}
) -> VaspInput:
    """
    Read in VASP inputs from carrier capture WAVECAR calculations and
    create VASP inputs for WSWQ calculations.

    Args
    ---------
        wav_dir (PathLike):
            Path to wavefunction calculation directory.
        charge (int):
            Integer charge state for the defect. Defaults to 0 (neutral).
        potcar (Potcar):
            `pymatgen` Potcar object. Defaults to None (same as
            wavefunction calc).
        incar_settings (dict):
            Dictionary of INCAR settings to to use for the single-energy
            displacement calculations. ALGO, LWSWQ, NELECT, and NUPDOWN
            are set within the function and everything else is taken
            from the previous wavefunction calc. Defaults to no
            additional settings.

    Returns
    ---------
        pymatgen.io.vasp.inputs.VaspInput set for electron-phonon matrix
        post-processing calcs.
    """
    # input files from displacement structure
    wav_inps = VaspInput.from_directory(wav_dir)
    poscar_wav = wav_inps.poscar

    # update POTCAR if specified
    potcar = wav_inps.potcar if potcar == None else potcar

    # get electron count from POSCAR/POTCAR
    nelect = 0
    for i, ele in enumerate(poscar_wav.site_symbols):
        for j in potcar:
            if ele == j.element:
                nelect += int(j.nelectrons)*poscar_wav.natoms[i]
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
    wswq_inps = VaspInput(
        poscar=poscar_wav,
        potcar=potcar,
        kpoints=wav_inps.kpoints,
        incar=incar
    )
    
    return wswq_inps


def setup_carrier_capture_wswq(
    defect_name: str,
    q_initial: int,
    q_final: int,
    ref_wavecar_path: PathLike,
    cc_path: Path = Path.cwd(),
    displacements: ArrayLike = np.array([
        -0.2, -0.1, 0., 0.1, 0.2
    ]),
    potcar: Potcar | None = None,
    incar_settings: dict = {}
) -> None:
    """
    Setup carrier capture WSWQ calculations from WAVECAR calculations.
    
    Args
    ---------
        defect_name (str):
            Name of defect in defect_site# format (e.g., Va_N1).
        q_initial (int):
            Initial charge state of defect.
        q_final (int):
            Final charge state of defect.
        ref_wavecar_path (PathLike):
            Path to reference WAVECAR file.
        cc_path (Path):
            Path to carrier_capture subdirectory. Defaults to Path.cwd().
        displacements (ArrayLike):
            Array of fractional displacements between the initial and
            final structures. 0 corresponds to the equilibrium structure
            of the final state and 1 corresponds to the equilibrium
            structure of the initial state. Defaults to -0.2 to 0.2 in
            0.1 increments (around final state equilibrium).
        potcar (Potcar):
            `pymatgen` Potcar object. Defaults to None (Potcar made from
            standard PBEv54 potentials).
        kpt (Kpoints):
            `pymatgen` Kpoints object. Defaults to None (Gamma-point
            only).
        incar_settings (dict):
            Dictionary of INCAR settings to to use for the single-energy
            displacement calculations. NELECT and NUPDOWN are set within
            the function and everything else is taken from the previous
            displacement calc. Defaults to no additional settings.

    Returns
    ---------
        Nothing.
    """
    capture_calc_path = cc_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    charge_diff = q_final - q_initial
    capture_initial_path = capture_calc_path / 'i_q'
    capture_final_path = capture_calc_path / f'f_q{charge_diff:+}'

    for i in range(displacements.shape[0]):
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'

        # setup WSWQ calcs for excited structures
        wav_calc_path_i = capture_initial_path / f'WAV_{disp_suffix}'
        wswq_calc_path_i =capture_initial_path / f'WSWQ_{disp_suffix}'
        vasp_wswq_inputs = create_carrier_capture_wswq_vasp_inputs(
            wav_calc_path_i,
            charge=q_initial,
            potcar=potcar,
            incar_settings=incar_settings
        )
        vasp_wswq_inputs.write_input(
            wswq_calc_path_i,
            make_dir_if_not_present=True,
            files_to_transfer={
                'WAVECAR.qqq': wav_calc_path_i / 'WAVECAR'
            }
        )
        shutil.copyfile(ref_wavecar_path, wswq_calc_path_i / 'WAVECAR')

        # setup WSWQ calcs for ground structures
        wav_calc_path_f = capture_final_path / f'WAV_{disp_suffix}'
        wswq_calc_path_f = capture_final_path / f'WSWQ_{disp_suffix}'
        vasp_wswq_inputs = create_carrier_capture_wswq_vasp_inputs(
            wav_calc_path_f,
            charge=q_final,
            potcar=potcar,
            incar_settings=incar_settings
        )
        vasp_wswq_inputs.write_input(
            wswq_calc_path_f,
            make_dir_if_not_present=True,
            files_to_transfer={
                'WAVECAR.qqq': wav_calc_path_f / 'WAVECAR'
            }
        )
        shutil.copyfile(ref_wavecar_path, wswq_calc_path_f / 'WAVECAR')
        
    return None


def setup_carrier_capture_perfect_ref(
    base_path: Path = Path.cwd(),
    wswq_mid_path: Path = Path('WSWQ_000'),
    incar_settings: dict = {}
) -> None:
    """
    Setup carrier capture perfect eigenvalue reference calculation.
    
    Args
    ---------
        base_path (Path):
            Base path to be used for setting up the carrier capture
            calculations. Should be the base directory for `pydefect`
            subdirectories and contain the defect and carrier_capture
            subdirectories. Defaults to Path.cwd().
        wswq_mid_path (Path):
            Relative path to subdirectory for the middle point of
            electron-phonon coupling matrix calculations corresponding
            to the reference configuration. Defaults to Path('WSWQ_000').
        incar_settings (dict):
            Dictionary of INCAR settings to to use for the single-energy
            displacement calculations. IBRION and NSW are set within the
            function and everything else is taken from the previous
            displacement calc. Defaults to no additional settings.

    Returns
    ---------
        Nothing.
    """
    perfect_path = base_path / 'defect' / 'perfect'
    eig_ref_path = perfect_path / 'WAV_ref'

    perfect_poscar = Poscar.from_file(perfect_path / 'CONTCAR')
    wswq_potcar = Potcar.from_file(wswq_mid_path / 'POTCAR')

    # match POTCAR from defect calculation, ensuring only elements
    # present in the perfect POSCAR are included
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
    
    eig_ref_inputs = VaspInput(
        poscar=perfect_poscar,
        potcar=eig_ref_potcar,
        kpoints=wswq_kpoints,
        incar=eig_ref_incar
    )
    eig_ref_inputs.write_input(eig_ref_path, make_dir_if_not_present=True)
        
    return None


def gather_qe_data(
    excited_poscar_path: PathLike,
    ground_poscar_path: PathLike,
    disp_path: PathLike,
    displacements: ArrayLike = np.array([
        -1.0, -0.6, -0.4, -0.2, -0.1, 0., 0.1, 0.2, 0.4, 0.6, 1.0
    ])
) -> pd.DataFrame:
    """
    Gather Q vs. E data for configuration coordinate diagram generation
    assuming 1D potential energy surfaces.

    Args
    ---------
        excited_poscar_path (PathLike):
            Path to displacement calculation directory.
        ground_poscar_path (PathLike):
            Path to displacement calculation directory.
        disp_path (PathLike):
            Path to displacement calculation directory.
        displacements (ArrayLike):
            Array of fractional displacements between the initial and
            final structures. 0 corresponds to the equilibrium structure
            of the final state and 1 corresponds to the equilibrium
            structure of the initial state. Defaults to -1.0 to 1.0 in
            varying increments (finer near equilibrium).

    Returns
    ---------
        DataFrame of Q vs. E data.
    """
    Q, E = [], []    
    for i in range(displacements.shape[0]):
        # assume displacements are organized as DISP_XXX
        # (e.g., DISP_001 or DISP_-01)
        disp_suffix = f'{str(displacements[i]).replace('.', ''):0>3}'
        
        # get excited & ground state structures
        excited_poscar = Poscar.from_file(excited_poscar_path)
        ground_poscar = Poscar.from_file(ground_poscar_path)
        
        # get intermediate structures and energy files
        inter_poscar = Poscar.from_file(disp_path / f'DISP_{disp_suffix}' / 'CONTCAR')
        oszi = Oszicar(disp_path / f'DISP_{disp_suffix}' / 'OSZICAR')
        
        Q.append(
            get_Q_from_struct(
                ground_poscar.structure,
                excited_poscar.structure,
                inter_poscar.structure
            )
        )
        E.append(oszi.final_energy)

    # create dataframe for QE data
    qe_df = pd.DataFrame({'Q': Q, 'E': E})

    return qe_df


def gather_qe_carrier_capture(
    defect_name: str,
    q_initial: int,
    q_final: int,
    cc_path: Path = Path.cwd(),
    displacements: ArrayLike = np.array([
        -1.0, -0.6, -0.4, -0.2, -0.1, 0., 0.1, 0.2, 0.4, 0.6, 1.0
    ]),
    qe_filename: PathLike = 'potential.csv'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gather Q vs. E data for carrier capture CCD.
    
    Args
    ---------
        defect_name (str):
            Name of defect in defect_site# format (e.g., Va_N1).
        q_initial (int):
            Initial charge state of defect.
        q_final (int):
            Final charge state of defect.
        cc_path (Path):
            Path to carrier_capture subdirectory. Defaults to Path.cwd().
        displacements (ArrayLike):
            Array of fractional displacements between the initial and
            final structures. 0 corresponds to the equilibrium structure
            of the final state and 1 corresponds to the equilibrium
            structure of the initial state. Defaults to -1.0 to 1.0 in
            varying increments (finer near equilibrium).
        qe_filename (PathLike):
            Filename for CSV files to save Q vs. E data as relative to
            initial and final state carrier capture directories.

    Returns
    ---------
        Tuple of DataFrames of Q vs. E data for initial and final states.
    """
    capture_calc_path = cc_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    charge_diff = q_final - q_initial
    capture_initial_path = capture_calc_path / 'i_q'
    capture_final_path = capture_calc_path / f'f_q{charge_diff:+}'

    # gather Q vs. E data for initial and final states for carrier capture
    excited_poscar_path = capture_initial_path / 'DISP_000' / 'CONTCAR'
    ground_poscar_path = capture_final_path / 'DISP_000' / 'CONTCAR'
    qe_i_df = gather_qe_data(
        excited_poscar_path,
        ground_poscar_path,
        capture_initial_path,
        displacements=displacements
    )
    qe_f_df = gather_qe_data(
        excited_poscar_path,
        ground_poscar_path,
        capture_final_path,
        displacements=displacements
    )
    
    qe_i_df.to_csv(capture_initial_path / qe_filename, index=False)
    qe_f_df.to_csv(capture_final_path / qe_filename, index=False)

    return qe_i_df, qe_f_df


def gather_qe_migration(
    defect_name: str,
    q_ext: int,
    q_initial: int,
    q_final: int,
    mig_path: Path = Path.cwd(),
    displacements: ArrayLike = np.array([
        -1.0, -0.6, -0.4, -0.2, -0.1, 0., 0.1, 0.2, 0.4, 0.6, 1.0
    ]),
    qe_filename: PathLike = 'potential.csv'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gather Q vs. E data for carrier-capture-enhanced athermal migration.
    
    Args
    ---------
        defect_name (str):
            Name of defect in defect_site# format (e.g., Va_N1).
        q_ext (int):
            Extra charge state of defect considered for migration
            pathway.
        q_initial (int):
            Initial charge state of defect.
        q_final (int):
            Final charge state of defect.
        mig_path (Path):
            Path to carrier_capture subdirectory. Defaults to Path.cwd().
        displacements (ArrayLike):
            Array of fractional displacements between the initial and
            final structures. 0 corresponds to the equilibrium structure
            of the final state and 1 corresponds to the equilibrium
            structure of the initial state. Defaults to -1.0 to 1.0 in
            varying increments (finer near equilibrium).
        qe_filename (PathLike):
            Filename for CSV files to save Q vs. E data as relative to
            initial and final state carrier capture directories.

    Returns
    ---------
        Tuple of DataFrames of Q vs. E data for initial and final states.
    """
    mig_calc_path = mig_path / '_'.join([defect_name, str(q_ext), f'{q_initial}to{q_final}'])
    capture_initial_path = mig_path / '_'.join([defect_name, str(q_initial), f'{q_initial}to{q_final}'])
    capture_final_path = mig_path / '_'.join([defect_name, str(q_final), f'{q_initial}to{q_final}'])

    # gather Q vs. E data for migration charge state
    excited_poscar_path = capture_initial_path / 'DISP_000' / 'CONTCAR'
    ground_poscar_path = capture_final_path / 'DISP_000' / 'CONTCAR'
    qe_mig_df = gather_qe_data(
        excited_poscar_path,
        ground_poscar_path,
        mig_calc_path,
        displacements=displacements
    )
    
    qe_mig_df.to_csv(mig_calc_path / qe_filename, index=False)

    return qe_mig_df


def calculate_g_capture(
    degen_initial: dict,
    degen_final: dict
) -> float:
    """
    Calculate capture pathway degeneracy from initial and final
    degeneracy dictionaries. Dictionaries include orientational and
    spin degeneracies as well as multiplicity of the defect site in the
    bulk cell.
    
    Args
    ---------
        degen_initial (dict):
            Dictionary of initial state degeneracy values.
        degen_final (dict):
            Dictionary of final state degeneracy values.

    Returns
    ---------
        Float of capture pathway degeneracy.
    """
    g_initial = float(degen_initial['g_Orient'])
    g_final = float(degen_final['g_Orient'])
    g_cap = round(max((g_final/g_initial), 1.), 0)
    g_cap *= float(degen_final['g_Spin'])
    return g_cap


def calculate_g_capture_e_and_h(
    q_1: int,
    q_2: int,
    degen_1: dict,
    degen_2: dict
) -> dict:
    """
    Calculate capture pathway degeneracies for hole and electron
    capture. Determines which state corresponds to hole vs. electron
    capture based on charge states and assigns capture degeneracies
    accordingly.
    
    Args
    ---------
        q_1 (int):
            Charge state of state 1.
        q_2 (int):
            Charge state of state 2.
        degen_1 (dict):
            Dictionary of state 1 degeneracy values.
        degen_2 (dict):
            Dictionary of state 2 degeneracy values.

    Returns
    ---------
        Dictionary of capture pathway degeneracies for hole and electron
        capture.
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


def parse_carrier_capture_info(
    defect_name: str,
    q_initial: int,
    q_final: int,
    coupling_state: str = 'final',
    g_e: float = 1,
    g_h: float = 1,
    m_e: float = 0.2,
    m_h: float = 0.8,
    dielectric_const: float = 10.,
    kpt_idx: int = 0,
    spin: int = 0,
    base_path: Path = Path.cwd(),
    displacements: ArrayLike = np.array([
        -0.2, -0.1, 0., 0.1, 0.2
    ]),
    cc_yaml_filename: PathLike = 'carrier_capture_params.yaml',
    ctl_jsonpath: PathLike = 'transition_levels.json',
    degen_csvpath: PathLike = 'degeneracies.csv',
    savefig_wif: PathLike | None = None
) -> dict:
    """
    Analyze defect calculations to gather parameters for carrier capture
    calculations.
    
    Args
    ---------
        defect_name (str):
            Name of defect in defect_site# format (e.g., Va_N1).
        q_initial (int):
            Initial charge state of defect.
        q_final (int):
            Final charge state of defect.
        coupling_state (str):
            Choose the reference coupling state. Implemented options are
            'final' and 'initial'. Defaults to 'final'.
        g_e (float):
            Electron capture pathway degeneracy. Defaults to 1.
        g_h (float):
            Hole capture pathway degeneracy. Defaults to 1.
        m_e (float):
            Electron effective mass in units of free electron mass.
            Defaults to 0.2.
        m_h (float):
            Hole effective mass in units of free electron mass.
            Defaults to 0.8.
        dielectric_const (float):
            Static dielectric constant of the material. Defaults to 10.
        kpt_idx (int):
            k-point index used for electron-phonon coupling matrix
            element calculation. Defaults to 0 (first k-point, should be
            the Gamma-point for Gamma-centered k-point meshes).
        spin (int):
            Spin channel used for electron-phonon coupling matrix
            element calculation. Defaults to 0 (spin-up).
        base_path (Path):
            Base path to be used for setting up the carrier capture
            calculations. Should be the base directory for `pydefect`
            subdirectories and contain the defect and carrier_capture
            subdirectories. Defaults to Path.cwd().
        displacements (ArrayLike):
            Array of fractional displacements between the initial and
            final structures. 0 corresponds to the equilibrium structure
            of the final state and 1 corresponds to the equilibrium
            structure of the initial state. Defaults to -0.2 to 0.2 in
            0.1 increments (around final state equilibrium).
        cc_yaml_filename (PathLike):
            Path to write YAML file with all carrier capture calculation
            info to. Will be relative to carrier_capture directory.
            Defaults to 'carrier_capture_params.yaml'.
        ctl_jsonpath (PathLike):
            Path to JSON file with charge transition levels, relative to
            defect directory path. Should follow the format from
            `pydefect`. Defaults to 'transition_levels.json'.
        degen_csvpath (PathLike):
            Path to CSV file with defect degeneracies, relative to
            defect directory path. Should follow the format from
            `doped`. Defaults to 'degeneracies.csv'.
        savefig_wif (PathLike or None):
            Relative path to save figure of electron-phonon coupling
            matrix element fitting. Defaults to None (no figure saved).

    Returns
    ---------
        Dictionary of capture pathway degeneracies for hole and electron
        capture.
    """
    # initialize yaml with given information
    cc_dict = {
        'defect': str(defect_name),
        'q_initial': int(q_initial),
        'q_final': int(q_final),
        'coupling state': str(coupling_state),
        'kpoint': int(kpt_idx + 1),
        'spin': int(spin)
    }

    # defect & carrier capture calculation paths
    defect_path = base_path / 'defect'
    capture_path = base_path / 'carrier_capture'
    defect_initial_path = defect_path / '_'.join([defect_name, str(q_initial)])
    defect_final_path = defect_path / '_'.join([defect_name, str(q_final)])
    capture_calc_path = capture_path / '_'.join([defect_name, str(q_initial), str(q_final)])
    
    charge_diff = q_final - q_initial
    capture_initial_path = capture_calc_path / 'i_q'
    capture_final_path = capture_calc_path / f'f_q{charge_diff:+}'

    # add charge transition level between q_i/q_f &
    # VBM/CBM if transition_levels.json exists
    try:
        transition_levels = loadfn(defect_path / ctl_jsonpath)
        for tl in transition_levels.transition_levels:
            tl.fermi_levels.sort()
            for name, charge, energy, fermi in zip_longest(
                [tl.name], tl.charges, tl.energies, tl.fermi_levels, fillvalue=tl.name
            ):
                if name in defect_name and q_initial in charge and q_final in charge:
                    cc_dict.update({'formation energy': float(energy), 'ctl': float(fermi)})
        cc_dict.update({'vbm': float(0.), 'cbm': float(transition_levels.cbm)})
    except:
        cc_dict.update({'formation energy': None, 'ctl': None, 'vbm': None, 'cbm': None})
        print(f'No {ctl_jsonpath} file found in defect directory.')

    # get VBM/CBM from EIGENVAL if not determined from CTL file
    # get band indices for degenerate valence band states
    perfect_eigenvals = Eigenval(
        defect_path / 'perfect' / 'WAV_ref' / 'EIGENVAL',
        separate_spins=True
    )
    # band properties tuple is (band gap, cbm, vbm, is_band_gap_direct),
    # each tuples of 2, with index 0 = spin-up & index 1 = spin-down
    band_props = perfect_eigenvals.eigenvalue_band_properties
    if cc_dict['vbm'] is None or cc_dict['cbm'] is None:
        cc_dict.update({
            'vbm': float(band_props[2][0]),
            'cbm': float(band_props[1][0]),
            'Eg': float(band_props[0][0])
        })
    else:
        cc_dict.update({'Eg': float(band_props[0][0])})
    valence_indices, conduction_indices, defect_indices = [], [], []
    # eigenvalues are dict of {(spin): NDArray(shape=(nkpt, nbands, 2))},
    # kpoint index is 0-based
    eig_occ_up = perfect_eigenvals.eigenvalues[Spin.up][kpt_idx]
    eig_occ_down = perfect_eigenvals.eigenvalues[Spin.down][kpt_idx]
    vbm_bandidx_up = np.where(np.isclose(eig_occ_up[:, 0], band_props[2][0], atol=0.0001))
    vbm_bandidx_down = np.where(np.isclose(eig_occ_down[:, 0], band_props[2][1], atol=0.0001))
    if isclose(vbm_bandidx_up[0][0], vbm_bandidx_down[0][0], abs_tol=0.1):
        valence_indices.append(vbm_bandidx_up[0][0])
        for band_idx in range(vbm_bandidx_up[0][0]):
            if isclose(
                eig_occ_up[:, 0][vbm_bandidx_up[0][0]],
                eig_occ_up[:, 0][band_idx],
                abs_tol=0.2
            ):
                valence_indices.append(band_idx)
            elif isclose(
                eig_occ_down[:, 0][vbm_bandidx_down[0][0]],
                eig_occ_down[:, 0][band_idx],
                abs_tol=0.2
            ):
                valence_indices.append(band_idx)
    valence_indices.sort()

    cbm_bandidx_up = np.where(np.isclose(eig_occ_up[:, 0], band_props[1][0], atol=0.0001))
    cbm_bandidx_down = np.where(np.isclose(eig_occ_down[:, 0], band_props[1][1], atol=0.0001))
    if isclose(cbm_bandidx_up[0][0], cbm_bandidx_down[0][0], abs_tol=0.1):
        conduction_indices.append(cbm_bandidx_up[0][0])
        for band_idx in range(cbm_bandidx_up[0][0]):
            if isclose(
                eig_occ_up[:, 0][cbm_bandidx_up[0][0]],
                eig_occ_up[:, 0][band_idx],
                abs_tol=0.2
            ):
                conduction_indices.append(band_idx)
            elif isclose(
                eig_occ_down[:, 0][cbm_bandidx_down[0][0]],
                eig_occ_down[:, 0][band_idx],
                abs_tol=0.2
            ):
                conduction_indices.append(band_idx)
    conduction_indices.sort()
    
    # determine dQ from relaxed defect calculation structures
    poscar_initial = Poscar.from_file(defect_initial_path / 'CONTCAR')
    poscar_final = Poscar.from_file(defect_final_path / 'CONTCAR')
    dQ = get_dQ(poscar_initial.structure, poscar_final.structure)
    
    Qmin, Qmax = 0., 0.
    for i in capture_calc_path.glob('*_q*/'):
        for j in i.glob('DISP_*'):
            Qmin = min(dQ*float(j.name.split('_')[-1])/10., Qmin)
            Qmax = max(dQ*float(j.name.split('_')[-1])/10., Qmax)

    # set default for displacements array corresponding to WSWQ calcs
    Q0 = displacements[displacements.shape[0]//2]
    
    # calculate electron-phonon coupling matrix elements from WSWQ calcs
    i_struc = Structure.from_file(defect_initial_path / 'CONTCAR')
    f_struc = Structure.from_file(defect_final_path / 'CONTCAR')
    i_WSWQs, f_WSWQs = [], []

    # create combined Wif figure
    wif_fig = plt.figure(figsize=(12, 10), layout='constrained')
    wif_subfigs = wif_fig.subfigures(2, 1)
    
    coupling_state_keystr = coupling_state.lower()[0]
    # check if coupling state is final or ground
    if coupling_state_keystr == 'f' or coupling_state_keystr == 'g':
        # path to initial vasprun
        ground_vr_path = capture_final_path / f'WAV_{str(Q0).replace('.', ''):0>3}' / 'vasprun.xml'
        # adjust valence/conduction band indices by electron difference
        # from perfect reference to coupling reference
        defect_eigenvals = Eigenval(
            capture_final_path / f'WAV_{str(Q0).replace('.', ''):0>3}' / 'EIGENVAL',
            separate_spins=True
        )
        nelect_diff = perfect_eigenvals.nelect - defect_eigenvals.nelect
        valence_indices = [i - floor(nelect_diff/2) for i in valence_indices]
        conduction_indices = [i - floor(nelect_diff/2) + 1 for i in conduction_indices]
        for d in capture_final_path.glob('WSWQ_*'):
            if str(d) in [f'{str(capture_final_path)}/WSWQ_{str(i).replace('.', ''):0>3}' \
                          for i in displacements]:
                Q_struc = Structure.from_file(d / 'CONTCAR')
                Q = get_Q_from_struct(f_struc, i_struc, Q_struc)
                f_WSWQs.append((Q, d / 'WSWQ'))
        # wavefunction indexing for get_Wif_from_WSWQ is 1-based
        # indexing, spin (0 - up, 1 - down), & kpoint defaults to first
        f_Wifs_vbm = get_Wif_from_WSWQ(
            f_WSWQs,
            str(ground_vr_path),
            int(max(valence_indices)+1),
            valence_indices,
            spin=spin,
            kpoint=kpt_idx+1,
            fig=wif_subfigs[0]
        )
        f_Wifs_cbm = get_Wif_from_WSWQ(
            f_WSWQs,
            str(ground_vr_path),
            int(min(conduction_indices)-1),
            conduction_indices,
            spin=spin,
            kpoint=kpt_idx+1,
            fig=wif_subfigs[1]
        )
        Wif_vbm = np.sqrt(np.mean([x[1]**2 for x in f_Wifs_vbm]))
        Wif_cbm = np.sqrt(np.mean([x[1]**2 for x in f_Wifs_cbm]))
    
    # check if coupling state is initial or excited
    elif coupling_state_keystr == 'i' or coupling_state_keystr == 'e':
        # path to initial vasprun
        wav_Q0 = f'WAV_{str(Q0).replace('.', ''):0>3}'
        ground_vr_path = capture_initial_path / wav_Q0 / 'vasprun.xml'
        # adjust valence/conduction band indices by electron difference
        # from perfect reference to coupling reference
        defect_eigenvals = Eigenval(
            capture_final_path / wav_Q0 / 'EIGENVAL',
            separate_spins=True
        )
        nelect_diff = perfect_eigenvals.nelect - defect_eigenvals.nelect
        valence_indices = [i - floor(nelect_diff/2) for i in valence_indices]
        conduction_indices = [i - floor(nelect_diff/2) + 1 for i in conduction_indices]
        for d in capture_initial_path.glob('WSWQ_*'):
            if str(d) in [f'{str(capture_initial_path)}/WSWQ_{str(i).replace('.', ''):0>3}' \
                          for i in displacements]:
                Q_struc = Structure.from_file(d / 'CONTCAR')
                Q = get_Q_from_struct(f_struc, i_struc, Q_struc)
                i_WSWQs.append((Q, d / 'WSWQ'))
        # wavefunction indexing for get_Wif_from_WSWQ is 1-based
        # indexing, spin (0 - up, 1 - down), & kpoint defaults to first
        i_Wifs_vbm = get_Wif_from_WSWQ(
            i_WSWQs,
            str(ground_vr_path),
            int(max(valence_indices)+1),
            valence_indices,
            spin=spin,
            kpoint=kpt_idx+1,
            fig=wif_subfigs[0]
        )
        i_Wifs_cbm = get_Wif_from_WSWQ(
            i_WSWQs,
            str(ground_vr_path),
            int(min(conduction_indices)-1),
            conduction_indices,
            spin=spin,
            kpoint=kpt_idx+1,
            fig=wif_subfigs[1]
        )
        Wif_vbm = np.sqrt(np.mean([x[1]**2 for x in i_Wifs_vbm]))
        Wif_cbm = np.sqrt(np.mean([x[1]**2 for x in i_Wifs_cbm]))
    else:
        raise ValueError(
            'Please choose either the final/initial state for use in ' +
            'calculating the electron-phonon coupling matrix element.'
        )

    # add band indices for defects considered in electron-phonon
    # coupling matrix element calculations
    defect_indices.append(int(max(valence_indices)+1))
    if int(min(conduction_indices)-1) not in defect_indices:
        defect_indices.append(int(min(conduction_indices)-1))
    defect_indices.sort()
    
    Wif_vbm_dec, Wif_vbm_exp = f'{Wif_vbm:.2E}'.split('E')
    Wif_cbm_dec, Wif_cbm_exp = f'{Wif_cbm:.2E}'.split('E')
    wif_subfigs[0].suptitle(
        (rf'$\tilde{{W}}_{{if}} \; \text{{(VBM):}} \; {float(Wif_vbm_dec)} \times'
         rf'10^{{{int(Wif_vbm_exp):d}}} \; \text{{eV}}/\text{{amu}}^{{1/2}} \text{{Å}}$'),
        fontsize=20,
        math_fontfamily='cm'
    )
    wif_subfigs[1].suptitle(
        (rf'$\tilde{{W}}_{{if}} \; \text{{(CBM):}} \; {float(Wif_cbm_dec)} \times'
         rf'10^{{{int(Wif_cbm_exp):d}}} \; \text{{eV}}/\text{{amu}}^{{1/2}} \text{{Å}}$'),
        fontsize=20,
        math_fontfamily='cm'
    )

    for sub in wif_subfigs:
        sub.supylabel(
            (r'$\left\langle \tilde{\psi}_{i} \left( 0 \right) \left| \hat{\tilde{S}}'
             r'\left( 0 \right) \right| \tilde{\psi}_{f} \left( Q \right) \right\rangle$'),
            fontsize=18,
            math_fontfamily='cm'
        )
    
    for ax in wif_fig.axes:
        ax.set_title(
            ax.get_title(),
            fontsize=18
        )
        ax.set_xlabel(
            r'$Q \; \text{(amu}^{1/2} \text{Å)}$',
            fontsize=18,
            math_fontfamily='cm'
        )
        ax.minorticks_on()
        ax.tick_params(
            top=True,
            bottom=True,
            right=True,
            left=True,
            which='both',
            direction='in',
            labelsize=16
        )
    
    if savefig_wif == None:
        plt.show()
    elif type(savefig_wif) == str:
        plt.savefig(capture_calc_path / savefig_wif, dpi=300)
    else:
        raise ValueError('Please choose a valid image name.')

    # calculate effective mass of the charge carrier
    m_h, m_e = m_h, m_e

    # calculate capture degeneracies of the defect
    if g_e is None or g_h is None:
        try:
            degen_df = pd.read_csv(defect_path / degen_csvpath)
            degen_initial = degen_df.loc[(degen_df['Defect'] == defect_name) & \
                                         (degen_df['q'] == q_initial)]
            degen_final = degen_df.loc[(degen_df['Defect'] == defect_name) & \
                                       (degen_df['q'] == q_final)]
        
            try:
                degen_initial_dict = degen_initial.to_dict(orient='records')[0]
                degen_final_dict = degen_final.to_dict(orient='records')[0]
                g_cap_dict = calculate_g_capture_e_and_h(
                    q_initial,
                    q_final,
                    degen_initial_dict,
                    degen_final_dict
                )
                g_e, g_h = g_cap_dict['g_e'], g_cap_dict['g_h']
            except IndexError:
                cc_dict.update({'g_e': None, 'g_h': None})
                print(f'Defect transition not present in {degen_csvpath}')
            
            if g_e.is_integer() == False:
                print('Electron capture degeneracy not an integer.')
            if g_h.is_integer() == False:
                print('Hole capture degeneracy not an integer.')
                
            cc_dict.update({'g_e': int(g_e), 'g_h': int(g_h)})
        except FileNotFoundError:
            cc_dict.update({'g_e': None, 'g_h': None})
            print(f'No {degen_csvpath} file found in defect directory.')
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
    Calculates the nonradiative capture rate for a given temperature
    from the nonradiative capture coefficient, defect concentration,
    and carrier concentration.
    
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
    Calculates the nonradiative capture rate for a given temperature
    from the nonradiative capture coefficient and carrier concentration.
    
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
    Calculates the temperature-dependent effective density of states for
    the valence/conduction band.
    
    Args
    ---------
        T (float):
            Temperature in units of (K).
        eff_mass (float):
            Effective mass of the carrier in the associated band in
            units of rest electron mass. Defaults to 1 (equal to
            electron rest mass).
        mc (int):
            Number of equivalent energy minima for the conduction band.
            Defaults to 1, should only be specified for conduction band
            effective DOS.

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
    Calculates the carrier thermal emission coefficient as a function of
    temperature using the capture coefficient and effective density of
    states for the valence/conduction band.
    
    Args
    ---------
        T (float):
            Temperature in units of (K).
        cap_coeff (float):
            Capture coefficient in units of (cm^3 s^-1).
        dE (float):
            Thermodynamic transition level in units of (eV).
        eff_mass (float):
            Effective mass of the carrier in the associated band in
            units of rest electron mass. Defaults to 1 (equal to
            electron rest mass).
        mc (int):
            Number of equivalent energy minima for the conduction band.
            Defaults to 1, should only be specified for conduction band
            effective DOS.

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
    Calculates the emission rate for a given temperature from the
    emission coefficient and defect concentration.
    
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
    Calculates the carrier thermal emission exponential factor times the
    defect concentration as a function of temperature using the
    effective density of states for the valence/conduction band.
    
    Args
    ---------
        T (float):
            Temperature in units of (K).
        defect_conc (float):
            Defect concentration in units of (cm^-3).
        dE (float):
            Thermodynamic transition level in units of (eV).
        eff_mass (float):
            Effective mass of the carrier in the associated band in
            units of rest electron mass. Defaults to 1 (equal to
            electron rest mass).
        mc (int):
            Number of equivalent energy minima for the conduction band.
            Defaults to 1, should only be specified for conduction band
            effective DOS.

    Returns
    ---------
        Emission exponential factor times the defect concentration with
        units of (cm^-6).
    """
    BOLTZMANN_EV = 8.617333262e-5  # eV / K

    eff_dos = effective_band_dos(T, eff_mass=eff_mass, mc=mc)
    emission_exp_factor = eff_dos*np.exp(-dE/(BOLTZMANN_EV*T))
    emission_rate_sans_coeff = emission_exp_factor*defect_conc
    
    return emission_rate_sans_coeff


def calc_all_cap_rates(
    cap_coeff_csv: PathLike,
    conc_csv: PathLike,
    temp: float = 300,
    def_conc_colname: str = 'Concentration (cm^-3)',
    p_conc_colname: str = 'Holes (cm^-3)',
    n_conc_colname: str = 'Electrons (cm^-3)'
) -> pd.DataFrame:
    """
    Calculates all nonradiative capture rates for a given temperature
    using DataFrames for nonradiative capture coefficients and defect/
    carrier concentrations. DataFrames are made from the paths to the
    CSV files for each type of data. Capture coefficients are made from
    `radDefects` and defect/carrier concentrations are calculated and
    formatted using `doped`.
    
    Args
    ---------
        cap_coeff_csv (PathLike):
            Path to CSV file containing nonradiative carrier capture
            coefficients.
        conc_csv (PathLike):
            Path to CSV file containing defect and carrier
            concentrations.
        temp (float):
            Temperature in units of (K).
        def_conc_colname (str):
            Column name for defect concentration in the concentration
            CSV file. Defaults to 'Concentration (cm^-3)'.
        p_conc_colname (str):
            Column name for hole concentration in the concentration CSV
            file. Defaults to 'Holes (cm^-3)'.
        n_conc_colname (str):
            Column name for electron concentrations in the concentration
            CSV file. Defaults to 'Electrons (cm^-3)'.

    Returns
    ---------
        DataFrame containing carrier capture rates.
    """
    cap_coeff_df = pd.read_csv(cap_coeff_csv, index_col=0)
    conc_df = pd.read_csv(conc_csv, index_col=0)
    
    cap_rate_df = cap_coeff_df.copy(deep=True)
    cap_rate_df.rename(columns={'C_p': 'R_p', 'C_n': 'R_n'}, inplace=True)
    
    # select concentrations for specified temperature
    conc_temp_df = conc_df.query(f'`Temperature (K)` == {temp}')

    # remove defects that don't have both capture coefficient values and
    # concentration values
    doped_defects, coeff_defects = conc_temp_df.index, cap_coeff_df.index
    simplified_defect_names = list(map(lambda x: re.sub(r'\d+', '', x), doped_defects))
    coeff_doped_cross_dict = {}
    
    for i, defect in enumerate(simplified_defect_names):
        if defect in coeff_defects:
            coeff_doped_cross_dict.update({defect: doped_defects[i]})
        else:
            # need to change if defect matches more than one alt_defect
            # (i.e., two site types for one defect)
            for j, alt_defect in enumerate(coeff_defects):
                if defect in alt_defect:
                    coeff_doped_cross_dict.update({alt_defect: doped_defects[i]})

    cap_rate_df = cap_rate_df.loc[coeff_doped_cross_dict.keys()]
    hole_cap_rates = cap_rate_df['R_p'].copy(deep=False)
    elec_cap_rates = cap_rate_df['R_n'].copy(deep=False)

    defect_tracking = []
    for capture in cap_rate_df.iterrows():
        capture_defect, capture_info = capture
        if capture_defect not in defect_tracking:
            if type(cap_rate_df.loc[capture_defect]['R_p']) == pd.Series:
                hole_cap_rates.loc[capture_defect] = cap_rate_df.loc[capture_defect]['R_p'].apply(
                    lambda x: capture_rate(
                        x,
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]][def_conc_colname],
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]][p_conc_colname]
                    )
                )
                elec_cap_rates.loc[capture_defect] = cap_rate_df.loc[capture_defect]['R_n'].apply(
                    lambda x: capture_rate(
                        x,
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]][def_conc_colname],
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]][n_conc_colname]
                    )
                )
            elif type(cap_rate_df.loc[capture_defect]['R_p']) == np.float64:
                hole_cap_rates.loc[capture_defect] = pd.Series(
                    cap_rate_df.loc[capture_defect]['R_p'],
                    index=[capture_defect]
                ).apply(
                    lambda x: capture_rate(
                        x,
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]][def_conc_colname],
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]][p_conc_colname]
                    )
                )
                elec_cap_rates.loc[capture_defect] = pd.Series(
                    cap_rate_df.loc[capture_defect]['R_n'], 
                    index=[capture_defect]
                ).apply(
                    lambda x: capture_rate(
                        x,
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]][def_conc_colname],
                        conc_temp_df.loc[coeff_doped_cross_dict[capture_defect]][n_conc_colname]
                    )
                )
            defect_tracking.append(capture_defect)
    
    return cap_rate_df


def estimate_transition_time(
    cap_coeff_path: PathLike,
    q_i: int,
    q_f: int,
    defect_name: str,
    temp: float = 300,
    p_conc: float = 1e18,
    n_conc: float = 1e18
) -> float:
    """
    Use to estimate time it takes to go from q_i to q_f, assuming single
    charge transitions
    
    Args
    ---------
        cap_coeff_path (PathLike):
            Path to carrier capture calculations directory.
        q_i (int):
            Initial charge state.
        q_f (int):
            Final charge state.
        defect_name (str):
            Defect name in defect_site# format (e.g., Va_N1).
        temp (float):
            Temperature in units of (K).
        p_conc (float):
            Hole concentration in units of (cm^-3).
        n_conc (float):
            Electron concentration in units of (cm^-3).

    Returns
    ---------
        DataFrame containing carrier capture rates.
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
            cap_coeff = cap_coeff_df.loc[defect_name].query(
                f'q_i=={q_all[i]} & q_f=={q_all[i+1]}'
            )[f'C_{cap_type}'].iloc[0]
        except IndexError:
            cap_coeff = cap_coeff_df.loc[defect_name].query(
                f'q_f=={q_all[i]} & q_i=={q_all[i+1]}'
            )[f'C_{cap_type}'].iloc[0]
        print(
            f'Capture coefficient ({cap_type}) for ({q_all[i]}/{q_all[i+1]}): {cap_coeff} cm^3/s'
        )

        if cap_type == 'p':
            t_i = partial_capture_rate(cap_coeff, p_conc)**-1
            print(f'Time for ({q_all[i]}/{q_all[i+1]}) transition @ p={p_conc}: {t_i} s')
        elif cap_type == 'n':
            t_i = partial_capture_rate(cap_coeff, n_conc)**-1
            print(f'Time for ({q_all[i]}/{q_all[i+1]}) transition @ n={n_conc}: {t_i} s')

        time += t_i
    
    return time
