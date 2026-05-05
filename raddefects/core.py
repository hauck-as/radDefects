"""Core functions and classes for analyzing rad-induced defects from VASP."""
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Optional
from numpy.typing import ArrayLike
from pymatgen.util.typing import PathLike

import shutil
import json
import yaml
from monty.serialization import dumpfn, loadfn

import math
import numpy as np
import pandas as pd

from pymatgen.core import Element, Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import Poscar, Potcar
from pymatgen.io.vasp.outputs import Locpot
from pymatgen.analysis.defects.core import DefectComplex, Vacancy
from mp_api.client import MPRester
from vise.util.logger import get_logger
from pydefect.input_maker.defect_set import DefectSet
from pydefect.input_maker.supercell_info import SupercellInfo

from pydefect.analyzer.band_edge_states import BandEdgeOrbitalInfos, PerfectBandEdgeState

logger = get_logger(__name__)


def create_pydefect_file_structure(
    base_path: Path = Path.cwd(),
    add_carrier_capture: bool = False,
    add_rad_defects: bool = False,
    create_dirs: bool = False
) -> dict:
    """
    Creates standard pydefect file structure under given base path.
    
    Args
    ---------
        base_path (Path):
            Base path to be used for setting up the carrier capture
            calculations. Should be the base directory for `pydefect`
            subdirectories and contain the defect and carrier_capture
            subdirectories. Defaults to Path.cwd().
        add_carrier_capture (bool):
            Whether to add carrier capture subdirectories to the file
            structure. Defaults to False.
        add_rad_defects (bool):
            Whether to add radiation defects subdirectories to the file
            structure. Defaults to False.
        create_dirs (bool):
            Whether to create the directories if they do not exist.
            Defaults to False.

    Returns
    ---------
        Dictionary of defect calculations directories.
    """
    cpd_path = base_path / 'cpd'
    defects_path = base_path / 'defect'
    unitcell_path = base_path / 'unitcell'

    band_path = unitcell_path / 'band'
    dielectric_path = unitcell_path / 'dielectric'
    dos_path = unitcell_path / 'dos'
    structure_opt_path = unitcell_path / 'structure_opt'

    pydefect_paths = {
        'cpd': cpd_path,
        'defect': defects_path,
        'unitcell': unitcell_path,
        'band': band_path,
        'dielectric': dielectric_path,
        'dos': dos_path,
        'structure_opt': structure_opt_path
    }

    if add_carrier_capture:
        cc_path = base_path / 'carrier_capture'
        pydefect_paths['carrier_capture'] = cc_path
    
    if add_rad_defects:
        rad_defects_path = base_path / 'rad_poscars'
        pydefect_paths['rad_defects'] = rad_defects_path

    if create_dirs:
        for path in pydefect_paths.values():
            os.makedirs(path, exist_ok=False)

    return pydefect_paths


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


def change_initial_defect_config(defect_path, initial_path):
    """
    Given two paths with directories of defect calculations, use the initial path to replace the
    atomic positions for the defect POSCARS in the defect path.
    """
    # read in lists of defects in both paths
    initial_list, defect_list = [i.name for i in initial_path.glob('*_*_*/')], [i.name for i in defect_path.glob('*_*_*/')]
    
    for i in defect_list:
        if i in initial_list:
            shutil.copy(defect_path / i / 'POSCAR', defect_path / i / 'POSCAR0')
            defect_pos, initial_pos = Poscar.from_file(defect_path / i / 'POSCAR'), Poscar.from_file(initial_path / i / 'CONTCAR')
            defect_pos.structure.sites = initial_pos.structure.sites
            defect_pos.write_file(defect_path / i / 'POSCAR')
        elif i.split('_')[:2] in list(map(lambda x: x.split('_')[:2], initial_list)):
            shutil.copy(defect_path / i / 'POSCAR', defect_path / i / 'POSCAR0')
            # assume neutral charge for initial defect configuration
            j = '_'.join(i.split('_')[:2]+['0'])
            defect_pos, initial_pos = Poscar.from_file(defect_path / i / 'POSCAR'), Poscar.from_file(initial_path / j / 'CONTCAR')
            defect_pos.structure.sites = initial_pos.structure.sites
            defect_pos.write_file(defect_path / i / 'POSCAR')
        else:
            print('No alternate initial configuration found for', i)
    
    return None


def create_vacancy_complex(
    vacs: Sequence[str],
    base_path: PathLike = Path.cwd(),
    vac_comp_idx: int = 1
) -> DefectComplex:
    """
    Generates a defective supercell with a vacancy complex and adds initial charge
    guesses to the rad_defect_in.yaml.
    
    Args
    ---------
        vacs (Sequence[str]):
            List/tuple of strings corresponding to elements to replace
            with vacant sites.
        base_path (Path):
            Base path to be used for setting up the carrier capture
            calculations. Should be the base directory for `pydefect`
            subdirectories and contain the defect subdirectory.
            Assumes a POSCAR file exists in base_path/defect/perfect/.
            Defaults to Path.cwd().
        vac_comp_idx (int):
            Index of the vacancy complex, used to differentiate
            distinct defects of the same type. Defaults to 1.

    Returns
    ---------
        `pymatgen-defect-analysis` DefectComplex object.
    """
    defects_path, rad_defects_path = base_path / 'defect', base_path / 'rad_poscars'
    
    perfect_poscar_path = defects_path / 'perfect' / 'POSCAR'
    perfect_pos = Poscar.from_file(perfect_poscar_path)
    
    # create vacancy complex name
    defect_name = '_'.join(['V'*len(vacs), ''.join(vacs)+str(vac_comp_idx)])
    
    # sum valence of each atom becoming vacancy and create charge list
    tot_valence = sum([Element(i).valence[1] for i in vacs])
    vac_complex_chgs = range(-tot_valence, tot_valence+1, 1)
    
    # try to add name: complex_charges to yaml
    try:
        with open(defects_path / 'rad_defect_in.yaml', 'r') as rdy:
            rad_defect_set = yaml.safe_load(rdy)

            try:
                # add defect if it doesn't appear in yaml
                if defect_name not in rad_defect_set:
                    rad_defect_set.update({defect_name: list(vac_complex_chgs)})
            except TypeError:
                # if yaml file is empty create new dict
                rad_defect_set = {defect_name: list(vac_complex_chgs)}
            
    except FileNotFoundError:
        # create rad_defect_in.yaml if it doesn't exist
        rad_defect_set = {defect_name: list(vac_complex_chgs)}
    
    with open(defects_path / 'rad_defect_in.yaml', 'w') as rdy:
        yaml.dump(rad_defect_set, rdy, default_flow_style=None)
    
    # copy perfect structure to add defect
    defect_struc = perfect_pos.structure.copy()
    
    # create vacant nearest neighbor sites for complex
    vac_site0 = defect_struc.sites[defect_struc.indices_from_symbol(vacs[0])[0]]
    vac_list = [Vacancy(structure=defect_struc, site=vac_site0)]
    vac0_neighbors = defect_struc.get_all_neighbors(r=1.1*len(vacs), sites=[vac_site0])[0]
    
    # find the closest neighbor that has the same site type as the next vacancy
    for v in range(1, len(vacs)):
        vac_site = vac0_neighbors[0]
        for n in range(len(vac0_neighbors)):
            if vacs[v] == vac0_neighbors[n].label:
                if math.dist(vac_site0.coords, vac_site.coords) < math.dist(vac_site0.coords, vac0_neighbors[n].coords):
                    vac_site = vac0_neighbors[n]
        vac_list.append(Vacancy(structure=defect_struc, site=vac_site))
    
    vac_complex = DefectComplex(defects=[vac for vac in vac_list])
    
    vac_complex_pos = Poscar(vac_complex.defect_structure)
    
    try:
        vac_complex_pos.write_file(rad_defects_path / ('POSCAR_' + defect_name))
    except FileExistsError:
        logger.info(f'POSCAR exists, so skipped...')
    
    return vac_complex


def create_vac_sub_complex(
    vac_site: str,
    sub_site: str,
    sub_type: str,
    base_path: PathLike = Path.cwd(),
    comp_idx: int = 1
) -> DefectComplex:
    """
    Generates a defective supercell with a vacancy-substitutional
    complex and adds initial charge guesses to rad_defect_in.yaml.
    
    Args
    ---------
        vac_site (str):
            String corresponding to element to replace with vacancy.
        sub_site (str):
            String corresponding to element at site to replace with
            the substitutional defect.
        sub_type (str):
            String corresponding to element to place at the
            substitutional site.
        base_path (Path):
            Base path to be used for setting up the carrier capture
            calculations. Should be the base directory for `pydefect`
            subdirectories and contain the defect subdirectory.
            Assumes a POSCAR file exists in base_path/defect/perfect/.
            Defaults to Path.cwd().
        comp_idx (int):
            Index of the defect complex, used to differentiate
            distinct defects of the same type. Defaults to 1.

    Returns
    ---------
        `pymatgen-defect-analysis` DefectComplex object.
    """
    defects_path, rad_defects_path = base_path / 'defect', base_path / 'rad_poscars'
    
    perfect_poscar_path = defects_path / 'perfect' / 'POSCAR'
    perfect_pos = Poscar.from_file(perfect_poscar_path)
    
    # create vacancy/substitutional complex name
    # future: DefectComplex should have a list with each type inside as an attribute
    # indicate defects in name based on capital letters
    defect_name = f'V{sub_type}_{vac_site}{sub_site}{comp_idx}'
    
    # sum valence of each atom becoming vacancy and create charge list
    tot_valence = sum([Element(i).valence[1] for i in [vac_site, sub_type]])
    complex_chgs = range(-tot_valence, tot_valence+1, 1)
    
    # try to add name: complex_charges to yaml
    try:
        with open(defects_path / 'rad_defect_in.yaml', 'r') as rdy:
            rad_defect_set = yaml.safe_load(rdy)

            try:
                # add defect if it doesn't appear in yaml
                if defect_name not in rad_defect_set:
                    rad_defect_set.update({defect_name: list(complex_chgs)})
            except TypeError:
                # if yaml file is empty create new dict
                rad_defect_set = {defect_name: list(complex_chgs)}
            
    except FileNotFoundError:
        # create rad_defect_in.yaml if it doesn't exist
        rad_defect_set = {defect_name: list(complex_chgs)}
    
    with open(defects_path / 'rad_defect_in.yaml', 'w') as rdy:
        yaml.dump(rad_defect_set, rdy, default_flow_style=None)
    
    # copy perfect structure to add defect
    defect_struc = perfect_pos.structure.copy()
    
    # create vacancy at first index
    vac_site0 = defect_struc.sites[defect_struc.indices_from_symbol(vac_site)[0]]
    defect_list = [Vacancy(structure=defect_struc, site=vac_site0)]
    # nearest neighbor sites for complex
    vac0_neighbors = defect_struc.get_all_neighbors(r=2.5, sites=[vac_site0])[0]
    
    # find the closest neighbor that has the same site type as the substitutional site
    sub_site = vac0_neighbors[0]
    for n in range(len(vac0_neighbors)):
        neighbor_site = vac0_neighbors[n]
        if sub_site == neighbor_site.label:
            if math.dist(vac_site0.coords, sub_site.coords) < math.dist(vac_site0.coords, neighbor_site.coords):
                sub_site = neighbor_site
    sub_site.species = Element(sub_type)
    defect_list.append(Substitution(structure=defect_struc, site=sub_site))
    
    vac_sub_complex = DefectComplex(defects=[pd for pd in defect_list])
    
    vac_sub_complex_pos = Poscar(vac_sub_complex.defect_structure)
    
    try:
        vac_sub_complex_pos.write_file(rad_defects_path / ('POSCAR_' + defect_name))
    except FileExistsError:
        logger.info(f'POSCAR exists, so skipped...')
    
    return vac_sub_complex


# want to generate directories for complex defects, similar to pydefect_vasp de
def make_rad_defect_entries(base_path=Path.cwd()):
    """
    based on make_defect_entries from pydefect
    
    what needs to happen
    need to read defect type and charge from name
    maybe set a directory for rad-induced defect POSCARs
    create directory structure and copy poscar multiple times
    convert poscar lattice constants to supercell?
    """
    supercell_info: SupercellInfo = loadfn(base_path / 'supercell_info.json')
    perfect = Path(base_path / 'perfect')
    
    try:
        perfect.mkdir()
        logger.info("Making perfect dir...")
        supercell_info.structure.to(filename=str(perfect / "POSCAR"))
    except FileExistsError:
        logger.info(f"perfect dir exists, so skipped...")
    
    rad_defect_set = DefectSet.from_yaml(base_path / 'rad_defect_in.yaml')
    rad_defect_dir = base_path.parent / 'rad_poscars'
    
    for defect in rad_defect_set:
        defect_poscar_title = 'POSCAR_' + defect.name
        poscar_path = rad_defect_dir / defect_poscar_title

        for i in range(len(defect.str_list)):
            charge, def_chg_str = defect.charges[i], defect.str_list[i]
            dir_path = base_path / def_chg_str
            
            try:
                dir_path.mkdir()
                logger.info(f"Making {dir_path} dir...")
                
                # copy POSCAR with defect from radiation damage simulation
                shutil.copy(poscar_path, dir_path / 'POSCAR')
                
                # adjust lattice vectors to be the same as perfect structure
                perfect_pos, defect_pos = Poscar.from_file(perfect / 'POSCAR'), Poscar.from_file(dir_path / 'POSCAR')

                defect_pos.structure.lattice = perfect_pos.structure.lattice
                defect_pos_adj = Poscar(defect_pos.structure)

                defect_pos_adj.write_file(dir_path / 'POSCAR')

                # create prior_info.yaml
                prior_info = {'charge': charge}
                with open(dir_path / 'prior_info.yaml', 'w') as piy:
                    yaml.dump(prior_info, piy)
                    
                # make defect_entry.json
                # defect_entry = make_defect_entry(name=defect.name, charge=charge, perfect_structure=perfect_pos.structure, defect_structure=defect_pos_adj.structure)
                # defect_entry.to_json_file(filename=str(dir_path / "defect_entry.json"))
                # defect_entry.to_prior_info(filename=str(dir_path / "prior_info.yaml"))
            
            except FileExistsError:
                logger.info(f"{dir_path} dir exists, so skipped...")

    return None


def setup_defect_subcalcs(
    subcalc_keys: str | list[str],
    defect_path: Path = Path.cwd(),
    defect_pattern: str = '*/',
    continue_from: str | None = None,
    potcar: Potcar | None = None
) -> None:
    """
    Creates defect subcalculation directories to relax defect structures
    in stages. Can create multiple subcalculations based on
    subcalc_keys.

    Creates <subcalc_key> directories for each defect in the defect path
    matching the defect pattern (e.g., '*_*_0/' corresponds to all
    neutral defects), and copies vise_<subcalc_key>.yaml, POSCAR, and
    any existing prior_info.yaml and defect_entry.json files to the new
    subcalculation directories. Creates defect_entry.json if needed.
    Modifies information in the vise.yaml files as needed (e.g.,
    MAGMOM).
    
    Afterward, execute `vise vs -t defect` for each subdirectory.
    Should do automatically in future.
    2x2x2 Gamma-centered k-point mesh example
    `vise vs -t defect -k 2.0 --uniform_kpt_mode --options
        gamma_centered True only_even_num_kpts True`
    
    Args
    ---------
        subcalc_keys (str | list[str]):
            Subcalculation key(s) to be used for setting up the defect
            relaxation steps. Assumes the first (or only) key
            corresponds to the first calculation stage. Keys should
            correspond to the suffix of the vise_<subcalc_key>.yaml
            files in the defect path. For example, if subcalc_keys is
            ['kpt1', 'kpt2'], then the function will look for
            vise_kpt1.yaml and vise_kpt2.yaml in the defect path to copy
            to the new subcalculation directories. The POSCAR file will
            be moved into the first stage directory.
        defect_path (Path):
            Defect directory for `pydefect`. Defaults to Path.cwd().
        defect_pattern (str):
            Pattern to match for defect directories in the defect path
            using Path.glob. The directories matching the pattern must
            have a POSCAR file if continue_from is None. Defaults to
            '*/', which corresponds to all defects and the perfect
            supercell relaxation.
        continue_from (str | None):
            Pattern and subcalc stage to continue from. Subcalc
            directory needs to have a CONTCAR file.
        potcar (Potcar | None):
            Potcar object to be used for determining the number of
            electrons for the defect structures. If None, defaults to
            PBE_54 POTCARs from pymatgen. Defaults to None.

    Returns
    ---------
        Nothing.
    """
    # ensure subcalc_keys is a list
    if type(subcalc_keys) == str:
        subcalc_keys = [subcalc_keys]

    # ensure defect_pattern ends with '/'
    if not defect_pattern.endswith('/'):
        defect_pattern += '/'

    # ensure defect_pattern ends with '/'
    if not continue_from.endswith('/'):
        continue_from += '/'
    
    defect_dirs = [_ for _ in defect_path.glob(f'{defect_pattern}')]
    if continue_from is not None:
        cont_paths = defect_path.glob(f'{continue_from}CONTCAR')
        prev_dirs = [p.parent for p in cont_paths]
        prev_stage_dict = {}

        # try to find closest match if exact match not found
        for stages_dir in defect_dirs:
            # if exact match found
            if stages_dir in prev_dirs:
                prev_stage_dict.update({stages_dir: stages_dir})
            else:
                for continue_from_dir in prev_dirs:
                    # defect type and site must match at least
                    prev_defect, next_defect = continue_from_dir.parent.name, stages_dir.name
                    if prev_defect.split('_')[:2] == next_defect.split('_')[:2]:
                        # check if defect charge closer than current match if multiple matches found
                        if continue_from_dir in prev_stage_dict:
                            current_defect = prev_stage_dict[continue_from_dir].name
                            current_match_charge = int(current_defect.split('_')[-1])
                            new_match_charge = int(next_defect.split('_')[-1])
                            target_charge = int(prev_defect.split('_')[-1])
                            if abs(new_match_charge - target_charge) < abs(current_match_charge - target_charge):
                                    prev_stage_dict.update({stages_dir: continue_from_dir})
                        else:
                            prev_stage_dict.update({stages_dir: continue_from_dir})
                    else:
                        logger.info(f'{next_defect} does not match {prev_defect}')
                        continue

            shutil.copy(
                prev_stage_dict[stages_dir] / 'CONTCAR',
                stages_dir / 'POSCAR'
            )
        print('Continued from values+CONTCAR to keys+POSCAR', prev_stage_dict, sep='\n')

    for calc_path in defect_dirs:
        # need to change calc path if continue_from
        poscar_path = calc_path / 'POSCAR'
        # ignore dotfile directories
        if calc_path.name[0] != '.':
            for k, subcalc in enumerate(subcalc_keys):
                # create subcalc directories
                subcalc_path = calc_path / subcalc
                subcalc_path.mkdir()

                # load POSCAR and get parameters for variable tags
                defect_poscar = Poscar.from_file(poscar_path)

                # get number of atoms for INCAR tags (e.g., MAGMOM)
                num_atoms = defect_poscar.structure.num_sites

                # get atom types
                atom_types = defect_poscar.site_symbols

                # default to PBE POTCAR, maybe base on x in vise.yaml
                potcar = Potcar(atom_types, 'PBE_54') if potcar == None else potcar
                charge = int(calc_path.name.split('_')[-1])

                # get electron count from POSCAR/POTCAR
                nelect = 0
                for i, ele in enumerate(atom_types):
                    for j in potcar:
                        if ele == j.element:
                            nelect += int(j.nelectrons)*defect_poscar.natoms[i]
                nelect -= charge

                # copy vise.yaml files to subcalc directories
                try:
                    with open(defect_path / f'vise_{subcalc}.yaml', 'r') as vy:
                        vise_yaml = vy.read()
                    
                    vise_yaml = vise_yaml.replace('!MAGMOM', str(num_atoms))
                    vise_yaml = vise_yaml.replace('!NUPDOWN', str(int(nelect%2)))
                    # add more variable tag replacements as needed

                    with open(subcalc_path / 'vise.yaml', 'w') as vy:
                        vy.write(vise_yaml)
                except FileNotFoundError:
                    logger.info(f'vise_{subcalc}.yaml not found, skipping')
                    continue

                # move POSCAR to first subcalc directory
                if k == 0:
                    shutil.move(
                        poscar_path,
                        subcalc_path / 'POSCAR'
                    )

                try:
                    shutil.copy(
                        calc_path / 'prior_info.yaml',
                        subcalc_path / 'prior_info.yaml'
                    )
                except FileNotFoundError:
                    logger.info(f'prior_info.yaml not found in {calc_path}, skipping')
                    pass

                try:
                    shutil.copy(
                        calc_path / 'defect_entry.json',
                        subcalc_path / 'defect_entry.json'
                    )
                except FileNotFoundError:
                    logger.info(f'defect_entry.json not found in {calc_path}, skipping')
                    pass
    
    return None


def update_defect_entry_pos(defect_entry, defect_pos_new, copy_filename=False):
    """
    Updates a pydefect defect_entry.json file with a new defect position.
    """
    # use to maintain a copy of original file under a new name
    if type(copy_filename) == str:
        shutil.copyfile(defect_entry, copy_filename)

    # replace defect position in pydefect file
    with open(defect_entry, 'r') as file_json:
        defect_data = json.load(file_json)
        defect_data['defect_center'] = defect_pos_new
        defect_data_new = json.dumps(defect_data)

    # create new defect_entry.json file
    with open(defect_entry, 'w') as file_json:
        file_json.write(defect_data_new)
    
    return defect_data


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


def lattice_param_thermal_expansion(x0, cte_x, dT):
    """
    Given the initial lattice parameter, the coefficient of thermal expansion for the
    given lattice parameter, and the temperature difference, returns the expected
    lattice parameter value at the desired temperature.
    """
    return x0*(1 + (cte_x*dT))
