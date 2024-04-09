"""Core functions and classes for analyzing rad-induced defects from VASP."""
import os
import subprocess
import numpy as np

from pymatgen.io.vasp.sets import VaspInputSet, Poscar
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import *
from pymatgen.analysis.defects.core import DefectComplex, Substitution, Vacancy
from pymatgen.analysis.defects.finder import DefectSiteFinder
from mp_api.client import MPRester

from pydefect.analyzer.band_edge_states import BandEdgeOrbitalInfos, BandEdgeState, PerfectBandEdgeState

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
    Something
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
        else:
            raise UserWarning('Multiple matching structures were found.')


# band edge orbital infos gives fermi level, states below should be occupied and above should be unoccupied
# interested in occupation of states between VBM & CBM: gap states
def find_gap_states(pbes_path, beois_path):
    """
    Something
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