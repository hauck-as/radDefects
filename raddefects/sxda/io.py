"""Parsing I/O for sxdefectalign calculations."""
import logging
from typing import TYPE_CHECKING, Optional
from io import StringIO

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


def parse_sxda_vatoms(filepath=os.getcwd()):
    """
    Parses sxdefectalign vAtoms.dat files.
    
    potential terms sxdefectalign = pydefect
        V(long-range) = V_{PC,q}
        V(defect)-V(ref) = V_{q/b}
        V(defect)-V(ref)-V(long-range) = dV_{PC,q/b}
    
    vAtoms.dat format
        r V(long-range) V(defect)-V(ref) V(defect)-V(ref)-V(long-range) x y z ...
        empty lines between different species (need to include in plotting now)
        with auto_sxda, empty lines are replaced with double empty lines to enable gnuplot index use
    """
    vatoms_filepath, contcar_filepath = os.path.join(filepath, 'vAtoms.dat'), os.path.join(filepath, 'CONTCAR')
    contcar = Poscar.from_file(contcar_filepath)
    
    vatoms_dict = {i.symbol:'' for i in contcar.structure.elements}
    
    with open(vatoms_filepath, 'r') as va_dat:
        va_dat_text = va_dat.read()
        
        # find double line breaks in vatoms data file string
        idx_split = [0]
        idx_split += re.search(r'\n\n\n', va_dat_text).span()
        idx_split.append(len(va_dat_text))
        
        # create dataframes for each element's atomic site potentials
        for i in range(int(len(idx_split)/2)):
            atomic_site_type = contcar.structure.elements[i].symbol
            vatoms_str = StringIO(va_dat_text[idx_split[2*i]:idx_split[2*i+1]].strip())
            vatoms_arr = np.genfromtxt(vatoms_str)
            vatoms_df = pd.DataFrame(
                data=vatoms_arr,
                columns=['r', 'V_lr', 'V_defect-V_ref', 'V_defect-V_ref-V_lr', 'x', 'y', 'z']
            )
            vatoms_dict.update({atomic_site_type:vatoms_df})
    
    return vatoms_dict

