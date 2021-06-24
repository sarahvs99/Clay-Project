#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:11:04 2021

@author: sarah
"""

import MDAnalysis as mda
from MDAnalysis.analysis import lineardensity as lin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create Universe
N1_Mg_eqd=mda.Universe("eq_d_spacing_06.gro", "eq_d_spacing_06.trr")
print("N1_Mg_eqd is: ", N1_Mg_eqd)
print("The universe N1_Mg_eqd has a trajectory: ", hasattr(N1_Mg_eqd, 'trajectory'))

# assign clay AtomGroup
clay=mda.AtomGroup(N1_Mg_eqd.select_atoms('resname NON*'))

# Minimum and maximum z-coordinate

clay_pos=clay.positions
clay_pos_max=clay_pos.max(axis=0)
clay_pos_min=clay_pos.min(axis=0)

print("Maximum z-coordinate of clay is ", clay_pos_max [2])
print("Minimum z-coordinate of clay is ", clay_pos_min [2]) 

# assign bulk ions AtomGroup
dynamic_ions=mda.AtomGroup(N1_Mg_eqd.select_atoms('(resname Mg or resname Cl) and (prop z > 89.049995 or prop z < 43.16)', updating=True))

# ionic z-density profiles
ionic_density=lin.LinearDensity(dynamic_ions, grouping='atoms').run()
