#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:39:04 2021

@author: sarah
"""

# Average radial distribution function for two groups of atoms

import MDAnalysis as mda
from MDAnalysis.analysis import lineardensity as lin
from MDAnalysis.analysis import density
from MDAnalysis.analysis import rdf
from MDAnalysis.topology import tpr
from MDAnalysis.topology import TPRParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

u=mda.Universe('top_file.tpr', 'traj_file.trr')
SOL_upper=u.select_atoms('resname SOL and prop z > 89')
Mg_upper=u.select_atoms('resname Mg and prop z > 89 and prop z < 95')
clay_oxy=u.select_atoms('resname NON2 and name O* and prop z > 89.32')
ags=[[clay_oxy, SOL_upper], [Mg_upper, SOL_upper]]

specific_rdf(u, ags, 75, (0.0, 10.0), True)


N1_Mg_sim1=mda.Universe('N1_Mg_sim_1.tpr', 'N1_Mg_sim_1.trr')

clay=mda.AtomGroup(N1_Mg_sim1.select_atoms('resname NON*'))
atomgroup_coords(clay)
position_density('N1_Mg_sim_1.tpr', 'N1_Mg_sim_1.trr', 'resname NON*', False)