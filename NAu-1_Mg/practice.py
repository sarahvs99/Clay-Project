#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:11:04 2021

@author: sarah
"""

import MDAnalysis as mda
from MDAnalysis.analysis import lineardensity as lin
from MDAnalysis.analysis import density
from MDAnalysis.topology import tpr
from MDAnalysis.topology import TPRParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create Universe
N1_Mg_eqd=mda.Universe("eq_d_spacing_06.tpr", "eq_d_spacing_06.trr")
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
dynamic_ions=mda.AtomGroup(N1_Mg_eqd.select_atoms('(resname Mg or resname Cl) and (prop z > 89.36 or prop z < 43.18)', updating=True))
dynamic_cations=mda.AtomGroup(N1_Mg_eqd.select_atoms('(resname Mg) and (prop z > 89.36 or prop z < 43.18)', updating=True))
dynamic_anions=mda.AtomGroup(N1_Mg_eqd.select_atoms('(resname Cl) and (prop z > 89.36 or prop z < 43.18)', updating=True))

# ionic z-density profiles
ionic_density=lin.LinearDensity(dynamic_ions, binsize=0.25).run()
cationic_density=lin.LinearDensity(dynamic_cations, binsize=0.25).run()
anionic_density=lin.LinearDensity(dynamic_anions, binsize=0.25).run()

#%%
fig, ion_dens=plt.plot(np.linspace(0, 133.3808, 534), ionic_density.results['z']['char'])
ions_dens.set_xlabel("z-coordinate (Å)")
ions_dens.set_ylabel("Charge density")
ion_dens.set_title("ionic z-density profile")

#%%
fig, cat_dens=plt.plot(np.linspace(0, 133.3808, 534), cationic_density.results['z']['char'])
cat_dens.set_xlabel("z-coordinate (Å)")
cat_dens.set_ylabel("Charge density")
cat_dens.set_title("cationic z-density profile")

#%%
fig, ani_dens=plt.plot(np.linspace(0, 133.3808, 534), anionic_density.results['z']['char'])
ani_dens.set_xlabel("z-coordinate (Å)")
ani_dens.set_ylabel("Charge density")
ani_dens.set_title("anionic z-density profile")

#%%
fig, cat_pos=plt.plot(np.linspace(0, 133.3808, 534), cationic_density.results['z']['pos'])
cat_pos.set_xlabel("z-coordinate (Å)")
cat_pos.set_ylabel("Position density")
cat_pos.set_title("cationic z-density profile")

#%%
fig, cat=plt.subplots()
cat.plot(np.linspace(0, 133.3808, 534), cationic_density.results['z']['char'], '--', color='black', label="charge density")
cat.plot(np.linspace(0, 133.3808, 534), cationic_density.results['z']['pos'], ':', label="mass density",)
cat.set_xlabel("z-coordinate (Å)")
cat.set_ylabel("density")
cat.legend()
