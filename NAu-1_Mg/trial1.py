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

# z-values
index=pd.Index(np.linspace(0, 133.3808, 534))

#%%

# create dataframe from profiles
ionic_density_df=pd.DataFrame(ionic_density.results['z'], columns = ['pos', 'char'])
ionic_density_df=ionic_density_df.set_index(index)
ionic_density_df.reset_index(inplace=True)
ionic_density_df=ionic_density_df.rename(columns = {'index':'z-coordinate'})

cationic_density_df=pd.DataFrame(cationic_density.results['z'], columns = ['pos', 'char'])
cationic_density_df=cationic_density_df.set_index(index)
cationic_density_df.reset_index(inplace=True)
cationic_density_df=cationic_density_df.rename(columns = {'index':'z-coordinate'})

anionic_density_df=pd.DataFrame(anionic_density.results['z'], columns = ['pos', 'char'])
anionic_density_df=anionic_density_df.set_index(index)
anionic_density_df.reset_index(inplace=True)
anionic_density_df=anionic_density_df.rename(columns = {'index':'z-coordinate'})

# normalisation of df
ionic_density_df_norm=ionic_density_df.copy()
ionic_density_df_norm['pos']=(ionic_density_df['pos']-ionic_density_df['pos'].mean())/ionic_density_df['pos'].std()
ionic_density_df_norm['char']=(ionic_density_df['char']-ionic_density_df['char'].mean())/ionic_density_df['char'].std()

cationic_density_df_norm=cationic_density_df.copy()
cationic_density_df_norm['pos']=(cationic_density_df['pos']-cationic_density_df['pos'].mean())/cationic_density_df['pos'].std()
cationic_density_df_norm['char']=(cationic_density_df['char']-cationic_density_df['char'].mean())/cationic_density_df['char'].std()

anionic_density_df_norm=anionic_density_df.copy()
anionic_density_df_norm['pos']=(anionic_density_df['pos']-anionic_density_df['pos'].mean())/anionic_density_df['pos'].std()
anionic_density_df_norm['char']=(anionic_density_df['char']-anionic_density_df['char'].mean())/anionic_density_df['char'].std()

#%%
fig, ion_char=plt.subplots()
ion_char.plot(np.linspace(0, 133.3808, 534), ionic_density.results['z']['char'])
ion_char.set_xlabel("z-coordinate (Å)")
ion_char.set_ylabel("Charge density")
ion_char.set_title("ionic z-density profile")

#%%
fig, cat_char=plt.subplots()
cat_char.plot(np.linspace(0, 133.3808, 534), cationic_density.results['z']['char'])
cat_char.set_xlabel("z-coordinate (Å)")
cat_char.set_ylabel("Charge density")
cat_char.set_title("cationic z-density profile")

#%%
fig, ani_char=plt.subplots()
ani_char.plot(np.linspace(0, 133.3808, 534), anionic_density.results['z']['char'])
ani_char.set_xlabel("z-coordinate (Å)")
ani_char.set_ylabel("Charge density")
ani_char.set_title("anionic z-density profile")

#%%
fig, cat_pos=plt.subplots()
cat_pos.plot(np.linspace(0, 133.3808, 534), cationic_density.results['z']['pos'])
cat_pos.set_xlabel("z-coordinate (Å)")
cat_pos.set_ylabel("Position density")
cat_pos.set_title("cationic z-density profile")

#%%
fig, cat_dens=plt.subplots()
cat_dens.plot(np.linspace(0, 133.3808, 534), cationic_density.results['z']['char'], '--', color='black', label="charge density")
cat_dens.plot(np.linspace(0, 133.3808, 534), cationic_density.results['z']['pos'], ':', label="mass density")
cat_dens.set_xlabel("z-coordinate (Å)")
cat_dens.set_ylabel("density")
cat_dens.legend()

#%%
fig, cat_norm=plt.subplots()
cat_norm.plot(cationic_density_df_norm['z-coordinate'], cationic_density_df_norm['char'], '--', color='black', label="normalised charge density")
cat_norm.plot(cationic_density_df_norm['z-coordinate'], cationic_density_df_norm['pos'], ':', label="normalised mass density")
cat_norm.set_xlabel("z-coordinate (Å)")
cat_norm.set_ylabel("normalised density")
cat_norm.legend()
