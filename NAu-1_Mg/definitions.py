#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:02:04 2021

@author: sarah
"""

import MDAnalysis as mda
from MDAnalysis.analysis import lineardensity as lin
from MDAnalysis.analysis import density
from MDAnalysis.analysis import rdf
from MDAnalysis.topology import tpr
from MDAnalysis.topology import TPRParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

# from trial1.py

def universe_help():
    '''Tells user how to create a universe
    '''
    print("To create a universe input: name=mda.Universe('top file', 'traj files')" )
    
def atomgroup_help():
    '''Tells user how to create an AtomGroup
    '''
    print("To create an AtomGroup input: name=mda.AtomGroup(universe.select_atoms('atoms'), updating=True/False)")
    print(' ')
    print("'atoms': called using resname etc with boolean logic, set bounds using prop '?' >/</<= etc '?'")
    print(' ')
    print("updating: Is the group dynamic? True if yes, False if no")
    
def atomgroup_coords(atomgroup):
    '''Generates the minimum and maximum x, y and z coordinates of atoms in AtomGroup
    Parameters
    ------------
    atomgroup: name of AtomGroup
    
    Returns
    -------
    Minimum and maximum coordinates'''
    at_pos=atomgroup.positions
    min_coords=at_pos.min(axis=0)
    max_coords=at_pos.max(axis=0)
    
    print("Minimum x-coordinate is ", min_coords[0]) 
    print("Maximum x-coordinate is ", max_coords[0])
    print(' ')
    print("Minimum y-coordinate is ", min_coords[1]) 
    print("Maximum y-coordinate is ", max_coords[1])
    print(' ')
    print("Minimum z-coordinate is ", min_coords[2]) 
    print("Maximum z-coordinate is ", max_coords[2])
    return at_pos, min_coords, max_coords

def position_density(top, traj, atoms, dyn):
    '''Creates dataframes of the mass-weighted position density of an AtomGroup along each axis
    Parameters
    -----------
    top: topology file used to create universe
    
    traj: trajectory file used to create universe
    
    atom: atom selection used to create AtomGroup
    
    dyn: dynamic selection used to create AtomGroup (True/False)
    
    Returns
    --------
    The size of created dataframes'''
    
    # Generate minimum and maximum coordinates
    u=mda.Universe(top, traj)
    at_group=mda.AtomGroup(u.select_atoms(atoms, updating=dyn)) 
    at_positions=at_group.positions
    min_coords=at_positions.min(axis=0)
    max_coords=at_positions.max(axis=0)
    
    min_x=min_coords[0]
    max_x=max_coords[0]
    
    min_y=min_coords[1]
    max_y=max_coords[1]
    
    min_z=min_coords[2]
    max_z=max_coords[2]
    
    atom_density=lin.LinearDensity(at_group, binsize=0.25).run()
    
    # Create dataframes
    x_pos_dens=pd.DataFrame(atom_density.results['x'], columns=['pos'])
    y_pos_dens=pd.DataFrame(atom_density.results['y'], columns=['pos'])
    z_pos_dens=pd.DataFrame(atom_density.results['z'], columns=['pos'])
    
    # Create coordinate column info
    x_index=pd.Index(np.linspace(min_x, max_x, atom_density.nbins))
    y_index=pd.Index(np.linspace(min_y, max_y, atom_density.nbins))
    z_index=pd.Index(np.linspace(min_z, max_z, atom_density.nbins))
    
    # Add coordinate column to dataframes
    x_pos_dens=x_pos_dens.set_index(x_index)
    x_pos_dens.reset_index(inplace=True)
    
    y_pos_dens=y_pos_dens.set_index(y_index)
    y_pos_dens.reset_index(inplace=True)
    
    z_pos_dens=z_pos_dens.set_index(z_index)
    z_pos_dens.reset_index(inplace=True)
    
    # Add normalised position column
    x_pos_dens['normalised position density']=(x_pos_dens['pos']-x_pos_dens['pos'].mean())/x_pos_dens['pos'].std()
    y_pos_dens['normalised position density']=(y_pos_dens['pos']-y_pos_dens['pos'].mean())/y_pos_dens['pos'].std()
    z_pos_dens['normalised position density']=(z_pos_dens['pos']-z_pos_dens['pos'].mean())/z_pos_dens['pos'].std()
    
    # Rename columns
    x_pos_dens=x_pos_dens.rename(columns = {'index':'x-coordinate', 'pos':'position density'})
    y_pos_dens=y_pos_dens.rename(columns = {'index':'y-coordinate', 'pos':'position density'})
    z_pos_dens=z_pos_dens.rename(columns = {'index':'z-coordinate', 'pos':'position density'})
    
    print("Created x_pos_dens with shape ", x_pos_dens.shape)
    print("Created y_pos_dens with shape ", y_pos_dens.shape)
    print("Created z_pos_dens with shape ", z_pos_dens.shape)
    
    return x_pos_dens, y_pos_dens, z_pos_dens

def charge_density(top, traj, atoms, dyn):
    '''Creates dataframes of the charge density of an AtomGroup along each axis
    Parameters
    -----------
    top: topology file used to create universe
    
    traj: trajectory file used to create universe
    
    atom: atom selection used to create AtomGroup
    
    dyn: dynamic selection used to create AtomGroup
    
    Returns
    --------
    The size of created dataframes'''
    
    # Generate minimum and maximum coordinates
    u=mda.Universe(top, traj)
    at_group=mda.AtomGroup(u.select_atoms(atoms, updating=dyn)) 
    
    at_positions=at_group.positions
    min_coords=at_positions.min(axis=0)
    max_coords=at_positions.max(axis=0)
    
    min_x=min_coords[0]
    max_x=max_coords[0]
    
    min_y=min_coords[1]
    max_y=max_coords[1]
    
    min_z=min_coords[2]
    max_z=max_coords[2]
    
    # Generate density results
    atom_density=lin.LinearDensity(at_group, binsize=0.25).run()
    
    # Create dataframes
    x_char_dens=pd.DataFrame(atom_density.results['x'], columns=['char'])
    y_char_dens=pd.DataFrame(atom_density.results['y'], columns=['char'])
    z_char_dens=pd.DataFrame(atom_density.results['z'], columns=['char'])
    
    # Create coordinate column info
    x_index=pd.Index(np.linspace(min_x, max_x, atom_density.nbins))
    y_index=pd.Index(np.linspace(min_y, max_y, atom_density.nbins))
    z_index=pd.Index(np.linspace(min_z, max_z, atom_density.nbins))
    
    # Add coordinate column to dataframes
    x_char_dens=x_char_dens.set_index(x_index)
    x_char_dens.reset_index(inplace=True)
    
    y_char_dens=y_char_dens.set_index(y_index)
    y_char_dens.reset_index(inplace=True)
    
    z_char_dens=z_char_dens.set_index(z_index)
    z_char_dens.reset_index(inplace=True)
    
    # Add normalised charge column
    x_char_dens['normalised charge density']=(x_char_dens['char']-x_char_dens['char'].mean())/x_char_dens['char'].std()
    y_char_dens['normalised charge density']=(y_char_dens['char']-y_char_dens['char'].mean())/y_char_dens['char'].std()
    z_char_dens['normalised charge density']=(z_char_dens['char']-z_char_dens['char'].mean())/z_char_dens['char'].std()
    
    # Rename columns
    x_char_dens=x_char_dens.rename(columns = {'index':'x-coordinate', 'char':'charge density'})
    y_char_dens=y_char_dens.rename(columns = {'index':'y-coordinate', 'char':'charge density'})
    z_char_dens=z_char_dens.rename(columns = {'index':'z-coordinate', 'char':'charge density'})
    
    print("Created x_char_dens with shape ", x_char_dens.shape)
    print("Created y_char_dens with shape ", y_char_dens.shape)
    print("Created z_char_dens with shape ", z_char_dens.shape)
    return x_char_dens, y_char_dens, z_char_dens

#%%

# from trial2.py

def average_rdf(atomgroup1, atomgroup2, bins, rdf_range):
    '''Plots the average radial distribution function for two AtomGroups
    Parameters
    -----------
    atomgroup1: first AtomGroup
    
    atomgroup2: second AtomGroup
    
    bins: number of bins in the histogram. For default use 75
    
    rdf_range: spherical shell limit around each atom. two numbers need to be given in form (?.?, ?.?). For default use (0.0, 15.0)
    
    Returns
    --------
    Plot of radius vs rdf'''
    av_rdf=rdf.InterRDF(atomgroup1, atomgroup2, 
                        nbins=bins,
                        range=rdf_range).run()
    av_rdf_plot=plt.plot(av_rdf.bins, av_rdf.rdf)
    plt.xlabel('Radius (Å)')
    plt.ylabel('Radial distribution')
    return av_rdf_plot

def intraAG_average_rdf(atomgroup1, bins, rdf_range, exclusion):
    '''Plots the average radial distribution function for an AtomGroup to iself
    Parameters
    -----------
    atomgroup1: AtomGroup being investigated
    
    bins: number of bins in the histogram. For default use 75
    
    rdf_range: spherical shell limit around each atom. Two numbers need to be given in form (?.?, ?.?). For default use (0.0, 15.0)
    
    exclusion: mask pairs within the same chunk of atoms. Two numbers to be given in the form (?, ?) which represents the size of the chunk excluded and does not need to be square. E.g. for the interaction between water molcules, give (3, 3) 
    
    Returns
    --------
    Plot of radius vs rdf'''
    ag_av_rdf=rdf.InterRDF(atomgroup1, atomgroup1, 
                        nbins=bins,
                        range=rdf_range, exclusion_block=(exclusion)).run()
    ag_av_rdf_plot=plt.plot(ag_av_rdf.bins, ag_av_rdf.rdf)
    plt.xlabel('Radius (Å)')
    plt.ylabel('Radial distribution')
    return ag_av_rdf_plot
    

def specific_rdf(universe, atom_pairs, bins, rdf_range, dens):
    '''Calculates the site-specific radial distribution function. Users can then choose to create .csv files and graphs of the outputs.
    This function can take a long time to run before an output is given.
    
    Parameters
    -----------
    universe: universe atomgroups are in
    
    atom_pairs: list of pairs of AtomGroups e.g. [[g1, g2], [g3, g4]] where g1 and g2 are two atomgroups whose interaction is under investigation
    
    bins: number of bins in the histogram. For default use 75
    
    rdf_range: spherical shell limit around each atom. two numbers need to be given in form (?.?, ?.?). For default use (0.0, 15.0)
    
    dens: if True, final density is averaged. if False, density not average so harder to compare between different sizes of AtomGroups
    
    Returns
    ----------
    If the user wishes, .csv files and graphs can be created
    '''
    print('The function "specific_rdf" is executing...')
    ss_rdf=rdf.InterRDF_s(universe, atom_pairs,
                          nbins=bins,
                          range=rdf_range,
                          density=dens).run()
    
    print(' ')
    i=0
    for list_i in atom_pairs:
        print(' ')
        print('Result array of', list_i, 'with Aatom_pairs index of', i, 'has shape: {}'.format(ss_rdf.rdf[i].shape))
        l, m, nbin = np.nonzero(ss_rdf.rdf[i])
        save=input('Do you want to save the indices of the atom numbers where there are non-zero values for this array? (y/n): ')
        if save == 'y':
            file_name1=input('Enter the name for the file containing indices for atom 1. Must end in .csv : ')
            np.savetxt(file_name1, l, delimiter=' ', fmt='%d')
            file_name2=input('Enter the name for the file containing indices for atom 2. Must end in .csv : ')
            np.savetxt(file_name2, m, delimiter=' ', fmt='%d')
        
        print('--------------------')    
        i += 1
    
    print(' ')
    end_plot=input("Do you want to plot a graph? (y/n): ")
    
    while end_plot == 'y':
        
        fig_name=input("Enter a name for graph(no spaces): ")
    
        i=int(input("Index of AtomGroup pair in atom_pairs: "))
        j=int(input("Index of atom 1: "))
        k=int(input("Index of atom 2: "))
    
        fig, fig_name=plt.subplots()
        fig_name.plot(ss_rdf.bins, ss_rdf.rdf[i][j][k])
        fig_name.set_xlabel('Radius (Å)')
        fig_name.set_ylabel('Radial Distribution Function')
        fig_name.set_title(input("Enter a title: "))

        print('--------------------') 
        print(' ')
        end_plot=input("Do you want to continue plotting? (y/n): ")
    
    print(' ')
    print('Thank you for using this function. Have a nice day!')
        
    
    
    
    
    
    
