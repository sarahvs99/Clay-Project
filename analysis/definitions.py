#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:02:04 2021

@author: sarah
"""

import MDAnalysis as mda
from MDAnalysis.topology import tpr
from MDAnalysis.topology import TPRParser

import scipy

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

def ions(universe):
    '''Tells user what input to use for selecting bulk ions as an AtomGroup'''
    
    clay=mda.AtomGroup(universe.select_atoms('resname NON*'))
    
    at_pos=clay.positions
    min_clay=at_pos.min(axis=0)
    max_clay=at_pos.max(axis=0)
    
    print('To select the ions in the bulk solution, use "resname (atom) and (prop <', min_clay[2], 'or prop >', max_clay[2], ')" for the atom selection')
    
#%%

# Atomic density
from MDAnalysis.analysis import lineardensity as lin

def position_density(top, traj, atoms, dyn):
    '''Creates dataframes of the mass-weighted position density of an AtomGroup along each axis and gives the option to plot the DataFrames
    Parameters
    -----------
    top: topology file used to create universe
    
    traj: trajectory file used to create universe
    
    atom: atom selection used to create AtomGroup
    
    dyn: dynamic selection used to create AtomGroup (True/False)
    
    Returns
    --------
    The size of created dataframes and any graphs plotted'''
    
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
    
    print(' ')
    print("Created x position density df with shape ", x_pos_dens.shape)
    print("Created y position density df with shape ", y_pos_dens.shape)
    print("Created z position density df with shape ", z_pos_dens.shape)
    print(' ')
    
    x_pos_np=x_pos_dens.to_numpy()
    y_pos_np=y_pos_dens.to_numpy()
    z_pos_np=z_pos_dens.to_numpy()
    
    pos_dens=np.dstack([x_pos_np, y_pos_np, z_pos_np])
    
    plot=input('Would you like to plot any of the dataframes? (y/n): ')
    while plot == 'y':
        
        axis=int(input('Which dataframe would you like to plot? Enter 0 for x, 1 for y and 2 for z: '))
        
        data_type=input('Which data would you like to plot on this graph? Options are non_normalised (non_norm), normalised (norm), both on the same graph (both): ')
        
        if data_type == 'non_norm':
                fig, fig_name=plt.subplots()
                fig_name.plot(pos_dens[:, 0, axis], pos_dens[:, 1, axis])
                fig_name.set_xlabel('axis coordinate (Å)')
                fig_name.set_ylabel('Position Density')
                fig_name.set_title(input("Enter a title: "))
        elif data_type == 'norm':
                fig, fig_name=plt.subplots()
                fig_name.plot(pos_dens[:, 0, axis], pos_dens[:, 2, axis])
                fig_name.set_xlabel('axis coordinate (Å)')
                fig_name.set_ylabel('Position Density')
                fig_name.set_title(input("Enter a title: "))
        elif data_type == 'both':
                fig, fig_name=plt.subplots()
                fig_name.plot(pos_dens[:, 0, axis], pos_dens[:, 1, axis], '--', color='black', label="position density")
                fig_name.plot(pos_dens[:, 0, axis], pos_dens[:, 2, axis], ':', label="normalised position density")
                fig_name.set_xlabel('axis coordinate (Å)')
                fig_name.set_ylabel('Position Density')
                fig_name.legend()
                fig_name.set_title(input("Enter a title: "))
            
        
        print(' ')
        plot=input('Would you like to plot another graph? (y/n): ')
    
    print(' ')
    print('Thank you for using this function. Have a nice day!')

def charge_density(top, traj, atoms, dyn):
    '''Creates dataframes of the charge density of an AtomGroup along each axis and gives the option to plot graphs of the DataFrames
    Parameters
    -----------
    top: topology file used to create universe
    
    traj: trajectory file used to create universe
    
    atom: atom selection used to create AtomGroup
    
    dyn: dynamic selection used to create AtomGroup
    
    Returns
    --------
    The size of created dataframes and any graphs plotted'''
    
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
    
    print(' ')
    print("Created x charge density df with shape ", x_char_dens.shape)
    print("Created y charge density df with shape ", y_char_dens.shape)
    print("Created z charge density df with shape ", z_char_dens.shape)
    print(' ')
    
    x_char_np=x_char_dens.to_numpy()
    y_char_np=y_char_dens.to_numpy()
    z_char_np=z_char_dens.to_numpy()
    
    char_dens=np.dstack([x_char_np, y_char_np, z_char_np])
    
    plot=input('Would you like to plot any of the dataframes? (y/n): ')
    while plot == 'y':
        
        axis=int(input('Which dataframe would you like to plot? Enter 0 for x, 1 for y and 2 for z: '))
        
        data_type=input('Which data would you like to plot on this graph? Options are non_normalised (non_norm), normalised (norm), both on the same graph (both): ')
        
        if data_type == 'non_norm':
                fig, fig_name=plt.subplots()
                fig_name.plot(char_dens[:, 0, axis], char_dens[:, 1, axis])
                fig_name.set_xlabel('axis coordinate (Å)')
                fig_name.set_ylabel('Charge Density')
                fig_name.set_title(input("Enter a title: "))
        elif data_type == 'norm':
                fig, fig_name=plt.subplots()
                fig_name.plot(char_dens[:, 0, axis], char_dens[:, 2, axis])
                fig_name.set_xlabel('axis coordinate (Å)')
                fig_name.set_ylabel('Charge Density')
                fig_name.set_title(input("Enter a title: "))
        elif data_type == 'both':
                fig, fig_name=plt.subplots()
                fig_name.plot(char_dens[:, 0, axis], char_dens[:, 1, axis], '--', color='black', label="charge density")
                fig_name.plot(char_dens[:, 0, axis], char_dens[:, 2, axis], ':', label="normalised charge density")
                fig_name.set_xlabel('axis coordinate (Å)')
                fig_name.set_ylabel('Charge Density')
                fig_name.legend()
                fig_name.set_title(input("Enter a title: "))
            
        
        print(' ')
        plot=input('Would you like to plot another graph? (y/n): ')
    
    print(' ')
    print('Thank you for using this function. Have a nice day!')
        
            
   

#%%

# RDF
from MDAnalysis.analysis import rdf

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
        print('Result array of', list_i, 'with Atom_pairs index of', i, 'has shape: {}'.format(ss_rdf.rdf[i].shape))
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

#%%

# MSD
from MDAnalysis.analysis import msd

from scipy.stats import linregress
from scipy.stats import t

def mean_sq_disp(u, selection, dimension, algorithm):
    '''Calculates and plots the mean-squared displacement using the Einstein relation. Estimates the diffusion coefficient using a linear model of the MSD.
    WARNING: This function only works in MDAnalysis version 2.0.0 and higher
    
    Parameters
    -----------
    u: universe or (non-updating)AtomGroup
    
    selection: atom selection to calculate the msd for. Default is all
    
    dimension: desired dimension of the msd. Option are 'xyz', 'xy', 'yz', 'xz', 'x', 'y', 'z'. Default is xyz
    
    algorithm: if True, the fast FFT algorithm is used and tidynamics package is required. If False the simple 'windowed algorithm is used'''
    
    
    MSD = msd.EinsteinMSD(u, select=selection, msd_type=dimension, fft=algorithm).run()
    
    nframes = MSD.n_frames
    print(' ')
    print('To calculate the time between frames use: dt x nstxout (from .mdp)')
    timestep=float(input('What is the time between frames? Give answer in nanoseconds: '))
    sim_time = np.arange(nframes)*timestep
    
    fig, fig_name=plt.subplots()
    fig_name.plot(sim_time, MSD.results.timeseries)
    fig_name.set_xlabel('Simulation time (ns)')
    fig_name.set_ylabel('Mean-Squared Displacement')
    fig_name.set_title(input("Enter a title: "))
    plt.show()
    
    start_time = int(input('What is the time at the start of the linear portion? '))
    start_index = int(start_time/timestep)
    end_time = int(input('What is the time at the end of the linear portion? '))
    end_index = int(end_time/timestep)
    linear_model = linregress(sim_time[start_index:end_index],
                              MSD.results.timeseries[start_index:end_index])
    
    fig, linear_plot=plt.subplots()
    linear_plot.plot(sim_time, linear_model.intercept + linear_model.slope*sim_time)
    linear_plot.set_xlabel('Simulation time (ns)')
    linear_plot.set_ylabel('Mean-Squared Displacement')
    linear_plot.set_title('Linear model of MSD')
    plt.show()
    
    slope = linear_model.slope
    error = linear_model.rvalue
    
    D = slope*1/(2*MSD.dim_fac)

    # Two-sided inverse Students t-distribution
    # p - probability, df - degrees of freedom
    tinv = lambda p, df: abs(t.ppf(p/2, df))   
    ts = tinv(0.05, len(sim_time)-2)

    print(' ')
    print(f'The coefficient of determination (R-squared) of the linear model is: {error**2:.6f}')
    print('The diffusion coefficient is estimated to be: ', D)
    print(f'The 95% confidence interval on the slope is: {slope:.6f} +/- {ts*linear_model.stderr:.6f}")')

#%%

# RMSD
from MDAnalysis.analysis import rms

def coordinate_rmsd(atom_group1, atom_group2, geometry, rot_trans):
    '''Calculates the Root Mean-Squared Deviation between two sets of coordinates
    Parameters
    ----------
    atom_group1: first AtomGroup
    
    atom_group2: second AtomGroup
    
    geometry: If True the center of geometry is subtracted before calculation
    
    rot_trans: If True, the function performs a rotational and translational superposition with the fast QCP algorithm before calculation. Must have geometry=True 
    '''
    
    a = atom_group1.positions
    b = atom_group2.positions
    
    ab_rmsd=rms.rmsd(a, b, centre=geometry, superposition=rot_trans)
    
    return ab_rmsd

def trajectory_rmsd(u1, u2, atom_sel, additional, reference):
    '''Calculates the Root Mean-Squared Deviation of an entire trajectory to a single reference point. Automatically performs a rotational and translational alignment of the target trajectory to the reference universe
    Warning: If using MDAnalysis version >=2.0.0 a depreciation warning will be printed during the use of the function
    Parameters
    -----------
    u1: universe to calculate RMSD for
    
    u2: reference universe.
    
    atom_sel: atom selection to superimpose. Can be a selection string for select_atoms(), a dictiomary of the form {'mobile': sel1, 'reference': sel2}, or a tuple of the form (sel1, sel2). (sel1 and sel2 are valid selection strings that are applied to u1 and u2 respectively and must generate groups of equivalent atoms)
    
    additional: list of selections that are always applied to the full universes. Each selection describes additional RMSDs to be computed after the structures have been superimposed according to atom_sel
    
    reference: frame index to select the frame from reference. Default is 0
    '''
    
    trj_rmsd=rms.RMSD(u1, u2, select=atom_sel, groupselections=additional, ref_frame=reference).run()
    
    cols = ['Frame', 'Time (ns)']
    
    selection=input('Enter a name for the atom_sel selection. This will be used as the column heading in the dataframe and the key on the graph of RMSD: ')
    
    cols.append(selection)
    
    print('------------------------')
    
    i=0
    for list_i in additional:
        print(' ')
        print('Enter a name for ', additional[i], ':')
        add_sel=input('This will be used as the column heading in the dataframe and the key on the graph of RMSD: ')
        print('------------------------')
        cols.append(add_sel)
        i += 1
      
    rmsd_df=pd.DataFrame(trj_rmsd.rmsd, columns=cols)
    
    rmsd_plot=rmsd_df.plot(x='Time (ns)', y=cols[2:], kind='line')
    rmsd_plot.set_ylabel('Root Mean-Squared Deviation (Å)')
    rmsd_plot.set_title(input('Enter a title for graph of RMSD: '))
    rmsd_plot.legend()
    
    return rmsd_df
    
#%%   

# distances and contacts
from MDAnalysis.analysis import distances
from MDAnalysis.analysis import contacts    
    
def atom_match_dist(atom_group1, atom_group2, resid_offset, dims):
    '''Calculates the distance between atoms in two atom groups. Most useful to show how atoms move throughout a simulation
    Parameters
    -----------
    atom_group1: first AtomGroup
    
    atom_group2: second AtomGroup. Must have the same number of atoms as atom_group1
    
    resid_offset: an offset added to the resids of atom_group1 and atom_group2. If an integer is given, it is applied to both. If a tuple is given, offset[0] is added to atom_group1 and offset[1] is added to atom_group2. Default is 0
    
    dims: unitcell dimensions of the system in the form [lx, ly, lz, alpha, beta, gamma] and used for simulations with periodic boundary conditions. If no pbc used, enter "None"
    '''
    
    resids1, resids2, dist = distances.dist(atom_group1, atom_group2,
                                        offset=resid_offset, box=dims)
    
    fig, dist_plot=plt.subplots()
    dist_plot.plot(resids1, dist)
    dist_plot.set_ylabel('Distance (Å)')
    dist_plot.set_title(input('Enter a title for graph of distance vs resid: '))

    
def atom_nonmatch_dist(atom_group1, atom_group2, dims):
    '''Calculates atom-to-atom distances between non-matching coordinate arrays
    Parameters
    -----------
    atom_group1: reference AtomGroup
    
    atom_group2: configuration AtomGroup
    
    dims: unitcell dimensions of the system used for simulations with periodic boundary conditions. Must be in the form obtained by u.dimensions. If no pbc used, enter "None"
    '''
    dist_arr = distances.distance_array(atom_group1.positions, 
                                    atom_group2.positions, 
                                    box=dims)
    
    fig, dist_plot = plt.subplots()
    im = dist_plot.imshow(dist_arr, origin='upper')
    cbar = fig.colorbar(im)
    cbar.dist_plot.set_ylabel('Distance (Å)')
    tick_interval = 5
    
    dist_plot.set_xlabel(input('Enter a name for atom_group1, to be used as an axis label: '))
    dist_plot.set_xticks(np.arange(len(atom_group1))[::tick_interval])
    dist_plot.set_xticklabels(atom_group1.resids[::tick_interval])
    
    dist_plot.set_ylabel(input('Enter a name for atom_group2, to be used as an axis label: '))
    dist_plot.set_yticks(np.arange(len(atom_group2))[::tick_interval])
    dist_plot.set_yticklabels(atom_group2.resids[::tick_interval])

    dist_plot.set_title(input('Enter a title for heatmap of distance: '))

def atom_contacts(u, atom_group1, atom_group2, contact_rad):
    '''Calculates the number of contacts between AtomGroups within the radius cutoff, for each frame in a trajectory and plots the information
    Parameters
    -------------
    u: universe AtomGroups are taken from
    
    atom_group1: reference AtomGroup
    
    atom_group2: configuration AtomGroup
    
    contact_rad: cutoff radius for defining a contact. Must be an integer or float
    
    '''
    
    timeseries = []
    for ts in u.trajectory:
        dist = contacts.distance_array(atom_group1.positions, atom_group2.positions)
        n_contacts = contacts.contact_matrix(dist, radius=contact_rad).sum()
        timeseries.append([ts.frame, n_contacts])
    
    at_con = np.arrray(timeseries)
    
    at_con_df = pd.DataFrame(at_con, columns=['Frame',
                                  '# Contacts'])
    
    timestep=float(input('What is the time between frames? Give answer in nanoseconds: '))
    at_con['Time (ns)']=(at_con['Frame']*timestep)
    
    at_con_plot=at_con_df.plot(x='Time (ns)', y='# Contacts')
    at_con_plot.set_ylabel('Number of Contacts')
    at_con_plot.set_title(input('Enter a title for graph of contacts vs time: '))

    
    
    
    
    
    
