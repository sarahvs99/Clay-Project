import MDAnalysis as mda
from MDAnalysis.topology import tpr
from MDAnalysis.topology import TPRParser
from MDAnalysis.analysis import lineardensity as lin
from MDAnalysis.analysis import rdf
from MDAnalysis.analysis import msd
from MDAnalysis.analysis import rms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# Atomic density - run for cation, Gly-N and Gly-OT*

def position_density(u, atoms):
    '''Creates dataframes of the mass-weighted position density of an AtomGroup along z
    
    Parameters
    -----------
    u: universe
    
    atoms: atom selection used to create AtomGroup
    
    Returns
    --------
    .csv file of the density, std and normalised density'''
    
    # Generate minimum and maximum coordinates
    at_group=mda.AtomGroup(u.select_atoms(atoms, updating=True)) 
    
    atom_density=lin.LinearDensity(at_group, binsize=0.25).run()
    
    # Create dataframe
    z_pos_dens=pd.DataFrame(atom_density.results.z.pos, columns=['pos'])
    
    # Create coordinate column info
    z_index=pd.Index(np.linspace(0, 139.0, atom_density.nbins))
    
    # Add coordinate column to dataframes
    z_pos_dens=z_pos_dens.set_index(z_index)
    z_pos_dens.reset_index(inplace=True)
    
    z_pos_dens['std of position density']=atom_density.results.z.pos_std
    
    # Rename columns
    z_pos_dens=z_pos_dens.rename(columns = {'index':'z-coordinate', 'pos':'position density'})
    
    z_pos_np=z_pos_dens.to_numpy()
    
    z_pos_dens['normalised density']=(atom_density.results.z.pos - np.min(atom_density.results.z.pos))/(np.max(atom_density.results.z.pos) - np.min(atom_density.results.z.pos))

    file_name=input('Enter the filename for z density. Must end in .csv : ')
    np.savetxt(file_name, z_pos_np, delimiter=' ', fmt='%f', header='pos pos_std norm_dens')

# RDF - run average_rdf for cation to Gly-OT1/2

def getcoord(rdf, r, dens):
    '''Calculates coordination number'''
    
    imax=rdf.argmax()
    imin=imax + rdf[imax:].argmin()
    
    dr = r[1] - r[0]
    
    integral = 0.0
    for i in range(imin):
        integral += 4 * np.pi * float(r[i]) * float(r[i]) * float(dr) * float(rdf[i])
    print(f'the coordination number is {float(integral) * dens}')

def average_rdf(atomgroup1, atomgroup2):
    '''Calculates the average radial distribution function within 8.0 Å for two AtomGroups and their first maximum coordination number. 
    If a second maximum is required, imin and imax in getcoord need altered.
    
    Parameters
    -----------
    atomgroup1: first AtomGroup
    
    atomgroup2: second AtomGroup
    
    Returns
    --------
    .csv file of rdf and prints the coordination number for first maximum'''
    
    av_rdf=rdf.InterRDF(atomgroup1, atomgroup2, 
                        nbins=75,
                        range=(0.0, 8.0)).run()
    
    
    df=pd.DataFrame(av_rdf.results.bins)
    df['rdf']=av_rdf.results.rdf
    array=df.to_numpy()
    file_name1=input('Enter the filename. Must end in .csv : ')
    np.savetxt(file_name1, array, delimiter=' ', fmt='%f', header='av_rdf')
    
    dens=float(input('What is the density of one of the atom groups?'))
    
    getcoord(av_rdf.results.rdf, av_rdf.results.bins, dens)
    
# MSD - run for Gly

def mean_sq_disp(u, selection):
    '''Calculates and plots the mean-squared displacement using the Einstein relation.
    WARNING: This function only works in MDAnalysis version 2.0.0 and higher
    
    Parameters
    -----------
    u: universe or (non-updating)AtomGroup
    
    selection: atom selection to calculate the msd for. Default is all
    
    Returns
    --------
    .csv file of the MSD
    '''
    
    MSD = msd.EinsteinMSD(u, select=selection, msd_type=z, fft=False).run()
    
    nframes = MSD.n_frames
    print(' ')
    print('To calculate the time between frames use: dt x nstxout (from .mdp)')
    timestep=float(input('What is the time between frames? Give answer in nanoseconds: '))
    sim_time = np.arange(nframes)*timestep
    
    #create df to save
    df=pd.DataFrame(sim_time)
    df['msd']=MSD.results.timeseries
    array=df.to_numpy()
    file_name1=input('Enter the filename. Must end in .csv : ')
    np.savetxt(file_name1, array, delimiter=' ', fmt='%f', header='msd')

# RMSD - run for Gly

def trajectory_rmsd(u, atom_sel):
    '''Calculates the Root Mean-Squared Deviation of an entire trajectory to a single reference point. 
    Automatically performs a rotational and translational alignment of the target trajectory to the reference universe
    Warning: If using MDAnalysis version >=2.0.0 a depreciation warning will be printed during the use of the function
    
    Parameters
    -----------
    u: universe to calculate RMSD for
    
    atom_sel: atom selection to superimpose
              Can be a selection string for select_atoms(), a dictiomary of the form {'mobile': sel1, 'reference': sel2}, or a tuple of the form (sel1, sel2)
              sel1 and sel2 are valid selection strings that are applied to u1 and u2 respectively and must generate groups of equivalent atoms
              
    Returns
    --------
    .csv file of RMSD
    '''
    
    trj_rmsd=rms.RMSD(u, u, select=atom_sel, groupselections=additional, ref_frame=0).run()
    
    selection=input('Enter a name for the atom_sel selection. This will be used as the column heading in the dataframe and the key on the graph of RMSD: ')
    
    cols = ['Frame', 'Time (ps)']
    cols.append(selection)
      
    rmsd_df=pd.DataFrame(trj_rmsd.results.rmsd, columns=cols)
    rmsd_np=rmsd_df.to_numpy()
    file_name1=input('Enter the filename. Must end in .csv : ')
    np.savetxt(file_name1, rmsd_np, delimiter=' ', fmt='%f', header=selection)
