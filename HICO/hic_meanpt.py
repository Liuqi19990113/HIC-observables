#@author: LiuQi 2023/2/6
'''This program is the used to calculate mean pt
'''

#Import module
import numpy as np
import multiprocessing as mpi
import h5py

#Function definition
def pt_one_hydro(hydro_path: str, PID: int) -> list:
    '''This function is used to get pt of PID particle in one
    hydro event

    Args:
        hydro_path:
            Path of hydro
        PID:
            Particle id of PDG
    Return:
        pT_array:
            Pt array
    '''
    pT_array = np.array([])
    hdf5_path, event_name = hydro_path.split('*')
    with h5py.File(hdf5_path, 'r') as f:
        nsample = f[event_name]['sample'][-1]
        sample = f[event_name]['sample']
        pT = f[event_name]['pT']
        y = f[event_name]['y']
        Pid = f[event_name]['ID']
        for i in range(1, nsample+1): 
            pT_ref = (pT[(sample == i) & (y > -0.1) & 
                        (y < 0.1) & (Pid == PID)])  #You can change the condition.
            pT_array = np.append(pT_array, pT_ref)
    return np.mean(pT_array)


def meanpT(hydro_path_list: str, PID: int) -> tuple:
    '''This function is used to get mean pT and std error

    Args:
        hydro_path_list:
            list of path of hydro.
        PID:
            Particle id of PDG.
    Return:
        A tuple:
            mean pT and std err.
    '''
    meanpT_array = np.array([])
    for hydro_path in hydro_path_list:
        meanpT_array = np.append(meanpT_array, pt_one_hydro(hydro_path, PID))
    meanpT = np.mean(meanpT_array)
    stderrpT = np.std(meanpT_array)
    return (meanpT, stderrpT)

#all charged meanpT 
def all_charged_pt_one_hydro(hydro_path: str) -> list:
    '''This function is used to get pt of PID particle in one
    hydro event

    Args:
        hydro_path:
            Path of hydro
        PID:
            Particle id of PDG
    Return:
        pT_array:
            Pt array
    '''
    pT_array = np.array([])
    hdf5_path, event_name = hydro_path.split('*')
    with h5py.File(hdf5_path, 'r') as f:
        nsample = f[event_name]['sample'][-1]
        sample = f[event_name]['sample']
        pT = f[event_name]['pT']
        y = f[event_name]['y']
        charge = f[event_name]['charge']
        for i in range(1, nsample+1): 
            pT_ref = (pT[(charge != 0) &(sample == i) & (y > -0.1) & 
                        (y < 0.1)])  #You can change the condition.
            pT_array = np.append(pT_array, pT_ref)
    return np.mean(pT_array)


def all_charged_meanpT(hydro_path_list: str) -> tuple:
    '''This function is used to get mean pT and std error

    Args:
        hydro_path_list:
            list of path of hydro.
        PID:
            Particle id of PDG.
    Return:
        A tuple:
            mean pT and std err.
    '''
    meanpT_array = np.array([])
    for hydro_path in hydro_path_list:
        meanpT_array = np.append(meanpT_array, all_charged_pt_one_hydro(hydro_path))
    meanpT = np.mean(meanpT_array)
    stderrpT = np.std(meanpT_array)
    return (meanpT, stderrpT)


