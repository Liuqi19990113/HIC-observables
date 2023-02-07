#!/usr/bin/python3
#@author: LiuQi 2023/1/2
'''This program is the used to sort centrality by raw hdf5 file
'''

#Import module
import hic_flow
import numpy as np
import multiprocessing as mpi
import h5py


#Variables definition
centrality_interval = np.array([0, 5,10, 20, 30, 40, 50, 60, 70, 80])
eta_cut = np.array([-1.3, 1.3])


#Function definition
def mult_dic(file_path: str) -> dict:
    '''This function is used to get multiplicity dict from
    raw hdf5 file. (filepath*hydroevent:multiplicity)

    Args:
        file_path_list:
            The list of path of hdf5 raw file.
    Return:
        this_dic
            The dictionary where key = hydro_name and value = mult
    '''
    this_dic = {}
    with h5py.File(file_path, 'r') as f:
        for event in f.keys():  #one f[event] is one hydro
            if f[event]['sample'].size == 0:
                continue  #Skip null events.
            nsample = f[event]['sample'][-1]  #The number of oversample
            charge = f[event]['charge']
            phi = f[event]['phi']
            eta = f[event]['eta']
            pt = f[event]['pT']  
            mult_ref = (phi[(charge != 0) & (eta > -1.3) & 
                        (eta < 1.3) &(pt > 0.15) &(pt <2 ) ] ).size/nsample  #You can change the condition.
            hydro_mult_key = file_path + '*' + event
            key_value = {hydro_mult_key: mult_ref}
            this_dic.update(key_value)
    return this_dic


def centrality_sort(hydro_mult_dic: dict, centrality_interval: np.ndarray) -> list:
    '''This function is used to get centrality dict from
    raw hdf5 file. (filepath*hydroevent:centrality interval)

    Args:
        hydro_mult_dic:
            Dictionary of hydro_name*event_number:mult
    Return:
        central_list:
            The list of file name list corrspond to centrality.
    '''
    print("Begin centrality sorting...")
    central_list = []
    hydro_mult_tuple_list = [(value1, key1) for key1, value1 
                             in hydro_mult_dic.items()]  #(mult, hydro_name)
    hydro_mult_tuple_list_sorted = sorted(hydro_mult_tuple_list, reverse=False)
    event_number = len(hydro_mult_tuple_list_sorted)
    cut_order_list = [int(np.percentile(range(0, event_number), 100 - percentile))
                      for percentile in centrality_interval]
    mult_cut = [hydro_mult_tuple_list_sorted[thing][0] for thing in cut_order_list]
    print("multiplicity cut list is {}".format(mult_cut))
    for i in range(0, len(cut_order_list) - 1):
        print('Centrality sorting in class {}'.format(i + 1))
        tmp_list = []
        mult_list = np.array([])
        high_cut = cut_order_list[i]
        low_cut = cut_order_list[i + 1]
        for j in range(low_cut, high_cut):
            mult_list = np.append(mult_list, hydro_mult_tuple_list_sorted[j][0])
            tmp_list.append(hydro_mult_tuple_list_sorted[j][1])
        print("In centrality {}, mult = {} and std error = {}".format(i+1,np.mean(mult_list),np.std(mult_list)))
        central_list.append(tmp_list)
    event_number = [len(thing) for thing in central_list]
    print("hydro number in each bin: {}".format(event_number))
    print('Centrality sorted!')
    return central_list


def return_somethings(obs_name: str, hdf5file_event_list: list) -> list:
    '''This function is used to get observables from
    one hydro*event list.

    Args:
        obs_name:
            A string like 'phi'
        hdf5_event_list:
            A string list whose elements is "hdf5_path*event"
    Return:
        elements.
    '''
    obs_ref_list_list =[]
    for hdf5file_event in hdf5file_event_list:
        hdf5_path, event_name = hdf5file_event.split('*')
        with h5py.File(hdf5_path, 'r') as f:
            nsample = f[event_name]['sample'][-1]
            sample = f[event_name]['sample']
            obs = f[event_name][obs_name]
            charge = f[event_name]['charge']
            eta = f[event_name]['eta']
            pt = f[event_name]['pT'] 
            for i in range(1, nsample+1): 
                obs_ref = (obs[(sample == i) & (charge != 0) & (eta > eta_cut[0]) & 
                            (eta < eta_cut[1]) & (pt > 0.15) & (pt <2 )])  #You can change the condition.
                obs_ref_list_list.append(obs_ref)
    return obs_ref_list_list           
            
            
    






    
    

   




