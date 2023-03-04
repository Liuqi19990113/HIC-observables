import numpy as np
from multiprocessing.pool import Pool
import hic_centrality
import h5py


def dNdeta(hdf5file_event_list):
    mult_list = np.array([])
    for hdf5file_event in hdf5file_event_list:
        hdf5_path, event_name = hdf5file_event.split('*')
        with h5py.File(hdf5_path, 'r') as f:
            charge = f[event_name]['charge']
            phi = f[event_name]['phi']
            eta = f[event_name]['eta']
            pt = f[event_name]['pT']
            nsample = f[event_name]['sample'][-1]  
            mult_ref = (phi[(charge != 0) & (eta > -0.5) & 
                        (eta < 0.5)  ] ).size/(nsample)  #You can change the condition.
            mult_list = np.append(mult_list, mult_ref)
    return np.mean(mult_list), np.std(mult_list)



def get_dNdeta(hdf5_files_list):
    all_hydro_mult_dic = {}
    mult_meanandstdtuple_list = []
    mult_list = []
    std = []
    for hdf5_file in hdf5_files_list:
        this_hdf5_dic = hic_centrality.mult_dic(hdf5_file)  #A hdf5_dic contains many hydro: mult key to value.
        all_hydro_mult_dic.update(this_hdf5_dic)
    central_hydroevent_list_list = hic_centrality.centrality_sort(
                    all_hydro_mult_dic, hic_centrality.centrality_interval)
    with Pool() as pool:
        mult_meanandstdtuple_list = pool.map(dNdeta, central_hydroevent_list_list)
        pool.close()
        pool.join()
    print(mult_meanandstdtuple_list)
    for thing in mult_meanandstdtuple_list:
        mult_list.append(thing[0])
        std.append(thing[1])
    return np.array(mult_list), np.array(std)


    

