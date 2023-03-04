import numpy as np
import h5py
from multiprocessing.pool import Pool

mult_bin_number = 100

def get_Nch_distribution(hdf5_file):
    mult_array = np.array([])
    with h5py.File(hdf5_file, 'r') as f:
        for ev in f:
            if f[ev]['sample'].size == 0:
                continue  #Skip null events.
            nsample = f[ev]['sample'][-1]
            sample = f[ev]['sample']
            pT = f[ev]['pT']
            charge = f[ev]['charge']
            phi = f[ev]['phi']
            eta = f[ev]['eta']
            for i in range(1, nsample+1):
                this_sample_mult = (phi[(charge != 0) & (eta > -0.8) & (eta < 0.8) & (sample == i) ] ).size
                mult_array = np.append(mult_array,this_sample_mult)
    mult_max = np.max(mult_array)
    return mult_array, mult_max

def distribution(mult_array, mult_max):
    bin_step = 100/mult_bin_number
    norm_mult_count = np.zeros(mult_bin_number)
    norm_mult_array = 100*mult_array/(mult_max + 1e-3)
    for norm_mult in norm_mult_array:
        this_bin = int(np.floor(norm_mult/bin_step))
        norm_mult_count[this_bin] = norm_mult_count[this_bin] + 1
    return norm_mult_count

def run(hdf_file_list):
    results_list = []
    with Pool() as pool:
        results_list = pool.map(get_Nch_distribution, hdf_file_list)
        pool.close()
        pool.join()
    all_mult = np.array([])
    all_max = np.array([])
    for result in results_list:
        all_mult = np.append(result[0])
        all_max = np.append(result[1])
    max_max = np.max(all_max)
    mult_distribution = distribution(all_mult, max_max)
    return mult_distribution



