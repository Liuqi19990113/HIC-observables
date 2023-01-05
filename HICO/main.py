#author@ Liuqi 2023/1/4
'''This python program file is main process used to calculate flow.
'''


#Import module
import numpy as np
import hic_flow
import hic_centrality
import os
from multiprocessing.pool import Pool
import time


#Function definition
def give_hydroevent_return_avea_and_mult(para_list: list):
    '''This function is designed for run mpi. 
    '''
    hydroevent = [para_list[0]]
    n = para_list[1]
    k = para_list[2]
    eveny_by_event_phi_array = hic_centrality.return_somethings('phi', hydroevent)
    mult_array = [len(phi_array) for phi_array in eveny_by_event_phi_array]
    single_ave_array_tuple = hic_flow.single_ave_array_outer_mpi(eveny_by_event_phi_array, mult_array, n, k)
    return (single_ave_array_tuple, mult_array)


def run_vn_serial(hdf5_files_list: list, n: int, k:int):
    time_start = time.time()
    '''This function is used to calculate vnk vs. centrality.

    Args:
        hdf5_files_list:
            A list of raw hdf5 file to calculate
        n: 
            n of vnk
        k:
            k of vnk
    Return:
        A list of tuple where tuple is (vnk, vnk_error).
    '''
    vnk_list = []
    vnk_error_list = []
    all_hydro_mult_dic = {}
    for hdf5_file in hdf5_files_list:
        this_hdf5_dic = hic_centrality.mult_dic(hdf5_file)  #A hdf5_dic contains many hydro: mult key to value.
        all_hydro_mult_dic.update(this_hdf5_dic)
    central_hydroevent_list_list = hic_centrality.centrality_sort(
                    all_hydro_mult_dic, hic_centrality.centrality_interval)
    for hydroevent_list in central_hydroevent_list_list:
        all_event_phi_array = hic_centrality.return_somethings('phi', hydroevent_list)
        mult_array = [len(phi_array) for phi_array in all_event_phi_array]
        my_cumu = hic_flow.CumulantsAndFlows(all_event_phi_array, mult_array)
        my_vnk = my_cumu.vn_k(n, k)
        my_vnk_error = my_cumu.error_vn_k(n, k)
        vnk_list.append(my_vnk)
        vnk_error_list.append(my_vnk_error)
    time_end = time.time()
    print("In serial run v{}{} is {}".format(n, k, vnk_list))
    print("In serial run v{}{} error is {}".format(n, k, vnk_error_list))
    print("run time: {}".format(time_end - time_start))
        

def run_vn_parallel(hdf5_files_list: list, n: int, k:int):
    time_start = time.time()
    '''This function is used to calculate vnk vs. centrality.

    Args:
        hdf5_files_list:
            A list of raw hdf5 file to calculate
        n: 
            n of vnk
        k:
            k of vnk
    Return:
        A list of tuple where tuple is (vnk, vnk_error).
    '''
    vnk_list = []
    vnk_error_list = []
    all_hydro_mult_dic = {}
    hydro_mult_dict_list = []
    with Pool() as pool:
        hydro_mult_dict_list = pool.map(hic_centrality.mult_dic, hdf5_files_list)
    pool.close()
    pool.join()
    for hydro_mult_dict in hydro_mult_dict_list:
        all_hydro_mult_dic.update(hydro_mult_dict)
    central_hydroevent_list_list = hic_centrality.centrality_sort(
                    all_hydro_mult_dic, hic_centrality.centrality_interval)
    for hydroevent_list in central_hydroevent_list_list:
        para_list_list = [[hydroevent, n, k] for hydroevent in hydroevent_list]
        all_event_single_ave_array = [[] for i in range(0, int(k/2))]
        all_event_mult_array = []
        avearray_multarray_tuple_list = []
        with Pool() as pool:
            avearray_multarray_tuple_list = pool.map(give_hydroevent_return_avea_and_mult, para_list_list)
        pool.close()
        pool.join()
        for avearray_multarray_tuple in avearray_multarray_tuple_list:
            all_event_mult_array += avearray_multarray_tuple[1]
            if k != 2:
                for j in range(0, int(k/2)):
                    all_event_single_ave_array[j] += list(avearray_multarray_tuple[0][j])
            else:
                all_event_single_ave_array[0] += list(avearray_multarray_tuple[0])
        my_mpi_cumu = hic_flow.CumulantsAndFlows_mpi(all_event_single_ave_array, all_event_mult_array)
        my_mpi_vnk = my_mpi_cumu.vn_k(n, k)
        my_mpi_vnk_error = my_mpi_cumu.error_vn_k(n, k)
        vnk_list.append(my_mpi_vnk)
        vnk_error_list.append(my_mpi_vnk_error)
    time_end = time.time()
    print("In parallel run v{}{} is {}".format(n, k, vnk_list))
    print("In parallel run v{}{} error is {}".format(n, k, vnk_error_list))
    print("run time: {}".format(time_end - time_start))


#go!
test1 = 'C:\\Users\\LiuQi\\Desktop\\test1'
test2 = 'C:\\Users\\LiuQi\\Desktop\\test2'
if __name__ == "__main__":
    run_vn_serial([test1, test2], 2, 2)
    run_vn_parallel([test1, test2], 2, 2)








