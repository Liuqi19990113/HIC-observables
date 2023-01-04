#author@ Liuqi 2023/1/4
'''This python program file is main process used to calculate flow.
'''


#Import module
import numpy as np
import hic_flow
import hic_centrality
import os


#Function definition
def run_vn(hdf5_files_list: list, n: int, k:int) -> list:
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
    all_hydro_mult_dic = {}
    for hdf5_file in hdf5_files_list:
        this_hdf5_dic = hic_centrality.mult_dic(hdf5_file)  #A hdf5_dic contains many hydro: mult key to value.
        all_hydro_mult_dic.update(this_hdf5_dic)
    central_hydroevent_list_list = hic_centrality.centrality_sort(all_hydro_mult_dic, hic_centrality.centrality_interval)
    for hydroevent_list in central_hydroevent_list_list:
        all_event_phi_array = hic_centrality.return_somethings('phi', hydroevent_list)
        mult_array = [len(phi_array) for phi_array in all_event_phi_array]
        my_cumu = hic_flow.CumulantsAndFlows(all_event_phi_array, mult_array)
        my_vnk = my_cumu.vn_k(n, k)
        my_vnk_error = my_cumu.error_vn_k(n, k)
        print(my_vnk, my_vnk_error)


#go!
test1 = 'C:\\Users\\LiuQi\\Desktop\\test1'
test2 = 'C:\\Users\\LiuQi\\Desktop\\test2'
run_vn([test1, test2], 2, 2)









