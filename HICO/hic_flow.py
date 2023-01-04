#@author: LiuQi 2023/1/1
'''This python program file is used to calculate flow observables by Q-cumulant. 
C_2{2}, C_2{4}, V_2{2}, V_2{4} and these error. The main reference article are: the
doctoral dissertation of Bilandzic. This program also take reference to jbernhard's
HIC package.
'''

# Import module
import numpy as np
import math
from hic import flow


# Function definition
def qn(phi_list: list, n: int) -> complex:
    '''This function is used to calculate the Q_n vector.

    Args:
        phi_list:
            A list of angle.
        n:
            The order 'n' of Q_n vector.
    Return:
        Return Q_n vector.
    '''
    phi_list = (np.array([phi_list])).astype(complex)
    qn_vector = np.exp(1j*n*phi_list).sum()
    return qn_vector


def eve_weight(mult: int, n: int) -> float:
    '''This function is used to calculate weight of single event.
    '''
    weight = 1.
    for i in range(mult - n + 1, mult + 1):
        i_float = float(i)
        weight = i_float*weight
    return weight
    

def single_ave2(qn: complex, mult: int) -> float:
    '''This function is used to calculate <2>.

    Args:
        qn:
            Q_n vector.
        mult:
            The multiplicity of this event.
    Return:
            <2>
    '''
    qn_conj = np.conjugate(qn)
    ave2_single_event = (((qn*qn_conj).real - mult)/(eve_weight(mult, 2)))
    return ave2_single_event 


def single_ave4(qn: complex, q2n: complex, mult: int) -> float:
    '''This function is used to calculate <4>. Args and return 
are looked like single_ave2.
    '''
    qn_conj = np.conjugate(qn)
    q2n_conj = np.conjugate(q2n)
    abs_qn_fp = (qn*qn*qn_conj*qn_conj).real
    abs_q2n_sq = (q2n_conj*q2n).real
    single_ave4_numerator = (abs_qn_fp + abs_q2n_sq - 2*(q2n*qn_conj*qn_conj).real - 
                4*(mult - 2)*(qn*qn_conj).real + 2*mult*(mult - 3))
    single_ave4_denominator = eve_weight(mult, 4)
    ave4_single_event = single_ave4_numerator/single_ave4_denominator
    return ave4_single_event


def single_ave6(qn: complex, q2n: complex, q3n: complex,mult: int) -> float:
    '''This function is used to calculate <4>. Args and return 
are looked like single_ave2, the calculation take reference of 
A10 of Phys. Rev. C 83, 044913.
    '''
    qn_conj = np.conjugate(qn)
    q2n_conj = np.conjugate(q2n)
    q3n_conj = np.conjugate(q3n)
    weight6 = eve_weight(mult, 6)
    part1 = (((qn*qn*qn*qn_conj*qn_conj*qn_conj).real + 
            9*(q2n*q2n_conj*qn*qn).real - 
            6*(q2n*qn*qn_conj*qn_conj*qn_conj).real)/(weight6))
    part2 = ((4*(q3n*qn_conj*qn_conj*qn_conj).real - 
            12*(q3n*q2n_conj*qn_conj).real)/(weight6))
    part3 = ((18*(mult - 4)*(q2n*qn_conj*qn_conj).real +
            4*(q3n*q3n_conj).real)/(weight6))
    part4 = (9*((qn*qn*qn_conj*qn_conj).real + 
            (q2n*q2n_conj).real))/(weight6*(mult - 4))
    mult_float = float(mult)  #make mult transfer to float in case of overflow.
    part5 = 18*(qn*qn_conj).real/(mult_float*(mult_float-1)*\
            (mult_float-3)*(mult_float-4))
    part6 = 6/((mult_float-1)*(mult_float-2)*(mult_float-3))
    ave6_single_event = (part1 + part2 + part3 - part4 + part5 - part6)
    return ave6_single_event


def all_event_ave(weight_list: list, single_ave_list: list) -> float:
    '''This function is used to calculate <<n>>.

    Args:
        weight_list:
            List of weight.
        single_ave_list:
            List of single event <n> result which corresponds to weight_list. 
    Return:
        <<n>>.
    '''
    all_event_ave_numerator = (weight_list*single_ave_list).sum()
    all_event_ave_denominator = weight_list.sum()
    ave_all_events = all_event_ave_numerator/all_event_ave_denominator
    return ave_all_events


def weighted_stand_error_sq(weight_list: np.ndarray, variable_list: np.ndarray, 
                    ave: float) -> float:
    '''This function is used to calculate the stand error square with weight,
    take reference of doctoral dissertation of Bilandzic equation(c.3).
        
    Args:
        weight_list:
            List of weight.
        variable_list:
            List of random variable.
        ave:
            Mean vaule of variable.
    Return:
        weighted_sq_err:
            Weighted square stand error.
    '''
    weight_list_sum = weight_list.sum()
    weight_list_sum_sq = weight_list.sum()*weight_list.sum()
    weight_list_sq_sum = (weight_list*weight_list).sum()
    weighted_sq_err = ((weight_list*np.power((variable_list - ave), 2)).sum()/(weight_list_sum*
                        (1 - weight_list_sq_sum/weight_list_sum_sq)))
    return weighted_sq_err


def weighted_covariance(variable1_list: np.ndarray, weight1_list: np.ndarray,
                        variable2_list: np.ndarray, weight2_list: np.ndarray) -> float:
    '''This function is used to calculate the covariance with weight,
    take reference of doctoral dissertation of Bilandzic equation(c.12).
        
    Args:
        weight_list:
            List of weight.
        variable_list:
            List of random variable.
    Return:
        covariance:
            Weighted covariance.
    '''
    numerator1 = ((variable1_list*weight1_list*variable2_list*weight2_list).sum()/
                    (weight1_list*weight2_list).sum())
    numerator2 = (((weight1_list*variable1_list).sum()/weight1_list.sum())
                    *((weight2_list*variable2_list).sum()/weight2_list.sum()))
    dominator = 1 - (weight1_list*weight2_list).sum()/(weight1_list.sum()*weight2_list.sum())
    covariance = (numerator1 - numerator2)/dominator
    return covariance 
    
    
# Class definition
class CumulantsAndFlows():
    '''This class use the function above to calculate the
    cumulants and flow vector and these error in No subevent.
    You need to initialize it by two dimension phi array and
    multiplicity array.
    '''

    def __init__(self, all_event_phi_array: list, mult_array: list):
        '''Initialize two dimension phi array of all event and
    multiplicity array 
        '''
        self.two_d_phi_array = all_event_phi_array
        self.multiplicity_array = mult_array
        self.weight2_array = np.array([eve_weight(mult, 2) for mult 
                                       in self.multiplicity_array])
        self.weight4_array = np.array([eve_weight(mult, 4) for mult 
                                       in self.multiplicity_array])
        self.weight6_array = np.array([eve_weight(mult, 6) for mult 
                                       in self.multiplicity_array])


    def single_ave_array_outer(self, n: int, k: int) -> tuple:
        '''Calculate <k> array with k = 2, 4, 6.

        Args:
            n:
                Order of Vn vector
            k:
                Number of correlate particles
        Return:
            A tuple of single event <k> array. 
        '''
        if k == 2:
            qn_array = np.array([qn(phi_list, n) for phi_list 
                                in self.two_d_phi_array])
            single_ave2_array = np.ravel(np.array([single_ave2(qn_array[i], 
                                         self.multiplicity_array[i]) for i in range(0,len(qn_array))]))
            return single_ave2_array
        elif k == 4:
            qn_array = np.array([qn(phi_list, n) for phi_list 
                                in self.two_d_phi_array])
            single_ave2_array = np.ravel(np.array([single_ave2(qn_array[i], 
                                         self.multiplicity_array[i]) for i in range(0,len(qn_array))]))
            q2n_array = np.array([qn(phi_list, 2*n) for phi_list
                                in self.two_d_phi_array])
            single_ave4_array = np.ravel(np.array([single_ave4(qn_array[i], q2n_array[i], 
                                        self.multiplicity_array[i]) for i in range(0,len(q2n_array))]))
            return single_ave2_array, single_ave4_array
        elif k == 6:
            qn_array = np.array([qn(phi_list, n) for phi_list 
                                in self.two_d_phi_array])
            single_ave2_array = np.ravel(np.array([single_ave2(qn_array[i], 
                                         self.multiplicity_array[i]) for i in range(0,len(qn_array))]))
            q2n_array = np.array([qn(phi_list, 2*n) for phi_list
                                in self.two_d_phi_array])
            single_ave4_array = np.ravel(np.array([single_ave4(qn_array[i], q2n_array[i], 
                                        self.multiplicity_array[i]) for i in range(0,len(q2n_array))]))
            q3n_array = np.array([qn(phi_list, 3*n) for phi_list
                                in self.two_d_phi_array])
            single_ave6_array = np.ravel(np.array([single_ave6(qn_array[i], q2n_array[i], q3n_array[i], 
                                        self.multiplicity_array[i]) for i in range(0,len(q3n_array))]))
            return single_ave2_array, single_ave4_array, single_ave6_array


    def cn_k(self, n: int, k: int) -> float:
        '''Calculate cn_{k} with k = 2, 4, 6
        s<k>

        Args:
            n:
                Order of Vn vector
            k:
                Number of correlate particles
        Return:
            cn{k}
        '''
        #cn{2}
        if k == 2:
            single_ave2_array = self.single_ave_array_outer(n, k)
            all_event_ave2 = all_event_ave(self.weight2_array, 
                                           single_ave2_array)
            cn_2 = all_event_ave2
            return cn_2
        #cn{4}
        elif k == 4:
            single_ave2_array, single_ave4_array = self.single_ave_array_outer(n, k)
            all_event_ave2 = all_event_ave(self.weight2_array, 
                                           single_ave2_array)
            all_event_ave4 = all_event_ave(self.weight4_array,
                                           single_ave4_array)
            cn_4 = all_event_ave4 - 2*all_event_ave2*all_event_ave2
            
            return cn_4
        #cn{6}
        elif k == 6:
            single_ave2_array, single_ave4_array, single_ave6_array = \
                self.single_ave_array_outer(n, k)
            all_event_ave2 = all_event_ave(self.weight2_array,
                                           single_ave2_array)
            all_event_ave4 = all_event_ave(self.weight4_array,
                                           single_ave4_array)
            all_event_ave6 = all_event_ave(self.weight6_array,
                                           single_ave6_array)
            cn_6 = (6*all_event_ave6 - 9*all_event_ave2*all_event_ave4 + 
                    12*all_event_ave2*all_event_ave2*all_event_ave2)
            return cn_6
        else:
            print("Error! No such k.")


    def vn_k(self, n: int, k: int) -> float:
        '''Calculate vnk by cnk. n k same to the def in cnk
        '''
        print('Calculating V{}{}'.format(n, k))
        #vn{2}
        if k == 2:
            cn2 = self.cn_k(n, k)
            if cn2 < 0:
                print('v{}{} is imag, let me show you c{}{}, c{}{} = {}'
                       .format(n, k, n, k, n, k, cn2))
                return np.nan
            else:
                vn2 = np.sqrt(cn2)
                return vn2
        #vn{4}
        elif k == 4:
            cn4 = self.cn_k(n, k)
            if cn4 > 0:
                print('v{}{} is imag, let me show you c{}{}, c{}{} = {}'
                       .format(n, k, n, k, n, k, cn4))
                return np.nan
            else:
                vn4 = np.power(-cn4, 0.25)
                return vn4
        #vn{6}
        elif k == 6:
            cn6 = self.cn_k(n, k)
            if cn6 < 0:
                print('v{}{} is imag, let me show you c{}{}, c{}{} = {}'
                       .format(n, k, n, k, n, k, cn6))
                return np.nan
            else:
                vn6 = np.power(0.25*cn6, 1/6)            
                return vn6
        else:
            print("Error! No such k.")
        print("done!")


    def error_vn_k(self, n: int, k: int) -> float:
        '''Output the stand error of vn_k. Take reference of 
        doctoral dissertation of Bilandzic equation(c.24), (c.28).
        The args is same to vn_k
        '''
        print('Calculating V{}{}_error'.format(n, k))
        if k == 2:
            single_ave2_array = self.single_ave_array_outer(n, k)
            s_2_sq = weighted_stand_error_sq(self.weight2_array, single_ave2_array, 
                                             np.mean(single_ave2_array))
            s_vn2_sq = ((self.weight2_array*self.weight2_array).sum()*s_2_sq/
                        (4*np.mean(single_ave2_array)*
                        np.power(self.weight2_array.sum(), 2)))
            if s_vn2_sq >= 0:
                return np.sqrt(s_vn2_sq)
            else:
                print('The stand error of v22 square < 0')
                return np.nan
        if k == 4:
            single_ave2_array, single_ave4_array = self.single_ave_array_outer(n, k)
            s_2_sq = weighted_stand_error_sq(self.weight2_array, single_ave2_array, 
                                            np.mean(single_ave2_array))
            s_4_sq = weighted_stand_error_sq(self.weight4_array, single_ave4_array, 
                                            np.mean(single_ave4_array))
            covar_24 = weighted_covariance(single_ave2_array, self.weight2_array, 
                                       single_ave4_array, self.weight4_array)
            s_vn4_sq_dom = np.power(2*np.mean(single_ave2_array)
                                    *np.mean(single_ave2_array)
                                    - np.mean(single_ave4_array), 3/2)
            s_vn4_sq_num1 = (s_2_sq*np.mean(single_ave2_array)
                             *np.mean(single_ave2_array)
                             *(self.weight2_array*self.weight2_array).sum()
                             /(self.weight2_array.sum()*self.weight2_array.sum()))
            s_vn4_sq_num2 = (((self.weight4_array*self.weight4_array).sum()
                               *s_4_sq)/(16*self.weight4_array.sum()*
                               self.weight4_array.sum()))
            s_vn4_sq_num3 = ((self.weight4_array*self.weight2_array).sum()
                             *np.mean(single_ave2_array)*covar_24
                             /(2*(self.weight4_array).sum()
                             *(self.weight2_array).sum()))
            s_vn4_sq = ((s_vn4_sq_num1 + s_vn4_sq_num2 - s_vn4_sq_num3)
                        /s_vn4_sq_dom)
            if s_vn4_sq >= 0:
                return np.sqrt(s_vn4_sq)
            else:
                print('The stand error of v22 square < 0')
                return np.nan            
        print("done!")

'''
#Let's test
import time
import cProfile
phi = np.random.uniform(-np.pi, np.pi, size=5000*10000).reshape(5000, 10000)
mult = np.array([p.size for p in phi])



my_cumu = CumulantsAndFlows(phi,mult)
my_v22 = cProfile.run('my_cumu.vn_k(2,2)')
my_v22_error = cProfile.run('my_cumu.error_vn_k(2,2)')
time_start_1 = time.time()
my_cumu = CumulantsAndFlows(phi,mult)
my_v22 = my_cumu.vn_k(2,2)
my_v22_error = my_cumu.error_vn_k(2,2)
print(my_v22, my_v22_error)
time_end_1 = time.time()
print(time_end_1 - time_start_1)

time_start_2 = time.time()
q2 = np.array([flow.qn(p, 2) for p in phi]).T
vnk = flow.Cumulant(mult, q2)
v22, v22_error = vnk.flow(2, 2, error=True)
print(v22)
time_end_2 = time.time()
print(time_end_2 - time_start_2)


my_cumu = CumulantsAndFlows(phi,mult)
#my_v24 = cProfile.run('my_cumu.vn_k(2,4)')
#my_v24_error = cProfile.run('my_cumu.error_vn_k(2,4)')
time_start_3 = time.time()
my_v24 = my_cumu.vn_k(2,4)
my_v24_error = my_cumu.error_vn_k(2,4)
print(my_v24, my_v24_error)
time_end_3 = time.time()
print(time_end_3 - time_start_3)

time_start_4 = time.time()
q2,q3,q4 = np.array([flow.qn(p, 2, 3, 4) for p in phi]).T
vnk = flow.Cumulant(mult, q2, q3, q4)
v24 = vnk.flow(2, 4)
print(v24)
time_end_4 = time.time()
print(time_end_4 - time_start_4)
'''













