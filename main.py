import numpy as np
import func
import time
import os
from func import Map, record, write_file
from benchmark import DOT, PLM, ILP, MIP_LR, other_iterations
from GP4 import GP4_iterations

'''
Note: The indexes of nodes and links start from 1 when being set or displayed, but start from 0 when stored and calculated.
'''

curr_model = "G" #model can be "G", "log", or "bi"
is_posterior = False #if True, the performance of the algorithm is evaluated in terms of its posterior probability, which is the same as how GP4 is evaluated.
S = 200
N = 150
MaxIter = 100
DOT_delta = 0.005

test_name = curr_model + " Simple"
curr_dir = os.getcwd()
map_dir = curr_dir + '/Networks/'
map_id = 0 #map_id can be integers from 0~7

file_name = test_name+'.txt' #results are stored in this file
fp = open(file_name, 'a+')
fp.write("model={}\n".format(test_name))
fp.write("S={}\n".format(S))
fp.write("N={}\n".format(N))
fp.write("MaxIter={}\n".format(MaxIter))
fp.write("DOT_delta={}\n".format(DOT_delta))
fp.write("\n")
fp.close()

mymap = Map()

######---Use the function below to generate map used in Experiment A---######
mymap.generate_simple_map(curr_model)
######---Use the function below to generate map used in Experiment B-D---######
# mymap.generate_real_map(map_id, map_dir)

T_factors = np.arange(0.7, 1.21, 0.01)

OD_pairs = [[1, 3]]

for OD in OD_pairs:
    write_file("OD", OD, file_name)
    write_file("============", "=============", file_name)
    print("OD={}".format(OD))

    mymap.update_OD(OD)

    for tf in T_factors:
        T = tf * mymap.dij_cost

        write_file("T", T, file_name)
        write_file("=========", "==========", file_name)
        print("T={}".format(T))

        ##############################-----------DOT-----------###################################################
        DOT_Solver = DOT(mymap, T, DOT_delta)
        res_DOT = DOT_Solver.get_DOT_prob(T), 0, 0, DOT_Solver.DOT_t_delta
        record("DOT", res_DOT, file_name)
        ##############################----------Pruning---------##################################################
        res_PAL = DOT_Solver.PA(T, N, S, is_posterior), 0, 0, DOT_Solver.PA_t_delta
        record("Pruning", res_PAL, file_name)
        ##############################-----------ILP-----------###################################################
        res_ILP = other_iterations(ILP, mymap, T, N, S, MaxIter, is_posterior)
        record("ILP", res_ILP, file_name)
        ##############################-----------OS-MIP-----------################################################
        res_MLR = other_iterations(MIP_LR, mymap, T, N, S, MaxIter, is_posterior)
        record("MLR", res_MLR, file_name)
        ##############################-----------PLM-----------####################################################
        res_PLM = other_iterations(PLM, mymap, T, N, S, MaxIter, is_posterior)
        record("PLM", res_PLM, file_name)
        ##############################-----------GP4-----------####################################################
        res_GP4 = GP4_iterations(mymap, T, N, S, MaxIter)
        record("GP4", res_GP4, file_name)

        fp = open(file_name, 'a+')
        fp.write("\n\n")
        fp.close()
