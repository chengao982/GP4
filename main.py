import numpy as np
import func
import time
import os
from func import Map, record, write_file
from benchmark import DOT, PLM, MIP_CPLEX, MIP_LR, other_iterations
from evaluation import calc_post_prob_DOT, calc_post_prob
from GP4 import GP4_iterations

'''
Note: The indexes of nodes and links start from 1 when being set or displayed, but start from 0 when stored and calculated.
'''

curr_model = "G" #model can be "G", "log", or "bi"
curr_method = "svd" #decompose method can be "svd", "eigh" or "cholesky"
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
fp.write("decom_method={}\n".format(curr_method))
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

    mymap.update_OD(OD, model=curr_model)

    for tf in T_factors:
        T = tf * mymap.dij_cost

        write_file("T", T, file_name)
        write_file("=========", "==========", file_name)
        print("T={}".format(T))

        ##############################-----------DOT-----------###################################################
        t1 = time.perf_counter()
        DOT_pro, J, U = DOT(mymap, T, DOT_delta, model=curr_model)
        t_delta = time.perf_counter() - t1
        res_DOT = DOT_pro, calc_post_prob_DOT(J,U,mymap,1,1,DOT_delta), 0, t_delta
        record("DOT", res_DOT, file_name)
        ##############################-----------OS-MIP_CPLEX-----------###########################################
        res_MCP = other_iterations(MIP_CPLEX, mymap, T, N, S, MaxIter, model=curr_model, decom_method=curr_method)
        record("MCP", res_MCP, file_name)
        ##############################-----------OS-MIP_LR-----------##############################################
        res_MLR = other_iterations(MIP_LR, mymap, T, N, S, MaxIter, model=curr_model, decom_method=curr_method)
        record("MLR", res_MLR, file_name)
        ##############################-----------PLM-----------####################################################
        res_PLM = other_iterations(PLM, mymap, T, N, S, MaxIter, model=curr_model, decom_method=curr_method)
        record("PLM", res_PLM, file_name)
        ##############################-----------GP4-----------####################################################
        res_GP4 = GP4_iterations(mymap, T, N, S, MaxIter, model=curr_model, decom_method=curr_method)
        record("GP4", res_GP4, file_name)

        fp = open(file_name, 'a+')
        fp.write("\n\n")
        fp.close()
