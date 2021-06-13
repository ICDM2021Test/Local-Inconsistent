#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:53:24 2020

"""

from __future__ import print_function
import numpy as np
from Util import *

import utilM
import util_opt
from CLT_class import CLT
from CNET_class import CNET
from MIXTURE_CLT import  load_mt
import time
import copy
import JT

from cnet_extend import CNET_ARR  # The array version of cutset network
from cnet_extend import save_cnet
#import control_study

from scipy.optimize import minimize
from opt_clt_ss import objective, derivative

import sys




def pertub_model(model, model_type='cnet', percent=0.1):
    
    
    if model_type=='cnet':
        
        updated_cpt_list = []
        
        for j in range (len(model.path)):
      
        
            topo_order = model.leaf_info_list[j][2]
        
            updated_cpt = np.copy(model.leaf_cpt_list[j])
            peturb_no = int(np.round(topo_order.shape[0]* percent))
            
            rand_number = np.random.choice(topo_order.shape[0], size=peturb_no, replace=False)
        
            rand_decimal = np.random.rand(peturb_no, 2, 2)
            
     
            # make a valid cpt
            norm_const = np.sum(rand_decimal, axis = 1)
            
            rand_decimal[:,:,0] = rand_decimal[:,:,0]/norm_const[:,0, np.newaxis]
            rand_decimal[:,:,1] = rand_decimal[:,:,1]/norm_const[:,1, np.newaxis]
            
        
            root = topo_order[0]
            if root in rand_number:
                sum_val = rand_decimal[0,0,0]  + rand_decimal[0,1,1] 
                rand_decimal[0,0,0]  = rand_decimal[0,0,1] = rand_decimal[0,0,0]/sum_val
                rand_decimal[0,1,0]  = rand_decimal[0,1,1] = rand_decimal[0,1,1]/sum_val
                
                       
            updated_cpt[rand_number,:,:] = rand_decimal

            
            updated_cpt_list.append(updated_cpt)
            
        return updated_cpt_list


    

def convert_cnet_to_arr(cnet):
    main_dict = {}
    main_dict['depth'] = cnet.depth
    main_dict['n_variables'] =cnet.nvariables
    main_dict['structure'] = {}
    
    # save the cnet to the structure that can be stored later
    save_cnet(main_dict['structure'], cnet.tree, np.arange(cnet.nvariables))
    
   
    cnet_a = CNET_ARR(main_dict['n_variables'], main_dict['depth'])
    cnet_a.convert_to_arr_ccnet(main_dict['structure'])
    
    cnet_a.path = cnet_a.print_all_paths_to_leaf()
    return cnet_a





    



def check(node):
    '''find the leaf node''' 
    # internal nodes
    if isinstance(node,list):
        #print ('*** in internal nodes ***')
        id,x,p0,p1,node0,node1=node
        print ('id, x: ', id, x)
        
        check(node0)
        check(node1)
        
    else:
        print ('parents: ', node.parents)
        

def check_arr(cnet_Q_arr):
    print ('path:', cnet_Q_arr.path)
    for i in range (len(cnet_Q_arr.path)):
        print ('sub tree: ', cnet_Q_arr.leaf_info_list[i][1])
    


def objective_cnode(x, P_cnode, Q_cnode):
    lamda = x[0]
    theta = x[1:]
    
    first_part = lamda * np.sum(P_cnode*np.log(theta))
    second_part = (1-lamda) * np.sum(Q_cnode*np.log(theta))
    
    return -(first_part+second_part)


def derivative_cnode(x, P_cnode, Q_cnode):
    lamda = x[0]
    theta = x[1:]
    
    n_cnodes = int(theta.shape[0]/2)
    der_lam = 0
    #der_lam = np.sum(P_cnode*np.log(theta)) - np.sum(Q_cnode*np.log(theta))
    
    der_theta = np.zeros_like(theta)
    
    #der_theta = (lamda*(P_cnode)- (1-lamda)*Q_cnode)/theta
    der_theta[:n_cnodes] =  (lamda*(P_cnode[:n_cnodes])- (1-lamda)*Q_cnode[:n_cnodes])/theta[:n_cnodes]
    der_theta[n_cnodes:] = der_theta[:n_cnodes]*(-1)
    

    der = np.zeros_like(x)
    der[0] = der_lam
    der[1:] = der_theta
    
    return der *(-1.0)
    



def main_opt_cnet():

    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    mt_dir = sys.argv[6]    
    depth = int(sys.argv[8])
    perturb_rate = float(sys.argv[10])
    std = float(sys.argv[12])
    lam = float(sys.argv[14])
    
    noise_mu = 0
    noise_std = std
    noise_percent = 1
    noise_parm = np.array([noise_mu, noise_std, noise_percent])
    noise_flag = True # Assume get noise distribtuion from P
    
    
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    
    
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    n_variables = train_dataset.shape[1]
    
    
    '''
    ### Load the trained mixture of clt, consider as P
    '''
    print ('Start reloading MT...')
    reload_mix_clt = load_mt(mt_dir, data_name)
    
    # Set information for MT
    for t in reload_mix_clt.clt_list:
        t.nvariables = n_variables
        # learn the junction tree for each clt
        jt = JT.JunctionTree()
        jt.learn_structure(t.topo_order, t.parents, t.cond_cpt)
        reload_mix_clt.jt_list.append(jt)
    
    
    model_P = reload_mix_clt
    """
    construct the cnet
    """
    
    print("Learning Cutset Networks from partial training data.....")
    n_rec = train_dataset.shape[0]
    rand_record =  np.random.randint(n_rec, size=int(n_rec/10))    
    half_data = train_dataset[rand_record,:]
    
    '''
    Cutset Network Learn from dataset
    '''
    print ('-------------- Cutset Network Learn from Data: (Q) ----------')
    cnet_Q = CNET(depth=depth)
    cnet_Q.learnStructure(half_data)
    cnet_Q_arr = convert_cnet_to_arr(cnet_Q)
    
    
    perturb_leaf_cpt_list = pertub_model(cnet_Q_arr, model_type='cnet', percent=perturb_rate)
    
    for j in range (len(cnet_Q_arr.path)):
        cnet_Q_arr.leaf_cpt_list[j] = perturb_leaf_cpt_list[j]
    
    
    
    print ('-------------- Cutset Network Learn from P and Q: (R) ----------')
    cnet_R = copy.deepcopy(cnet_Q)
    cnet_R_arr = copy.deepcopy(cnet_Q_arr)

    
    
    '''
    Inference P to get list of marginals and pairwise marginals for
    each leaf tree in Q
    '''
    pair_marginal_P = []
    marginal_P = []
    
    for i in range (len(cnet_Q_arr.path)):
        path = cnet_Q_arr.path[i]
        

        evid_list =[]  # evidence list
        for var_sign in path:
            var = int(var_sign[:-1])
            sign = var_sign[-1]
           
            if sign == '-': # going to left
                
                '''add evidence to P'''
                evid_list.append([var,0])
                                
            
            elif sign == '+': # going to right
                
                '''add evidence to P'''
                evid_list.append([var,1])
              
        
        evid_arr = np.asarray(evid_list)
        ids = np.delete(np.arange(n_variables), np.sort(evid_arr[:,0]))

        #pxy, px = model_P.inference_jt([],ids)
        pxy, px = model_P.inference_jt(evid_list,ids)
        pair_marginal_P.append(pxy)
        marginal_P.append(px)
        
    
    # for weights assigned to cnode
    cnode_ind = np.where(cnet_R_arr.cnode_info[0] >= 0)[0]
    n_cnodes = cnode_ind.shape[0] # number of cnods
    
    marginal_P_cnodes = np.zeros((2,n_cnodes))
    
    # for each branch in cutset network
    for i in range (len(cnet_R_arr.path)):
        path = cnet_R_arr.path[i]
    
        P_temp = copy.deepcopy(model_P)
        
        var_ind = 0
        
        evid_list =[]  # evidence regarding to distribution P
        for var_sign in path:
            var = int(var_sign[:-1])
            sign = var_sign[-1]
            
           
            incremental_evid_list =[]  # evidence that increased in every depth
            if sign == '-': # going to left
                
                '''add evidence to P'''
                evid_list.append([var,0])
                incremental_evid_list.append([var,0])
                #P_temp.cond_cpt = P_temp.instantiation(evid_list)
                for k, t in enumerate(P_temp.clt_list):
                    t.cond_cpt = t.instantiation(incremental_evid_list)
                    
                
                                
                if marginal_P_cnodes[0, var_ind] == 0: # not calculated
                    
                    '''P_marginal should be calculated based on different distribution P'''
                    
                    P_marginal = 0
                    for k, t in enumerate(P_temp.clt_list):                    
                        P_marginal += utilM.ve_tree_bin(t.topo_order, t.parents, t.cond_cpt) * P_temp.mixture_weight[k]
                    
                    marginal_P_cnodes[0, var_ind] = P_marginal
                    
                
                var_ind = 2*var_ind+1
                
                
            
            if sign == '+': # going to right
                
                '''add evidence to P'''
                evid_list.append([var,1])
                incremental_evid_list.append([var,1])
                #P_temp.cond_cpt = P_temp.instantiation(evid_list)
                for k, t in enumerate(P_temp.clt_list):
                    t.cond_cpt = t.instantiation(incremental_evid_list)
                
                if marginal_P_cnodes[1, var_ind] == 0: # not calculated
                   
                    
                    '''P_marginal should be calculated based on different distribution P'''
                    P_marginal = 0
                    for k, t in enumerate(P_temp.clt_list):                    
                        P_marginal += utilM.ve_tree_bin(t.topo_order, t.parents, t.cond_cpt) * P_temp.mixture_weight[k]
                    #P_marginal = utilM.ve_tree_bin(P_temp.topo_order, P_temp.parents, P_temp.cond_cpt)
                    
                    marginal_P_cnodes[1, var_ind] = P_marginal
                    
                var_ind = 2*var_ind+2
            
    

    '''
    Add noise to pairwise marginals
    '''
    pair_marginal_P_noise = []
    for i in range (len(pair_marginal_P)):
        pair_marginal_P_noise.append( util_opt.add_noise (pair_marginal_P[i], n_variables-len(cnet_Q_arr.path[i]), noise_mu, noise_std, percent_noise=noise_percent))
        
    marginal_P_noise = marginal_P
    
    
    '''
    Update cnet R leaf parameters
    '''
    for j in range (len(cnet_Q_arr.path)):
        if noise_flag == True:
            '''apply noise to P''' # 2 is topo order, 1 is parent
            args = (cnet_Q_arr.leaf_info_list[j][2], cnet_Q_arr.leaf_info_list[j][1], cnet_Q_arr.leaf_cpt_list[j], marginal_P_noise[j], pair_marginal_P_noise[j])
        else:
            args = (cnet_Q_arr.leaf_info_list[j][2], cnet_Q_arr.leaf_info_list[j][1], cnet_Q_arr.leaf_cpt_list[j], marginal_P[j], pair_marginal_P[j])
        
        sub_nvariables = n_variables-len(cnet_Q_arr.path[j])
        # set the bound for all variables
        bnd = (0.001,0.999)
        bounds = [bnd,]*(4*sub_nvariables+1)
        
        x0 = np.zeros(4*sub_nvariables+1)
        x0[0] = 0.5  # initial value for lamda
        x0[1:] = cnet_R_arr.leaf_cpt_list[j].flatten()
        
        # constraint: valid prob
        normalize_cons = []
        for i in range (sub_nvariables):
    
            
            normalize_cons.append({'type': 'eq',
               'fun' : lambda x: np.array([x[i*4+1] + x[i*4+3] - 1, 
                                           x[i*4+2] + x[i*4+4] - 1])})
       
        
       
        
        #res = minimize(objective, x0, method='SLSQP', jac=derivative, constraints=normalize_cons,  # with normalization constriant
        res = minimize(objective, x0, method='SLSQP', jac=derivative, # without normalization constraint
                   options={'ftol': 1e-6, 'disp': True, 'maxiter': 1000},
                   bounds=bounds, args = args)
        
        cnet_R_arr.leaf_cpt_list[j] = res.x[1:].reshape(sub_nvariables,2,2)
    
    
    
    '''
    Update cnet R cnode parameters
    '''
    x0= np.zeros(2*n_cnodes+1)
    x0[0] = lam # initial value for lamda
    
    
    x0[1:] = cnet_R_arr.cnode_info[1:3,:n_cnodes].flatten()
    
    
    args_cnode = (marginal_P_cnodes.flatten(), cnet_Q_arr.cnode_info[1:3,:n_cnodes].flatten())
    
    bnd = (0.001,0.999)
    bounds_cnode = [bnd,]*(2*n_cnodes+1)
    
    
    res = minimize(objective_cnode, x0, method='SLSQP', jac=derivative_cnode, # without normalization constraint
                   options={'ftol': 1e-4, 'disp': True, 'maxiter': 100},
                   bounds=bounds_cnode, args = args_cnode)
    
    updated_cnode_weights = res.x[1:].reshape(2, n_cnodes)
    
    sum_val = np.sum(updated_cnode_weights, axis =0)
   
    updated_cnode_weights /= sum_val
    cnet_R_arr.cnode_info[1,:n_cnodes] = updated_cnode_weights[0]
    cnet_R_arr.cnode_info[2,:n_cnodes] = updated_cnode_weights[1]
    
    
    cross_PQ = util_opt.compute_cross_entropy_cnet(reload_mix_clt, cnet_Q_arr)
    print ('P||Q:', cross_PQ)
   
    
    cross_PR2 = util_opt.compute_cross_entropy_cnet(reload_mix_clt, cnet_R_arr)
    print ('P||R:', cross_PR2)
    
   
    
    # output_rec = np.array([cross_PQ, cross_PR])
    # output_file = '../output_results/'+data_name+'/cnet_'+str(perturb_rate)
    # with open(output_file, 'a') as f_handle:
    #     np.savetxt(f_handle, output_rec.reshape(1,2), fmt='%f', delimiter=',')


if __name__=="__main__":
 
    start = time.time()
    main_opt_cnet()
    print ('Total running time: ', time.time() - start)