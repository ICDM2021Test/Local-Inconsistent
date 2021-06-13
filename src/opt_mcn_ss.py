#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
P is MCN, QR are both MCNs
use sampling to do the KL
using all pairwise maginals to update every parameter

Apply gaussian noise to marginals of P
Q is built from trainning data

"""

from __future__ import print_function
import numpy as np
from Util import *
import utilM
import util_opt
from CLT_class import CLT
from CNET_class import CNET
from MIXTURE_CLT import MIXTURE_CLT, load_mt
import time
import copy
import JT
import sys

from scipy.optimize import minimize



def get_single_var_marginals(topo_order, parents, cond_cpt):
    # get marginals:
    marginals= np.zeros((topo_order.shape[0],2))
    #marginal_R[topo_order[0]] = theta[0,:,0]
    marginals[topo_order[0]] = cond_cpt[0,:,0]
    for k in range (1,topo_order.shape[0]):
        c = topo_order[k]
        p = parents[c]
        marginals[c] = np.einsum('ij,j->i',cond_cpt[k], marginals[p])
    
    return marginals


# ordered by topo order
def get_edge_marginals(topo_order, parents, cond_cpt, single_marginal):
        
    # edge_marginals ordered by topo order
    edge_marginals = np.zeros_like(cond_cpt)
    edge_marginals[0,0,0] = cond_cpt[0,0,0]
    edge_marginals[0,1,1] = cond_cpt[0,1,1]
        
    parents_order = parents[topo_order]
    topo_marginals = single_marginal[parents_order[1:]]   # the parent marignals, ordered by topo_order 
        
    edge_marginals[1:] = np.einsum('ijk,ik->ijk',cond_cpt[1:], topo_marginals)

    return edge_marginals

'''
Compute the cross entropy of PlogQ using samples from P
Q is mixtrue of trees
'''
def compute_cross_entropy_mt_sampling(P, Q, samples):
    LL_P = P.computeLL_each_datapoint(samples)
    LL_Q = Q.computeLL_each_datapoint(samples)

    #approx_cross_entropy = np.sum(np.exp(LL_P)*LL_Q)
    #approx_cross_entropy = np.sum((LL_P - LL_Q))
    approx_cross_entropy = np.sum(LL_Q)
    return approx_cross_entropy 


def pertub_model(model, model_type='mt', percent=0.1):
    
    
    if model_type=='mt':
        
        updated_cpt_list = []
        
        for c in range (model.n_components):
      
            sub_tree =model.clt_list[c]
            topo_order = sub_tree.topo_order
            updated_cpt = np.copy(sub_tree.cond_cpt)
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


'''
Sample from tree distribution
'''
def sample_from_tree(clt, n_samples):

    
    topo_order = clt.topo_order
    parents = clt.parents
    
    
    cpt = np.copy(clt.cond_cpt)
    
    
    tree_samples = np.zeros((n_samples, topo_order.shape[0]), dtype = int)
    
    # tree root

    nums_0_r = int(np.rint(cpt[0,0,0] * n_samples))
    
    tree_samples [:nums_0_r, topo_order[0]] = 0
    tree_samples [nums_0_r:, topo_order[0]] = 1
    
    for j in range (1, topo_order.shape[0]):
 
        t_child = topo_order[j]
        t_parent = parents[t_child]
      
        # find where parent = 0 and parent = 1
        par_0 = np.where(tree_samples[:,t_parent]==0)[0]
        par_1 = np.where(tree_samples[:,t_parent]==1)[0]
       
 
        num_10 = int(np.round(cpt[j,1,0] * par_0.shape[0], decimals =0))
        num_11 = int(np.round(cpt[j,1,1] * par_1.shape[0], decimals =0))
  
        arr_pa0 = np.zeros(par_0.shape[0],dtype = int)
        arr_pa0[:num_10] = 1
        
        np.random.shuffle(arr_pa0)
        
        tree_samples[par_0, t_child] = arr_pa0
        
        
        
        arr_pa1 = np.zeros(par_1.shape[0],dtype = int)
        arr_pa1[:num_11] = 1
       
        np.random.shuffle(arr_pa1)
       
        tree_samples[par_1, t_child] = arr_pa1
      
    
    return tree_samples
        
    


'''
Sample from mixture of trees
'''    

def sample_from_mt (mt, n_samples):
    
    samples = []
    
    ''' for each component '''
    for i in range (mt.n_components):
        '''make sure num of samples >= n_samples'''  
        sub_n_samples = int(mt.mixture_weight[i]*n_samples)+1
        sub_samples = sample_from_tree(mt.clt_list[i],sub_n_samples)
        samples.append(sub_samples)
        

    samples = np.vstack(samples)
    np.random.shuffle(samples)
    
    
    # correct the number of samples
    diff = samples.shape[0] - n_samples 
    
    # too many samples
    if diff > 0:
        rand_ind = np.random.randint(samples.shape[0], size=diff)
       
        samples = np.delete(samples, rand_ind, 0)
        
    return samples






# the objective function
def objective(x, mt_R, mt_Q,  marginal_P, n_variables):
    
    n_variables = marginal_P.shape[0]
    n_components = mt_Q.n_components
    
    lamda = x[0]
    
    marginal_R = np.zeros_like(marginal_P)
    for c in range (n_components):
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[c] = x[start] 
        mt_R.clt_list[c].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
              
        # get marginals:
        marginal_R +=mt_R.mixture_weight[c] * get_single_var_marginals(mt_R.clt_list[c].topo_order, mt_R.clt_list[c].parents, mt_R.clt_list[c].cond_cpt)
    
    # first part:
    first_part = lamda*(np.sum(marginal_P*np.log(marginal_R)))
    
    
    # second part:
    second_part = 0
    for c in range (n_components):
        second_part += mt_R.mixture_weight[c]* np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
   
    sec_part = (1.0-lamda)*second_part
    
    
    return -(first_part+sec_part)
    
    
    
# the derivative function
def derivative(x, mt_R, mt_Q,  marginal_P, n_variables):

    #n_variables = marginal_P.shape[0]
    n_components = mt_Q.n_components
    der = np.zeros_like(x)
    
   
    lamda = x[0]
    
    
    ''' pre calculation '''
    marginal_R = np.zeros_like(marginal_P)
    sub_marginal_R = []
    for c in range (n_components):
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[c] = x[start] 
        mt_R.clt_list[c].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
        
        
        # get marginals:
        sub_marginal_R.append( get_single_var_marginals(mt_R.clt_list[c].topo_order, mt_R.clt_list[c].parents, mt_R.clt_list[c].cond_cpt))
        marginal_R +=mt_R.mixture_weight[c] * sub_marginal_R[c]
    
    
    marginal_P_divide_R = marginal_P/ marginal_R
    

    # first part:
    first_part = np.sum(marginal_P*np.log(marginal_R))
    
    # second part:
    second_part = 0
    for c in range (n_components):
        second_part += mt_R.mixture_weight[c]* np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
    
    '''deravertive of lamda'''
    der_lam = 0
    #der_lam = first_part-second_part   # test, not update lam
    der[0] = der_lam
    
    der_h_arr = np.zeros(n_components)
    '''deravertive of theta, h, For each subtree'''
    for c in range (n_components):
        sub_tree = mt_R.clt_list[c]
        h_weight = mt_R.mixture_weight[c]
        theta = sub_tree.cond_cpt
        jt = mt_R.jt_list[c]
        # dervative of hidden variable H
        der_h = 0        
        
        #der_h=lamda*np.sum(marginal_P_divide_R*sub_marginal_R[i]) 
        der_h=lamda*np.sum(marginal_P_divide_R*sub_marginal_R[c]) +(1-lamda)*np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
        der_h_arr[c] = der_h
        
        # derivativ of thetas
        der_theta = np.zeros_like(theta)
        
        
        jt.clique_potential = np.copy(theta)
        jt.clique_potential[0,0,1] = jt.clique_potential[0,1,0] = 0
        # add 1 varialbe in JT
        jt_var = copy.deepcopy(jt)
            
        for var in range(n_variables):
    
            new_potential = jt_var.add_query_var(var)
     
            jt_var.propagation(new_potential)
                   
            # normalize
            norm_const=np.einsum('ijkl->i',new_potential)
            new_potential /= norm_const[:,np.newaxis,np.newaxis,np.newaxis]
    
            der_theta[:,:,:] += (marginal_P[var,0]/marginal_R[var,0])*(new_potential[:,:,:,0]/theta[:,:,:]) + \
                    (marginal_P[var,1]/marginal_R[var,1])*(new_potential[:,:,:,1]/theta[:,:,:])               
            

        der_theta[:,:,:] = h_weight * (lamda*der_theta[:,:,:]+(1.0-lamda)*(mt_Q.clt_list[c].cond_cpt[:,:,:]/theta[:,:,:]))
        
        
        '''Apply theta_{\bar{b}|a} = 1-theta_{b|a}'''
        # root: special case
        der_theta[0,0,0] -= der_theta[0,1,1]
        der_theta[0,1,1] = -der_theta[0,0,0]
        der_theta[0,0,1] = der_theta[0,0,0]    
        der_theta[0,1,0] = der_theta[0,1,1]
    
        der_theta[1:,0,:] -= der_theta[1:,1,:]
        der_theta[1:,1,:] = -der_theta[1:,0,:]
    
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        der[start] = der_h
        der[start+1: end+1] = der_theta.flatten()
    

    
    '''make h to be sum to 1'''
    der_h_adj = np.sum(der_h_arr)/n_components
    
    for i in range (n_components):
        start = i*(4*n_variables+1)+1
        der[start] -= der_h_adj
        
    return der *(-1.0)

'''   
# the objective function use pairwise marginal of P
'''
def objective_pair(x, mt_R, mt_Q,  pair_marginal_P, n_variables):
    
    n_components = mt_Q.n_components
    
   
    lamda = x[0]
    
    pair_marginal_R = np.zeros_like(pair_marginal_P)
    for c in range (n_components):
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[c] = x[start] 
        mt_R.clt_list[c].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
        
        mt_R.jt_list[c].clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
        mt_R.jt_list[c].clique_potential[0,0,1] = mt_R.jt_list[c].clique_potential[0,1,0] = 0
              
    # get marginals of R:
    pair_marginal_R, temp_marginal_R =mt_R.inference_jt([],np.arange(n_variables))
    
    
    # first part:
    first_part = lamda*(np.sum(pair_marginal_P*np.log(pair_marginal_R)))
    
    
    # second part:
    second_part = 0
    for c in range (n_components):
        second_part += mt_R.mixture_weight[c]* np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))

    sec_part = (1.0-lamda)*second_part
    
    # maximize is the negation of minimize
    return -(first_part+sec_part)
    
    
'''   
# the derivative function
'''
def derivative_pair(x, mt_R, mt_Q,  pair_marginal_P, n_variables):

    #n_variables = pair_marginal_P.shape[0]
    n_components = mt_Q.n_components
    ids = np.arange(n_variables)
    der = np.zeros_like(x)
    
   
    lamda = x[0]
    
    
    ''' pre calculation '''
    pair_marginal_R = np.zeros_like(pair_marginal_P)
    sub_marginal_R = []
    for c in range (n_components):
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[c] = x[start] 
        mt_R.clt_list[c].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
        
        
        sub_jt = mt_R.jt_list[c]
        sub_jt.clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
        sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0

        p_xy =  JT.get_marginal_JT(sub_jt, [], np.arange(n_variables))
        
        # get marginals:
        
        p_x = np.zeros((n_variables, 2))
        #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
        
        p_x[:,0] = p_xy[0,:,0,0] + p_xy[0,:,1,0]
        p_x[:,1] = p_xy[0,:,0,1] + p_xy[0,:,1,1]        
        p_x[0,0] = p_xy[1,0,0,0] + p_xy[1,0,1,0]
        p_x[0,1] = p_xy[1,0,0,1] + p_xy[1,0,1,1]
        
        # Normalize        
        p_x = Util.normalize1d(p_x)
        
        for j in range (ids.shape[0]):
            p_xy[j,j,0,0] = p_x[j,0] - 1e-8
            p_xy[j,j,1,1] = p_x[j,1] - 1e-8
            p_xy[j,j,0,1] = 1e-8
            p_xy[j,j,1,0] = 1e-8
        
        
        sub_marginal_R.append(p_xy)
        pair_marginal_R += p_xy * mt_R.mixture_weight[c]
        #p_xy_all = Util.normalize2d(p_xy_all)
        
    
    
    pair_marginal_P_divide_R = pair_marginal_P/ pair_marginal_R

    # first part:
    first_part = np.sum(pair_marginal_P*np.log(pair_marginal_R))

    # second part:
    second_part = 0
    for c in range (n_components):
        second_part += mt_R.mixture_weight[c]* np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
    
    '''deravertive of lamda, not update'''
    der_lam = 0
    #der_lam = first_part-second_part   
    der[0] = der_lam
    
    der_h_arr = np.zeros(n_components)
    '''deravertive of theta, h, For each subtree'''
    for c in range (n_components):
        sub_tree = mt_R.clt_list[c]
        h_weight = mt_R.mixture_weight[c]
        theta = sub_tree.cond_cpt
        
        
        sub_jt = copy.deepcopy(mt_R.jt_list[c])
        sub_jt.clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
        sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0

        der_h = 0        
        
        der_h=lamda*np.sum(pair_marginal_P_divide_R*sub_marginal_R[c]) + (1-lamda)*np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
        der_h_arr[c] = der_h
        
        # derivativ of thetas
        der_theta = np.zeros_like(theta)
        
        

        binary_arr = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        #sub_pxy_list =[]
        for j in range (n_variables): 
            #temp_tree.cond_cpt = np.copy(sub_tree.cond_cpt) # reset
            t = sub_tree.topo_order[j]
            u = sub_tree.parents[t]
            
            '''
            size = 4*nVar*nVar*2*2, 4 represent the 4 values of theta_c|u
            '''
            pxy_regarding_theta = []
            for k in range (binary_arr.shape[0]):
                val_t = binary_arr[k,0]
                val_u = binary_arr[k,1]
                
                evid_theta = []
                evid_theta.append([t,val_t])
                if u != -9999:
                    evid_theta.append([u,val_u])

                
               
                sub_jt.clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
                sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0
               
                
                sub_pxy = JT.get_marginal_JT(sub_jt, evid_theta, np.arange(n_variables))
               
                
                pxy_regarding_theta.append(sub_pxy)
            
            pxy_regarding_theta_arr = np.asarray(pxy_regarding_theta)
            for y in range(n_variables):
                for z in range (y+1, n_variables):
                    
                    #  val_c=0, val_u=0
                    der_theta[t,0,0] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[0,y,z,0,0] + \
                        pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[0,y,z,0,1]+ \
                        pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[0,y,z,1,0] + \
                        pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[0,y,z,1,1])/theta[t,0,0]
                    
                    
                    
                    der_theta[t,1,1] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[3,y,z,0,0] + \
                        pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[3,y,z,0,1]+ \
                        pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[3,y,z,1,0] + \
                        pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[3,y,z,1,1])/theta[t,1,1]
                        
                    
                    if u != 9999:
                        der_theta[t,0,1] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[1,y,z,0,0] + \
                            pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[1,y,z,0,1]+ \
                            pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[1,y,z,1,0] + \
                            pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[1,y,z,1,1])/theta[t,0,1]
                    
                        der_theta[t,1,0] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[2,y,z,0,0] + \
                            pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[2,y,z,0,1]+ \
                            pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[2,y,z,1,0] + \
                            pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[2,y,z,1,1])/theta[t,1,0]
                        
                    
       
 


        der_theta[:,:,:] = h_weight * (lamda*der_theta[:,:,:]+(1.0-lamda)*(mt_Q.clt_list[c].cond_cpt[:,:,:]/theta[:,:,:]))
        
       
        
        '''Apply theta_{\bar{b}|a} = 1-theta_{b|a}'''
        # root: special case
        der_theta[0,0,0] -= der_theta[0,1,1]
        der_theta[0,1,1] = -der_theta[0,0,0]
        der_theta[0,0,1] = der_theta[0,0,0]    
        der_theta[0,1,0] = der_theta[0,1,1]
    
        der_theta[1:,0,:] -= der_theta[1:,1,:]
        der_theta[1:,1,:] = -der_theta[1:,0,:]
    
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        der[start] = der_h
        der[start+1: end+1] = der_theta.flatten()
    

   
    '''make h to be sum to 1'''
    der_h_adj = np.sum(der_h_arr)/n_components
    
    for i in range (n_components):
        start = i*(4*n_variables+1)+1
        der[start] -= der_h_adj
       
    return der *(-1.0)
'''

'''

def main_opt_mt():
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    mt_dir = sys.argv[6]    
    n_components = int(sys.argv[8])
    perturb_rate = float(sys.argv[10])
    std = float(sys.argv[12])
    lam = float(sys.argv[14])
    
    
    # for EM to generate MCN
    max_iter = 1000
    epsilon = 1e-4
    
    # for optimization
    max_iter_opt = 1000
    
    n_samples = 100000 # number of samples used to do the evaluation
    
   
    P_type = 'mt'
    pair = True  # using pairwise marginals
    
    

    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    
    
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    
    n_variables = train_dataset.shape[1]
    
    
    if P_type == 'mt':
        '''
        ### Load the trained mixture of clt, consider as P
        '''
        #print ('Start reloading MT...')
        #mt_dir =  'mt_output/'
        reload_mix_clt = load_mt(mt_dir, data_name)
        
        # Set information for MT
        for t in reload_mix_clt.clt_list:
            t.nvariables = n_variables
            # learn the junction tree for each clt
            jt = JT.JunctionTree()
            jt.learn_structure(t.topo_order, t.parents, t.cond_cpt)
            reload_mix_clt.jt_list.append(jt)
        
        # using mixture of trees as P
        model_P = reload_mix_clt
        
        p_xy_all = np.zeros((n_variables, n_variables, 2, 2))
        p_x_all = np.zeros((n_variables, 2))
        for i, jt in enumerate(model_P.jt_list):
            p_xy = JT.get_marginal_JT(jt, [], np.arange(n_variables))
            p_xy_all += p_xy * model_P.mixture_weight[i]


        p_x_all[:,0] = p_xy_all[0,:,0,0] + p_xy_all[0,:,1,0]
        p_x_all[:,1] = p_xy_all[0,:,0,1] + p_xy_all[0,:,1,1]
        
        p_x_all[0,0] = p_xy_all[1,0,0,0] + p_xy_all[1,0,1,0]
        p_x_all[0,1] = p_xy_all[1,0,0,1] + p_xy_all[1,0,1,1]
        
        
        # Normalize        
        marginal_P = Util.normalize1d(p_x_all)
        
        
        for i in range (n_variables):
            p_xy_all[i,i,0,0] = p_x_all[i,0] - 1e-8
            p_xy_all[i,i,1,1] = p_x_all[i,1] - 1e-8
            p_xy_all[i,i,0,1] = 1e-8
            p_xy_all[i,i,1,0] = 1e-8
        
        pair_marginal_P = Util.normalize2d(p_xy_all)
        
        
        '''
        Sampling from P
        '''
        samples_P = sample_from_mt(model_P, n_samples)
    
#    elif P_type == 'bn':
#        '''
#        # Learn BNET as P
#        '''
#        order = np.arange(train_dataset.shape[1])
#        np.random.shuffle(order)
#        print("Learning Bayesian Network.....")
#        bnet = BNET()
#        bnet.learnStructure_PE(train_dataset, order, option=1)
#        #print("done")
#        samples_P = bnet.getSamples(n_samples)
#        model_P = bnet

    
    # 10% to generate Q
    n_rec = train_dataset.shape[0]
    rand_record =  np.random.randint(n_rec, size=int(n_rec/10))    
    half_data = train_dataset[rand_record,:]
    eval_data = samples_P

#    '''statistics of P'''
#    xycounts_P = Util.compute_xycounts(samples_P) + 1 # laplace correction
#    xcounts_P = Util.compute_xcounts(samples_P) + 2 # laplace correction
#    pair_marginal_P = Util.normalize2d(xycounts_P)
#    marginal_P = Util.normalize1d(xcounts_P)
#    

    '''
    Get the noise
    '''
    noise_mu = 0
    noise_std = std
    noise_percent = 1
    
  
    pair_marginal_P_blur = util_opt.add_noise (pair_marginal_P, n_variables, noise_mu, noise_std, percent_noise=noise_percent) 
    
    '''
    Q Learn from dataset
    '''
    #print ('-------------- Mixture of trees Learn from partial data: (Q) ----------')
    mt_Q = MIXTURE_CLT()
    mt_Q.learnStructure(half_data, n_components)
    mt_Q.EM(half_data, max_iter, epsilon)
    
    
    
    
    if perturb_rate > 0:
        
        perturbed_list = pertub_model(mt_Q, 'mt', perturb_rate)
        

        for c in range (n_components):
            mt_Q.clt_list[c].cond_cpt= perturbed_list[c]
            
    
    
    cross_PP = compute_cross_entropy_mt_sampling (model_P, model_P, eval_data)
    
    cross_PQ = compute_cross_entropy_mt_sampling (model_P, mt_Q, eval_data)
    
    

    
    #print ('-------------- Mixture of trees Learn Learn from P and Q using samples: (R) ----------')
    mt_R = copy.deepcopy(mt_Q)
    
    '''construct junction tree list for R'''
    for i in range (n_components):
        jt = JT.JunctionTree()
        sub_tree = mt_R.clt_list[i]
        jt.learn_structure(sub_tree.topo_order, sub_tree.parents, sub_tree.cond_cpt)
        mt_R.jt_list.append(jt)
    
    
    # set the bound for all variables
    bnd = (0.001,0.999)
    n_parm = (4*n_variables+1)*n_components+1 # number of parameters that needs to update
    bounds = [bnd,]*n_parm
    
    x0 = np.zeros(n_parm)
    x0[0] = lam  # initial value for lamda
    
    for i in range (n_components):
        start = i*(4*n_variables+1)+1
        end = start+4*n_variables
        x0[start] = mt_R.mixture_weight[i]   #mixture weight H
        x0[start+1: end+1] = mt_R.clt_list[i].cond_cpt.flatten()
    
    
    if pair == True:
        
        pair_marginal_P_no_dup = np.copy(pair_marginal_P_blur)
        # eliminate the duplication
        for i in range (n_variables):
            for j in range (i+1):
                pair_marginal_P_no_dup[i,j] = 0
                
        args = (mt_R, mt_Q,  pair_marginal_P_no_dup, n_variables)
    
        res = minimize(objective_pair, x0, method='SLSQP', jac=derivative_pair, # without normalization constraint
               options={'ftol': 1e-6, 'disp': True, 'maxiter': max_iter_opt},
               bounds=bounds, args = args)
    else:
        
        args = (mt_R, mt_Q,  marginal_P, n_variables)
        #res = minimize(objective, x0, method='SLSQP', jac=derivative, constraints=normalize_cons,  # with normalization constriant
        res = minimize(objective, x0, method='SLSQP', jac=derivative, # without normalization constraint
               options={'ftol': 1e-6, 'disp': True, 'maxiter': max_iter_opt},
               bounds=bounds, args = args)
    

    x = res.x
    for i in range (n_components):
        start = i*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[i] = x[start] 
        mt_R.clt_list[i].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
    
    
    
    print ('P||Q:', cross_PQ/n_samples)
    
    cross_PR = compute_cross_entropy_mt_sampling (model_P, mt_R, eval_data)
    print ('P||R:', cross_PR/n_samples)
    
    
    # output_rec = np.array([cross_PQ/n_samples, cross_PR/n_samples])
    # output_file = '../output_results/'+data_name+'/mt_'+str(perturb_rate)
    # with open(output_file, 'a') as f_handle:
    #     np.savetxt(f_handle, output_rec.reshape(1,2), fmt='%f', delimiter=',')
    


if __name__=="__main__":
    
    start = time.time()
    main_opt_mt()
    print ('Total running time: ', time.time() - start)