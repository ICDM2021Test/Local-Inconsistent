#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:26:43 2020
instead of marginal, we get the sufficiet statistics from P
P is MCNET

Pertub the statitics of Q in order to see the performance
"""

# optimization problem with cutset networks 
import numpy as np
from scipy.optimize import minimize

from CLT_class import CLT
from MIXTURE_CLT import MIXTURE_CLT, load_mt
from Util import *
import JT

import sys
import time
import copy

import util_opt


'''
Replace cpt with random numbers
'''
def pertub_model(model, model_type='clt', percent=0.1):
    
    
    if model_type=='clt':
        topo_order = model.topo_order
        parents = model.parents
        updated_cpt = np.copy(model.cond_cpt)
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
        
        return updated_cpt
    
    


"""
using theta_{\bar{b}|a} = 1-theta_{b|a}
Cleaned all the commented code based on version 0704
"""

def get_single_var_marginals(topo_order, parents, cond_cpt):
    # get marginals:
    marginals= np.zeros((topo_order.shape[0],2))
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
        

# the objective function
def objective(x, topo_order, parents, cpt_Q, marginal_P, pair_marginal_P):
    
    lamda = x[0]
    theta = x[1:].reshape(marginal_P.shape[0],2,2)
           

    # first part:
    cross_PlogR =  util_opt.compute_cross_entropy_parm(pair_marginal_P,marginal_P, parents, topo_order, theta)
    first_part = lamda*cross_PlogR
    
    # second part:
    sec_part = (1.0-lamda)*(np.sum(cpt_Q *np.log(theta)))
    
    # maximize is the negation of minimize
    return -(first_part+sec_part)
    
    
    
# the derivative function
def derivative(x, topo_order, parents, cpt_Q, marginal_P, pair_marginal_P):

    lamda = x[0]
    theta = x[1:].reshape(marginal_P.shape[0],2,2)
    nvariable = topo_order.shape[0]
    

    
    # derivative of lambda
    der_lam = 0
    cross_PlogR =  util_opt.compute_cross_entropy_parm(pair_marginal_P,marginal_P, parents, topo_order, theta)
    #der_lam = cross_PlogR - np.sum(cpt_Q *np.log(theta))
    
    # derivativ of thetas
    der_theta = np.zeros_like(theta)
        
    '''marginals of (x,u) where (x,u) is one edge in R, ordered in topo_order of R'''
    edge_marginal_P = np.zeros_like(cpt_Q)
    for i in range (nvariable-1):
        cld = topo_order[i+1]
        pa = parents[cld]
        edge_marginal_P[i+1] = pair_marginal_P[cld, pa]
    
    root = topo_order[0]
    edge_marginal_P[0,0,:] = marginal_P[root,0]
    edge_marginal_P[0,1,:] = marginal_P[root,1]
        
    der_theta[:,:,:] = lamda*edge_marginal_P/theta+(1.0-lamda)*(cpt_Q[:,:,:]/theta[:,:,:])
    
    
    '''Apply theta_{\bar{b}|a} = 1-theta_{b|a}'''
    # root: special case
    der_theta[0,0,0] -= der_theta[0,1,1]
    der_theta[0,1,1] = -der_theta[0,0,0]
    der_theta[0,0,1] = der_theta[0,0,0]    
    der_theta[0,1,0] = der_theta[0,1,1]

    der_theta[1:,0,:] -= der_theta[1:,1,:]
    der_theta[1:,1,:] = -der_theta[1:,0,:]
    
   

    der = np.zeros_like(x)
    der[0] = der_lam
    der[1:] = der_theta.flatten() 
    
    return der *(-1.0)


'''
Update the parameters of R directly from P
'''
def update_S_use_P(P_pair,P_single, S):
    return Util.compute_conditional_CPT(P_pair, P_single, S.topo_order, S.parents)





def main_opt_clt():
#    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    mt_dir = sys.argv[6]
    perturb_rate = float(sys.argv[8])
    std = float(sys.argv[10])
    lam = float(sys.argv[12])
    
    blur_flag = True

    print('------------------------------------------------------------------')
    print('Construct CLT using optimization methods')
    print('------------------------------------------------------------------')
    
    
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    full_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    full_dataset = np.concatenate((full_dataset, test_dataset), axis=0)
    
    n_variables = train_dataset.shape[1]
    
    P_type = 'mt'
    
    if P_type == 'mt':
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
        

        
        
    
    else:
        print("Learning Chow-Liu Trees on full data ......")
        clt_P = CLT()
        clt_P.learnStructure(full_dataset)
        
        marginal_P = clt_P.xprob # the single marginals of P
        # get the pairwise marginals of P
        jt_P = JT.JunctionTree()
        jt_P.learn_structure(clt_P.topo_order, clt_P.parents, clt_P.cond_cpt)
        pair_marginal_P = JT.get_marginal_JT(jt_P, [], np.arange(n_variables))
    

    
    
    # Use half of the training data to bulid Q
    n_rec = train_dataset.shape[0]
    rand_record =  np.random.randint(n_rec, size=int(n_rec/10))    
    half_data = train_dataset[rand_record,:]
    
    clt_Q = CLT()
    clt_Q.learnStructure(half_data)
    
    clt_Q.cond_cpt = pertub_model(clt_Q, model_type='clt', percent=perturb_rate)
    
    
    # Initialize R as P
    clt_R = copy.deepcopy(clt_Q)
        
   
    cpt_Q = clt_Q.cond_cpt
    cpt_R = clt_R.cond_cpt
    
    '''
    Get the noise
    '''
    noise_mu = 0
    noise_std = std
    noise_percent = 1
    
    
    pair_marginal_P_blur = util_opt.add_noise (pair_marginal_P, n_variables, noise_mu, noise_std, percent_noise=noise_percent)
    marginal_P_blur = marginal_P
    if blur_flag == True:
        '''apply noise to P'''
        args = (clt_R.topo_order, clt_R.parents, cpt_Q, marginal_P_blur, pair_marginal_P_blur)
    else:
        args = (clt_R.topo_order, clt_R.parents, cpt_Q, marginal_P, pair_marginal_P)
    
    # set the bound for all variables
    bnd = (0.001,0.999)
    bounds = [bnd,]*(4*n_variables+1)
    
    x0 = np.zeros(4*n_variables+1)
    x0[0] = lam  # initial value for lamda
    x0[1:] = cpt_R.flatten()
    
    # constraint: valid prob
    normalize_cons = []
    for i in range (n_variables):
    
        
        normalize_cons.append({'type': 'eq',
           'fun' : lambda x: np.array([x[i*4+1] + x[i*4+3] - 1, 
                                       x[i*4+2] + x[i*4+4] - 1])})
   
    #res = minimize(objective, x0, method='SLSQP', jac=derivative, constraints=normalize_cons,  # with normalization constriant
    res = minimize(objective, x0, method='SLSQP', jac=derivative, # without normalization constraint
               options={'ftol': 1e-6, 'disp': True, 'maxiter': 1000},
               bounds=bounds, args = args)
    clt_R.cond_cpt = res.x[1:].reshape(n_variables,2,2)
    
    
    
    print ('------Cross Entropy-------')
        
    P_Q = util_opt.compute_KL(pair_marginal_P, marginal_P, clt_Q)
    P_R = util_opt.compute_KL(pair_marginal_P, marginal_P, clt_R)
    print ('P||Q:', P_Q)
    print ('P||R:', P_R)
    
    # output_rec = np.array([P_Q, P_R])
    # output_file = '../output_results/'+data_name+'/clt_'+str(perturb_rate)
    # with open(output_file, 'a') as f_handle:
    #     np.savetxt(f_handle, output_rec.reshape(1,2), fmt='%f', delimiter=',')
    


if __name__=="__main__":

    start = time.time()
    main_opt_clt()
    print ('Total running time: ', time.time() - start)


