#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:26:43 2020
instead of marginal, we get the sufficiet statistics from P
This version we have evidences

use the common format to update derivative

"""

# optimization problem with chow-liu tree
import numpy as np
from scipy.optimize import minimize


from CLT_class import CLT
from Util import *
import JT
from MIXTURE_CLT import MIXTURE_CLT, load_mt

import sys
import time
import copy

import util_opt
import utilM





def compute_cross_entropy_mt_sampling_evid(Q, samples, evid_list):
    LL_Q = Q.getWeights(samples)
    #approx_cross_entropy = np.sum(np.exp(LL_P)*LL_Q)
    #approx_cross_entropy = np.sum((LL_P - LL_Q))
    
    cond_cpt_evid = Q.instantiation(evid_list)
    evid_prob = utilM.ve_tree_bin(Q.topo_order, Q.parents, cond_cpt_evid) 

    approx_cross_entropy = np.sum(LL_Q)/samples.shape[0] - np.log(evid_prob)
    return approx_cross_entropy 



'''
Replace cpt with random numbers
'''
def pertub_model(model, model_type='clt', percent=0.1):
    
    
    if model_type=='clt':
        topo_order = model.topo_order
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


'''
# compute P(x|e)log(R(x|e))
#
'''
def cross_entropy_evid(P, R, evid_list, non_evid_var):
    cross_entropy = 0
    
    total_n_var = len(evid_list)+non_evid_var.shape[0] #evid+non-evid
    
    # assume P is a tree    
    jt_P = JT.JunctionTree()
    jt_P.learn_structure(P.topo_order, P.parents, P.cond_cpt)

        
    P_xy_evid =  JT.get_marginal_JT(jt_P, evid_list, non_evid_var)
    #p_xy_norm = Util.normalize2d(p_xy)
    P_x_evid = np.zeros((non_evid_var.shape[0], 2))
    #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
    
    P_x_evid[:,0] = P_xy_evid[0,:,0,0] + P_xy_evid[0,:,1,0]
    P_x_evid[:,1] = P_xy_evid[0,:,0,1] + P_xy_evid[0,:,1,1]        
    P_x_evid[0,0] = P_xy_evid[1,0,0,0] + P_xy_evid[1,0,1,0]
    P_x_evid[0,1] = P_xy_evid[1,0,0,1] + P_xy_evid[1,0,1,1]
    
    # Probablity of evidence according to P
    P_e = np.sum(P_x_evid[0,:])
    
    
    # Probablity of evidence according to R
    cond_cpt_e = R.instantiation(evid_list)
    R_e = utilM.ve_tree_bin(R.topo_order, R.parents, cond_cpt_e)
    
    # mark which variable is evidence
    evid_flag = np.full(total_n_var,-1) #-1 means non evidence
    evid_arr = np.asarray(evid_list)
    evid_flag[evid_arr[:,0]] = evid_arr[:,1]    
    
    P_xy_evid_full = np.zeros((total_n_var, total_n_var, 2,2))
    P_xy_evid_full[non_evid_var[:,None],non_evid_var] = P_xy_evid
    P_x_evid_full = np.zeros((total_n_var, 2))
    P_x_evid_full[non_evid_var,:] = P_x_evid
    

    # root is the special case
    for i in range (1, R.topo_order.shape[0]):
        cld = R.topo_order[i]
        par = R.parents[cld]
        
        val_c =  evid_flag[cld]
        val_p =  evid_flag[par]
        # both cld and par are not evid
        if  val_c ==-1 and val_p ==-1:
            cross_entropy += np.sum(P_xy_evid_full[cld, par] * np.log(cond_cpt_e[i]))
        # cld is evidence   
        elif val_c !=-1 and val_p ==-1:
            cross_entropy += np.sum(P_x_evid_full[cld] * np.log(cond_cpt_e[i,val_c,:]))
        # par is evidence   
        elif val_c ==-1 and val_p !=-1:
            cross_entropy += np.sum(P_x_evid_full[par] * np.log(cond_cpt_e[i,:,val_p]))
        # else both cld and par are evidence
        else:
            cross_entropy += P_e * np.log(cond_cpt_e[i,val_c,val_p])
    
    # root
    R_root = R.topo_order[0]
    val_root = evid_flag[R_root]
    # not evid
    if val_root == -1:
        R_root_marginal = np.array([cond_cpt_e[0,0,0], cond_cpt_e[0,1,1]])
        cross_entropy += np.sum(P_x_evid_full[R_root]* np.log(R_root_marginal))
    else:
        cross_entropy += P_e * np.log(cond_cpt_e[i,val_root,val_root])
    
           
    cross_entropy -= P_e * np.log(R_e)
    
    
    return cross_entropy




def cross_entropy_evid_parm(R, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag):
    
    cross_entropy = 0

    
    cond_cpt_e = R.instantiation(evid_list)
    #cond_cpt_e =  np.nan_to_num(cond_cpt_e)
    #R_e = utilM.ve_tree_bin(R.topo_order, R.parents, cond_cpt_e)
    
    #cross_PlogR =  util_opt.compute_cross_entropy_parm(pair_marginal_P,marginal_P, parents, topo_order, theta)
    for i in range (1, R.topo_order.shape[0]):
        cld = R.topo_order[i]
        par = R.parents[cld]
        
        val_c =  evid_flag[cld]
        val_p =  evid_flag[par]
        # both cld and par are not evid
        if  val_c ==-1 and val_p ==-1:
            cross_entropy += np.sum(pair_marginal_P[cld, par] * np.log(cond_cpt_e[i]))
        # cld is evidence   
        elif val_c !=-1 and val_p ==-1:
            cross_entropy += np.sum(marginal_P[cld] * np.log(cond_cpt_e[i,val_c,:]))
        # par is evidence   
        elif val_c ==-1 and val_p !=-1:
            cross_entropy += np.sum(marginal_P[par] * np.log(cond_cpt_e[i,:,val_p]))
        # else both cld and par are evidence
        else:
            cross_entropy += np.log(cond_cpt_e[i,val_c,val_p])
    
    # root
    R_root = R.topo_order[0]
    val_root = evid_flag[R_root]
    # not evid
    if val_root == -1:
        R_root_marginal = np.array([cond_cpt_e[0,0,0], cond_cpt_e[0,1,1]])
        cross_entropy += np.sum(marginal_P[R_root]* np.log(R_root_marginal))
    else:
        cross_entropy += np.log(cond_cpt_e[0,val_root,val_root])
      
    
    #cross_entropy -= np.log(R_e)
    
    return cross_entropy
    

# the objective function
def objective(x, clt_R, cpt_Q, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag):
    
    
    n_variable = evid_flag.shape[0]
    lamda = x[0]
    theta = x[1:].reshape(n_variable,2,2)
    
    clt_R.cond_cpt = theta
           
    # get marginals:
    
    
    # first part
    cross_PlogR = cross_entropy_evid_parm(clt_R, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag)
    first_part = lamda*(cross_PlogR)    
   
    # second part:
    sec_part = (1.0-lamda)*(np.sum(cpt_Q *np.log(theta)))
    
    # maximize is the negation of minimize
   
    return -(first_part+sec_part)
    
    
    
# the derivative function
def derivative(x, clt_R, cpt_Q, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag):

    lamda = x[0]
    theta = x[1:].reshape(marginal_P.shape[0],2,2)
    n_variable = evid_flag.shape[0]
    
    
    clt_R.cond_cpt = theta
    
    # derivative of lambda
    cross_PlogR =  cross_entropy_evid_parm(clt_R, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag)
    #der_lam = cross_PlogR - np.sum(cpt_Q *np.log(theta))
    der_lam = 0
    
    # derivativ of thetas
    der_theta = np.zeros_like(theta)
                
    #cond_cpt_e = clt_R.instantiation(evid_list)    
    jt_R = JT.JunctionTree()
    jt_R.learn_structure(clt_R.topo_order, clt_R.parents, clt_R.cond_cpt)
        
    #R_xy_evid =  JT.get_marginal_JT(jt_R, evid_list, non_evid_var)
    R_xy_evid =  JT.get_marginal_JT(jt_R, evid_list, np.arange(n_variable))
    R_x_evid = np.zeros((n_variable, 2))
    #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
    
    
    R_x_evid[:,0] = R_xy_evid[0,:,0,0] + R_xy_evid[0,:,1,0]
    R_x_evid[:,1] = R_xy_evid[0,:,0,1] + R_xy_evid[0,:,1,1]        
    R_x_evid[0,0] = R_xy_evid[1,0,0,0] + R_xy_evid[1,0,1,0]
    R_x_evid[0,1] = R_xy_evid[1,0,0,1] + R_xy_evid[1,0,1,1]
    

    
    R_xy_given_evid = Util.normalize2d(R_xy_evid)
    R_x_given_evid = Util.normalize1d(R_x_evid)
    
#    R_xy_given_evid_full = np.zeros((n_variable, n_variable, 2,2))
#    R_xy_given_evid_full[non_evid_var[:,None],non_evid_var] = R_xy_given_evid
#    R_x_given_evid_full = np.zeros((n_variable, 2))
#    R_x_given_evid_full[non_evid_var,:] = R_x_given_evid
    
    
    R_xy_given_evid_full = R_xy_given_evid
    R_x_given_evid_full = R_x_given_evid
    
    '''P(x,u|e)-R(x,u|e), where (x,u) is one edge in R, ordered in topo_order of R'''
    edge_marginal_diff = np.zeros_like(cpt_Q)
    for i in range (1,n_variable):
        cld = clt_R.topo_order[i]
        par = clt_R.parents[cld]
        
        val_c =  evid_flag[cld]
        val_p =  evid_flag[par]
        # both cld and par are not evid
        if  val_c ==-1 and val_p ==-1:
            edge_marginal_diff[i] = pair_marginal_P[cld, par] *(1- R_xy_given_evid_full[cld, par])
        # cld is evidence   
        elif val_c !=-1 and val_p ==-1:
            edge_marginal_diff[i,val_c,:] = marginal_P[cld] *(1- R_x_given_evid_full[cld])
        # par is evidence   
        elif val_c ==-1 and val_p !=-1:
            edge_marginal_diff[i,:, val_p] = marginal_P[par] *(1- R_x_given_evid_full[par])
        # else both cld and par are evidence
        else:
            edge_marginal_diff[i, val_c,val_p] = 0
        
        #edge_marginal_P[i+1] = pair_marginal_P[cld, pa]
    
    root = clt_R.topo_order[0]    
    val_root = evid_flag[root]
    # not evid
    if val_root == -1:
        edge_marginal_diff[0,0,:] = marginal_P[root,0]* (1 - R_x_given_evid_full[root,0])
        edge_marginal_diff[0,1,:] = marginal_P[root,1]* (1 - R_x_given_evid_full[root,1])
    else:
        edge_marginal_diff = 0
    
        
    der_theta[:,:,:] = lamda*edge_marginal_diff/theta+(1.0-lamda)*(cpt_Q[:,:,:]/theta[:,:,:])
    
   

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




def main_opt_clt():

   
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    mt_dir = sys.argv[6]
    e_percent = float(sys.argv[8])
    perturb_rate = float(sys.argv[10])
    std = float(sys.argv[12])
    lam = float(sys.argv[14])
    

    n_samples = 100000
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
    
    evids = util_opt.read_evidence_file('../evidence/', e_percent, 'nltcs')
    
    evid_var =  evids[:,0]
    non_evid_var = np.setdiff1d(np.arange(n_variables), evid_var)
    evid_list = list(evids)
    
    evid_flag = np.full(n_variables,-1) #-1 means non evidence
    evid_flag[evids[:,0]] = evids[:,1]    

    
    P_type = 'mt'
    
    if P_type == 'mt':
        '''
        ### Load the trained mixture of clt, consider as P
        '''
        #print ('Start reloading MT...')
        #mt_dir =  '../mt_output/'
        reload_mix_clt = load_mt(mt_dir, data_name)
        non_evid_size = non_evid_var.shape[0]
        
        # Set information for MT
        for t in reload_mix_clt.clt_list:
            t.nvariables = non_evid_size
            # learn the junction tree for each clt
            jt = JT.JunctionTree()
            jt.learn_structure(t.topo_order, t.parents, t.cond_cpt)
            reload_mix_clt.jt_list.append(jt)
        
        # using mixture of trees as P
        model_P = reload_mix_clt
                
        p_xy_all = np.zeros((n_variables, n_variables, 2, 2))
        p_x_all = np.zeros((n_variables, 2))
        for i, jt in enumerate(model_P.jt_list):
            p_xy = JT.get_marginal_JT(jt, evid_list, np.arange(n_variables))
            p_xy_all += p_xy * model_P.mixture_weight[i]


        p_x_all[:,0] = p_xy_all[0,:,0,0] + p_xy_all[0,:,1,0]
        p_x_all[:,1] = p_xy_all[0,:,0,1] + p_xy_all[0,:,1,1]
        
        p_x_all[0,0] = p_xy_all[1,0,0,0] + p_xy_all[1,0,1,0]
        p_x_all[0,1] = p_xy_all[1,0,0,1] + p_xy_all[1,0,1,1]
         
        # Normalize        
        P_x_given_evid = Util.normalize1d(p_x_all)
        
        P_xy_given_evid = Util.normalize2d(p_xy_all)
        for i in range (non_evid_size):
            P_xy_given_evid[i,i,0,0] = p_x_all[i,0] - 1e-8
            P_xy_given_evid[i,i,1,1] = p_x_all[i,1] - 1e-8
            P_xy_given_evid[i,i,0,1] = 1e-8
            P_xy_given_evid[i,i,1,0] = 1e-8
        
        #P_xy_given_evid = Util.normalize2d(p_xy_all)
    else:
    
        print("Learning Chow-Liu Trees on full data ......")
        clt_P = CLT()
        clt_P.learnStructure(full_dataset)
        
        
        
        jt_P = JT.JunctionTree()
        jt_P.learn_structure(clt_P.topo_order, clt_P.parents, clt_P.cond_cpt)
    
        P_xy_evid =  JT.get_marginal_JT(jt_P, evid_list, non_evid_var)
        
        #p_xy_norm = Util.normalize2d(p_xy)
        P_x_evid = np.zeros((non_evid_var.shape[0], 2))
        #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
        
        P_x_evid[:,0] = P_xy_evid[0,:,0,0] + P_xy_evid[0,:,1,0]
        P_x_evid[:,1] = P_xy_evid[0,:,0,1] + P_xy_evid[0,:,1,1]        
        P_x_evid[0,0] = P_xy_evid[1,0,0,0] + P_xy_evid[1,0,1,0]
        P_x_evid[0,1] = P_xy_evid[1,0,0,1] + P_xy_evid[1,0,1,1]
        
    
        # normalize
        P_xy_given_evid = Util.normalize2d(P_xy_evid)
        P_x_given_evid = Util.normalize1d(P_x_evid)
    

    
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

    '''apply noise to P'''
    if blur_flag == True:                
        '''
        Get the noise
        '''
        noise_mu = 0
        noise_std = std
        noise_percent = 1
        
        P_xy_given_evid_blur = util_opt.add_noise (P_xy_given_evid, n_variables, noise_mu, noise_std, percent_noise=noise_percent)
        
        P_xy_given_evid_full = P_xy_given_evid_blur
        P_x_given_evid_full = P_x_given_evid
        
 
    else:
        
        P_xy_given_evid_full = P_xy_given_evid
        P_x_given_evid_full = P_x_given_evid
    
    
    
    
    args = (clt_R, cpt_Q, P_x_given_evid_full, P_xy_given_evid_full, evid_list, non_evid_var, evid_flag)
    
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
               options={'ftol': 1e-4, 'disp': True, 'maxiter': 1000},
               bounds=bounds, args = args)
    clt_R.cond_cpt = res.x[1:].reshape(n_variables,2,2)
    clt_R.log_cond_cpt = np.log(clt_R.cond_cpt)
    

    
    
    print ('------Cross Entropy-------')
    
    if P_type == 'mt':
#        P_xy_given_evid_full = np.zeros((n_variables, n_variables, 2,2))
#        P_xy_given_evid_full[non_evid_var[:,None],non_evid_var] = P_xy_given_evid
#        P_x_given_evid_full = np.zeros((n_variables, 2))
#        P_x_given_evid_full[non_evid_var,:] = P_x_given_evid
#        
#        P_Q = cross_entropy_evid_parm(clt_Q, P_x_given_evid_full, P_xy_given_evid_full, evid_list, non_evid_var, evid_flag)
#        P_R = cross_entropy_evid_parm(clt_R, P_x_given_evid_full, P_xy_given_evid_full, evid_list, non_evid_var, evid_flag)
        
        samples = util_opt.sample_from_mt_evid_posterior(model_P, n_samples, evids, non_evid_var)
        
        
        P_Q = compute_cross_entropy_mt_sampling_evid(clt_Q, samples, evid_list)
        P_R = compute_cross_entropy_mt_sampling_evid(clt_R, samples, evid_list)
    else: # P is tree
        P_P = cross_entropy_evid(clt_P, clt_P, evid_list, non_evid_var)
        P_Q = cross_entropy_evid(clt_P, clt_Q, evid_list, non_evid_var)
        P_R = cross_entropy_evid(clt_P, clt_R, evid_list, non_evid_var)
        print ('P||P:', P_P)
    print ('P||Q:', P_Q)
    print ('P||R:', P_R)
    
   
    
    # output_rec = np.array([P_Q, P_R])
    # #print (output_rec.shape)
    # output_file = '../output_results/'+data_name+'/clt_e_'+str(e_percent) +'_'+str(perturb_rate)
    # with open(output_file, 'a') as f_handle:
    #     np.savetxt(f_handle, output_rec.reshape(1,2), fmt='%f', delimiter=',')


if __name__=="__main__":

    start = time.time()
    main_opt_clt()
    print ('Total running time: ', time.time() - start)