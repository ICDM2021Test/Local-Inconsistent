
"""
Created on Thu Sep 24 20:13:07 2020

"""

import numpy as np
import utilM
from CLT_class import CLT
import copy
import JT
from Util import *

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


def compute_KL_tree(T, R):
    print ('compute the KL divergence between 2 trees T||R:')
    
    KL = 0
    R_log_cpt = np.log(R.cond_cpt)
    
    # root is the special case
    for i in range (1, R.topo_order.shape[0]):
        c = R.topo_order[i]
        p = R.parents[c]
    
        T_pair_marginal = utilM.ve_tree_bin2(T.topo_order, T.parents, T.cond_cpt, c, p)
        KL += np.sum(T_pair_marginal* R_log_cpt[i])
    
    # root
    R_root_marginal = np.array([R_log_cpt[0,0,0], R_log_cpt[0,1,1]])
    R_root = R.topo_order[0]
    T_single_marginal = utilM.get_var_prob (T.topo_order, T.parents, T.cond_cpt, R_root)
    KL += np.sum(T_single_marginal* R_root_marginal)
    
    return KL


'''
Compute the KL divergence between 2 distributions.
P is pairwise marginal distribution
R is a tree distribution
'''
def compute_KL(P_pair,P_single, R):
    #print ('compute the KL divergence between 2 distributions P||R:')
    
    KL = 0
    R_log_cpt = np.log(R.cond_cpt)
    
    # root is the special case
    for i in range (1, R.topo_order.shape[0]):
        c = R.topo_order[i]
        p = R.parents[c]
    
        KL += np.sum(P_pair[c,p]* R_log_cpt[i])
    
    # root
    R_root_marginal = np.array([R_log_cpt[0,0,0], R_log_cpt[0,1,1]])
    R_root = R.topo_order[0]
    KL += np.sum(P_single[R_root]* R_root_marginal)
    
    return KL



'''
Compute the cross entropy between 2 distributions.
P is pairwise marginal distribution
R is a tree distribution， but provided in topo order, conditinal CPT format
'''
def compute_cross_entropy_parm(P_pair,P_single, R_parents, R_topo_order, R_cond_cpt):
    #print ('compute the KL divergence between 2 distributions P||R:')
    
    KL = 0
    R_log_cpt = np.log(R_cond_cpt)
    
    # root is the special case
    for i in range (1, R_topo_order.shape[0]):
        c = R_topo_order[i]
        p = R_parents[c]
    
        KL += np.sum(P_pair[c,p]* R_log_cpt[i])
    
    # root
    R_root_marginal = np.array([R_log_cpt[0,0,0], R_log_cpt[0,1,1]])
    R_root = R_topo_order[0]
    KL += np.sum(P_single[R_root]* R_root_marginal)
    
    return KL


'''
Compute the cross entropy of pairwise marginal probabilities. 
P log(R)
'''
def compute_KL_pairwise(P, R):
    
    n_var = P.shape[0]
    
    KL = 0
    for i in range (n_var):
        for j in range(i+1, n_var):
            KL += np.sum(P[i,j] * np.log(R[i,j]))
    
    return KL


'''
Compute the cross entropy of single variable marginal probabilities. 
P log(R)
'''
def compute_KL_single(P, R):
    
    
    return np.sum (P*np.log(R))



'''
Get the single variable marginals from pairwise marginals
'''
def get_single_from_pairwise (p_xy_all):
    
    
    p_x_all = np.zeros((p_xy_all.shape[0],2))
    p_x_all[:,0] = p_xy_all[0,:,0,0] + p_xy_all[0,:,1,0]
    p_x_all[:,1] = p_xy_all[0,:,0,1] + p_xy_all[0,:,1,1]
    
    p_x_all[0,0] = p_xy_all[1,0,0,0] + p_xy_all[1,0,1,0]
    p_x_all[0,1] = p_xy_all[1,0,0,1] + p_xy_all[1,0,1,1]
    

    return p_x_all


'''
P is n*n matrix, represent pairwise marginals
'''

def add_noise (P, n_var, noise_mu, noise_std, percent_noise=1):
    
    
    
    # how many potential function that has noise
    num_edges = int(n_var*(n_var-1)/2)
    num_pair_noise = int(num_edges * percent_noise) 
    
    noise = np.random.normal(loc=noise_mu, scale=noise_std, size=(num_pair_noise,2,2))
    noise_seq = np.random.choice(num_edges, size=num_pair_noise)
    
    edges = []
    for i in range (n_var):
        for j in range(i+1, n_var):
            edges.append([i,j])
    
    
    Q_noise = np.copy(P)
    
    for k,s in enumerate(noise_seq):
        
        [i,j] = edges[s]
        
        '''apply noise'''
        Q_noise[i,j] += noise[k]
        
        '''Set all value between [0.01 ~ 0.99]'''
        Q_noise[i,j][Q_noise[i,j] < 0.01] = 0.01
        Q_noise[i,j][Q_noise[i,j] > 0.99] = 0.99
        
        '''normalize'''
        Q_noise[i,j] /= np.sum(Q_noise[i,j])
        # avoid nan
        Q_noise[np.isnan(Q_noise)] = 0
        
        '''symetric'''

        Q_noise[j,i,0,0] = Q_noise[i,j,0,0]
        Q_noise[j,i,0,1] = Q_noise[i,j,1,0]
        Q_noise[j,i,1,0] = Q_noise[i,j,0,1]
        Q_noise[j,i,1,1] = Q_noise[i,j,1,1]
                
       

    return Q_noise




'''
Compute the cross entropy between 2 distributions.
P is pairwise marginal distribution
Q is a cutset network, in cnet array representation
'''
# P  is the oracle, Q is a cutset network
def compute_cross_entropy_cnet(P, Q):
    
    '''
    # P * log(a)
    # line 0 is left side
    # line 1 is right side
    '''
    cnode_entropy = np.zeros((2,int((Q.cnode_info.shape[1]-1)/2))) 
    kl = 0
    
    # for each branch in cutset network
    for i in range (len(Q.path)):
        path = Q.path[i]
    
        P_temp = copy.deepcopy(P)
        
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
                
                for k, t in enumerate(P_temp.clt_list):
                    t.cond_cpt = t.instantiation(incremental_evid_list)
                    
                
                                
                if cnode_entropy[0, var_ind] == 0: # not calculated
                    
                    '''P_marginal should be calculated based on different distribution P'''
                    
                    P_marginal = 0
                    for k, t in enumerate(P_temp.clt_list):                    
                        P_marginal += utilM.ve_tree_bin(t.topo_order, t.parents, t.cond_cpt) * P_temp.mixture_weight[k]
                   
                    cnode_entropy[0, var_ind] = P_marginal * np.log(Q.cnode_info[1,var_ind])
                    
                    kl += cnode_entropy[0, var_ind]
                var_ind = 2*var_ind+1
                
                
            
            if sign == '+': # going to right
                
                '''add evidence to P'''
                evid_list.append([var,1])
                incremental_evid_list.append([var,1])
                
                for k, t in enumerate(P_temp.clt_list):
                    t.cond_cpt = t.instantiation(incremental_evid_list)
                
                if cnode_entropy[1, var_ind] == 0: # not calculated
                    
                    '''P_marginal should be calculated based on different distribution P'''
                    P_marginal = 0
                    for k, t in enumerate(P_temp.clt_list):                    
                        P_marginal += utilM.ve_tree_bin(t.topo_order, t.parents, t.cond_cpt) * P_temp.mixture_weight[k]
                    
                    cnode_entropy[1, var_ind] = P_marginal * np.log(Q.cnode_info[2,var_ind])
                    
                
                    kl += cnode_entropy[1, var_ind]
                var_ind = 2*var_ind+2
        
        
        
        '''now, reach for the leaf of cnet'''
        leaf_tree_ind = np.where(Q.leaf_ind==var_ind)[0][0]  # the index of tree
    
        leaf_tree = CLT()
        leaf_tree.ids = Q.leaf_info_list[leaf_tree_ind][0]
        leaf_tree.parents = Q.leaf_info_list[leaf_tree_ind][1]
        leaf_tree.topo_order = Q.leaf_info_list[leaf_tree_ind][2]
        leaf_tree.cond_cpt = Q.leaf_cpt_list[leaf_tree_ind]
        
        P_xy_prob, P_x_prob = P_temp.inference_jt_wo_norm(evid_list,leaf_tree.ids)
        
        kl += compute_KL(P_xy_prob,P_x_prob, leaf_tree)
        
        
        
    return kl
   



'''
Compute the cross entropy of PlogQ using samples from P
Q is mixtrue of trees
Assume we can always get the Pr(e) from P and Q
Pr(x|e) = Pr(x,e)|Pr(e)
'''

def compute_cross_entropy_mt_sampling_evid(P, Q, samples, evid_list):
    LL_P = P.compute_cond_LL_each_datapoint(samples, evid_list)
    LL_Q = Q.compute_cond_LL_each_datapoint(samples, evid_list)
    
    approx_cross_entropy = np.sum(LL_Q)
    return approx_cross_entropy 















'''
------------------------------------------------------------------
Different sample methods to sample from Mixture of trees
------------------------------------------------------------------
'''



'''
Sample from tree distribution
'''
def sample_from_tree_evid(clt, n_samples, evid, non_evid_var):

  
    topo_order = clt.topo_order
    parents = clt.parents
   
    n_variables = topo_order.shape[0]
    evid_var =  evid[:,0]
    
    tree_samples = np.zeros((n_samples, topo_order.shape[0]), dtype = int)
    
    # set the evidence
    for i in range (evid.shape[0]):
        tree_samples[:, evid[i,0]] = evid[i,1]
    
   
    '''
    Compute the posterior distribtuion Pr(xi|par, e) by reconstruct the posterior
    CPT
    P(x|pa,e) = P(x,pa,e)|P(pa,e) = P(x,pa|e)/P(pa|e)
    If pa is in e, then P(x|pa,e) = P(x,e)|P(e) = P(x|e)
    '''
    
    jt = JT.JunctionTree()
    jt.learn_structure(clt.topo_order, clt.parents, clt.cond_cpt)
    P_xy_evid =  JT.get_marginal_JT(jt, list(evid), non_evid_var)
        
        
    P_x_evid = np.zeros((non_evid_var.shape[0], 2))
    
    P_x_evid[:,0] = P_xy_evid[0,:,0,0] + P_xy_evid[0,:,1,0]
    P_x_evid[:,1] = P_xy_evid[0,:,0,1] + P_xy_evid[0,:,1,1]        
    P_x_evid[0,0] = P_xy_evid[1,0,0,0] + P_xy_evid[1,0,1,0]
    P_x_evid[0,1] = P_xy_evid[1,0,0,1] + P_xy_evid[1,0,1,1]
    

    # normalize
    P_xy_given_evid = Util.normalize2d(P_xy_evid)
    P_x_given_evid = Util.normalize1d(P_x_evid)
    P_e = np.sum(P_x_evid[0,:])
    
    P_xy_given_evid_full = np.zeros((n_variables, n_variables, 2,2))

    P_xy_given_evid_full[non_evid_var[:,None],non_evid_var] = P_xy_given_evid
    P_x_given_evid_full = np.zeros((n_variables, 2))
    P_x_given_evid_full[non_evid_var,:] = P_x_given_evid
    
    cpt = np.zeros_like(clt.cond_cpt)
    
    for i in range (1,n_variables):
        cld = topo_order[i]
        par = parents[cld]
        
        if cld in evid_var:
            continue
        
        if par in evid_var:
            cpt[i,0,:] = P_x_given_evid_full[cld,0]
            cpt[i,1,:] = P_x_given_evid_full[cld,1]
            continue
        
        cpt[i,0,0] = P_xy_given_evid_full[cld,par,0,0]/P_x_given_evid_full[par,0]
        cpt[i,0,1] = P_xy_given_evid_full[cld,par,0,1]/P_x_given_evid_full[par,1]
        cpt[i,1,0] = P_xy_given_evid_full[cld,par,1,0]/P_x_given_evid_full[par,0]
        cpt[i,1,1] = P_xy_given_evid_full[cld,par,1,1]/P_x_given_evid_full[par,1]
    
    # root
    root = topo_order[0]
    if root not in evid_var:
        cpt[0,0,:] = P_x_given_evid_full[root,0]
        cpt[0,1,:] = P_x_given_evid_full[root,1]
        

    # tree root
    if topo_order[0] not in evid[:,0]:
   
        nums_0_r = int(np.rint(cpt[0,0,0] * n_samples))

        tree_samples [:nums_0_r, topo_order[0]] = 0
        tree_samples [nums_0_r:, topo_order[0]] = 1
   
    for j in range (1, topo_order.shape[0]):
        
        #evidence, do not sample
        if topo_order[j] in evid[:,0]:
            continue
        
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
Sample from mixture of trees with evidence
Reject sampling
'''    

def sample_from_mt_evid (mt, n_samples, evids):
    
    samples = []
    
    ''' for each component '''
    for i in range (mt.n_components):
        '''make sure num of samples >= n_samples'''  
        sub_n_samples = int(mt.mixture_weight[i]*n_samples)+1
        
        sub_samples = sample_from_tree(mt.clt_list[i],sub_n_samples)
        samples.append(sub_samples)
        

    samples = np.vstack(samples)
    
    np.random.shuffle(samples)
 
    for i in range (evids.shape[0]):
        var = evids[i,0]
        val = evids[i,1]
        ind = np.where(samples[:,var]==val)[0]
        samples = samples[ind]

    # correct the number of samples
    diff = samples.shape[0] - n_samples 
    
    # too many samples
    if diff > 0:
        rand_ind = np.random.randint(samples.shape[0], size=diff)
        
        samples = np.delete(samples, rand_ind, 0)
        
    return samples



'''
Sample from mixture of trees with evidence
Direct sample from posterior distribution
Check slides in Sampling Algorithmsfor Probablistic Graphical models: Gibbs sampling
''' 
def sample_from_mt_evid_posterior (mt, n_samples, evids, non_evid_var):
    
    samples = []
    

    '''
    Compute Pr(H|e)
    '''
    P_he = np.zeros(mt.n_components)
    for i in range (mt.n_components):
        sub_tree = mt.clt_list[i]
        inst_cpt = sub_tree.instantiation(list(evids))
        P_he[i] = utilM.ve_tree_bin(sub_tree.topo_order, sub_tree.parents, inst_cpt)* mt.mixture_weight[i]
        
    p_h_given_e = P_he/np.sum(P_he)
    
   
    ''' for each component '''
    for i in range (mt.n_components):
        '''make sure num of samples >= n_samples'''  
        sub_n_samples = int(p_h_given_e[i]*n_samples)+1
        sub_samples = sample_from_tree_evid(mt.clt_list[i],sub_n_samples, evids, non_evid_var)
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
        print ('sub_n_samples: ', sub_n_samples)
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


def read_evidence_file(file_dir, evid_percent, data_name):
    input =  open(file_dir + 'evids_'+ str(int(evid_percent*100)))
    in_lines =  input.readlines()
    input.close()
    
    total_datasets = int(len(in_lines) / 3)
    for i in range(total_datasets):
        if in_lines[3*i].strip() == data_name:
            # var
            evid = in_lines[3*i+1].strip().split(',')
            evid[0] = evid[0][1:]
            evid[-1] = evid[-1][:-1]
            
            #v value of var
            evid_val = in_lines[3*i+2].strip().split(',')
            evid_val[0] = evid_val[0][1:]
            evid_val[-1] = evid_val[-1][:-1]
            
            break
    
    evid_arr = np.zeros((2, len(evid)), dtype = int)
    evid_arr[0] = np.array(evid).astype(int)
    evid_arr[1] = np.array(evid_val).astype(int)
    
    return evid_arr.T
