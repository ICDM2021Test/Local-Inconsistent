
"""
Created on Mon Jun 25 20:28:19 2018

@author: 
"""

import numpy as np
import numba
from Util import *
import time

LOG_ZERO = -np.inf

@numba.jit
def get_sample_ll(samples,topo_order, parents, log_cond_cpt):
    
    nvariables= samples.shape[1]
    probs = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        for j in range(nvariables):
            x = topo_order[j]
            assignx=samples[i,x]
            # if root sample from marginal
            if parents[x] == -9999:
                probs[i] += log_cond_cpt[0, assignx, 0]
            else:
                # sample from p(x|y)
                y = parents[x]
                assigny = samples[i,y]
                probs[i] += log_cond_cpt[j, assignx, assigny]
    return probs

@numba.jit
def get_tree_dataset_ll(dataset, topo_order, parents, log_cond_cpt):
    
    prob = 0.0
    nvariables= dataset.shape[1]
    for i in range(dataset.shape[0]):
        for j in range(nvariables):
            x = topo_order[j]
            assignx=dataset[i,x]
            # if root sample from marginal
            if parents[x] == -9999:
                prob += log_cond_cpt[0, assignx, 0]
            else:
                # sample from p(x|y)
                y = parents[x]
                assigny = dataset[i,y]
                prob += log_cond_cpt[j, assignx, assigny]
    return prob

@numba.jit
def get_single_ll(sample,topo_order, parents, log_cond_cpt):
    
    nvariables= sample.shape[0]
    prob = 0.0
    for j in range(nvariables):
        x = topo_order[j]
        assignx=sample[x]
        # if root sample from marginal
        if parents[x] == -9999:
            prob += log_cond_cpt[0, assignx, 0]
        else:
            # sample from p(x|y)
            y = parents[x]
            assigny = sample[y]
            prob += log_cond_cpt[j, assignx, assigny]
    return prob




@numba.jit
def updata_coef(curr_rec, total_rec, lamda, function):
    
    
    ratio = float(curr_rec) / total_rec 
    
    if function == 'linear':
        return lamda * ratio
    
    if function == 'square':
        return lamda * ratio **(2)
    
    if function == 'root':
        return lamda * ratio**(0.5)
    
    
    return lamda
    

@numba.jit
# The varible eliminate for tree structure with only binary variables
def ve_tree_bin_log(topo_order, parents, log_cond_cpt):
    
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.zeros((nvariables,2))
    
    # loop in reverse order, this loop exclude the root 
    for i in range(nvariables-1, 0, -1):
    
        x = topo_order[i]
        y = parents[x]
    
        single_cpt = np.copy(log_cond_cpt[i])
        cpt = np.zeros(2) #only handle binary
        cpt_child = cpt_income[x]  
        
        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]
        
        
        cpt[0] += np.logaddexp(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += np.logaddexp(single_cpt[0,1], single_cpt[1,1])
        
        
        cpt_income[y] += cpt
        
       
    # the root node:
    root_cpt = np.zeros(2)
    root_cpt = cpt_income[0] + log_cond_cpt[0,:,0]
    
    return np.logaddexp(root_cpt[0], root_cpt[1])


@numba.jit
# Using max instead of sum in varible eliminate for tree structure with only binary variables
# return the max probablity as well as 
def max_tree_bin_log(topo_order, parents, log_cond_cpt):
    
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.zeros((nvariables,2))
    
    # loop in reverse order, this loop exclude the root 
    for i in range(nvariables-1, 0, -1):
        
        
        x = topo_order[i]
        
        y = parents[x]

        single_cpt = np.copy(log_cond_cpt[i])
        
        cpt = np.zeros(2) #only handle binary
        cpt_child = cpt_income[x]  
        
        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]
        
        cpt[0] += max(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += max(single_cpt[0,1], single_cpt[1,1])
        
        cpt_income[y] += cpt
        
    
    # the root node:
    root_cpt = np.zeros(2)
    root_cpt = cpt_income[0] + log_cond_cpt[0,:,0]
    
    
    return max(root_cpt[0], root_cpt[1])


@numba.jit
# Using max instead of sum in varible eliminate for tree structure with only binary variables
# return the max probablity as well as the map tuple
def max_tree_bin_map(topo_order, parents, log_cond_cpt):
    
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.zeros((nvariables,2))
    # This array contains the max assignment of child node given parent value
    # the index in parent assignment, the value is child assignment
    # [1,0] means when p=0, max assginment of  c is 1, when p=1, max assginment of  c is 0
    # based on natual incremental order
    max_reserve_arr = np.zeros((nvariables, 2))
    
    # loop in reverse order, this loop exclude the root 
    for i in range(nvariables-1, 0, -1):
        
        x = topo_order[i]
       
        y = parents[x]
    
        single_cpt = np.copy(log_cond_cpt[i])
        
        cpt = np.zeros(2) #only handle binary
        cpt_child = cpt_income[x]  
        
        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]
        
        # when tie, always choose 0
        if single_cpt[0,0] >= single_cpt[1,0]:
            max_reserve_arr[x,0] = 0
            cpt[0] += single_cpt[0,0]
        else:
            max_reserve_arr[x,0] = 1
            cpt[0] += single_cpt[1,0]
        
        if single_cpt[0,1] >= single_cpt[1,1]:
            max_reserve_arr[x,1] = 0
            cpt[1] += single_cpt[0,1]
        else:
            max_reserve_arr[x,1] = 1
            cpt[1] += single_cpt[1,1]
        
            
        
        cpt_income[y] += cpt
       
    # the root node:
    root_cpt = np.zeros(2)
    root_cpt = cpt_income[0] + log_cond_cpt[0,:,0]
    
    
    max_prob = 0.0
    
    if root_cpt[0] >= root_cpt[1]:
        max_prob = root_cpt[0]
        max_reserve_arr[0,:] = 0
    else:
        max_prob = root_cpt[1]
        max_reserve_arr[0,:] = 1
        
    # back propgation to find the assignment
    assign_x =  np.zeros(topo_order.shape[0], dtype =int)
    assign_x[0] = max_reserve_arr[0,0]    
    for i in range(1,topo_order.shape[0]):
        x = topo_order[i]
        y = parents[x]
        assign_x[x] = max_reserve_arr[x,assign_x[y]]
    
   
    return max_prob, assign_x
    


@numba.jit
# The varible eliminate for tree structure with only binary variables
def ve_tree_bin(topo_order, parents, cond_cpt):
    
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.ones((nvariables,2))
    
    # loop in reverse order, this loop exclude the root 
    for i in range(nvariables-1, 0, -1):
        
        
        x = topo_order[i]
       
        y = parents[x]
     
        single_cpt = np.copy(cond_cpt[i])
        cpt = np.ones(2) #only handle binary
        cpt_child = cpt_income[x]  
        
        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]
        
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])
        
        
        cpt_income[y] *= cpt
        
        
        
    
    # the root node:
    root_cpt = np.zeros(2)
    root_cpt = cpt_income[0] * cond_cpt[0,:,0]
    
    return root_cpt[0]+ root_cpt[1]


"""
For deep
"""
@numba.jit
def get_var_prob (topo_order, parents, cond_cpt, var):
    
    nvariables= topo_order.shape[0]
    cpt_income = np.ones((nvariables,2))
    # save for efficient 
    cpt_income_save = np.ones((nvariables,2))
    topo_loc = nvariables-1
    xprob = np.zeros(2)
    
    # loop in reverse order, this loop exclude the root 
    #------------------------------------------
    # 0
    #------------------------------------------    
    for i in range(nvariables-1, 0, -1):
                
        x = topo_order[i]
    
        y = parents[x]
        
        single_cpt = np.copy(cond_cpt[i])
        cpt = np.ones(2) #only handle binary
        cpt_child = cpt_income[x]  
        
        if x==var:
            # locate the x in topo_order
            topo_loc = i
            cpt_income_save = np.copy(cpt_income) # for 1
          
            # x as child
            single_cpt[1,:] = 0
            # x as parent
            cpt_child[1] = 0
       
        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]
        
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])
        
        cpt_income[y] *= cpt
       
        
    # the root node:
    root_cpt = np.copy(cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    
    if topo_order[0] == var:
        root_cpt[1] = 0
        root_cpt_income[1] = 0
    
    root_cpt *= root_cpt_income
    

    xprob[0] = root_cpt[0]+ root_cpt[1]

    
    
    #------------------------------------------
    # 1
    #------------------------------------------ 
    cpt_income = cpt_income_save
    for i in range(topo_loc, 0, -1):
        x = topo_order[i]
        y = parents[x]
        
        single_cpt = np.copy(cond_cpt[i])

        cpt = np.ones(2) #only handle binary
        cpt_child = cpt_income[x]  
        
        
        if x==var:

            # x as child
            single_cpt[0,:] = 0
            # x as parent
            cpt_child[0] = 0
      
        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]
        
        
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])
      
        cpt_income[y] *= cpt
        
       
    # the root node:
    root_cpt = np.copy(cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    
    if topo_order[0] == var:
        root_cpt[0] = 0
        root_cpt_income[0] = 0
    
    root_cpt *= root_cpt_income    
    
    xprob[1] = root_cpt[0]+ root_cpt[1]
   
    return xprob


"""
For deep
"""
def get_edge_prob(topo_order, parents, cond_cpt, edges):
    xyprob = np.zeros((edges.shape[0], 2, 2))
    for i in range (edges.shape[0]):
        x = edges[i,0]
        y = edges[i,1]
        xyprob[i,:,:] = ve_tree_bin2(topo_order, parents, cond_cpt, x, y)
    return xyprob
    
    
    

# The varible eliminate for tree structure with only binary variables
def ve_tree_bin_fast(topo_order, parents, cond_cpt, var1, var2):
    
    
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.ones((nvariables,2))
    cpt_income_extra = np.ones((2,2,2)) # store the cpt that contain var1, var2 as child
    # the current parenents for var1 and var2    
    curr_p_var = np.array([parents[var1], parents[var2]])
    ind = np.zeros(2, dtype = int)   
    ind[0] =  np.argwhere (topo_order == var1)# the index in topo for var1
    ind[1] =  np.argwhere (topo_order == var2)# the index in topo for var2
    
    cpt_income_extra[0] = cond_cpt[ind[0]]
    cpt_income_extra[1] = cond_cpt[ind[1]]
    
    ind_remain = np.delete(np.arange(nvariables), ind)
    elimi_order = topo_order[ind_remain]
    elimi_cpt = cond_cpt[ind_remain]
    
    
    
    if elimi_order[0] == 0:
        last_ind = 0
    else:
        last_ind = -1

    # loop in reverse order, this loop exclude the root 
    for i in range(elimi_order.shape[0]-1, last_ind, -1):
        
        
        x = elimi_order[i]
        
        y = parents[x]

        single_cpt = np.copy(elimi_cpt[i])
        
        cpt = np.ones(2) #only handle binary
        cpt_child = cpt_income[x]  
        
        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]
        
        # x has child that is var1 or var2, here x may not be the direct parents, could be ancestor
        if x in curr_p_var:
            # has the same ancenstor
            if curr_p_var.shape[0] == 1:
                cpt_income_extra = np.einsum('lk,ijl ->ijk',single_cpt,cpt_income_extra)
                curr_p_var[0] = y
                
                
            else:
                
                # the cloest ancestor
                if curr_p_var[0] == curr_p_var[1]:
                    cpt_income_extra = np.einsum('lk,il, jl ->ijk',single_cpt,cpt_income_extra[0], cpt_income_extra[1])
                    curr_p_var = np.array([y])
                    
                else:
                    if x == curr_p_var[0]:
                        cpt_income_extra[0] = np.einsum('ki,jk ->ji',single_cpt,cpt_income_extra[0])
                        curr_p_var[0] = y
                        
                    elif x == curr_p_var[1]:
                         cpt_income_extra[1] = np.einsum('ki,jk ->ji',single_cpt,cpt_income_extra[1])
                         curr_p_var[1] = y
                        
                    
            
        else:
            cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
            cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])
            
    
            cpt_income[y] *= cpt
        
    # the root node
    if var1 == 0:
        pxy = np.einsum('i,i, ji ->ij',cpt_income[0],cpt_income_extra[0,:,0], cpt_income_extra[1])
        
        pxy = np.einsum('j, ij ->ij',cpt_income[var2],pxy)
        
        
    elif var2==0:
        pxy = np.einsum('j,j, ij ->ij',cpt_income[0],cpt_income_extra[1,:,0], cpt_income_extra[0])
        pxy = np.einsum('i, ij ->ij',cpt_income[var1],pxy)
    # sumout the root value
    else:  
        # root is not the first ancestor of var1 and var2
        # we have var1, var2, 0 in cpt_income_extra
        if curr_p_var.shape[0] == 1: 
            pxy = np.einsum('k,k, ijk ->ij',cpt_income[0],cond_cpt[0,:,0], cpt_income_extra)
        else:
            if curr_p_var[0] == 0:
                if curr_p_var[1] == 0:  # we have var1, 0 and var2, 0
                    pxy = np.einsum('k,k, ik, jk ->ij',cpt_income[0],cond_cpt[0,:,0], cpt_income_extra[0], cpt_income_extra[1])
                else:               # we have va1, 0 and var2, var1
                    pxy = np.einsum('k,k, ik, ji ->ij',cpt_income[0],cond_cpt[0,:,0], cpt_income_extra[0], cpt_income_extra[1])
            else:         # we have va1, var2 and var2, 0
                pxy = np.einsum('k,k, ij, jk ->ij',cpt_income[0],cond_cpt[0,:,0], cpt_income_extra[0], cpt_income_extra[1])                   
            
        pxy = np.einsum('i,j, ij ->ij',cpt_income[var1],cpt_income[var2], pxy)
       
    
    return pxy





#@numba.jit
# The varible eliminate for tree structure with only binary variables
# compute P(0,0), P(0,1), P(1,0), P(1,1) at the same time
def ve_tree_bin2(topo_order, parents, cond_cpt, var1, var2):
    
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.ones((nvariables,2))
    cpt_income_save =  np.ones((3,nvariables,2))
    topo_loc = np.zeros(2, dtype=np.uint32)
    p_xy = np.zeros((2,2))
    flag = False
    
    #-------------------------------------------------------
    # (0, 0)  along the topo_order 
    #-------------------------------------------------------
    # loop in reverse order, this loop exclude the root 
    for i in range(nvariables-1, 0, -1):
        
        
        x = topo_order[i]
        
        y = parents[x]
       
        single_cpt = np.copy(cond_cpt[i])
        
        cpt = np.ones(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])
        
        if x==var1 or x== var2:
            # locate the x in topo_order
            if flag == False:
                topo_loc[0] = i
                cpt_income_save[1] = np.copy(cpt_income) # for 11
                flag = True
            else:
                topo_loc[1] = i
                cpt_income_save[0] = np.copy(cpt_income) # for 01
    
            # x as child
            single_cpt[1,:] = 0
            # x as parent
            cpt_child[1] = 0
            
        
        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]
        
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])
        
     
        cpt_income[y] *= cpt
        
       
    
    # the root node:
    root_cpt = np.copy(cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    
    if topo_order[0] == var1 or topo_order[0] == var2:
        cpt_income_save[0] = np.copy(cpt_income) # for 01, special case
        root_cpt[1] = 0
        root_cpt_income[1] = 0
    
    root_cpt *= root_cpt_income
    
    p_xy[0,0] =  root_cpt[0]+ root_cpt[1]
    
    
    #
    #-------------------------------------------------------
    # (1, 0) along the topo_order
    #-------------------------------------------------------  
    cpt_income = cpt_income_save[0]
   
    for i in range(topo_loc[1], 0, -1):
        
        x = topo_order[i]
        
        y = parents[x]
        single_cpt = np.copy(cond_cpt[i])
        cpt = np.ones(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])
        
        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[0,:] = 0
            # x as parent
            cpt_child[0] = 0
            
        
        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]
        
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])
        
        cpt_income[y] *= cpt
        
        
        
    
    # the root node:
    root_cpt = np.copy(cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    
    if topo_order[0] == var1 or topo_order[0] == var2:
        root_cpt[0] = 0
        root_cpt_income[0] = 0
    
    root_cpt *= root_cpt_income
    
   
    if topo_order[topo_loc[1]] == var1:
        p_xy[1,0] =  root_cpt[0]+ root_cpt[1]
    else:
        p_xy[0,1] =  root_cpt[0]+ root_cpt[1]
    
    
    #-------------------------------------------------------
    # (1,1) along the topo_order
    #-------------------------------------------------------
    cpt_income = cpt_income_save[1]
    for i in range(topo_loc[0], 0, -1):
        
        
        x = topo_order[i]
        y = parents[x]
       
        single_cpt = np.copy(cond_cpt[i])
        
        cpt = np.ones(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])
        
        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[0,:] = 0
            # x as parent
            cpt_child[0] = 0
            
            if i == topo_loc[1]:
                cpt_income_save[2] = np.copy(cpt_income) # for 01
                
            
        
        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]
        
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])
        
       
        cpt_income[y] *= cpt
        
   # the root node:
    root_cpt = np.copy(cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
   
    if topo_order[0] == var1 or topo_order[0] == var2:
        cpt_income_save[2] = np.copy(cpt_income) # for 10, special case
        root_cpt[0] = 0
        root_cpt_income[0] = 0
    
    root_cpt *= root_cpt_income
    
    p_xy[1,1] =  root_cpt[0]+ root_cpt[1]
    
    
    #-------------------------------------------------------
    # (0,1) along the topo_order
    #-------------------------------------------------------
    cpt_income = cpt_income_save[2]
    for i in range(topo_loc[1], 0, -1):
      
        x = topo_order[i]
        
        y = parents[x]
    
        single_cpt = np.copy(cond_cpt[i])
        
        cpt = np.ones(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])
        
        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[1,:] = 0
            # x as parent
            cpt_child[1] = 0
            
        
        single_cpt[0,:] *= cpt_child[0]
        single_cpt[1,:] *= cpt_child[1]
        
        cpt[0] *= (single_cpt[0,0] + single_cpt[1,0])
        cpt[1] *= (single_cpt[0,1] + single_cpt[1,1])
        
        
        cpt_income[y] *= cpt
        
        
        
    
    # the root node:
    root_cpt = np.copy(cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    
    if topo_order[0] == var1 or topo_order[0] == var2:
        root_cpt[1] = 0
        root_cpt_income[1] = 0
    
    root_cpt *= root_cpt_income
    
    
    if topo_order[topo_loc[1]] == var1:
        p_xy[0,1] =  root_cpt[0]+ root_cpt[1]
    else:
        p_xy[1,0] =  root_cpt[0]+ root_cpt[1]


    # Do Not normalize
    return p_xy

@numba.jit
def ve_tree_bin_log2(topo_order, parents, log_cond_cpt, var1, var2):
    
    
    # all orders are based on topo_order
    nvariables= topo_order.shape[0]
    cpt_income = np.zeros((nvariables,2))
    cpt_income_save =  np.zeros((3,nvariables,2))
    topo_loc = np.zeros(2, dtype=np.uint32)
    p_xy = np.zeros((2,2))
    flag = False
    
    #-------------------------------------------------------
    # (0, 0)  along the topo_order 
    #-------------------------------------------------------
    # loop in reverse order, this loop exclude the root 
    for i in range(nvariables-1, 0, -1):
 
        x = topo_order[i]
        
        y = parents[x]
      
        single_cpt = np.copy(log_cond_cpt[i])
        
        cpt = np.zeros(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])
        
        if x==var1 or x== var2:
            # locate the x in topo_order
            if flag == False:
                topo_loc[0] = i
                cpt_income_save[1] = np.copy(cpt_income) # for 11
                flag = True
            else:
                topo_loc[1] = i
                cpt_income_save[0] = np.copy(cpt_income) # for 01
            #set x = 0
            # x as child
            single_cpt[1,:] = LOG_ZERO
            # x as parent
            cpt_child[1] = LOG_ZERO
            
        
        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]
        
        cpt[0] += np.logaddexp(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += np.logaddexp(single_cpt[0,1], single_cpt[1,1])
        
        cpt_income[y] += cpt
      
    
    # the root node:
    root_cpt = np.copy(log_cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])

    
    if topo_order[0] == var1 or topo_order[0] == var2:
        cpt_income_save[0] = np.copy(cpt_income) # for 01, special case
        root_cpt[1] = LOG_ZERO
        root_cpt_income[1] = LOG_ZERO
    
    root_cpt += root_cpt_income

    p_xy[0,0] =  np.logaddexp(root_cpt[0], root_cpt[1])
    

    #-------------------------------------------------------
    # (1, 0) along the topo_order
    #-------------------------------------------------------
    
    cpt_income = cpt_income_save[0]
    
    for i in range(topo_loc[1], 0, -1):
       
        x = topo_order[i]
       
        y = parents[x]
        
        single_cpt = np.copy(log_cond_cpt[i])
       
        cpt = np.zeros(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])
        
        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[0,:] = LOG_ZERO
            # x as parent
            cpt_child[0] = LOG_ZERO
            
        
        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]
 
        cpt[0] += np.logaddexp(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += np.logaddexp(single_cpt[0,1], single_cpt[1,1])
        
       
        cpt_income[y] += cpt
        
    
    # the root node:
    root_cpt = np.copy(log_cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
   
    if topo_order[0] == var1 or topo_order[0] == var2:
        root_cpt[0] = LOG_ZERO
        root_cpt_income[0] = LOG_ZERO
    
    root_cpt += root_cpt_income
    
    if topo_order[topo_loc[1]] == var1:
        p_xy[1,0] =  np.logaddexp(root_cpt[0], root_cpt[1])
    else:
        p_xy[0,1] =  np.logaddexp(root_cpt[0], root_cpt[1])
    
    
    #-------------------------------------------------------
    # (1,1) along the topo_order
    #-------------------------------------------------------
    
    cpt_income = cpt_income_save[1]
    for i in range(topo_loc[0], 0, -1):
        
        
        x = topo_order[i]
       
        y = parents[x]
       
        single_cpt = np.copy(log_cond_cpt[i])
        cpt = np.zeros(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])
        
        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[0,:] = LOG_ZERO
            # x as parent
            cpt_child[0] = LOG_ZERO
            
            if i == topo_loc[1]:
                cpt_income_save[2] = np.copy(cpt_income) # for 01
                
            
        
        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]
       
        cpt[0] += np.logaddexp(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += np.logaddexp(single_cpt[0,1], single_cpt[1,1])
        
        cpt_income[y] += cpt
       
  # the root node:
    root_cpt = np.copy(log_cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    
    if topo_order[0] == var1 or topo_order[0] == var2:
        cpt_income_save[2] = np.copy(cpt_income) # for 10, special case
        root_cpt[0] = LOG_ZERO
        root_cpt_income[0] = LOG_ZERO
    
    root_cpt += root_cpt_income
    
    
    p_xy[1,1] =  np.logaddexp(root_cpt[0], root_cpt[1])
    
    
    #-------------------------------------------------------
    # (0,1) along the topo_order
    #-------------------------------------------------------
    cpt_income = cpt_income_save[2]
    for i in range(topo_loc[1], 0, -1):

        x = topo_order[i]
        y = parents[x]
        
        single_cpt = np.copy(log_cond_cpt[i])
    
        cpt = np.zeros(2) #only handle binary
        cpt_child = np.copy(cpt_income[x])
        
        if x==var1 or x== var2:
            #set x = 1
            # x as child
            single_cpt[1,:] = LOG_ZERO
            # x as parent
            cpt_child[1] = LOG_ZERO
            
        
        single_cpt[0,:] += cpt_child[0]
        single_cpt[1,:] += cpt_child[1]

        cpt[0] += np.logaddexp(single_cpt[0,0], single_cpt[1,0])
        cpt[1] += np.logaddexp(single_cpt[0,1], single_cpt[1,1])
        
        
        cpt_income[y] += cpt
        
        
    
    # the root node:
    root_cpt = np.copy(log_cond_cpt[0,:,0])
    root_cpt_income = np.copy(cpt_income[0])
    
    if topo_order[0] == var1 or topo_order[0] == var2:
        root_cpt[1] = LOG_ZERO
        root_cpt_income[1] = LOG_ZERO
    
    root_cpt += root_cpt_income
    
    
    if topo_order[topo_loc[1]] == var1:
        p_xy[0,1] =  np.logaddexp(root_cpt[0], root_cpt[1])
    else:
        p_xy[1,0] =  np.logaddexp(root_cpt[0], root_cpt[1])

    return p_xy


   
def get_prob_matrix(topo_order, parents, cond_cpt, ids):

    dim = topo_order.shape[0]
    p_xy = np.zeros((dim, dim, 2, 2))
    
    for i, x in enumerate(ids):       
        # find edges that x is the parent
        for j in range(i+1, ids.shape[0]):
            y = ids[j]
            p_xy[x,y,:,:] = ve_tree_bin2(topo_order, parents, cond_cpt, x, y)
            p_xy[y,x,:,:] = p_xy[x,y,:,:]
            # swap
            p_xy[y,x,0,1], p_xy[y,x,1,0] = p_xy[y,x,1,0], p_xy[y,x,0,1]
    
    p_xy = p_xy[:,ids][ids,:]
    
    
    return p_xy

def get_prob_matrix_log(topo_order, parents, log_cond_cpt, ids):
    
    dim = topo_order.shape[0]
    p_xy = np.zeros((dim, dim, 2, 2))
    
    
    
    for i, x in enumerate(ids):       
        # find edges that x is the parent
        for j in range(i+1, ids.shape[0]):
            y = ids[j]
            p_xy[x,y,:,:] = ve_tree_bin_log2(topo_order, parents, log_cond_cpt, x, y)
            p_xy[y,x,:,:] = p_xy[x,y,:,:]
            # swap
            p_xy[y,x,0,1], p_xy[y,x,1,0] = p_xy[y,x,1,0], p_xy[y,x,0,1]
    
    p_xy = p_xy[:,ids][ids,:]
    
    
    # compute p_x    
    p_x = np.zeros((ids.shape[0], 2))
    p_x[:,0] = p_xy[0,:,0,0] + p_xy[0,:,1,0]
    p_x[:,1] = p_xy[0,:,0,1] + p_xy[0,:,1,1]
    
    p_x[0,0] = p_xy[1,0,0,0] + p_xy[1,0,1,0]
    p_x[0,1] = p_xy[1,0,0,1] + p_xy[1,0,1,1]
    
    
    
    for i in range (ids.shape[0]):
        p_xy[i,i,0,0] = p_x[i,0] - 1e-8
        p_xy[i,i,1,1] = p_x[i,1] - 1e-8
        p_xy[i,i,0,1] = 1e-8
        p_xy[i,i,1,0] = 1e-8
    
    
    return p_xy, p_x

def save_cutset(main_dict, node, ids, ccpt_flag = False):
    if isinstance(node,list):
        id,x,p0,p1,node0,node1=node
        main_dict['type'] = 'internal'
        main_dict['id'] = id
        main_dict['x'] = x
        main_dict['p0'] = p0
        main_dict['p1'] = p1
        main_dict['c0'] = {}  # the child associated with p0
        main_dict['c1'] = {}  # the child associated with p0
        
        ids=np.delete(ids,id,0)
        save_cutset(main_dict['c0'], node0, ids, ccpt_flag)
        save_cutset(main_dict['c1'], node1, ids, ccpt_flag)
    else:
        main_dict['type'] = 'leaf'
        
        if ccpt_flag == False:
            node.get_log_cond_cpt()
        main_dict['log_cond_cpt'] =  node.log_cond_cpt
        main_dict['topo_order'] = node.topo_order
        main_dict['parents'] = node.parents
        #main_dict['ids'] = node.ids           #2
        #main_dict['p_xy'] = node.xyprob          #3
        main_dict['p_x'] = node.xprob           #4
        return
            
def computeLL_reload(reload_cutset, dataset):
    probs = np.zeros(dataset.shape[0])
    #cnet = copy.deepcopy(reload_cutset)
    for i in range(dataset.shape[0]):
        cnet = reload_cutset
        prob = 0.0
        ids=np.arange(dataset.shape[1])
     
        while cnet['type'] == 'internal':
            id = cnet['id']
            x  = cnet['x']
            p0 = cnet['p0']
            p1 = cnet['p1']
            c0 = cnet['c0']
            c1 = cnet['c1']

            assignx=dataset[i,x]
            ids=np.delete(ids,id,0)
            if assignx==1:
                prob+=np.log(p1/(p0+p1))
                cnet=c1
            else:
                prob+=np.log(p0/(p0+p1))
                cnet=c0
            
        if cnet['type'] == 'leaf':
            log_cond_cpt = cnet['log_cond_cpt']
            topo_order = cnet['topo_order']
            parents = cnet['parents']
            
            prob += get_single_ll(dataset[i][ids], topo_order, parents, log_cond_cpt)
            probs[i] = prob
    
        else:
            print ("*****ERROR******")
            exit()
        
    return probs


#@numba.jit    
#-------------------------------------------------------------------------------
# log space subtraction
# return log (exp(x) - exp (y)) if x > y
#-------------------------------------------------------------------------------
def log_subtract(x, y):
    if(x < y):
        print ("Error!! computing the log of a negative number \n")
        #return np.log(-1)
        # under our assumption, x < y could not happen, if happens, we believe it is caused by numeric issue
        return LOG_ZERO  
    if (x == y) :
        return LOG_ZERO
    
    return x + np.log1p(-np.exp(y-x))


#@numba.jit    
#-------------------------------------------------------------------------------
# Add an array in log space
#-------------------------------------------------------------------------------
def log_add_arr(log_arr):
    sum_val = LOG_ZERO
    for i in range(log_arr.shape[0]):
        sum_val = np.logaddexp(sum_val, log_arr[i])
        
    return sum_val

