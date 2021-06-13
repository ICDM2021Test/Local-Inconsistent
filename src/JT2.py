
"""
Created on Sun Sep 16 11:39:28 2018
# This is the workable verion that only works for tree converted junction tree
# We need LOG here
# This is the version:
#  1) it seems deepcopy in loop is expensive, try to avoid it
#  2) Not the fastest version, if you need faster, replace 
#     the numpy einsum function with plain numpy matrix add and multiplication
#     then import numba.jit
# Running time:
dataset  old    new     new_matrix  remove deepcopy
nltcs:   0.17   0.02    0.01        0.008
jester:  5.11   0.75    0.33        0.25
ad:      4652   202     115         99

"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order
from CLT_class import CLT
import copy
import time
import numba

import sys
#import numba
import utilM
from Util import *

LOG_ZERO = -np.inf

#@numba.jit
def msg_leaf_to_root(topo_order, parents, potential_orig):
    
    msg = np.ones((parents.shape[0],2))
    for i in range(topo_order.shape[0]-1, 0, -1):
        cid = topo_order[i]
        cid_pa  = parents[cid] # parent cid
        # get the msg

        msg[cid,0] = np.logaddexp(potential_orig[cid,0,0], potential_orig[cid,1,0])
        msg[cid,1] = np.logaddexp(potential_orig[cid,0,1], potential_orig[cid,1,1])
        
        potential_orig[cid_pa,0,:] += msg[cid,0]
        potential_orig[cid_pa,1,:] += msg[cid,1]

    return   msg

#@numba.jit
def msg_root_to_leaf(topo_order, children, no_of_chlidren, potential_orig, msg_sent_prev):
    

    for cid in topo_order:
        n_child = no_of_chlidren[cid]
        if n_child == 0:  # no child, pass
            continue
        curr_children  = children[cid, 0: n_child]

    
        
        msg = np.zeros(2)
        msg[0] = np.logaddexp(potential_orig[cid,0,0], potential_orig[cid,0,1])
        msg[1] = np.logaddexp(potential_orig[cid,1,0], potential_orig[cid,1,1])

        msg_sent_prev[msg_sent_prev == LOG_ZERO] = 0
        msg = msg - msg_sent_prev  # exclude the msg that sent by self when doing leaf -> root
        

        potential_orig[curr_children,0,0] +=  msg[curr_children,0]
        potential_orig[curr_children,1,0] +=  msg[curr_children,0]
        potential_orig[curr_children,0,1] +=  msg[curr_children,1]
        potential_orig[curr_children,1,1] +=  msg[curr_children,1]
    
    

    return potential_orig
   


#@numba.jit     
def get_marginal(potential_orig, var_arr):
    
    marginals = np.zeros((var_arr.shape[0], 2))
    marginals[:,0] = np.logaddexp(potential_orig[:,0,0], potential_orig[:,0,1])
    marginals[:,1] = np.logaddexp(potential_orig[:,1,0], potential_orig[:,1,1])
    
    
    return marginals[np.argsort(var_arr)]




# Assume in cliques, all sequence is child|parent
# all pothentail is in log space
class Clique:
     
    def __init__(self,cid, varibles, potential):
        
        # '-1' is reserved to add new node        
        self.cid=cid #the unique id for each clique
        self.var = np.full(3, -1)
        self.var[:2] = varibles  # the varible array that clique contains
        
        self.potential = potential  # the potential functions
        self.parent = None
        self.children = []
        
 
    
       
    # when initial the children list
    def set_child_list(self, child_list):
        self.children = child_list

        
    def set_parent(self, parent):
        self.parent = parent
        
    

 


class JunctionTree:
    
    def _init_(self):
        self.clique_list = []
        self.n_cliques = 0
        self.n_varibles = 0
        self.jt_order = None
        self.jt_parents = None
        self.var_in_clique = {} # a dictionary contains the information indicate the  variable in which clique
                                # make sure the first element under each key is the smallest one
                                # which is the one that cantain the actual information about the key var
        self.ids = None
        self.log_value = -np.inf
        self.evid_var =[]  # this is in sequece of jt, not the original var number in dataset
        
                                 
    
    def learn_structure(self, topo_order, parents, cond_prob):
        self.clique_list = []
        self.n_cliques = topo_order.shape[0]
        self.jt_parents = np.zeros(self.n_cliques)
        self.var_in_clique = {}
        self.n_varibles = topo_order.shape[0]
        
        
        # create a very special clique as root
        root_cpt = np.copy(cond_prob[0])
        root_cpt[0,1] = root_cpt[1,0] = 0
        root_clique = Clique(0, np.array([0, 0]), root_cpt)
        self.clique_list.append(root_clique)
        self.var_in_clique[topo_order[0]] = [topo_order[0]]
        
        # exclude the root node
        for i in range(1, topo_order.shape[0]):
            child = topo_order[i]
            parent = parents[child]
            
            clique_id = i
            new_clique = Clique(clique_id, np.array([child, parent]), cond_prob[i])
            self.clique_list.append(new_clique)
            
            if child in self.var_in_clique:
                self.var_in_clique[child].append(clique_id)
            else:
                self.var_in_clique[child] = [clique_id]                
            if parent in self.var_in_clique:
                self.var_in_clique[parent].append(clique_id)
            else:
                self.var_in_clique[parent] = [clique_id]
                
        

        self.clique_to_tree()
        # Convert clique to matrix
        self.clique_to_matrix()
    
    
    
    def clique_to_tree(self):
                
        neighbors = np.zeros((self.n_cliques, self.n_cliques))
        for k in self.var_in_clique.keys():
            
            nb_val = self.var_in_clique[k]
            nb_num = len(nb_val) # how many cliques that conatain this variable
            
            # for cliques connected to root clique
            if k==0:
                for i in range(nb_num):
                    neighbors[0, nb_val[i]] =1
                    neighbors[nb_val[i], 0] =1                    
                continue
                
            
            if nb_num > 1:
                for i in range(nb_num):
                    for j in range(i+1, nb_num):                        
                        # connect only parent and child, for tree only
                        if self.clique_list[nb_val[i]].var[0] == self.clique_list[nb_val[j]].var[1] \
                        or self.clique_list[nb_val[i]].var[1] == self.clique_list[nb_val[j]].var[0] :
                            neighbors[nb_val[i], nb_val[j]] =1
                            neighbors[nb_val[j], nb_val[i]] =1
                    
    
                    
        # compute the minimum spanning tree
        Tree = minimum_spanning_tree(csr_matrix(neighbors * (-1)))
        # Convert the spanning tree to a Bayesian network
        self.jt_order, self.jt_parents = depth_first_order(Tree, 0, directed=False)
        
        # Get child array
        for i in range(self.n_cliques):
            child_index = np.where(self.jt_parents==i)[0]
            
            if child_index.shape[0] > 0:
                child_list = []
                for c in child_index:
                    child_list.append(self.clique_list[c])
                self.clique_list[i].set_child_list(child_list)
            
            if self.jt_parents[i] != -9999:
                self.clique_list[i].set_parent(self.clique_list[self.jt_parents[i]])
            

        
        
    def set_evidence(self, evid_list):
        # no evidence
        if len(evid_list) == 0:
            return
        for k in range(len(evid_list)):
            evid_id = evid_list[k][0]
            evid_val = evid_list[k][1]
          
            ind = self.var_in_clique[evid_id]
            
            # leaf node in original clt
            ops_val = 1-evid_val  # the oppsite value
            if ind.shape[0] == 0:
                self.clique_potential[ind[0],ops_val,:] = 0
            else:
                # as child
                self.clique_potential[ind[0],ops_val,:] = 0
                # as parent
                self.clique_potential[ind[1:],:,ops_val] = 0
            
            
    
    def set_evidence_log(self, evid_list):
        # no evidence
        if len(evid_list) == 0:
            return
        for k in range(len(evid_list)):
            evid_id = evid_list[k][0]
            evid_val = evid_list[k][1]

            ind = self.var_in_clique[evid_id]
            
            ops_val = 1-evid_val  # the oppsite value
            if ind.shape[0] == 0:
                self.clique_potential[ind[0],ops_val,:] = LOG_ZERO
            else:
                # as child
                self.clique_potential[ind[0],ops_val,:] = LOG_ZERO
                # as parent
                self.clique_potential[ind[1:],:,ops_val] = LOG_ZERO
            
            

    '''
    Start from here, we convert cliques to matrix, no clique will be available
    '''

    # remove object clique, convert everthing to matrix
    def clique_to_matrix(self):
        self.clique_var_arr = np.zeros((self.n_cliques, 3), dtype = int)    # The variable each clique contains
        self.clique_potential = np.zeros((self.n_cliques, 2,2))  # the potential functions
        # parent is the same as jt.parent
        
        self.clique_children = None
        
        # Under our assumption, this is the 'clique id' where the 'var' acctually has information
        self.clique_id_var_asChild = np.zeros(self.n_varibles, dtype = int) 
        
        
        max_width = np.max(np.bincount(self.jt_parents[1:])) # the max number of child in jt
        self.clique_children = np.full((self.n_cliques, max_width),-1)
        self.no_of_chlidren = np.zeros(self.n_cliques, dtype = int)  # how many child for each clique
        
        for clq in self.clique_list:
            self.clique_var_arr[clq.cid] = clq.var
            self.clique_potential[clq.cid] = clq.potential
            
            self.clique_id_var_asChild[clq.var[0]] = clq.cid
            
            for j, ch in enumerate(clq.children):
                self.clique_children[clq.cid,j] = ch.cid
            self.no_of_chlidren[clq.cid] = len(clq.children)
        
        # Delete the clique list
        self.clique_list = None
        

        
        # convert from list to numpy array
        for j in range(0, self.n_varibles):
            # convert list to numpy array
            self.var_in_clique[j] = np.asarray(self.var_in_clique[j])
            
    
        

    
    def propagation(self):
 
        clique_msg_out = msg_leaf_to_root(self.jt_order, self.jt_parents, self.clique_potential)
        
        msg_root_to_leaf(self.jt_order, self.clique_children, self.no_of_chlidren, self.clique_potential, clique_msg_out)

    
    def set_clique_potential(self,clt_cond_cpt):
        self.clique_potential = np.copy(clt_cond_cpt)
        self.clique_potential[0,0,1] = self.clique_potential[0,1,0] = 0
        

                
        
def get_marginal_JT(jt, evid_list, n_varible):
     
     #n_variable = query_var.shape[0]
     jt_dup = copy.deepcopy(jt)
     
     # convert clique_potential to log
     jt_dup.clique_potential = np.log(jt_dup.clique_potential)
     
     jt_dup.set_evidence_log(evid_list)
     
     marginal_var = np.zeros((n_varible, 2))

     jt_dup.propagation()
     
     
     marginal_var = get_marginal(jt_dup.clique_potential, jt_dup.clique_var_arr[:,0])
 
     return marginal_var


def main_jt():
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    
       
    train_name = dataset_dir + data_name +'.ts.data'
    #valid_name = dataset_dir + data_name +'.valid.data'
    test_name = dataset_dir + data_name +'.test.data'
    data_train = np.loadtxt(train_name, delimiter=',', dtype=np.uint32)
    #data_valid = np.loadtxt(valid_name, delimiter=',', dtype=np.uint32)
    data_test = np.loadtxt(test_name, delimiter=',', dtype=np.uint32)
    
    clt = CLT()
    clt.learnStructure(data_train)
    print ('clt testset loglikihood score: ', clt.computeLL(data_test) / data_test.shape[0])
    
    n_variable = data_train.shape[1]
    clt.get_log_cond_cpt()
    
    start_jt = time.time()
    jt = JunctionTree()
    jt.learn_structure(clt.topo_order, clt.parents, clt.cond_cpt)
    print ('time for learning JT: ', time.time() - start_jt)
    
    
    # Compare, using the orignal way to compute pairwise marginal
    evid_list = []
    evid_list.append([0,1])
    evid_list.append([7,0])
    evid_list.append([9,1])
    
    
    start = time.time()
    marginal = get_marginal_JT(jt, evid_list, n_variable)

    print ('running time for new: ', time.time()-start)

if __name__=="__main__":

    start = time.time()
    main_jt()
    print ('Total running time: ', time.time() - start)