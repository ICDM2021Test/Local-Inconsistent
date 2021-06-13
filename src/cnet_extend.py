
"""
Convert the cnet into an array format
Created on Wed Oct  2 22:38:36 2019

"""



import numpy as np
LOG_ZERO = -np.inf 
import JT2
import time
import sys

from CNET_class import CNET   

"""
Array represent chow-liu Tree
"""

class CLT_ARR():
    
    def __init__(self, t_vars):
        print ('Chow liu tree in array format')
    

"""
Array represent cnet
"""

class CNET_ARR() :
    
    def __init__(self, n_variables, depth):
    
        self.n_variables = n_variables
        self.depth = depth
        self.cnode_info = []         # 4 * m matrix, m represent (number of cnode + number of leaf clt noder)
                                        # 1st row var_id of the cnode, negative number indicate leaf, -1 is the first clt, -2 is the second, .etc
                                        # 2nd row is the left weight of the cnode
                                        # 3rd row is the right weight of the cnode
                                        # 4th row is the path (product from root to itself) value
                                        
        self.leaf_info_list = []    # each element in the list represent a leaf clt using array
                                    # In every element, the array has 3 rows:
                                        # 1st row is var_ids
                                        # 2nd row is parents
                                        # 3rd row is topo order
        
        self.leaf_cpt_list =[]      #each element in the list represent a leaf clt's conditional cpt, using array
        
        self.cnode_ind = []                                
        self.leaf_ind = []          # the index of leaf node that in cnode_arr
        self.non_ind = []           # the index of none node that in cnode_arr
        
        self.path_value = []
        
        
        # for getting marginals
        self.JT_list = []
        self.path =  []
        self.node_in_path = {}
    '''
    Convert CCNET to arr CNET
    '''
    
    def convert_to_arr_ccnet(self,root):
        # DFS        
        
        
        cnode_list =[]
        leaf_ind = 0   # indicate the sequence number of leaf
    
        
        # BFS
        # we need to form the complete tree in order to perform tricks
        non_value = -np.inf  # fill the none 
        curr_depth = 0
        nodes_to_process = [root]
        
    
        while nodes_to_process:
            child_level = []
            curr_depth += 1
            for curr_node in nodes_to_process:
                
                
                # None value
                if curr_node is None:
                    cnode_list.append(np.array([non_value,1,1,1])) 
                    if curr_depth < self.depth:
                        child_level.append(None)
                        child_level.append(None)
                
                elif curr_node['type'] == 'internal':
            
            
                    # get the weights
                    p0 = curr_node['p0']
                    p1 = curr_node['p1']
                    p0 = float(p0) / (p0+p1)
                    p1 = 1.0 - p0
                    cnode_list.append(np.array([curr_node['x'], p0, p1,1]))
                    
                    child_level.append(curr_node['c0'])
                    child_level.append(curr_node['c1'])
                # leaf node
                elif curr_node['type'] == 'leaf':
                
                    cnode_list.append(np.array([leaf_ind-1,1,1,1])) # indicate the leaf node
                    leaf_ind -= 1
                    
                    single_leaf_info_list = []  # 1st line is var_ids
                                                # 2nd is parents
                                                # 3rd topo order                
                    single_leaf_info_list.append(curr_node['ids']) 
                    single_leaf_info_list.append(curr_node['parents']) 
                    single_leaf_info_list.append(curr_node['topo_order']) 
                    self.leaf_info_list.append(np.asarray(single_leaf_info_list))
                    
                    
                    self.leaf_cpt_list.append(curr_node['cond_cpt'])
                    
                    if curr_depth < self.depth:
                        child_level.append(None)
                        child_level.append(None)
                
                
                        
            nodes_to_process = child_level
        

        
        self.cnode_info = np.array(cnode_list).T
        
        
        self.non_ind = np.where(self.cnode_info[0] == non_value)[0]
        self.leaf_ind = np.setdiff1d(np.where(self.cnode_info[0] < 0)[0], self.non_ind)
        self.cnode_ind = np.where(self.cnode_info[0] >= 0)[0]
        
    
    '''
    Compute the path value without any evidence
    '''
    def compute_path_value(self):
        
        # number of paths = number of leaves
        #path_values = np.zeros(self.leaf_ind.shape[0])
        
        for i in range(int((self.cnode_info.shape[1]-1)/2)):
            self.cnode_info[3,2*i+1] = self.cnode_info[3,i] * self.cnode_info[1,i]  # left branch
            self.cnode_info[3,2*i+2] = self.cnode_info[3,i] * self.cnode_info[2,i]  # right branch
            
        
            
        self.path_value = self.cnode_info[3, self.leaf_ind]
        
        return self.path_value
    
    '''
    Get which path in the  cnet is not valid base on evidences
    '''
    
    def update_path_val_with_evids(self, path_val, evid_var, evid_val):
        
        evid_str = set()
        for i in range (len(evid_var)):
            if evid_val[i] == 0:
                sign = '+'
            else:
                sign = '-'
            evid_str.add(str(evid_var[i]) + sign)
            
        cnode_eliminate = evid_str.intersection(self.node_in_path.keys())
        
        
        path_eliminate = []
        for x in cnode_eliminate:
            path_eliminate += self.node_in_path[x]
        
        # remove duplicate
        path_eliminate = list(set(path_eliminate))
        
        path_val[np.asarray(path_eliminate)]  = 0
        
            
        
        return path_val
    
    
    def get_tree_evid_info(self, t_vars, evid_var, evid_val):
        
        evid_flag = np.full(self.n_variables,-1)
        evid_flag[evid_var] = evid_val
        
    
        tree_evid_flag = np.delete(evid_flag, np.setdiff1d(np.arange(self.n_variables), t_vars))
        
        tree_evid_ind = np.where(tree_evid_flag > -1)[0]
        
        tree_evid_list = list(np.vstack((tree_evid_ind, tree_evid_flag[tree_evid_ind])).T)

        
        return tree_evid_flag, tree_evid_list
    
    
    """
    Latin Hypercube Sampling
    """
    def sampling(self, path_value_, total_sample_num, evid_var = [], evid_val = []):
        
        print ('-----in sampling----')
   
        path_value = np.copy(path_value_)
        
        if len(self.node_in_path.keys()) == 0:
            
            self.path = self.print_all_paths_to_leaf()
            self.node_in_path = self.get_variable_in_path(self.path)
            
        
        if len(evid_var) > 0:

            path_value = self.update_path_val_with_evids(path_value, evid_var, evid_val)
            
       
        # normalize path_value
                
        path_value /= np.sum(path_value)
        
        
        samples = []
        
        cnode_id = self.cnode_info[0].astype(int) # extract cnode id, to use it as int array, not float array
        
        
        num_of_samples = np.rint(total_sample_num * path_value) # round to int
        
        for i in range(self.leaf_ind.shape[0]):
            
            # pass the branches 
            if path_value[i] == 0:
                continue
            
            #-------------------
            # sample cnodes
            #-------------------
            start = self.leaf_ind[i]
            n_samples = int(num_of_samples[i])
            
            
            
            one_batch_samples = np.zeros((n_samples, self.n_variables), dtype = int)

            
            while start > 0:
               
                parent = np.right_shift(start-1,1)
                

                # left is assign '0', so we don't need to mention
                if start % 2 == 0:    #right
                    
                    one_batch_samples[:,cnode_id[parent]] = 1
                start = parent
            
            
            
            
            #------------------
            # sample leaf tree
            #-----------------
            tree = self.leaf_info_list[i]
            
            t_vars = tree[0]
            topo_order = tree[2]
            parents = tree[1]
            
            cpt = np.copy(self.leaf_cpt_list[i])
   
            
            tree_evid_flag = np.full(t_vars.shape[0],-1)
            if len(evid_var) > 0:
                tree_evid_flag = self.get_tree_evid_flag(t_vars, evid_var, evid_val)
   
            
            tree_samples = np.zeros((n_samples, tree.shape[1]), dtype = int)
            
            # tree root
            if tree_evid_flag[topo_order[0]] > -1:  # evidence
                tree_samples [:, topo_order[0]] = tree_evid_flag[topo_order[0]]
            else:
                nums_0_r = int(np.rint(cpt[0,0,0] * n_samples))
                
                tree_samples [:nums_0_r, topo_order[0]] = 0
                tree_samples [nums_0_r:, topo_order[0]] = 1
            
            
            for j in range (1, tree.shape[1]):
                
                if tree_evid_flag[topo_order[j]] > -1:  # evidence
                    tree_samples [:, topo_order[j]] = tree_evid_flag[topo_order[j]]
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
               
            
            
            one_batch_samples[:,t_vars] = tree_samples
            
            
            
            samples.append(one_batch_samples)
            
        
        samples = np.vstack(samples)
        np.random.shuffle(samples)
        
        
        
        # correct the number of samples
        diff = samples.shape[0] - total_sample_num 
        
        # too many samples
        if diff > 0:
            rand_ind = np.random.randint(samples.shape[0], size=diff)
            
            samples = np.delete(samples, rand_ind, 0)
            
       
        
        if diff < 0:
            extra_sample = self.get_one_sample(path_value_, -diff)
            samples = np.vstack((samples,extra_sample))
            
        
        return samples

                
                
    """
    Sample one by one
    """
    def get_one_sample(self, path_value_, total_sample_num, evid_var =[], evid_val = []):
        
        
        path_value = np.copy(path_value_)
        
        if len(evid_var) > 0:

            
            
            path_eliminate = self.get_eliminated_path(evid_var, evid_val)
            
            path_value[np.asarray(path_eliminate)]  = 0
            
       
        # normalize path_value
        path_value /= np.sum(path_value)
        
        # get the intervals
        path_itervals = np.cumsum(path_value)
        rand_val = np.random.random_sample((total_sample_num,))
        
        
        
        
        samples = []
        for i in range(total_sample_num):
            
            # the indices of the path
            ind = np.argmax(path_itervals > rand_val[i])
            
            
            # sample cnodes
            start = self.leaf_ind[ind]
            
    
            
            
            one_sample = np.zeros( self.n_variables, dtype = int)
            
            while start > 0:
            
                parent = np.right_shift(start-1,1)
                if start % 2 == 0:    #right
                    one_sample[int(self.cnode_info[0,parent])] = 1
                start = parent
            
            
            
            # sample leaf tree
            tree = self.leaf_info_list[ind]
            
            
            
            t_vars = tree[0]
            
            topo_order = tree[2]
            parents = tree[1]
            
            cpt = self.leaf_cpt_list[ind]
            
            
            
        
            
            tree_evid_flag = np.full(t_vars.shape[0],-1)
            if len(evid_var) > 0:
                tree_evid_flag = self.get_tree_evid_flag(t_vars, evid_var, evid_val)
            
            tree_samples = np.zeros(tree.shape[1], dtype = int)
            
            
            
            if tree_evid_flag[topo_order[0]] > -1: # root is evidence
                tree_samples [topo_order[0]] = tree_evid_flag[topo_order[0]]
            else:
                tree_samples [topo_order[0]] = np.random.choice(2, 1, p=cpt[0, :,0])[0]
            
            for j in range (1, tree.shape[1]):
                
                
                if tree_evid_flag[topo_order[j]] > -1: # node is evidence
                    tree_samples [topo_order[j]] = tree_evid_flag[topo_order[j]]
                    continue
                
                
                p_val = tree_samples[parents[topo_order[j]]] # parent value in this sample
                
                tree_samples [topo_order[j]] = np.random.choice(2, 1, p=cpt[j, :,p_val])[0]
            
            one_sample[t_vars] = tree_samples
            
            samples.append(one_sample)
            
        
        return np.asarray(samples)
    
    
    
    """
    # Get the junction list of the leaf tree
    """
    def get_leaf_jt_list(self):
        
        jt_list = []
        
        for t in self.leaf_info_list:        
            jt = JT2.JunctionTree()
            jt.learn_structure(t[2], t[1], np.ones((t[2].shape[0],2,2)))
            jt.ids = t[0]
            
            jt_list.append(jt)
        
        
        self.JT_list = jt_list
        return jt_list
    
    
    
    """
    # function to print all path from root 
    # to leaf in binary tree
    """
    def print_all_paths_to_leaf(self): 
        # list to store path 
        path = [] 
        root_index = 0
        path_len = 0
        self.print_paths_rec(root_index, path, path_len, '-') 
        self.print_paths_rec(root_index, path, path_len, '+') 
        
        self.path = path[len(path[-1]):]
        return self.path
    
    
    """
    # Helper function to print path from root  
    # to leaf in binary tree 
    """
    def print_paths_rec(self, node_ind, path, path_len, sign): 
        
        
        node_var = self.cnode_info[0, node_ind]
        # Base condition - if reach the leaf clt
        if  node_var < 0: 
            return
      

    
        if(len(path) > path_len):  
    
            path[path_len] = str(int(node_var)) + sign
        else: 
            path.append( str(int(node_var)) + sign) 
      
        # increment pathLen by 1 
        path_len +=  1
        
        
        left_child_ind = 2*node_ind+1
        right_child_ind = 2*node_ind+2
        # reach the end of the path
        if self.cnode_info[0, left_child_ind] < 0 and sign == '-':
            path.append(path[:path_len])
            
        
        elif self.cnode_info[0, right_child_ind] < 0 and sign == '+':
            path.append(path[:path_len])
              
            
        else:
            # try for left and right subtree 
            if sign == '-':
                self.print_paths_rec(left_child_ind, path, path_len, '-') 
                self.print_paths_rec(left_child_ind, path, path_len, '+') 
            if sign == '+':
                self.print_paths_rec(right_child_ind, path, path_len, '-') 
                self.print_paths_rec(right_child_ind, path, path_len, '+') 
    
    
    """
    return a dictionary
    key is each node var,+ means when the value of the var is 1 while - means 0
    value is the index of the path, in list format
    """
    def get_variable_in_path(self, path):
        node_in_path = {}
        for i, p in enumerate(path):
            for n in p:
                if n in node_in_path.keys():
                    node_in_path[n].append(i)
                else:
                    node_in_path[n] = [i]
        
        
        
        self.node_in_path = node_in_path
        return node_in_path

    
    # Get marginals for all variables
    def get_marginals(self, path, node_in_path, jt_list, evid_var = [], evid_val = []):
        if len(self.path_value) == 0:
            self.compute_path_value()
        
        temp_path_value = np.copy(self.path_value)
        marginals = np.zeros((self.n_variables, 2))
        all_vars = np.arange(self.n_variables)
        
        # eliminate paths regards to the evidence
        
        if len(evid_var) > 0:
            
            
            evid_str = set()
            for i in range (len(evid_var)):
                if evid_val[i] == 0:
                    sign = '+'
                else:
                    sign = '-'
                evid_str.add(str(evid_var[i]) + sign)
                
            cnode_eliminate = evid_str.intersection(node_in_path.keys())
            
           
            
            path_eliminate = []
            for x in cnode_eliminate:
                path_eliminate += node_in_path[x]
            
            # remove duplicate
            path_eliminate = list(set(path_eliminate))
            
            
            if len(path_eliminate) > 0:
                temp_path_value[np.asarray(path_eliminate)]  = 0
            
            
       
        for i in range(len(path)):
            # if path contains opposite value of the evidence, then skip
            if temp_path_value[i] == 0:
                continue
            
            
                    
            
            # leaf part
            jt = jt_list[i]
            # update clique potential
            jt.set_clique_potential(self.leaf_cpt_list[i])
            
            
            jt.clique_potential = np.log(jt.clique_potential)
            
            
            # map evidence to the evidence in junction tree
            
            jt_evid_list = []
            if len(evid_var) > 0:
                evid_flag = np.full(self.n_variables,-1)
                evid_flag[evid_var] = evid_val
                
                temp_flag = np.delete(evid_flag, np.setdiff1d(all_vars, jt.ids))
                jt_evid_ind = np.where(temp_flag > -1)[0]
                jt_evid_list = list(np.vstack((jt_evid_ind, temp_flag[jt_evid_ind])).T)
                
             
            
            
            jt.set_evidence_log(jt_evid_list)
            marginal_var = np.zeros((jt.ids.shape[0], 2))
             
            jt.propagation()
            
            
            marginal_var = np.exp(JT2.get_marginal(jt.clique_potential, jt.clique_var_arr[:,0]))
            
            marginals[jt.ids] += marginal_var * self.path_value[i]
            
            
            
            
            for cn in path[i]:
                leaf_evid_prob = marginal_var[0,0]+marginal_var[0,1]
                if cn[1] == '-':
                    marginals[int(cn[0:-1]),0] += self.path_value[i] * leaf_evid_prob
                else:
                    marginals[int(cn[0:-1]),1] += self.path_value[i] * leaf_evid_prob
        
        return marginals
        

                
    


def save_cnet(main_dict, node, ids):
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
        save_cnet(main_dict['c0'], node0, ids)
        save_cnet(main_dict['c1'], node1, ids)
    else:
        main_dict['type'] = 'leaf'
        
        main_dict['cond_cpt'] =  node.cond_cpt
        main_dict['topo_order'] = node.topo_order
        main_dict['parents'] = node.parents
        main_dict['ids'] = node.ids           #2
        return




def main_cnet_arr():
    """
    Test functions
    """         
    
 
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    depth = int(sys.argv[6])
    

    #train_filename = sys.argv[1]
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    
    
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    
    
    n_variables = train_dataset.shape[1]
    
    """
    cnet
    """

    print("Learning Cutset Networks only Training data.....")
    
    
    cnet = CNET(depth=depth)
    cnet.learnStructure(train_dataset)
    
    
    # save to folder
    main_dict = {}
    main_dict['depth'] = depth
    main_dict['n_variables'] = n_variables
    main_dict['structure'] = {}
    save_cnet(main_dict['structure'], cnet.tree, np.arange(n_variables))
    
    cnet_a = CNET_ARR(main_dict['n_variables'], main_dict['depth'])
    cnet_a.convert_to_arr_ccnet(main_dict['structure'])
    
    start = time.time()
    path_value = cnet_a.compute_path_value()
    print ('path value: ', path_value)
    
    
    '''
    Get samples without evidence
    '''
    samples = cnet_a.sampling(path_value, 10000)
    samples = cnet_a.get_one_sample(path_value, 10000)
    print ('total running time: ', time.time() - start)

        
    print (np.sum(samples, axis = 0) / float(samples.shape[0]))
    print (np.sum(train_dataset, axis = 0) / float(train_dataset.shape[0]))
    
    
    # '''
    # Get samples with evidence
    # '''
    # evid_var = np.array([5,12,1,8])
    # evid_val = np.array([0,1,1,0])
    # samples = cnet_a.sampling(path_value, 1000000, evid_var, evid_val)
    
    # print (np.sum(samples, axis = 0) / float(samples.shape[0]))
    
    # train_with_evid = train_dataset[np.where(train_dataset[:,5]==0 )[0]]
    # train_with_evid = train_with_evid[np.where(train_with_evid[:,12]==1 )[0]]
    # train_with_evid = train_with_evid[np.where(train_with_evid[:,1]==1 )[0]]
    # train_with_evid = train_with_evid[np.where(train_with_evid[:,8]==0 )[0]]
    
    # print (np.sum(train_with_evid, axis = 0) / float(train_with_evid.shape[0]))
    
    
"""
Using the cnet array to get marginals for all variables
"""
def main_cnet_arr_marginals():
    """
    Test functions
    """         
    
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    depth = int(sys.argv[6])
    
    
    #train_filename = sys.argv[1]
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    
    
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    
    n_variables = train_dataset.shape[1]
    
    """
    cnet
    """
    print("Learning Cutset Networks only Training data.....")
    
    
    cnet = CNET(depth=depth)
    cnet.learnStructure(train_dataset)
    
    
    # save to folder
    main_dict = {}
    main_dict['depth'] = depth
    main_dict['n_variables'] = n_variables
    main_dict['structure'] = {}
    save_cnet(main_dict['structure'], cnet.tree, np.arange(n_variables))
    
    
    cnet_a = CNET_ARR(main_dict['n_variables'], main_dict['depth'])
    cnet_a.convert_to_arr_ccnet(main_dict['structure'])
    
    

    JT_list = cnet_a.get_leaf_jt_list()
    path = cnet_a.print_all_paths_to_leaf()
    node_in_path = cnet_a.get_variable_in_path(path)
    print ('path:', path)
    
    # first test: get marginals without evidence
    marginals_no_evid = cnet_a.get_marginals( path, node_in_path, JT_list)
    print ('marginals:')
    print (marginals_no_evid)
    print (np.sum(marginals_no_evid, axis =1))
    
    # first test: get marginals with evidence
    evid_var = np.array([5,12,1,8])
    evid_val = np.array([0,1,1,0])
    marginals_w_evid = cnet_a.get_marginals( path, node_in_path, JT_list, evid_var, evid_val)
    print ('marginals with evidence:')
    print (marginals_w_evid)
    print (np.sum(marginals_w_evid, axis =1))
    
    
    print ('normalized marginals:')
    print (marginals_w_evid / np.sum(marginals_w_evid, axis =1)[0])
    

    


if __name__=="__main__":
    
    start = time.time()
    main_cnet_arr_marginals()
    main_cnet_arr()
    print ('Total running time: ', time.time() - start) 