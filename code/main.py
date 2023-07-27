import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import pdb
import utils
import time

num_nodes = 24818
start_line = 100000
GAP = 20000
epsilon = 1e-4
epsilon2 = 1e-4
alpha = 0.85
load = 1

def get_graph_info_sx(graph_file_name):
    user_set = set()
    graph_file = open(graph_file_name)
    user_to_index = {}
    all_line = graph_file.readlines()
    for line in all_line:
        items = line.strip().split(' ')
        user_set.add(int(items[0]))
        user_set.add(int(items[1]))
    graph_file.close()
    user_list = sorted(user_set)
    for i in range(len(user_set)):
        user_to_index[user_list[i]] = i
    return user_set, all_line, user_to_index


def to_prmatrix(P: sp.spmatrix):
    sums = P.sum(axis = 0)
    Q = sp.lil_matrix(P.shape)
    P_t = P.transpose()
    counter = 0
    for i in range(P.shape[0]):
        if sums[0, i] != 0:
            Q[i, :] = P_t[i, :]/sums[0, i]
        counter += 1
        if counter % 2000 == 0:
            print(counter)
    Q = Q.transpose()
    return Q.tocsr()

def construct_pre_knowledge(P: sp.spmatrix):
    total_edge = P.count_nonzero()
    sums = P.sum(axis = 0)
    h = np.ones(P.shape[0])
    for i in range(P.shape[0]):
        h[i] = sums[0, i]
    h = h/total_edge
    return h
    

if __name__ == '__main__':
    user_set, all_line, user_to_index = get_graph_info_sx('./sx-mathoverflow.txt')
    end_line = len(all_line)
    v_exact_list = []
    v_eveppr_app_list = []
    v_eveppr_list = []
    tv_exact_list = []
    tv_eveppr_app_list = []
    tv_eveppr_list = []
    P_dict = {}
    adj_matrix = sp.lil_matrix((num_nodes, num_nodes))
    pr_matrix = sp.lil_matrix((num_nodes, num_nodes))
    # construct initial graph
    for i in range(start_line):
        items = all_line[i].strip().split(' ')
        user_index1 = (user_to_index[int(items[0])])
        user_index2 = (user_to_index[int(items[1])])
        adj_matrix[user_index1, user_index2] = 1
    if load == 0:
        pr_matrix = to_prmatrix(adj_matrix)
        h = construct_pre_knowledge(adj_matrix)
        sp.save_npz('./saved_data' + '/P_matrix_start' + '.npz', pr_matrix)
        np.save('./saved_data' + '/h_array_start.npy', h)
    else:
        pr_matrix = sp.load_npz('./saved_data' + '/P_matrix_start.npz')
        h = np.zeros(num_nodes)
        h[0] = 1


    P_dict[0] = pr_matrix
    # the one-hot solution
    q = {}
    M = {}
    P = pr_matrix.tocsr()
    onehot_ppr = utils.calc_onehot_ppr_matrix(P, alpha, 2).tolil().transpose()
    for i in range(num_nodes):
        q[i] = onehot_ppr[i, :]
        M[i] = P_dict[0]

    v_initial = utils.calc_ppr_by_power_iteration(pr_matrix, alpha, h, 30)
    v_exact_list.append(v_initial)
    v_eveppr_app_list.append(v_initial)
    v_eveppr_list.append(v_initial)

    timestamp = 0
    max_iter = 5

    for line_id in range(start_line, end_line):
        items = all_line[line_id].strip().split(' ')
        user_index1 = (user_to_index[int(items[0])])
        user_index2 = (user_to_index[int(items[1])])
        adj_matrix[user_index1, user_index2] = 1
        
        if ((line_id + 1) % GAP == 0):
            max_iter -= 1
            if (max_iter < 0):
                break    
            timestamp += 1
            print("change num:", timestamp)
            if load == 0:
                pr_matrix_new = to_prmatrix(adj_matrix)
                h_new = construct_pre_knowledge(adj_matrix)
                sp.save_npz('./saved_data' + '/P_matrix_' + str(timestamp) + '.npz', pr_matrix_new)
                np.save('./saved_data' + '/h_array_' + str(timestamp) + '.npy', h_new)
            else:
                pr_matrix_new = sp.load_npz('./saved_data' + '/P_matrix_' + str(timestamp) + '.npz')
                h_new = np.zeros(num_nodes)
                h_new[timestamp] = 1


            start_time = time.time()

            # Exact PPR by power iteration
            v_exact_list.append(utils.calc_ppr_by_power_iteration(pr_matrix_new, alpha, h_new, 30))
            exact_time = time.time()
            tv_exact_list.append(exact_time - start_time)

            # EvePPR-APP
            v_eveppr_app_list.append(utils.eveppr_app(v_eveppr_app_list[timestamp-1], pr_matrix, pr_matrix_new, h_new, alpha, epsilon, epsilon2))
            app_time = time.time()
            tv_eveppr_app_list.append(app_time - exact_time)

            # EvePPR
            P_dict[timestamp] = pr_matrix_new.copy()
            v_mid = utils.osp(v_eveppr_list[timestamp-1], pr_matrix, pr_matrix_new, alpha, epsilon, 0)
            delta_h = h_new - h
            for i in range(num_nodes):
                if delta_h[i] != 0:
                    q_new = utils.osp(q[i].toarray().ravel(), M[i], pr_matrix_new, alpha, epsilon, 0)
                    q[i] = sp.lil_matrix(q_new)
                    M[i] = P_dict[timestamp]
                    v_mid = v_mid + delta_h[i] * q_new
            v_eveppr_list.append(v_mid)

            eveppr_time = time.time()
            tv_eveppr_list.append(eveppr_time - app_time)

            pr_matrix = pr_matrix_new.copy()
            h = h_new.copy()

    r_eveppr_app_list = []
    r_eveppr_list = []

    # 2-norm
    for i in range(5):
        r_eveppr_app_list.append(la.norm(v_exact_list[i+1] - v_eveppr_app_list[i+1]))
        r_eveppr_list.append(la.norm(v_exact_list[i+1] - v_eveppr_list[i+1]))


    pdb.set_trace()