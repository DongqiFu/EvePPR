import numpy as np
import scipy.sparse as sp

# calculate the exact PPR vector by power iteration
def calc_ppr_by_power_iteration(P: sp.spmatrix, alpha: float, h: np.ndarray, t: int) -> np.ndarray:
    iterated = (1 - alpha) * h
    result = iterated.copy()
    for iteration in range(t):
        iterated = (alpha * P).dot(iterated)
        result += iterated
    return result



# udpate v from $\mathbf{v} =  \alpha \mathbf{A} \mathbf{v} + (1-\alpha) \mathbf{h}$ to $\mathbf{v} =  \alpha \mathbf{B} \mathbf{v} + (1-\alpha) \mathbf{h}$
def osp(v: np.ndarray, A: sp.spmatrix, B: sp.spmatrix, alpha: float, epsilon: float, whether_print: int) -> np.ndarray:
    assert A.shape == B.shape, "in osp, the dimension of matrix A should be the same as the dimension of matrix B"
    q_offset = alpha * (B - A) @ v
    v_offset = q_offset.copy()
    x_offset = q_offset.copy()
    number = 0
    # pdb.set_trace()
    while (np.linalg.norm(x_offset, 1) > epsilon):
        number += 1 
        x_offset = alpha * B @ x_offset
        v_offset += x_offset
    if whether_print == 1:
        print("x_offset norm of osp:", np.linalg.norm(x_offset, 1), "iteration number of osp:", number)
        pass
    return v + v_offset



# udpate v from a previous solution. The return value approximately satisfies $\mathbf{v} =  \alpha \mathbf{P} \mathbf{v} + (1-\alpha) \mathbf{h}$
def gauss_southwell(v: np.ndarray, P: sp.spmatrix, h: np.ndarray, alpha: float, epsilon: float) -> np.ndarray:
    dimension_P = P.shape[0]
    x = v
    r = (1 - alpha) * h - (sp.eye(dimension_P) - alpha * P) @ v
    max_index = np.argmax(r)
    number = 0
    while r[max_index] > epsilon:
        e = np.zeros(dimension_P,)
        e[max_index] = 1
        x = x + r[max_index] * e
        r = r - r[max_index] * e + alpha * r[max_index] * P @ e
        max_index = np.argmax(r)
        number += 1 
    print("final residual maximum element:", r[np.argmax(r)])
    return x



# udpate v from $\mathbf{v} =  \alpha \mathbf{P_old} \mathbf{v} + (1-\alpha) \mathbf{h_old}$ to $\mathbf{v} =  \alpha \mathbf{P_new} \mathbf{v} + (1-\alpha) \mathbf{h_new}$
def eveppr_app(v: np.ndarray, P_old: sp.spmatrix, P_new: sp.spmatrix, h_new:np.ndarray, alpha: float, epsilon: float, epsilon2: float):
    v_mid = osp(v, P_old, P_new, alpha, epsilon, 1)
    return gauss_southwell(v_mid, P_new, h_new, alpha, epsilon2)



def calc_onehot_ppr_matrix(P: sp.spmatrix, alpha: float, t: int) -> np.ndarray:
    iterated = (1 - alpha) * sp.eye(P.shape[0])
    matrix_result = iterated.copy()
    for iteration in range(t):
        iterated = (alpha * P).dot(iterated)
        matrix_result += iterated
    return matrix_result            