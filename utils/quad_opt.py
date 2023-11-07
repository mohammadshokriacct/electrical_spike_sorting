import quadprog
import numpy as np

def quadprog_solve_qp(P:np.ndarray, q:np.ndarray, G:np.ndarray=None, \
    h:np.ndarray=None, A:np.ndarray=None, b:np.ndarray=None) -> np.ndarray:
    """Solve the following quadratic optimization:
        argmin_x(0.5 x^T P x + q x)
        s.t. A x = b
             G x <= h

    Args:
        P (np.ndarray): P matrix
        q (np.ndarray): q vector
        G (np.ndarray, optional): G matrix. Defaults to None.
        h (np.ndarray, optional): h vector. Defaults to None.
        A (np.ndarray, optional): A matrix. Defaults to None.
        b (np.ndarray, optional): b vector. Defaults to None.

    Returns:
        np.ndarray: solution of optimization
    """
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]