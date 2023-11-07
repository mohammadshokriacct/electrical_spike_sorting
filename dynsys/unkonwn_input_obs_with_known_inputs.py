import logging
import numpy as np
from scipy.linalg import null_space
from scipy.signal import convolve2d
from utils.quad_opt import quadprog_solve_qp

def output_matrices_with_known_input(A, B, B_k, C, D, D_k, L_min=1, \
    L_max=10000):
    """Calculate output matrice from the matrice of the control system.
    """
    P = D.shape[0]
    P_k = D.shape[0]
    M = D.shape[1]
    M_k = D_k.shape[1]

    # Initialization
    J_mat = D
    J_mat_k = D_k
    J_mat_prev = D
    O_mat = C

    done = False

    for L in range(1, L_max):
        J_mat_prev = J_mat
        
        J_mat = np.concatenate([np.concatenate(\
            [D, np.zeros([P, L*M])], axis=1), \
            np.concatenate([np.matmul(O_mat, B), J_mat], axis=1)], axis=0)
        J_mat_k = np.concatenate([np.concatenate(\
            [D_k, np.zeros([P_k, L*M_k])], axis=1), \
            np.concatenate([np.matmul(O_mat, B_k), J_mat_k], axis=1)], axis=0)
        O_mat = np.concatenate([C, np.matmul(O_mat, A)], axis=0)
        
        new_rank = np.linalg.matrix_rank(J_mat)
        prev_rank = np.linalg.matrix_rank(J_mat_prev)
        
        if (L >= L_min):
            if (new_rank - prev_rank == M):
                done = True
                break
    return J_mat, J_mat_k, J_mat_prev, O_mat, L, done

def unkown_input_observer_matrice_with_known_input(A, B, B_k, C, D, L, J_mat, \
    J_mat_k, J_mat_prev, O_mat, isolation_data=None, \
    isolation_input_indice=None, isolation_data_indice=None, \
    isolation_coefficient=0):
    """Calculating matrices for designing unknown input observer considering 
        known inputs. 
    Methodology: https://ece.uwaterloo.ca/~ssundara/courses/notes/uiobs.pdf
    """
    N = A.shape[0]
    M = B.shape[1]
    P = C.shape[0]

    # S_total creation
    N_bar = null_space(J_mat_prev.T).T
    N_bar_rank = np.linalg.matrix_rank(N_bar)

    eye_like_mat = np.concatenate([np.zeros([P - M + N_bar_rank, (L+1)*M]), \
        np.concatenate([np.eye(M), np.zeros([M, L*M])], axis=1)], axis=0)
    N_mat = np.matmul(eye_like_mat, np.linalg.pinv(J_mat))
    N_mat[0:(P - M + N_bar_rank),:] = null_space(J_mat.T).T
    S_total = np.matmul(N_mat, O_mat)
    logging.info('S is calculated.')

    # splitting S_total
    S1 = S_total[0:(S_total.shape[0]-M), :]
    S2 = S_total[(S_total.shape[0]-M):, :]

    # observer matrices
    F1 = np.zeros([N, S1.shape[0]])
    E_mat = A - np.matmul(B, S2) - np.matmul(F1, S1)
    if np.any(np.linalg.norm(np.linalg.eig(E_mat)[0]) >= 1):
        logging.info('Matrix shape: ' + str(S1.T.shape))
        A_new = (A - np.matmul(B, S2)).T
        B_new = S1.T
        
        # isolation matrix
        wind = np.zeros([N_mat.shape[1], L + 1])
        wind[(L + 1)*np.arange(L + 1)[::-1], :] = np.eye(L + 1) 

        def isolation_data_convolve(isl_data):
            return convolve2d(isl_data, wind, mode='full', boundary='fill', \
                fillvalue=0)[0:wind.shape[0],:]

        collected_data = list(map(isolation_data_convolve, \
            [x for x in isolation_data]))

        for ind, (startp,endp,isl_data_indice) in \
            enumerate(zip(isolation_input_indice[:-1], \
            isolation_input_indice[1:], isolation_data_indice)):
            logging.info('indice ' + str(ind) + ' out of ' + \
                str(len(isolation_input_indice)-1))
            
            column_isolation_data = np.concatenate([x for i,x in \
                enumerate(collected_data) if i in isl_data_indice], axis=1)
            
            col_isolation_matrix = np.matmul(N_mat[:(-B.shape[1]),:], \
                np.matmul(np.matmul(column_isolation_data, \
                    column_isolation_data.T), N_mat[:(-B.shape[1]),:].T))
            
            # F cols
            den_mat = np.matmul(B_new.T, B_new) + \
                isolation_coefficient*col_isolation_matrix
            num_mat = np.matmul(B_new.T, A_new[:,startp:endp]) - \
                isolation_coefficient*np.matmul(np.matmul(\
                    N_mat[:(-B.shape[1]),:], N_mat[(-B.shape[1]):,:].T), \
                        (B[startp:endp, :]).T)
            
            F1[startp:endp, :] = (np.matmul(np.linalg.pinv(den_mat), num_mat)).T
    
    E_mat = A - np.matmul(B, S2) - np.matmul(F1, S1)
    F_mat = np.matmul(np.concatenate([F1, B], axis=1), N_mat)
    F_mat_k = np.concatenate([B_k, np.zeros([B_k.shape[0], J_mat_k.shape[1] - \
        B_k.shape[1]])], axis=1) - np.matmul(F_mat, J_mat_k)
    G_mat = np.matmul(np.eye(M), np.linalg.pinv(np.concatenate([B, D], axis=0)))
    
    return E_mat, F_mat, F_mat_k, G_mat

def unkown_input_observer_design_with_known_input(A, B, B_k, C, D, D_k, \
    L_min=1, L_max=10000, isolation_data=None, isolation_input_indice=None, \
    isolation_data_indice=None, isolation_coefficient=0):
    """Design unknown input observer considering known inputs. 
    Methodology: https://ece.uwaterloo.ca/~ssundara/courses/notes/uiobs.pdf
    """

    for L_min in range(L_min, L_max):
        # output matrices
        J_mat, J_mat_k, J_mat_prev, O_mat, L, L_found = \
            output_matrices_with_known_input(A, B, B_k, C, D, D_k, \
            L_min=L_min, L_max=L_max)
        logging.info('L = ' + str(L))

        # observer matrices
        E_mat, F_mat, F_mat_k, G_mat = \
            unkown_input_observer_matrice_with_known_input(A, B, B_k, C, D, L, \
                J_mat, J_mat_k, J_mat_prev, O_mat, \
                isolation_data=isolation_data, \
                isolation_input_indice=isolation_input_indice, \
                isolation_data_indice=isolation_data_indice, \
                isolation_coefficient=isolation_coefficient)
        
        # check for asymptomatic stability
        max_eigen_val_norm = np.max(np.abs(np.linalg.eig(E_mat)[0]))
        logging.info(max_eigen_val_norm)
        if (max_eigen_val_norm < 1) and (L_found is True):
            break
    
    # logging the result of designing
    if (max_eigen_val_norm < 1) and (L_found is True):
        logging.info('Unkown input observer has been successfully designed!')
    else:
        logging.info('Unkown input observer has been failed to be designed!')

    return E_mat, F_mat, F_mat_k, G_mat, L, J_mat, J_mat_k, O_mat

def partial_measurement_UIO_design_with_known_input(ignored_indice, A, B_k, \
    F_mat, L, J_mat, J_mat_k, O_mat, output_dim):
    """Redesign unkown input observer's matrice with partial measurement.
    """
    R_mat = np.eye(output_dim*(L+1))
    R_mat = np.delete(R_mat, ignored_indice, axis=0)

    reduced_J = np.matmul(R_mat, J_mat)
    reduced_J_k = np.matmul(R_mat, J_mat_k)
    F_new = np.matmul(np.matmul(np.matmul(F_mat, J_mat), reduced_J.T), \
                    np.linalg.pinv(np.matmul(reduced_J, reduced_J.T)))
    E_new = A - np.matmul(F_new, np.matmul(R_mat, O_mat))
    F_mat_k = np.concatenate([B_k, np.zeros([B_k.shape[0], J_mat_k.shape[1] - \
        B_k.shape[1]])], axis=1) - np.matmul(F_new, reduced_J_k)
    
    return E_new, F_new, F_mat_k

class UIO():
    """Simulator of unknown input observer.
    """
    def __init__(self, A, B, B_k, C, D, D_k, E_mat, F_mat, F_mat_k, G_mat, L, \
        J_mat, J_mat_k, O_mat) -> None:
        self.A = A
        self.B = B
        self.B_k = B_k
        self.C = C
        self.D = D
        self.D_k = D_k
        self.E_mat = E_mat
        self.F_mat = F_mat
        self.F_mat_k = F_mat_k
        self.G_mat = G_mat
        self.L = L
        self.J_mat = J_mat
        self.J_mat_k = J_mat_k
        self.O_mat = O_mat

        # dimensions
        self.state_dim = self.A.shape[0]
        self.input_dim = self.B.shape[1]
        self.output_dim = self.C.shape[0]
    
    def simulate(self, output_seq, input_k_seq, initial_state=None, \
        is_optim_based=True, uncertain_electrode_mask=None, \
        E_list=None, F_list=None, F_k_list=None):
        if initial_state is not None:
            state = initial_state 
        else:
            state = np.zeros([self.state_dim])
        state_seq = np.reshape(state, [1, -1])

        input_seq = np.zeros([0, self.input_dim])
        output_estim_seq = np.zeros([0, self.output_dim])

        # partial UIO
        is_partial_UIO = (uncertain_electrode_mask is not None)

        T = output_seq.shape[0]
        extended_output_seq = np.concatenate([output_seq, \
            np.tile(output_seq[-1,:], [self.L, 1])], axis=0)
        extended_input_k_seq = np.concatenate([input_k_seq, \
            np.tile(input_k_seq[-1,:], [self.L, 1])], axis=0)
        
        # specify the partial observation mask
        if is_partial_UIO is True:
            if extended_output_seq.shape[0] > uncertain_electrode_mask.shape[0]:
                extended_uncertain_electrode_mask = \
                    np.concatenate([uncertain_electrode_mask, \
                    np.tile(uncertain_electrode_mask[-1,:], \
                    [extended_output_seq.shape[0] - \
                    uncertain_electrode_mask.shape[0], 1])], axis=0)
            else:
                extended_uncertain_electrode_mask = \
                    uncertain_electrode_mask[:,0:extended_output_seq.shape[1]]

        # Sequence loop
        last_ignored_indice = []
        for i in range(T):
            obs_out = extended_output_seq[i,:]
            out_obs_vec = np.reshape(extended_output_seq[i:(i+self.L+1), :], \
                [-1])
            inp_k = extended_input_k_seq[i,:]
            inp_k_vec = np.reshape(extended_input_k_seq[i:(i+self.L+1), :], \
                [-1])
            prev_state = state

            # partial output UIO
            if (is_partial_UIO is True) and \
                ((E_list is None) or (F_list is None) or (F_k_list is None)):
                ignored_indice = np.where(np.reshape(\
                    extended_uncertain_electrode_mask[i:(i+self.L+1), :] == 1, \
                        [-1]))[0]
                if len(ignored_indice) > 0:
                    if not np.array_equal(ignored_indice, last_ignored_indice):
                        E_new, F_new, F_new_k = \
                            partial_measurement_UIO_design_with_known_input(\
                            ignored_indice, self.A, self.B_k, self.F_mat, \
                            self.L, self.J_mat, self.J_mat_k, self.O_mat, \
                            self.output_dim)
                    out_obs_vec = np.delete(out_obs_vec, ignored_indice)
                else:
                    E_new, F_new, F_new_k = self.E_mat, self.F_mat, self.F_mat_k
                last_ignored_indice = ignored_indice
            elif (is_partial_UIO is True) and ((E_list is not None) and \
                (F_list is not None) and (F_k_list is not None)):
                ignored_indice = np.where(np.reshape(\
                    extended_uncertain_electrode_mask[i:(i+self.L+1), :] == 1, \
                        [-1]))[0]
                E_new, F_new, F_new_k = E_list[i], F_list[i], F_k_list[i]
                out_obs_vec = np.delete(out_obs_vec, ignored_indice)
            else:
                E_new, F_new, F_new_k = self.E_mat, self.F_mat, self.F_mat_k
    
            # state
            state = self.state_first_calc(prev_state, out_obs_vec, inp_k_vec, \
                E_new, F_new, F_new_k)
            
            # unknown input determination
            if is_optim_based:
                obs_input = self.optimal_unkown_input_calc(state, obs_out, \
                    prev_state, inp_k)
            else:
                obs_input = self.algebraic_unkown_input_calc(state, obs_out, \
                    prev_state, inp_k)
            
            obs_state = self.state_normal_calc(prev_state, obs_input, inp_k)
            output_estim = self.output_calc(obs_state, obs_input, inp_k)

            # measurement
            state_seq = np.append(state_seq, np.reshape(obs_state, [1, -1]), \
                axis=0)
            input_seq = np.append(input_seq, np.reshape(obs_input, [1, -1]), \
                axis=0)
            output_estim_seq = np.append(output_estim_seq, \
                np.reshape(output_estim, [1, -1]), axis=0)
            
        return input_seq, state_seq, output_estim_seq

    def state_first_calc(self, state, out_obs_vec, inp_k_vec, E_new, F_new, \
        F_new_k):
        return np.matmul(E_new, state) + np.matmul(F_new, out_obs_vec) + \
            np.matmul(F_new_k, inp_k_vec)

    def state_normal_calc(self, state, inp, inp_k):
        return np.matmul(self.A, state) + np.matmul(self.B, inp) + \
            np.matmul(self.B_k, inp_k)

    def output_calc(self, state, inp, inp_k):
        return np.matmul(self.C, state) + np.matmul(self.D, inp) + \
            np.matmul(self.D_k, inp_k)

    def optimal_unkown_input_calc(self, state, obs_out, prev_state, inp_k):
        BD_mat = np.concatenate([self.B, self.D], axis=0)
        rhs_vec = np.reshape(np.concatenate(\
            [state - np.matmul(self.A, prev_state) - np.matmul(self.B_k, inp_k), \
            obs_out - np.matmul(self.C, prev_state) - np.matmul(self.D_k, inp_k)], \
            axis=0), [-1, 1])
    
        P = np.dot(BD_mat.T, BD_mat)
        q = - np.reshape(np.dot(BD_mat.T, rhs_vec), [-1])
        G = np.concatenate([-np.eye(self.input_dim), np.eye(self.input_dim)], \
            axis=0)
        h = np.concatenate([-np.zeros(self.input_dim), np.ones(self.input_dim)], \
            axis=0)
        
        return quadprog_solve_qp(P, q, G, h)

    def algebraic_unkown_input_calc(self, state, obs_out, prev_state, inp_k):
        return np.reshape(np.matmul(self.G_mat, np.reshape(np.concatenate(\
            [state - np.matmul(self.A, prev_state) - np.matmul(self.B_k, inp_k), \
            obs_out - np.matmul(self.C, prev_state) - np.matmul(self.D_k, inp_k)], \
            axis=0), [-1, 1])), [-1])

