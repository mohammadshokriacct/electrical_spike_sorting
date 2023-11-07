import logging
import numpy as np

from dynsys.unkonwn_input_obs_with_known_inputs import \
    partial_measurement_UIO_design_with_known_input
from utils.io import DataHandler

def determine_uncertain_electrode_mask(dh:DataHandler):
    """Determining the mask of the electrodes that have to be ignored or trusted
    for the UIO procedure. Redesigning UIO based on the partial measurements.
    """
    # Load data
    stim_data = dh.read_stimulation_data()
    stimulation_trace = stim_data.stimulation_trace
    stimulation_electrode_index = stim_data.stimulation_electrode_index
    
    uio_params = dh.load_uio_design()
    E_mat, F_mat, F_mat_k, L = uio_params['E_mat'], uio_params['F_mat'], \
        uio_params['F_mat_k'], uio_params['L']
    J_mat, J_mat_k, O_mat = uio_params['J_mat'], uio_params['J_mat_k'], \
        uio_params['O_mat']
    A_aug, B_k_aug, C_aug = uio_params['A_aug'], uio_params['B_k_aug'], \
        uio_params['C_aug']
    
    # mean artifact signal
    mean_artifact_data = [np.mean(sig, axis=0) for sig in stimulation_trace]

    # Simulate for each amplitudes
    last_sample_size = None
    for a_ind,a_input_data in enumerate(mean_artifact_data):
        logging.info('>>> Amplitude: ' + str(a_ind) + ' <<<')
        measurement_vec = a_input_data.T
        
        if last_sample_size is not measurement_vec.shape[0]: 
            # if the sample sizes are equal, the results don't change
            last_sample_size = measurement_vec.shape[0]
            
            # Ignorance mask
            data_mask = np.zeros(measurement_vec.shape, dtype=int)
            data_mask[:, stimulation_electrode_index] = 1

            # New UIO matrices
            E_list = list()
            F_list = list()
            F_k_list = list()
            output_dim = C_aug.shape[0]
            last_ignored_indice = list()
            matrix_indice = list()
            E_new, F_new, F_new_k = E_mat, F_mat, F_mat_k
            for i in range(len(data_mask)):
                logging.info('time ' + str(i))
                
                # Determine the uncertain points
                extended_output_seq = np.concatenate([measurement_vec, \
                    np.tile(measurement_vec[-1,:], [L, 1])], axis=0)
                if extended_output_seq.shape[0] > data_mask.shape[0]:
                    extended_uncertain_electrode_mask = \
                        np.concatenate([data_mask, np.tile(data_mask[-1,:], \
                            [extended_output_seq.shape[0] - \
                            data_mask.shape[0], 1])], axis=0)
                else:
                    extended_uncertain_electrode_mask = \
                        data_mask[:,0:extended_output_seq.shape[1]]

                # Redesign the UIO
                ignored_indice = np.where(np.reshape(\
                    extended_uncertain_electrode_mask[i:(i+L+1), :] == 1, \
                        [-1]))[0]
                if len(ignored_indice) == 0:
                    E_new, F_new, F_new_k = E_mat, F_mat, F_mat_k
                elif not np.array_equal(ignored_indice, last_ignored_indice):
                    E_new, F_new, F_new_k = \
                        partial_measurement_UIO_design_with_known_input(\
                            ignored_indice, A_aug, B_k_aug, F_mat, \
                            L, J_mat, J_mat_k, O_mat, output_dim)
                else:
                    E_new, F_new, F_new_k = E_new, F_new, F_new_k
                
                # Matrix repetition check
                matrix_indice.append(max(len(E_list) - 1, 0))
                if (not np.array_equal(ignored_indice, last_ignored_indice)) \
                    or (i == 0): 
                    # New matrix
                    E_list.append(E_new)
                    F_list.append(F_new)
                    F_k_list.append(F_new_k)

                last_ignored_indice = ignored_indice

        # saving the results
        result_dict = {
            'data_mask': data_mask,
            'matrix_indice': matrix_indice, 
            'E_list':E_list, 
            'F_list':F_list,
            'F_k_list':F_k_list
        }
        dh.save_uncertain_mask(result_dict, a_ind)
