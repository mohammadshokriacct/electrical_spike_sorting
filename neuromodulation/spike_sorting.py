import copy
import logging
import numpy as np

from dynsys.unkonwn_input_obs_with_known_inputs import UIO
from neuromodulation.artifact_system_identification import \
    generate_input_templates
from utils.detection_functions import cross_corrolation_match,square_error_match
from utils.io import DataHandler

def do_spike_sorting(dh:DataHandler, zero_threshold:float=0.8, \
    activation_template_size:int=15):
    """Spike sorting via UIO.

    Args:
        zero_threshold (float, optional): threshold to consider input signal to 
            be zero. Defaults to 0.8.
        activation_template_size (int, optional): horizon for estimation. 
            Defaults to 15.
    """
    # Load the data
    stim_data = dh.read_stimulation_data()
    stimulation_trace = stim_data.stimulation_trace
    amplitude_list = stim_data.amplitude_list
    amp_num = len(amplitude_list)
    sample_num = np.shape(stimulation_trace[0][0])[1]
    trial_num = np.shape(stimulation_trace[0])[0]
    
    # Neural systems
    neural_systems_data = dh.load_neural_sysid()
    neuron_num = len(neural_systems_data)
    input_start_list = list()
    input_mid_list = list()
    input_end_list = list()
    significant_electrodes_list = list()
    activation_template_list = list()
    for neuron_ind in range(neuron_num):
        neuron_result_data = neural_systems_data[neuron_ind]
        input_start_list.append(neuron_result_data['input_start'])
        input_mid_list.append(neuron_result_data['input_mid'])
        input_end_list.append(neuron_result_data['input_end'])
        significant_electrodes_list.append(\
            neuron_result_data['significant_electrodes'])

        # activation templates
        activation_template = np.zeros([activation_template_size, 2])
        activation_template[0:(input_mid_list[neuron_ind] - \
            input_start_list[neuron_ind]), 0] = 1
        activation_template[(input_mid_list[neuron_ind] - \
            input_start_list[neuron_ind]):(input_end_list[neuron_ind] - \
                input_start_list[neuron_ind]), 1] = 1

        activation_template_list.append(activation_template)
    
    # Artifact's system
    artifact_system_data = dh.load_artifact_sysid()
    input_start_list = artifact_system_data['input_start']
    input_end_list = artifact_system_data['input_end']
    input_template = generate_input_templates(sample_num, input_start_list, \
        input_end_list)
    
    # UIO parameters
    uio_params = dh.load_uio_design()
    E_mat, F_mat, F_mat_k, G_mat = uio_params['E_mat'], uio_params['F_mat'], \
        uio_params['F_mat_k'], uio_params['G_mat']
    L = uio_params['L']
    J_mat, J_mat_k, O_mat = uio_params['J_mat'], uio_params['J_mat_k'], \
        uio_params['O_mat']
    A_aug, B_aug, B_k_aug, C_aug, D_aug, D_k_aug = uio_params['A_aug'], \
        uio_params['B_aug'], uio_params['B_k_aug'], uio_params['C_aug'], \
        uio_params['D_aug'], uio_params['D_k_aug']

    # Configuration save for the folder of spike sorting
    conf_dict = {
        'amplitude_num':amp_num, 
        'trial_num':trial_num, 
        'neuron_num':neuron_num, 
        'amplitude_list':amplitude_list.tolist(),
    }
    dh.save_spike_sorting_config(conf_dict)

    # simulation model
    observer_sys = UIO(A_aug, B_aug, B_k_aug, C_aug, D_aug, D_k_aug, E_mat, \
        F_mat, F_mat_k, G_mat, L, J_mat, J_mat_k, O_mat)

    # Simulate for each amplitudes
    uncertain_electrode_mask = []
    for a_ind,a_data in enumerate(stimulation_trace):
        logging.info('>>> Amplitude: ' + str(a_ind) + ' <<<')
        amp = amplitude_list[a_ind]

        # New UIO matrices
        result_dict = dh.load_uncertain_mask(a_ind)
        uncertain_electrode_mask = result_dict['data_mask']
        matrix_indice = result_dict['matrix_indice']
        E_list = result_dict['E_list']
        F_list = result_dict['F_list']
        F_k_list = result_dict['F_k_list']
        
        E_list_total, F_list_total, F_k_list_total = list(), list(), list()
        for ind in matrix_indice: 
            # Converting to list
            E_list_total.append(E_list[ind])
            F_list_total.append(F_list[ind])
            F_k_list_total.append(F_k_list[ind])

        # Spike sorting via UIO
        for t_ind,t_data in enumerate(a_data):
            logging.info('* Trial: ' + str(t_ind) + ' *')
            output_seq = t_data.T

            # artifact input recreation
            a_input_vec = amp*input_template

            # simulation
            input_seq, _, output_estim_seq = observer_sys.simulate(output_seq, \
                a_input_vec, initial_state=None, is_optim_based=True, \
                uncertain_electrode_mask=uncertain_electrode_mask, \
                E_list=E_list_total, F_list=F_list_total, \
                F_k_list=F_k_list_total)

            # spike template detection
            cc_sig_list = list()
            se_sig_list = list()
            for neuron_ind,activation_template in enumerate(\
                activation_template_list):
                inp_sig = copy.deepcopy(\
                    input_seq[:,(2*neuron_ind):(2*(neuron_ind+1))])
                inp_sig[np.where(inp_sig < zero_threshold)] = 0
                
                cc_sig_list.append(cross_corrolation_match(inp_sig, \
                    activation_template))
                se_sig_list.append(square_error_match(inp_sig, \
                    activation_template))
                
            # Saving the results
            result_list = {
                    'input_seq':input_seq,
                    'output_seq':output_seq,
                    'output_estim_seq':output_estim_seq,
                    'cc_sig_list':cc_sig_list,
                    'se_sig_list':se_sig_list, 
                    'activation_template_list':activation_template_list
                }
            dh.save_spike_sorting_results(result_list, a_ind, t_ind)