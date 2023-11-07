import numpy as np

from dynsys.unkonwn_input_obs_with_known_inputs import unkown_input_observer_design_with_known_input
from dynsys.augmentation import two_input_augment_dynsys
from utils.io import DataHandler

def uio_design(dh:DataHandler, isolation_coefficient:float=10000000, \
    L_min:int=10):
    """Design UIO robust against isolation data

    Args:
        isolation_coefficient (float, optional): the coefficient to adjust 
            robustness against isolation data. Defaults to 10000000.
        L_min (int, optional): minimume horozon length. Defaults to 5.
    """
    # Reading the data file
    elec_imaging = dh.read_neural_imaging_data()
    templates = elec_imaging.templates
    
    stim_data = dh.read_stimulation_data()
    stimulation_trace = stim_data.stimulation_trace
    
    neural_systems_data = dh.load_neural_sysid()
    artifact_system_data = dh.load_artifact_sysid()
    neuron_num = np.shape(templates)[0]

    # Gathering and augmenting artifact and neural systems
    A_list = list()
    B_list = list()
    B_k_list = list()
    C_list = list()
    D_list = list()
    D_k_list = list()

    A_list.append(artifact_system_data['A'])
    B_list.append(np.zeros([artifact_system_data['A'].shape[0], 0]))
    B_k_list.append(artifact_system_data['B'])
    C_list.append(artifact_system_data['C'])
    D_list.append(np.zeros([artifact_system_data['C'].shape[0], 0]))
    D_k_list.append(artifact_system_data['D'])

    significant_electrodes_list = list()
    for neuron_ind in range(neuron_num):
        neuron_sys = neural_systems_data[neuron_ind]
        A_list.append(neuron_sys['A'])
        B_list.append(neuron_sys['B'])
        B_k_list.append(np.zeros([neuron_sys['A'].shape[0], 0]))
        C_list.append(neuron_sys['C'])
        D_list.append(neuron_sys['D'])
        D_k_list.append(np.zeros([neuron_sys['C'].shape[0], 0]))
        significant_electrodes_list.append(neuron_sys['significant_electrodes'])
        
    A_aug, B_aug, B_k_aug, C_aug, D_aug, D_k_aug = \
        two_input_augment_dynsys(A_list, B_list, B_k_list, C_list, D_list, \
        D_k_list, additive_output=True)

    # Managing isolation data 
    mean_artifact_data = [np.mean(sig, axis=0) for sig in stimulation_trace]
    isolation_data = mean_artifact_data + list(templates)

    artifact_isolation_indice = list(range(len(mean_artifact_data)))
    templates_isolation_indice = list(range(len(mean_artifact_data), \
        len(mean_artifact_data) + len(templates)))

    isolation_input_indice = np.cumsum(\
        [0, np.shape(artifact_system_data['A'])[0]] + \
        [len(elem) for elem in significant_electrodes_list])

    isolation_data_indice = \
        [templates_isolation_indice] + \
        [artifact_isolation_indice + [item for i,item in \
            enumerate(templates_isolation_indice) if j!=i] for j in \
                range(len(templates_isolation_indice))]

    # designing unkown input observer
    E_mat, F_mat, F_mat_k, G_mat, L, J_mat, J_mat_k, O_mat = \
        unkown_input_observer_design_with_known_input(A_aug, B_aug, B_k_aug, \
            C_aug, D_aug, D_k_aug, \
            isolation_data=isolation_data, \
            isolation_input_indice=isolation_input_indice, \
            isolation_data_indice=isolation_data_indice, \
            isolation_coefficient=isolation_coefficient, L_min=L_min)

    # saving the results
    result_list = {
        "E_mat":E_mat,
        "F_mat":F_mat, 
        "F_mat_k":F_mat_k, 
        "G_mat":G_mat,
        "L":L, 
        "A_aug":A_aug,
        "B_aug":B_aug,
        "B_k_aug":B_k_aug,
        "C_aug":C_aug,
        "D_aug":D_aug,
        "D_k_aug":D_k_aug,
        "J_mat":J_mat, 
        "J_mat_k":J_mat_k, 
        "O_mat":O_mat,
    }
    dh.save_uio_design(result_list)
        