import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

from dynsys.lin_sysid import SparseLinearSys
from utils.io import DataHandler
from .multi_electrode_array import calculate_electrode_distance
from .neural_models import ArtifactSystem

def identify_artifact_system(dh:DataHandler, dist_thr_A_coef:float=6, \
    dist_thr_B_coef:float=6, insignificance_distance_coef:float=8, \
    regularization_param:float=0.000001, epochs:int=4000):
    """identify the model of the artifact.

    Args:
        dist_thr_A_coef (float, optional): threshold for the mask of A matrix. 
            Defaults to 6.
        dist_thr_B_coef (float, optional): threshold for the mask of B matrix. 
            Defaults to 6.
        insignificance_distance_coef (float, optional): coefficient for being 
            significant electrode. Defaults to 0.3.
        regularization_param (float, optional): regularization parameter for the
            regression. Defaults to 0.000001.
        epochs (int, optional): _description_. Defaults to 4000.
    """
    # Load data
    stim_data = dh.read_stimulation_data()
    stimulation_trace = stim_data.stimulation_trace
    amplitude_list = stim_data.amplitude_list
    stimulation_electrode_index = stim_data.stimulation_electrode_index
    electrode_x = stim_data.electrode_x
    electrode_y = stim_data.electrode_y
    electrode_spacing = stim_data.electrode_spacing
    sample_num = np.shape(stimulation_trace[0][0])[1]
    electrode_num = len(electrode_x)
    electrode_distance = calculate_electrode_distance(electrode_x, electrode_y)
        
    # Load system identification parameters
    sysid_params = dh.read_artifact_sysid_parameters()
    learning_rate = sysid_params.learning_rate
    input_start_list = sysid_params.input_start_list
    input_end_list = sysid_params.input_end_list
    b_scale = sysid_params.b_scale
    template_inp_dim = len(input_start_list)

    # input template
    input_template = generate_input_templates(sample_num, input_start_list, \
        input_end_list)
    mean_artifact_data = [np.mean(sig, axis=0) for sig in stimulation_trace]

    # Regressors
    input_vec = np.zeros([0, input_template.shape[1]])
    state_vec = np.zeros([0, electrode_num])
    next_state_vec = np.zeros([0, electrode_num])
    for a_ind,a_input_data in enumerate(mean_artifact_data):
        amp = amplitude_list[a_ind][0]
        normalize_value = amp

        a_input_vec = input_template[:-1,:]

        a_state_vec = a_input_data[:, :-1].T/normalize_value
        a_next_state_vec = a_input_data[:, 1:].T/normalize_value

        input_vec = np.concatenate([input_vec, a_input_vec], axis=0)
        state_vec = np.concatenate([state_vec, a_state_vec], axis=0)
        next_state_vec = np.concatenate([next_state_vec, a_next_state_vec], \
            axis=0)
        
    # dependency masks
    significant_electrodes, insignificant_electrodes = \
        split_electrodes_by_dominancy(electrode_distance, \
        stimulation_electrode_index, insignificance_distance_coef, \
        electrode_spacing)
        
    mask_mat_A = generate_A_mask(electrode_distance, electrode_spacing, \
        insignificant_electrodes, dist_thr_A_coef)
    mask_mat_B = generate_B_mask(electrode_distance, electrode_spacing, \
        insignificant_electrodes, dist_thr_B_coef, \
        stimulation_electrode_index, template_inp_dim)
    
    # Create a Keras model for linear control system
    input_dim = np.shape(input_vec)[1]
    state_dim = np.shape(state_vec)[1]
    data_num = np.shape(input_vec)[0]
    linear_layer = SparseLinearSys(state_dim, input_dim, \
        mask_matrix_A=mask_mat_A, mask_matrix_B=mask_mat_B, \
        b_scale=list(b_scale))

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_fn = keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            output = linear_layer(state_vec, input_vec)

            A_mat = linear_layer.A_mat()
            regularization_loss = tf.reduce_sum(tf.pow(A_mat, 2))
            output_loss = loss_fn(next_state_vec, output)/data_num

            loss_value = output_loss + regularization_loss*regularization_param

        grads = tape.gradient(loss_value, linear_layer.trainable_weights)
        optimizer.apply_gradients(zip(grads, linear_layer.trainable_weights))
        
        logging.info('Epoch: ' + str(epoch) + ',' + ' Loss: ' + \
            str(loss_value.numpy()))

    # saving the results
    A = linear_layer.A_mat().numpy()[significant_electrodes,:]\
        [:, significant_electrodes]
    B = linear_layer.B_mat().numpy()[significant_electrodes,:]
    C = np.eye(electrode_num)[:, significant_electrodes]
    D = np.zeros([electrode_num, input_dim])
    
    # Saving the results
    artif_sys = ArtifactSystem(A=A, B=B, C=C, D=D, \
        input_start=input_start_list, input_end=input_end_list, \
        input_template=input_template, \
        stimulation_electrode_index=stimulation_electrode_index, \
        significant_electrodes=significant_electrodes)
    dh.save_artifact_sysid(artif_sys)
    
def generate_input_templates(sample_num:int, input_start_list:list, \
    input_end_list:list):
    """Generating input vector with size sample_num with multiple pulses from 
        start to end points.

    Args:
        sample_num (int): number of samples
        input_start (list): staring point of first pulse
        input_end (list): end point of second pulse

    Returns:
        np.array: input templates
    """
    template_inp_dim = len(input_start_list)
    input_template = np.zeros([sample_num, template_inp_dim])
    for inp_ind, (start_ind,end_ind) in enumerate(zip(input_start_list, \
        input_end_list)):
        start_ind = start_ind if start_ind is not None else sample_num
        end_ind = end_ind if end_ind is not None else sample_num
        input_template[start_ind:end_ind, inp_ind] = 1
    return input_template

def generate_A_mask(electrode_distance, electrode_spacing:float, \
    insignificant_electrodes:list[int], dist_thr_A_coef:float) -> np.ndarray:
    """Generating mask for elements of A mat.

    Args:
        electrode_distance (np.array): distance between each two electrodes
        electrode_spacing (float): electrodes' spacing
        dist_thr_A_coef (float): threshold for the mask of A mat

    Returns:
        np.ndarray: masks of A mat
    """
    mask_mat_A = (electrode_distance < \
        dist_thr_A_coef*electrode_spacing).astype(int)
    mask_mat_A[insignificant_electrodes, :] = 0
    mask_mat_A[:, insignificant_electrodes] = 0
        
    return mask_mat_A

def generate_B_mask(electrode_distance, electrode_spacing:float, \
    insignificant_electrodes:list[int], dist_thr_B_coef:float, \
    stimulation_electrode_index:int, template_inp_dim:int):
    """Generating mask for elements of B mat, and calculating maximum power for 
        all electrodes (useful for normalization)

    Args:
        electrode_distance (np.array): distance between each two electrodes
        neuron_data (np.array): time series of the neuron
        electrode_spacing (float): electrodes' spacing
        insignificant_electrodes (list[int]): list of insignificant electrodes
        dist_thr_B_coef (float): threshold for the mask of B mat
        stimulation_electrode_index (int): index of stimulating electrode
        template_inp_dim (int): input template size

    Returns:
        np.array: masks of B mat
        float: maximum power for all electrodes
    """
    mask_mat_B = np.reshape((electrode_distance[stimulation_electrode_index] < \
        dist_thr_B_coef*electrode_spacing).astype(int), [-1, 1])
    mask_mat_B = np.tile(mask_mat_B, [1,template_inp_dim])
    mask_mat_B[insignificant_electrodes] = 0
    
    return mask_mat_B

def split_electrodes_by_dominancy(electrode_distance, \
    stimulation_electrode_index:int, insignificance_distance_coef:float, \
    electrode_spacing:float) -> tuple[list[int], list[int]]:
    """spliting the electrodes into significant and insignificant electrodes 

    Args:
        electrode_distance (np.array): distance between each two electrodes
        stimulation_electrode_index (int): index of stimulating electrode
        insignificance_distance_coef (float): coefficient for electrodes for 
            being insignificant 
        electrode_spacing (float): electrodes' spacing

    Returns:
        list[int]: significant neuron list
        list[int]: insignificant neuron list
    """
    insignificant_electrodes = np.where(\
        electrode_distance[stimulation_electrode_index] > \
        insignificance_distance_coef*electrode_spacing)[0]
    insignificant_electrodes = np.append(insignificant_electrodes, \
        stimulation_electrode_index)
    significant_electrodes = np.where(\
        electrode_distance[stimulation_electrode_index] <= \
            insignificance_distance_coef*electrode_spacing)[0]
    
    return significant_electrodes, insignificant_electrodes