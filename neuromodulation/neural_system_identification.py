import logging
import numpy as np
from numpy.lib.shape_base import tile
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from dynsys.lin_sysid import SparseLinearSys
from .multi_electrode_array import calculate_electrode_distance
from .neural_models import NeuronSystem
from utils.io import DataHandler

def identify_neural_system(dh:DataHandler, dist_thr_A_coef:float=6, \
    dist_thr_B_coef:float=6, abs_max_threshold:float=0.3, epochs:int=8000):
    """identify the models of all neurons with control systems

    Args:
        dist_thr_A_coef (float, optional): threshold for the mask of A 
            matrix. Defaults to 6.
        dist_thr_B_coef (float, optional): threshold for the mask of B 
            matrix. Defaults to 6.
        abs_max_threshold (float, optional): threshold for being significant 
            electrode. Defaults to 0.3.
        epochs (int, optional): _description_. Defaults to 8000.
    """
    # Load electrical imaging
    elec_imaging = dh.read_neural_imaging_data()
    templates = elec_imaging.templates
    electrode_x = elec_imaging.electrode_x
    electrode_y = elec_imaging.electrode_y, 
    electrode_spacing = elec_imaging.electrode_spacing
    electrode_num = len(electrode_x)
    electrode_distance = calculate_electrode_distance(electrode_x, electrode_y)
    
    # Load system identification parametes
    sysid_params = dh.read_neural_sysid_parameters()
    learning_rate_list = sysid_params.learning_rate_list
    input_start_list = sysid_params.input_start_list
    input_mid_list = sysid_params.input_mid_list
    input_end_list = sysid_params.input_end_list
    b_scale_list = sysid_params.b_scale_list
    
    # Spike identification for specific neuron
    neuron_systems_list = list()
    for n_ind, (neuron_data,learning_rate, input_start, input_mid, input_end, \
            b_scale) in enumerate(zip(templates, learning_rate_list, \
            input_start_list, input_mid_list, \
        input_end_list, b_scale_list)):
        logging.info('Sysid for neuron ' + str(n_ind))
        
        # Regressors
        sample_num = np.shape(neuron_data)[1]
        input_vec = generate_input_vector(sample_num, input_start, input_mid, \
            input_end)
        input_num = input_vec.shape[1]

        state_vec = np.transpose(neuron_data[:, 0:(sample_num - 1)])
        next_state_vec = np.transpose(neuron_data[:, 1:sample_num])

        # dependency masks 
        significant_electrodes, insignificant_electrodes = \
            split_electrodes_by_dominancy(neuron_data, abs_max_threshold)
        
        mask_mat_A = generate_A_mask(electrode_distance, electrode_spacing, \
            insignificant_electrodes, dist_thr_A_coef)
        mask_mat_B, max_pow_elec_ind = generate_B_mask(electrode_distance, \
            neuron_data, electrode_spacing, insignificant_electrodes, \
            dist_thr_B_coef)

        # Create a keras model for the linear system
        input_dim = np.shape(input_vec)[1]
        state_dim = np.shape(state_vec)[1]
        lin_layer = SparseLinearSys(state_dim, input_dim, \
            mask_matrix_A=mask_mat_A, mask_matrix_B=mask_mat_B, b_scale=b_scale)

        # Optimize the keras model
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        loss_fn = keras.losses.MeanSquaredError()

        state_abs_max = np.max(np.abs(state_vec))
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                output = lin_layer(state_vec, input_vec)
                loss_value = loss_fn(next_state_vec, output)/state_abs_max

            grads = tape.gradient(loss_value, lin_layer.trainable_weights)
            optimizer.apply_gradients(zip(grads, lin_layer.trainable_weights))
            
            logging.info('Epoch: ' + str(epoch) + ',' + ' Loss: ' + \
                str(loss_value.numpy()))

        # saving the results
        A = lin_layer.A_mat().numpy()[significant_electrodes,:]\
            [:, significant_electrodes]
        B = lin_layer.B_mat().numpy()[significant_electrodes,:]
        C = np.eye(electrode_num)[:, significant_electrodes]
        D = np.zeros([electrode_num, input_num])
        
        neuron_systems_list.append(NeuronSystem(A=A, B=B, C=C, D=D, \
            input_start=input_start, input_mid=input_mid, input_end=input_end, \
            input_vec=input_vec, max_pow_elec_ind=max_pow_elec_ind, \
            state_abs_max=state_abs_max, \
            significant_electrodes=significant_electrodes))

    # saving the neural systems
    dh.save_neural_sysid(neuron_systems_list)

def generate_input_vector(sample_num:int, input_start:int, input_mid:int, \
    input_end:int) -> np.ndarray:
    """Generating input vector with size sample_num with 2 pulses.
    * first pulse from input_start to input_mid
    * second pulse from input_mid to input_end

    Args:
        sample_num (int): number of samples
        input_start (int): staring point of first pulse
        input_mid (int): end point of first pulse and starting point of second 
            pulse
        input_end (int): end point of second pulse

    Returns:
        np.array: input vector
    """
    input_vec = np.zeros([sample_num - 1, 2])
    input_vec[input_start:input_mid, 0] = 1
    input_vec[input_mid:input_end, 1] = 1
    return input_vec

def generate_A_mask(electrode_distance:np.ndarray, electrode_spacing:float, \
    insignificant_electrodes:list[int], dist_thr_A_coef:float) -> np.ndarray:
    """Generating mask for elements of A matrix.

    Args:
        electrode_distance (np.ndarray): distance between each two electrodes
        electrode_spacing (float): electrodes' spacing
        dist_thr_A_coef (float): threshold for the mask of A matrix

    Returns:
        np.array: masks of A matrix
    """
    mask_mat_A = (electrode_distance < \
        dist_thr_A_coef*electrode_spacing).astype(int)
    mask_mat_A[insignificant_electrodes, :] = 0
    mask_mat_A[:, insignificant_electrodes] = 0
        
    return mask_mat_A

def generate_B_mask(electrode_distance, neuron_data, electrode_spacing:float, \
    insignificant_electrodes:list[int], dist_thr_B_coef:float) -> \
        tuple[np.ndarray, float]:
    """Generating mask for elements of B matrix, and calculating maximum power 
        for all electrodes (useful for normalization)

    Args:
        electrode_distance (np.array): distance between each two electrodes
        neuron_data (np.array): time series of the neuron
        electrode_spacing (float): electrodes' spacing
        insignificant_electrodes (list[int]): list of insignificant electrodes
        dist_thr_B_coef (float): threshold for the mask of B matrix

    Returns:
        np.array: masks of B matrix
        float: maximum power for all electrodes
    """
    powers = np.sqrt(np.sum(neuron_data**2, axis=1))
    max_pow_elec_ind = np.argmax(powers)
    
    mask_mat_B = np.reshape((electrode_distance[max_pow_elec_ind] < \
        dist_thr_B_coef*electrode_spacing).astype(int), [-1, 1])
    mask_mat_B = np.tile(mask_mat_B, [1,2])
    mask_mat_B[insignificant_electrodes] = 0
    
    return mask_mat_B, max_pow_elec_ind

def split_electrodes_by_dominancy(neuron_data:np.ndarray, \
    abs_max_threshold:float) -> tuple[list[int], list[int]]:
    """spliting the electrodes into significant and insignificant electrodes 

    Args:
        neuron_data (np.ndarray): time series of the neuron
        abs_max_threshold (float): threshold for being significant

    Returns:
        list[int]: significant neuron list
        list[int]: insignificant neuron list
    """
    power_abs = np.max(np.abs(neuron_data), axis=1)
    insignificant_electrodes = np.where(power_abs < abs_max_threshold)[0]
    significant_electrodes = np.where(power_abs >= abs_max_threshold)[0]
    
    return significant_electrodes, insignificant_electrodes