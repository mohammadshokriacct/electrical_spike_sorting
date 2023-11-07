import numpy as np

from utils.io import DataHandler

def detect_spikes(dh:DataHandler, cc_threshold=1.6, se_threshold=1.3):
    """Spike detectiona based on the thresholds of cross corrolation and squared
        error.

    Args:
        cc_threshold (float, optional): threshold of cross corrolation. Defaults
            to 1.6.
        se_threshold (float, optional): threshold of squared error. Defaults to 
            2.1.
    """
    # Load the data
    spike_sorting_config = dh.load_spike_sorting_config()
    amplitude_num = spike_sorting_config['amplitude_num']
    trial_num = spike_sorting_config['trial_num']
    neuron_num = spike_sorting_config['neuron_num']

    # Simulate for each amplitudes
    cc_detection = np.zeros([amplitude_num, trial_num, neuron_num], \
        dtype=np.int_)
    se_detection = np.zeros([amplitude_num, trial_num, neuron_num], \
        dtype=np.int_)
    for a_ind in range(amplitude_num):
        for t_ind in range(trial_num):
            # Loading the results
            spike_sorting_list = dh.load_spike_sorting_results(a_ind, t_ind)
            
            cc_detection[a_ind, t_ind, :] = [np.sign(sum(x > cc_threshold)) \
                for x in spike_sorting_list['cc_sig_list']]
            se_detection[a_ind, t_ind, :] = [np.sign(sum(x < se_threshold)) \
                for x in spike_sorting_list['se_sig_list']]
            
    # Saving the results
    result_dict = {
        'cc_detection':cc_detection,
        'se_detection':se_detection,
    }
    dh.save_spike_detection(result_dict)