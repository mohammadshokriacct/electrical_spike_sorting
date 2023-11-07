import numpy as np

from config import DETECTION_TARGET_NEURON_PATH
from utils.io import DataHandler
from utils.arg_parser import ArgParser
from visualization.detection import activation_curve_plt

if __name__ == '__main__':
    # Parsing arguments
    arg_parser = ArgParser()
    arg_parser.parse()

    # Create a data handeler
    dh = DataHandler(arg_parser.dataset)
    
    # load the data
    result_dict = dh.load_spike_detection()
    cc_detection = result_dict['cc_detection']    
    se_detection = result_dict['se_detection'] 

    amplitude_list, activation_curve, target_neuron = \
        dh.read_activation_curve()

    # visualization of first neuron
    file_path = dh.join_subpath(DETECTION_TARGET_NEURON_PATH, is_result=True)
    activation_curve_plt(np.reshape(amplitude_list, [-1]), \
        np.sum(cc_detection[:,:,target_neuron]*se_detection[:,:,target_neuron], \
            axis=1)/cc_detection.shape[1], \
        activation_curve=activation_curve, \
        image_path=file_path)
    
    
