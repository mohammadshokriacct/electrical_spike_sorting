import os
import logging

# Dataset
DEFAULT_DATASET = 'main'

# Folders
DATA_FOLDER = 'data'
RESULT_FOLDER = 'results'
SPIKE_SORTING_FOLDER = 'spike_sorting'
ARTIFACT_MASK_FOLDER = 'artifact_mask'

# Paths 
DATA_PATH = 'electrode_data.pickle'
NEURAL_SYSID_PARAMS_PATH = 'neural_iden_params.csv'
ARTIFACT_SYSID_PARAMS_PATH = 'artifact_iden_params.json'
NEURAL_SYSID_RESULT_PATH = 'neural_sysid.pickle'
ARTIFACT_SYSID_RESULT_PATH = 'artifact_sysid.pickle'
UIO_DESIGN_PATH = 'uio_design.pickle'
SPIKE_SORTING_CONFIG_PATH = 'spike_sorting_conf.cfg'
SPIKE_DETECTION_PATH = 'spike_detection.pickle'

# Arguments
DATASET_ARG = '--dataset'
NEURAL_SYSID_ARG = '--nsysid'
ARTIFACT_SYSID_ARG = '--asysid'
UIO_DESIGN_ARG = '--uiod'
UNCERTAIN_MASK_ARG = '--uncmask'
SPIKE_SORTING_ARG = '--spikesorting'
SPIKE_DETECTION_ARG = '--spikedetection'

# Visualization
DETECTION_TARGET_NEURON_PATH = "activation_curve_target_neuron.pdf"

# Configure the code
def initial_config():
    """Prepare the initial configurations.
    """
    # Configure logging 
    logging.basicConfig(level=logging.INFO)