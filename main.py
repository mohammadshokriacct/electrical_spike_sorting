import logging

from config import initial_config
from neuromodulation.neural_system_identification import \
    identify_neural_system
from neuromodulation.artifact_system_identification import \
    identify_artifact_system
from neuromodulation.neuromodulation_uio_design import uio_design
from neuromodulation.uncertain_electrode_mask import \
    determine_uncertain_electrode_mask
from neuromodulation.spike_sorting import do_spike_sorting
from neuromodulation.spike_detection import detect_spikes
from utils.arg_parser import ArgParser
from utils.io import DataHandler

# Initial configuration
initial_config()

if __name__ == '__main__':
    # Parsing arguments
    arg_parser = ArgParser()
    arg_parser.parse()
    
    # Create a data handeler
    dh = DataHandler(arg_parser.dataset)
    
    # System identification for the neurons
    if arg_parser.neural_sysid_active:
        logging.info('Starting the system identification for neurons')
        identify_neural_system(dh)
        
    # System identification for the artifact
    if arg_parser.artifact_sysid_active:
        logging.info('Starting the system identification for artifact')
        identify_artifact_system(dh)
        
    # UIO design
    if arg_parser.uio_design_active:
        logging.info('Starting the uio design')
        uio_design(dh)
        
    # Uncertain mask calculation
    if arg_parser.uncertain_mask_active:
        logging.info('Starting the uncertain mask calculation')
        determine_uncertain_electrode_mask(dh)
        
    # Spkie sorting
    if arg_parser.spike_sorting_active:
        logging.info('Starting the spike sorting')
        do_spike_sorting(dh)
        
    # Spkie detection
    if arg_parser.spike_detection_active:
        logging.info('Starting the spike detection')
        detect_spikes(dh)
    
    # Finishing of the process
    logging.info('The process has been sucessfully finished')
