from argparse import ArgumentParser
from dataclasses import dataclass, field

from config import DEFAULT_DATASET, DATASET_ARG, NEURAL_SYSID_ARG, \
    ARTIFACT_SYSID_ARG, UIO_DESIGN_ARG, UNCERTAIN_MASK_ARG, SPIKE_SORTING_ARG, \
    SPIKE_DETECTION_ARG

@dataclass
class ArgParser:
    """Class for handeling system arguments."""
    arg_parser: ArgumentParser = field(default_factory=ArgumentParser) # parser
    dataset: str = DEFAULT_DATASET
    neural_sysid_active: bool = True
    artifact_sysid_active: bool = True
    uio_design_active: bool = True
    uncertain_mask_active: bool = True
    spike_sorting_active: bool = True
    spike_detection_active: bool = True
    
    def __post_init__(self):
        self.__set_arguments()
    
    def __set_arguments(self):
        """Setting the arguments to the parser.
        """
        # Specify dataset
        self.arg_parser.add_argument(DATASET_ARG, action='store', \
            default=DEFAULT_DATASET)
        
        # Flags for the procedures
        self.arg_parser.add_argument(NEURAL_SYSID_ARG, action='store_true', \
            default=False)
        self.arg_parser.add_argument(ARTIFACT_SYSID_ARG, action='store_true', \
            default=False)
        self.arg_parser.add_argument(UIO_DESIGN_ARG, action='store_true', \
            default=False)
        self.arg_parser.add_argument(UNCERTAIN_MASK_ARG, action='store_true', \
            default=False)
        self.arg_parser.add_argument(SPIKE_SORTING_ARG, action='store_true', \
            default=False)
        self.arg_parser.add_argument(SPIKE_DETECTION_ARG, action='store_true', \
            default=False)
    
    def parse(self):
        """Parse the system arguments. If no True flag, set all flags True.
        """
        # Fetch argument
        args = self.arg_parser.parse_args()
        
        # Dataset
        self.dataset = getattr(args, DATASET_ARG.replace('-', ''))
        
        # Flags
        self.neural_sysid_active = getattr(args, \
            NEURAL_SYSID_ARG.replace('-', ''))
        self.artifact_sysid_active = getattr(args, \
            ARTIFACT_SYSID_ARG.replace('-', ''))
        self.uio_design_active = getattr(args, \
            UIO_DESIGN_ARG.replace('-', ''))
        self.uncertain_mask_active = getattr(args, \
            UNCERTAIN_MASK_ARG.replace('-', ''))
        self.spike_sorting_active = getattr(args, \
            SPIKE_SORTING_ARG.replace('-', ''))
        self.spike_detection_active = getattr(args, \
            SPIKE_DETECTION_ARG.replace('-', ''))
        
        if not any([self.neural_sysid_active, self.artifact_sysid_active, \
            self.uio_design_active, self.uncertain_mask_active, \
            self.spike_sorting_active, self.spike_detection_active]):
            self.neural_sysid_active = True
            self.artifact_sysid_active = True
            self.uio_design_active = True
            self.uncertain_mask_active = True
            self.spike_sorting_active = True
            self.spike_detection_active = True