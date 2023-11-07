from argparse import ArgumentParser
from dataclasses import dataclass

@dataclass
class ArgParser:
    """Class for handeling system arguments."""
    _arg_parser: ArgumentParser
    neural_sysid_active: bool = True
    artifact_sysid_active: bool = True
    uio_design_active: bool = True
    uncertain_mask_active: bool = True
    spike_sorting_active: bool = True
    spike_detection_active: bool = True
    
    def __post_init__(self):
        self._arg_parser = ArgumentParser()
        pass