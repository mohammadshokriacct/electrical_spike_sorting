from dataclasses import dataclass
import numpy as np

@dataclass
class ElectricalImaging:
    """Class for electrical Imaging data of mutiple neurons
    """
    templates: np.ndarray
    electrode_x: np.ndarray
    electrode_y: np.ndarray
    electrode_spacing: float
    
@dataclass
class StimulationDataModel:
    """Class for stimulation data with different stimulating amplitudes and 
        different trials from a single electrode
    """
    stimulation_trace: np.ndarray
    amplitude_list: np.ndarray
    stimulation_electrode_index: int
    electrode_x: np.ndarray
    electrode_y: np.ndarray
    electrode_spacing: float

@dataclass
class NeuralSysidParams:
    """Class for the parameters of neural system identification of mutiple 
        neurons
    """
    learning_rate_list: list[float]
    input_start_list: list[int]
    input_mid_list: list[int]
    input_end_list: list[int]
    b_scale_list: list[float]

@dataclass
class ArtifactSysidParams:
    """Class for the parameters of artifact system identification of mutiple 
        neurons
    """
    learning_rate: float
    input_start_list: list[int]
    input_end_list: list[int]
    b_scale: float
    
@dataclass
class NeuronSystem:
    """Class for control system of a neuron
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    input_start: int
    input_mid: int
    input_end: int
    input_vec: np.ndarray
    max_pow_elec_ind: int
    state_abs_max: float
    significant_electrodes: np.ndarray
    
@dataclass
class ArtifactSystem:
    """Class for control system of a neuron
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    input_start: list[int]
    input_end: list[int]
    input_template: np.ndarray
    stimulation_electrode_index: int
    significant_electrodes: np.ndarray
    
    