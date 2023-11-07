from dataclasses import asdict, dataclass
import json
import logging
import numpy as np
import os
import pickle
import pandas as pd

from config import DATA_PATH, DATA_FOLDER, RESULT_FOLDER, NEURAL_SYSID_PARAMS_PATH, \
    NEURAL_SYSID_RESULT_PATH, ARTIFACT_SYSID_PARAMS_PATH, \
    ARTIFACT_SYSID_RESULT_PATH, UIO_DESIGN_PATH, ARTIFACT_MASK_FOLDER, \
    SPIKE_SORTING_CONFIG_PATH, SPIKE_SORTING_FOLDER, SPIKE_DETECTION_PATH
from neuromodulation.neural_models import ElectricalImaging, \
    NeuralSysidParams, NeuronSystem, StimulationDataModel, \
    ArtifactSysidParams, ArtifactSystem

@dataclass
class DataHandler:
    """Data handler class for loading and savind data. 
    """
    dataset_folder: str
    
    def __post_init__(self):
        """Create essential folders if they do not exist
        """
        essential_subfolders = [SPIKE_SORTING_FOLDER, ARTIFACT_MASK_FOLDER]
        for subfolder in essential_subfolders:
            folder = self.join_subpath(subfolder, is_result=True)
            if not os.path.exists(folder):
                os.makedirs(folder)
                
    def join_subpath(self, subfolder:str, is_result:bool=True) -> str:
        if is_result:
            # Result folder
            return os.path.join(RESULT_FOLDER, self.dataset_folder, subfolder)
        else: 
            # Data folder
            return os.path.join(DATA_FOLDER, self.dataset_folder, subfolder)
    
    def read_neural_imaging_data(self) -> ElectricalImaging:
        """Load the neural imaging data.

        Raises:
            err: error for loading the dataset.

        Returns:
            ElectricalImaging: electrical imaging data
        """
        # Reading the data
        file_path = self.join_subpath(DATA_PATH, is_result=False)
        try:
            with open(file_path, "rb") as file_obj:
                input_data = pickle.load(file_obj)
        except Exception as err:
            logging.error("Error in loading neural imaging data")
            raise err
        
        templates = np.array(input_data["templates"])
        electrode_x = np.array(input_data["electrode_x"])
        electrode_y = np.array(input_data["electrode_y"])
        electrode_spacing = input_data["electrode_spacing"]

        return ElectricalImaging(templates=templates, electrode_x=electrode_x, \
                electrode_y=electrode_y, electrode_spacing=electrode_spacing)

    def read_stimulation_data(self) -> StimulationDataModel:
        """Load the stimulation data for each atimulating amplitude.

        Raises:
            err: error for loading the dataset.

        Returns:
            StimulationDataModel: stimulation data
        """
        # Reading the data
        file_path = self.join_subpath(DATA_PATH, is_result=False)
        try:
            with open(file_path, "rb") as file_obj:
                input_data = pickle.load(file_obj)
        except Exception as err:
            logging.error("Error in loading neural imaging data")
            raise err
        
        stimulation_trace = input_data['trace_all']
        amplitude_list = input_data['amplitude_list_all_trace']
        stimulation_electrode_index = input_data['stimulation_electrode_index']
        
        electrode_x = np.array(input_data["electrode_x"])
        electrode_y = np.array(input_data["electrode_y"])
        electrode_spacing = input_data["electrode_spacing"]

        return StimulationDataModel(stimulation_trace=stimulation_trace, \
            amplitude_list=amplitude_list, \
            stimulation_electrode_index=stimulation_electrode_index, \
            electrode_x=electrode_x, electrode_y=electrode_y, \
            electrode_spacing=electrode_spacing
        )

    def read_activation_curve(self):
        """Load the activation curve (spiking probabilities).

        Raises:
            err: error for loading the dataset.

        Returns:
            np.array: stimulating amplitude list
            np.array: activation curve
        """
        # Reading the data
        file_path = self.join_subpath(DATA_PATH, is_result=False)
        try:
            with open(file_path, "rb") as file_obj:
                input_data = pickle.load(file_obj)
        except Exception as err:
            logging.error("Error in loading neural imaging data")
            raise err
        
        amplitude_list = input_data['amplitude_list_all_trace']
        activation_curve = input_data['activation_curve']
        target_neuron_index = input_data['target_neuron_index']
        
        return amplitude_list, activation_curve, target_neuron_index

    def read_neural_sysid_parameters(self) -> NeuralSysidParams:
        """Load the patameters of neural system identification.

        Raises:
            err: error for loading the dataset.

        Returns:
            NeuralSysidParams: params of neural systems
        """
        # Reading the parameters data of neural systems
        file_path = self.join_subpath(NEURAL_SYSID_PARAMS_PATH, is_result=False)
        try:
            config_data = pd.read_csv(file_path)
        except Exception as err:
            logging.error("Error in loading neural system identification params")
            raise err
        
        learning_rate_list = list(config_data.learning_rate)
        input_start_list = list(config_data.input_start)
        input_mid_list = list(config_data.input_mid)
        input_end_list = list(config_data.input_end)
        b_scale_list = list(config_data.b_scale)

        return NeuralSysidParams(learning_rate_list=learning_rate_list, \
            input_start_list=input_start_list, input_mid_list=input_mid_list, \
            input_end_list=input_end_list, b_scale_list=b_scale_list)

    def read_artifact_sysid_parameters(self) -> ArtifactSysidParams:
        """Load the patameters of artifact system identification.

        Raises:
            err: error for loading the dataset.

        Returns:
            ArtifactSysidParams: params of artifact system identification
        """
        # Reading the parameters data of neural systems
        file_path = self.join_subpath(ARTIFACT_SYSID_PARAMS_PATH, is_result=False)
        try:
            with open(file_path, 'rb') as file_obj:
                config_data = json.load(file_obj)
        except Exception as err:
            logging.error("Error in loading neural system identification params")
            raise err
        
        learning_rate = config_data['learning_rate']
        input_start_list = list(config_data['input_start'])
        input_end_list = list(config_data['input_end'])
        b_scale = list(config_data['b_scale'])

        return ArtifactSysidParams(learning_rate=learning_rate, \
            input_start_list=input_start_list, input_end_list=input_end_list, \
            b_scale=b_scale)

    def save_neural_sysid(self, neuron_systems_list:list[NeuronSystem]):
        """Save the neural systems

        Args:
            neuron_systems_list (list[NeuronSystem]): list of parameters of each 
                neural system

        Raises:
            err: error in saving the data
        """
        # Saving the neural systems
        list_of_dicts = [asdict(ns) for ns in neuron_systems_list]
        file_path = self.join_subpath(NEURAL_SYSID_RESULT_PATH, is_result=True)
        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(list_of_dicts, file_obj)
        except Exception as err:
            logging.error("Error in saving neural systems")
            raise err
        
    def save_artifact_sysid(self, artifact_system:ArtifactSystem):
        """Save the artifact systems

        Args:
            artifact_system (ArtifactSystem): artifact's system

        Raises:
            err: error in saving the data
        """
        # Saving the artifact systems
        file_path = self.join_subpath(ARTIFACT_SYSID_RESULT_PATH, is_result=True)
        try:
            artifact_system_dict = asdict(artifact_system)
            with open(file_path, "wb") as file_obj:
                pickle.dump(artifact_system_dict, file_obj)
        except Exception as err:
            logging.error("Error in saving artifact model")
            raise err
        
    def load_neural_sysid(self):
        """Load the neural systems

        Raises:
            err: error in loading the data
        
        Returns:
            dict: the dictionary for the models' parameters
        """
        # Saving the neural systems
        file_path = self.join_subpath(NEURAL_SYSID_RESULT_PATH, is_result=True)
        try:
            with open(file_path, "rb") as file_obj:
                result_data = pickle.load(file_obj)
        except Exception as err:
            logging.error("Error in loading neural systems")
            raise err
        
        return result_data
        
    def load_artifact_sysid(self):
        """Loading the artifact system

        Raises:
            err: error in loading the data
            
        Returns:
            dict: the dictionary for the model's parameters
        """
        # Loading the artifact systems
        file_path = self.join_subpath(ARTIFACT_SYSID_RESULT_PATH, is_result=True)
        try:
            with open(file_path, "rb") as file_obj:
                result_data = pickle.load(file_obj)
        except Exception as err:
            logging.error("Error in loading artifact model")
            raise err
        
        return result_data
        
    def save_uio_design(self, uio_design_dict:dict):
        """Save the uio design

        Args:
            uio_design_dict (dict): list of parameters of the artifact system

        Raises:
            err: error in saving the data
        """
        # Saving the UIO parameters
        file_path = self.join_subpath(UIO_DESIGN_PATH, is_result=True)
        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(uio_design_dict, file_obj)
        except Exception as err:
            logging.error("Error in saving uio design")
            raise err
        
    def load_uio_design(self):
        """Load the uio design

        Raises:
            err: error in loading the data
            
        Returns:
            dict: the dictionary for the uio's parameters
        """
        # Saving the artifact systems
        file_path = self.join_subpath(UIO_DESIGN_PATH, is_result=True)
        try:
            with open(file_path, "rb") as file_obj:
                result_data = pickle.load(file_obj)
        except Exception as err:
            logging.error("Error in saving uio design")
            raise err
        
        return result_data

    def save_uncertain_mask(self, uncertain_mask_dict:dict, a_ind:int):
        """Save the uncertain mask of amplitude a_ind

        Args:
            uncertain_mask_dict (dict): parameters of the uncertain mask
            a_ind (int): index of amplitude

        Raises:
            err: error in saving the data
        """
        # Saving the uncertain mask
        file_path = self.join_subpath(\
            ARTIFACT_MASK_FOLDER + '/' + 'amp_' + str(a_ind) + '.pickle', \
            is_result=True)
        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(uncertain_mask_dict, file_obj)
        except Exception as err:
            logging.error("Error in saving uncertain mask")
            raise err
        
    def load_uncertain_mask(self, a_ind:int):
        """Load the uncertain mask of amplitude a_ind

        Args:
            a_ind (int): index of amplitude

        Raises:
            err: error in saving the data

        Returns:
            dict: the dictionary for the uncertain mask's parameters
        """
        # Loading the uncertain mask
        file_path = self.join_subpath(\
            ARTIFACT_MASK_FOLDER + '/' + 'amp_' + str(a_ind) + '.pickle', \
            is_result=True)
        try:
            with open(file_path, "rb") as file_obj:
                uncertain_mask_dict = pickle.load(file_obj)
        except Exception as err:
            logging.error("Error in loading uncertain mask")
            raise err
        
        return uncertain_mask_dict

    def save_spike_sorting_config(self, config_dict:dict):
        """Save the configuration parameters of the spike sorting

        Args:
            config_dict (dict): configuration parameters of the spike sorting

        Raises:
            err: error in saving the data
        """
        # Saving the spike sorting configuration
        file_path = self.join_subpath(SPIKE_SORTING_CONFIG_PATH, is_result=True)
        try:
            with open(file_path, 'w') as file_obj:
                json.dump(config_dict, file_obj)
        except Exception as err:
            logging.error("Error in saving spike sorting configuration")
            raise err

    def load_spike_sorting_config(self):
        """Load the configuration parameters of the spike sorting

        Raises:
            err: error in saving the data
            
        Returns:
            dict: configuration parameters of the spike sorting
        """
        # Saving the spike sorting configuration
        file_path = self.join_subpath(SPIKE_SORTING_CONFIG_PATH, is_result=True)
        try:
            with open(file_path, 'r') as file_obj:
                config_dict = json.load(file_obj)
        except Exception as err:
            logging.error("Error in loading spike sorting configuration")
            raise err
        
        return config_dict
        
    def save_spike_sorting_results(self, spike_sorting_results:dict, a_ind:int, \
        t_ind:int):
        """Saving the spiking sorting result of amplitude a_ind and trial t_ind

        Args:
            uncertain_mask_dict (dict): parameters of the uncertain mask
            a_ind (int): index of amplitude
            t_ind (int): index of trial

        Raises:
            err: error in saving the data
        """
        # Saving the spike sorting results
        file_path = self.join_subpath(\
            SPIKE_SORTING_FOLDER + '/' + 'amp_' + str(a_ind) + '_trial_' + \
            str(t_ind) + '.pickle', is_result=True)
        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(spike_sorting_results, file_obj)
        except Exception as err:
            logging.error("Error in saving spike sorting results")
            raise err
        
    def load_spike_sorting_results(self, a_ind:int, t_ind:int):
        """Loading the spiking sorting result of amplitude a_ind and trial t_ind

        Args:
            a_ind (int): index of amplitude
            t_ind (int): index of trial

        Raises:
            err: error in saving the data

        Returns:
            dict: the spiking sorting result
        """
        # Saving the uncertain mask
        file_path = self.join_subpath(\
            SPIKE_SORTING_FOLDER + '/' + 'amp_' + str(a_ind) + '_trial_' + \
            str(t_ind) + '.pickle', is_result=True)
        try:
            with open(file_path, "rb") as file_obj:
                spike_sorting_results = pickle.load(file_obj)
        except Exception as err:
            logging.error("Error in loading uncertain mask")
            raise err
        
        return spike_sorting_results

    def save_spike_detection(self, spike_detection_dict:dict):
        """Saving the spike detection results

        Args:
            spike_detection_dict (dict): results of spike detection

        Raises:
            err: error in saving the data
        """
        # Saving the spike detection parameters
        file_path = self.join_subpath(SPIKE_DETECTION_PATH, is_result=True)
        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(spike_detection_dict, file_obj)
        except Exception as err:
            logging.error("Error in saving uio design")
            raise err
        
    def load_spike_detection(self):
        """Loading the spike detection results

        Raises:
            err: error in saving the data
            
        Returns:
            dict: results of spike detection
        """
        # Loading the spike detection parameters
        file_path = self.join_subpath(SPIKE_DETECTION_PATH, is_result=True)
        try:
            with open(file_path, "rb") as file_obj:
                spike_detection_dict = pickle.load(file_obj)
        except Exception as err:
            logging.error("Error in loading uio design")
            raise err
        
        return spike_detection_dict
        