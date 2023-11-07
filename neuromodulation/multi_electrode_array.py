import numpy as np

def calculate_electrode_distance(electrode_x:np.ndarray, electrode_y:np.ndarray) -> np.ndarray:
    """Calculate the distance between electrodes

    Args:
        electrode_x (np.ndarray): position x of the electrodes
        electrode_y (np.ndarray): position y of the electrodes

    Returns:
        np.ndarray: distance between each two electrodes
    """
    elec_num = len(electrode_x)
    xx_diff = np.tile(electrode_x, [elec_num, 1]) - np.transpose(np.tile(electrode_x, [elec_num, 1]))
    yy_diff = np.tile(electrode_y, [elec_num, 1]) - np.transpose(np.tile(electrode_y, [elec_num, 1]))
    electrode_distance = np.sqrt(xx_diff**2 + yy_diff**2)
    
    return electrode_distance