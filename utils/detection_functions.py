import numpy as np
from scipy.signal import convolve2d

def cross_corrolation_match(sig1:np.ndarray, sig2:np.ndarray) -> np.ndarray:
    """Calculate the cross corrolation of sig1 and sig2

    Args:
        sig1 (np.ndarray): first signal
        sig2 (np.ndarray): second signal

    Returns:
        np.ndarray: cross corrolation of signals
    """
    w_mat = np.diag(np.sum(sig2,axis=0)**-1)/np.sum(np.sum(sig2,axis=0)**-1)
    conv_result = convolve2d(sig1, sig2, mode='same', boundary='fill', \
        fillvalue=0)[:,0:sig2.shape[1]]
    return np.sum(np.matmul(conv_result, w_mat), axis=1)
    
def square_error_match(sig1:np.ndarray, sig2:np.ndarray) -> np.ndarray:
    """Calculate the square error of sig1 and sig2

    Args:
        sig1 (np.ndarray): first signal
        sig2 (np.ndarray): second signal

    Returns:
        np.ndarray: square error of signals
    """
    w_mat = np.diag(np.sum(sig2,axis=0)**-1)/np.sum(np.sum(sig2,axis=0)**-1)
    temp_len = sig2.shape[0]
    detected_input = np.concatenate([sig1, np.tile(sig1[-1,:], \
        [temp_len-1,1])], axis=0)

    detection_error_result = list()
    for i in range(sig1.shape[0]):
        detection_error_result.append(\
            np.sum(np.matmul((\
                np.abs(detected_input[i:(i+temp_len), :] - sig2))**2, w_mat)))
    
    return np.array(detection_error_result)
