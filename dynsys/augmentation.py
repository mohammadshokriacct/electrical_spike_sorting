import numpy as np
from scipy.linalg import block_diag

def augment_dynsys(A_list:list[np.ndarray], B_list:list[np.ndarray], \
    C_list:list[np.ndarray], D_list:list[np.ndarray], \
    additive_output:bool=False) -> \
    tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    # augmentation
    A_aug = block_diag(*A_list)
    B_aug = block_diag(*B_list)
    if additive_output is True:
        C_aug = np.concatenate(C_list, axis=1)
        D_aug = np.concatenate(D_list, axis=1)
    else:
        C_aug = block_diag(C_list, axis=1)
        D_aug = block_diag(D_list, axis=1)
    
    return A_aug, B_aug, C_aug, D_aug

def two_input_augment_dynsys(A_list:list[np.ndarray], \
    B1_list:list[np.ndarray], B2_list:list[np.ndarray], \
    C_list:list[np.ndarray], D1_list:list[np.ndarray], \
    D2_list:list[np.ndarray], additive_output:bool=False) -> \
    tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    # augmentation
    A_aug = block_diag(*A_list)
    B1_aug = block_diag(*B1_list)
    B2_aug = block_diag(*B2_list)
    if additive_output is True:
        C_aug = np.concatenate(C_list, axis=1)
        D1_aug = np.concatenate(D1_list, axis=1)
        D2_aug = np.concatenate(D2_list, axis=1)
    else:
        C_aug = block_diag(*C_list)
        D1_aug = block_diag(*D1_list)
        D2_aug = block_diag(*D2_list)
    
    return A_aug, B1_aug, B2_aug, C_aug, D1_aug, D2_aug