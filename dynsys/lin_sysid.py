import tensorflow as tf
from tensorflow import keras
import numpy as np

class SparseLinearSys(keras.layers.Layer):
    """Keras model for control systems with sparse variables.
    x_new = A x + B u
    """
    def __init__(self, state_dim:int=2, input_dim:int=1, mask_matrix_A=None, \
        mask_matrix_B=None, b_scale:float=1):
        """Constructor of linear system model with sparse elements that are 
            specified by the masks. 

        Args:
            state_dim (int, optional): state dimension. Defaults to 2.
            input_dim (int, optional): input dimension. Defaults to 1.
            mask_matrix_A (np.ndarray, optional): mask of A matrix. Defaults to 
                None.
            mask_matrix_B (np.ndarray, optional): mask of B matrix. Defaults to 
                None.
            b_scale (float, optional): scale of B matrix. Defaults to 1.
        """
        super(SparseLinearSys, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        if mask_matrix_A is None:
            mask_matrix_A = np.ones(self.state_dim, self.state_dim)
        self.mask_matrix_A = mask_matrix_A
        
        if mask_matrix_B is None:
            mask_matrix_B = np.ones(self.input_dim, self.state_dim)
        self.mask_matrix_B = mask_matrix_B
        
        self.b_scale = b_scale
        
        number_of_one = sum(sum(mask_matrix_A))
        self.trainable_A_weight = self.add_weight(
            shape=(number_of_one,), initializer="zeros", dtype=tf.float64, \
                trainable=True
        )
        self.indices_of_A = np.transpose(np.stack(np.where(self.mask_matrix_A \
            == 1)))
            
        number_of_one = sum(sum(mask_matrix_B))
        self.trainable_B_weight = self.add_weight(
            shape=(number_of_one,), initializer="zeros", dtype=tf.float64, \
                trainable=True
        )
        self.indices_of_B = np.transpose(np.stack(np.where(self.mask_matrix_B \
            == 1)))
            
    def call(self, state, input):
        """calling the Keras model.
        """
        state = tf.cast(state, dtype=tf.float64)
        input = tf.cast(input, dtype=tf.float64)
        A = self.A_mat()
        B = self.B_mat()
        return tf.matmul(state, tf.transpose(A)) + \
            tf.matmul(input, tf.transpose(B))
    
    def A_mat(self):
        """A matrix getter.
        """
        return tf.scatter_nd(self.indices_of_A, self.trainable_A_weight, \
            [self.state_dim, self.state_dim])
    
    def B_mat(self):
        """B matrix getter.
        """
        if type(self.b_scale) is not list:
            return self.b_scale*tf.scatter_nd(self.indices_of_B, \
                self.trainable_B_weight, [self.state_dim, self.input_dim])
        else:
            return tf.matmul(tf.scatter_nd(self.indices_of_B, \
                self.trainable_B_weight, [self.state_dim, self.input_dim]), \
                np.diag(self.b_scale))
    