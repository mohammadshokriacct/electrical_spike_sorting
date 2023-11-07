import numpy as np

class LinSysSimulator():
    """Simulator class for linear system with state space representation:
    x_new = A x + B u
    y = C x + D u
    """
    def __init__(self, A:np.ndarray, B:np.ndarray, C:np.ndarray, D:np.ndarray) \
        -> None:
        """Constructor of the class by getting matrices and determining the 
            dimensions.
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        # dimensions
        self.state_dim = self.A.shape[0]
        self.input_dim = self.B.shape[1]
        self.output_dim = self.C.shape[0]
    
    def simulate(self, input_seq:np.ndarray, initial_state:np.ndarray=None, \
        is_state_output_size_match:bool=True) -> tuple[np.ndarray,np.ndarray]:
        """Simulate the system based on a trajectory of input.

        Args:
            input_seq (np.ndarray): the trajectory of input.
            initial_state (np.ndarray, optional): initial state. Defaults to 
                None.
            is_state_output_size_match (bool, optional): Determine if the state 
                and output sequence must have the same size. Defaults to True.

        Returns:
            np.ndarray: output sequence of the simulation
            np.ndarray: state sequence of the simulation
        """
        if initial_state is not None:
            state = initial_state 
        else:
            state = np.zeros([self.state_dim])
        state_seq = np.reshape(state, [1, -1])

        if is_state_output_size_match is True:
            output = self.output_calc(state, np.zeros(self.input_dim))
            output_seq = np.reshape(output, [1, -1])
        else:
            output_seq = np.zeros([self.output_dim, 0])

        # Sequence loop
        for inp in input_seq:
            state = self.state_calc(state, inp)
            output = self.output_calc(state, inp)
            
            state_seq = np.append(state_seq, np.reshape(state, [1, -1]), \
                axis=0)
            output_seq = np.append(output_seq, np.reshape(output, [1, -1]), \
                axis=0)
        
        return output_seq, state_seq

    def state_calc(self, state:np.ndarray, inp:np.ndarray) -> np.ndarray:
        """Calculate the state transition equation: x_new = A x + B u

        Args:
            state (np.ndarray): state vector
            inp (np.ndarray): input vector

        Returns:
            np.ndarray: next state
        """
        return np.matmul(self.A, state) + np.matmul(self.B, inp)

    def output_calc(self, state:np.ndarray, inp:np.ndarray) -> np.ndarray:
        """Calculate the output equation: y = C x + D u

        Args:
            state (np.ndarray): state vector
            inp (np.ndarray): input vector

        Returns:
            np.ndarray: output
        """
        return np.matmul(self.C, state) + np.matmul(self.D, inp)

