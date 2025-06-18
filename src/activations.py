import numpy as np

class Activation:
    def __init__(self, name):
        self.name = name

    def forward(self, Z):
        raise NotImplementedError

    def backward(self, dA):
        raise NotImplementedError

    def get_config(self):
        return {'name': self.name}

class ReLU(Activation):
    def __init__(self):
        super().__init__('relu')

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.Z <= 0] = 0
        return dZ

class Sigmoid(Activation):
    def __init__(self):
        super().__init__('sigmoid')

    def forward(self, Z):
        self.Z = Z
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dA):
        s = self.A
        dZ = dA * s * (1 - s)
        return dZ

class Tanh(Activation):
    def __init__(self):
        super().__init__('tanh')

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        """
        dA is dL/dA
        Returns dL/dZ = dL/dA * dA/dZ = dA * (1 - A^2)
        """
        dZ = dA * (1 - np.power(self.A, 2))
        return dZ

class Softmax(Activation):
    def __init__(self):
        super().__init__('softmax')

    def forward(self, Z):
        """
        Z is the input (logits)
        Numerically stable softmax: max_val = np.max(Z, axis=-1, keepdims=True)
                                 exp_Z = np.exp(Z - max_val)
        """
        # Subtract max for numerical stability (prevents overflow)
        exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        self.A = exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)
        return self.A

    def backward(self, dA):
        """
        dA is dL/dA (gradient of Loss w.r.t. the output of Softmax)
        Returns dL/dZ (gradient of Loss w.r.t. the input of Softmax)

        For a single sample, if A = softmax(Z):
        dZ_i = sum_j (dA_j * d(A_j)/d(Z_i))
        d(A_j)/d(Z_i) = A_j * (delta_ji - A_i)

        So, dZ_i = sum_j (dA_j * A_j * (delta_ji - A_i))
                 = dA_i * A_i * (1 - A_i) + sum_{j!=i} (dA_j * A_j * (-A_i))
                 = A_i * (dA_i - sum_j (dA_j * A_j))

        This can be computed for a batch as:
        S = self.A (output of softmax, shape N x C)
        dS = dA (gradient dL/dA, shape N x C)
        dL_dZ = S * (dS - np.sum(dS * S, axis=-1, keepdims=True))
        """
        # Note: This is the general case. If combined with CrossEntropyLoss,
        # the dL/dZ often simplifies to (A - Y_true).
        # Here, we assume dA is the gradient from the subsequent step/loss.

        # Ensure A has been computed and stored from forward pass
        if not hasattr(self, 'A'):
            raise ValueError("Forward pass must be called before backward pass for Softmax.")

        # For each sample in the batch
        batch_size = dA.shape[0]
        num_classes = dA.shape[1]
        dZ = np.empty_like(dA)

        for i in range(batch_size):
            # Get the softmax output for the i-th sample
            s_i = self.A[i, :].reshape(1, -1) # Shape (1, C)
            # Get the gradient dL/dA for the i-th sample
            da_i = dA[i, :].reshape(1, -1)   # Shape (1, C)

            # Compute the Jacobian of softmax output w.r.t its input for the i-th sample
            # J_mn = d(s_m)/d(z_n) = s_m * (delta_mn - s_n)
            # jacobian_matrix is C x C
            jacobian_matrix = np.diagflat(s_i) - np.dot(s_i.T, s_i)

            # dL/dZ_i = dL/dA_i @ jacobian_matrix
            dZ[i, :] = np.dot(da_i, jacobian_matrix)

        return dZ

# Factory function for activations
def get_activation(name):
    if name is None:
        return None
    name = name.lower()
    if name == 'relu':
        return ReLU()
    elif name == 'sigmoid':
        return Sigmoid()
    elif name == 'tanh':
        return Tanh()
    elif name == 'softmax':
        return Softmax()
    else:
        raise ValueError(f"Unknown activation function: {name}")
