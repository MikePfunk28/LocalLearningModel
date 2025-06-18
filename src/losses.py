import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_pred, y_true):
        # Number of samples in the batch for averaging the gradient
        m = y_true.shape[0]
        return 2 * (y_pred - y_true) / m

class CrossEntropyLoss(Loss):
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        y_pred: Predicted probabilities (output of Softmax), shape (batch_size, num_classes)
        y_true: True labels (one-hot encoded), shape (batch_size, num_classes)
        """
        # Clip y_pred to prevent log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1. - self.epsilon)

        # Number of samples in the batch
        m = y_true.shape[0]

        # Compute cross-entropy loss
        log_likelihood = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, y_pred, y_true):
        """
        y_pred: Predicted probabilities, shape (batch_size, num_classes)
        y_true: True labels (one-hot encoded), shape (batch_size, num_classes)
        Returns: Gradient of the loss w.r.t. y_pred (dL/dy_pred)
        """
        # Clip y_pred to prevent division by zero
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1. - self.epsilon)

        # Number of samples in the batch
        m = y_true.shape[0]

        # Gradient of cross-entropy loss
        # dL/dy_pred = -y_true / y_pred
        # The division by m is to average the gradient over the batch
        gradient = -y_true / y_pred_clipped
        return gradient / m
