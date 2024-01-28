import torch
import torch.nn as nn
from torch import Tensor


class MultivariateSeq2SeqModel(nn.Module):
    """A PyTorch neural network model for multivariate data.

    Args:
        T (int): The input dimension.
        D (int): The hidden dimension.
        tau (int): The output dimension.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        relu (nn.ReLU): The ReLU activation function.

    Methods:
        encode(x: Tensor) -> Tensor: Encodes the input tensor using the first fully connected layer and ReLU activation.
        query(h: Tensor) -> Tensor: Performs a query on the hidden state tensor using the second fully connected layer.
        forward(x_seq: Tensor) -> Tensor: Performs forward propagation on the input sequence tensor.

    """

    def __init__(self, T: int, D: int, tau: int) -> None:
        super(MultivariateSeq2SeqModel, self).__init__()
        print("T : ", T)
        print("D : ", D)
        self.fc1 = nn.Linear(T, D)
        self.fc2 = nn.Linear(D, tau)
        self.relu = nn.ReLU()

    def encode(self, x: Tensor) -> Tensor:
        """Encodes the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The encoded tensor.

        """
        h = self.fc1(x)
        return self.relu(h)

    def query(self, h: Tensor) -> Tensor:
        """Performs a query on the hidden state tensor.

        Args:
            h (Tensor): The hidden state tensor.

        Returns:
            Tensor: The query result.

        """
        return self.fc2(h)

    def forward(self, x_seq: Tensor) -> Tensor:
        """Performs forward propagation on the input sequence tensor.

        Args:
            x_seq (Tensor): The input sequence tensor.

        Returns:
            Tensor: The output tensor.

        """
        h = self.encode(x_seq)
        h = self.query(h)
        return h
