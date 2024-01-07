import pytest
import torch
from torch import Tensor
from models import MultivariateModel  # Replace 'your_module' with the actual module name

@pytest.fixture
def model():
    T, D, tau = 5, 3, 2
    return MultivariateModel(T, D, tau)

def test_encode(model):
    x = torch.randn(10, model.fc1.in_features)
    encoded = model.encode(x)
    assert isinstance(encoded, Tensor)
    assert encoded.shape == (10, model.fc1.out_features)

def test_query(model):
    h = torch.randn(10, model.fc1.out_features)
    result = model.query(h)
    assert isinstance(result, Tensor)
    assert result.shape == (10, model.fc2.out_features)

def test_forward(model):
    x_seq = torch.randn(5, model.fc1.in_features)
    output = model.forward(x_seq)
    assert isinstance(output, Tensor)
    assert output.shape == (5, model.fc2.out_features)

def test_forward_with_different_input_dimensions(model):
    with pytest.raises(RuntimeError, match=".*mat1 and mat2 shapes cannot be multiplied.*"):
        x_seq = torch.randn(5, model.fc1.in_features + 1)  # Adding 1 to cause size mismatch
        model.forward(x_seq)

def test_forward_with_invalid_input_type(model):
    with pytest.raises(TypeError, match=".*linear.*input.*must be Tensor.*"):
        x_seq = "invalid_input"
        model.forward(x_seq)