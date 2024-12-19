import pytest
import torch
from model import Net
from utils import get_data_loaders
from config import DEVICE, BATCH_SIZE

@pytest.fixture
def model():
    """Fixture for the model."""
    return Net().to(DEVICE)

@pytest.fixture
def train_loader():
    """Fixture for the training data loader."""
    train_loader, _ = get_data_loaders(BATCH_SIZE)
    return train_loader

@pytest.fixture
def test_loader():
    """Fixture for the test data loader."""
    _, test_loader = get_data_loaders(BATCH_SIZE)
    return test_loader

def test_parameter_count(model):
    """Test to check the total number of parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameter Count Test:")
    print(f"Total parameters in model: {total_params:,}")
    assert total_params < 20000, f"Model has {total_params:,} parameters, should be < 20,000"

def test_required_components(model):
    """Test to verify required components are present in the model."""
    has_batchnorm = any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.children())
    has_dropout = any(isinstance(layer, torch.nn.Dropout) for layer in model.children())
    has_conv2d = any(isinstance(layer, torch.nn.Conv2d) for layer in model.children())
    has_linear = any(isinstance(layer, torch.nn.Linear) for layer in model.children())
    has_gap = any(isinstance(layer, torch.nn.AdaptiveAvgPool2d) for layer in model.children())

    print("Required Components Test:")
    print(f"BatchNorm layer present: {has_batchnorm}")
    print(f"Dropout layer present: {has_dropout}")
    print(f"Conv2d layer present: {has_conv2d}")
    print(f"Linear layer present: {has_linear}")
    print(f"Global Average Pooling layer present: {has_gap}")

    assert has_batchnorm, "BatchNorm layer is missing"
    assert has_dropout, "Dropout layer is missing"
    assert has_conv2d, "Conv2d layer is missing"
    assert has_linear, "Linear layer is missing"
    assert has_gap, "Global Average Pooling layer is missing"

    print("All required components are present in the model.")
