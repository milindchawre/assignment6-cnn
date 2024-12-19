# MNIST Classification with CNN

![Build Status](https://github.com/milindchawre/assignment6-cnn/actions/workflows/ml-pipeline.yml/badge.svg)

A PyTorch implementation of MNIST digit classification achieving over 99.4% test accuracy with approximately 16k parameters.

## Features
- Convolutional Neural Network (CNN) architecture
- Batch Normalization for improved training stability
- Dropout layers for regularization
- Global Average Pooling (GAP) for dimensionality reduction
- Data augmentation techniques for better generalization
- Comprehensive testing suite to verify model components and performance

## Requirements
- Python 3.8+
- PyTorch 1.7+
- torchvision
- pytest
- tqdm
- matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/milindchawre/assignment6-cnn.git
   cd assignment6-cnn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train the model from scratch, run:
```bash
python src/train.py
```
The training process will:
- Load and preprocess the MNIST dataset
- Train the model for up to 19 epochs
- Save the best model weights to `model.pth`
- Display training progress and final results

### Testing the Model
To run the model tests, execute:
```bash
pytest src/test_model.py -s
```
This will verify:
- Parameter count is within limits
- Model architecture includes required components (BatchNorm, Dropout, GAP, etc.)
- Model achieves target accuracy

## Model Architecture Highlights
- Initial channels: 10
- Progressive channel expansion: 10 → 14 → 20
- Convolutional layers with Batch Normalization and Dropout
- Global Average Pooling before the final fully connected layer
- Final FC layer: 20 → 10

## Model Training Logs (Github Action - https://github.com/milindchawre/assignment6-cnn/actions/runs/12412505708/job/34652442221)
Training logs will be generated after running the updated model. You can monitor the training process through the console output.
```
Total Model Parameters: 16,042

Dataset Split:

Training samples: 50,000
Validation/Test samples: 10,000
Split ratio: 50000/10000

Epoch 1: Test set: Average loss: 0.0787, Accuracy: 97.47%
Epoch 2: Test set: Average loss: 0.0527, Accuracy: 98.22%
Epoch 3: Test set: Average loss: 0.0475, Accuracy: 98.45%
Epoch 4: Test set: Average loss: 0.0398, Accuracy: 98.61%
Epoch 5: Test set: Average loss: 0.0314, Accuracy: 99.04%
Epoch 6: Test set: Average loss: 0.0271, Accuracy: 99.05%
Epoch 7: Test set: Average loss: 0.0273, Accuracy: 98.99%
Epoch 8: Test set: Average loss: 0.0229, Accuracy: 99.14%
Epoch 9: Test set: Average loss: 0.0288, Accuracy: 99.06%
Epoch 10: Test set: Average loss: 0.0240, Accuracy: 99.15%
Epoch 11: Test set: Average loss: 0.0239, Accuracy: 99.19%
Epoch 12: Test set: Average loss: 0.0267, Accuracy: 99.05%
Epoch 13: Test set: Average loss: 0.0238, Accuracy: 99.15%
Epoch 14: Test set: Average loss: 0.0304, Accuracy: 98.94%
Epoch 15: Test set: Average loss: 0.0165, Accuracy: 99.46%
Reached target accuracy of 99.4% at epoch 15
Training Complete!

==================================================
Dataset Split Summary:
Training Set: 50,000 samples
Validation/Test Set: 10,000 samples
Split Ratio: 50000/10000
--------------------------------------------------
Total Model Parameters: 16,042
Best Validation/Test Accuracy: 99.46%
Final Training Loss: 0.0286
Final Validation/Test Loss: 0.0165
Training stopped at epoch: 15/19
==================================================
```

## Model Test Logs (Github Action - https://github.com/milindchawre/assignment6-cnn/actions/runs/12412505708/job/34652442221)
```
============================= test session starts ==============================
platform linux -- Python 3.8.18, pytest-8.3.4, pluggy-1.5.0
rootdir: /home/runner/work/assignment6-cnn/assignment6-cnn
collected 2 items
src/test_model.py 
Model Parameter Count Test:
Total parameters in model: 16,042
.Required Components Test:
BatchNorm layer present: True
Dropout layer present: True
Conv2d layer present: True
Linear layer present: True
Global Average Pooling layer present: True
All required components are present in the model.
.
============================== 2 passed in 1.97s ===============================
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
