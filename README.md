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

## Model Training Logs
Training logs will be generated after running the updated model. You can monitor the training process through the console output.
```
Total Model Parameters: 16,042

Dataset Split:
Training samples: 50,000
Validation/Test samples: 10,000
Split ratio: 50000/10000
Epoch 1: Test set: Average loss: 0.0840, Accuracy: 97.30%
Epoch 2: Test set: Average loss: 0.0812, Accuracy: 97.44%
Epoch 3: Test set: Average loss: 0.0415, Accuracy: 98.74%
Epoch 4: Test set: Average loss: 0.0363, Accuracy: 98.82%
Epoch 5: Test set: Average loss: 0.0342, Accuracy: 98.88%
Epoch 6: Test set: Average loss: 0.0330, Accuracy: 98.96%
Epoch 7: Test set: Average loss: 0.0261, Accuracy: 99.19%
Epoch 8: Test set: Average loss: 0.0280, Accuracy: 99.17%
Epoch 9: Test set: Average loss: 0.0269, Accuracy: 99.13%
Epoch 10: Test set: Average loss: 0.0235, Accuracy: 99.24%
Epoch 11: Test set: Average loss: 0.0230, Accuracy: 99.22%
Epoch 12: Test set: Average loss: 0.0212, Accuracy: 99.28%
Epoch 13: Test set: Average loss: 0.0261, Accuracy: 99.17%
Epoch 14: Test set: Average loss: 0.0202, Accuracy: 99.46%

Reached target accuracy of 99.4% at epoch 14

Training Complete!
==================================================
Dataset Split Summary:
Training Set: 50,000 samples
Validation/Test Set: 10,000 samples
Split Ratio: 50000/10000
--------------------------------------------------
Total Model Parameters: 16,042
Best Validation/Test Accuracy: 99.46%
Final Training Loss: 0.0343
Final Validation/Test Loss: 0.0202
Training stopped at epoch: 14/19
==================================================
```

## Model Test Logs
```

```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
