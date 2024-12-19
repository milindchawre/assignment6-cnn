import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First Block
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)  # Reduced from 12 to 10
        self.bn1 = nn.BatchNorm2d(10)
        
        # Second Block
        self.conv2 = nn.Conv2d(10, 14, 3, padding=1)  # Reduced from 16 to 14
        self.bn2 = nn.BatchNorm2d(14)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)
        
        # Third Block
        self.conv3 = nn.Conv2d(14, 20, 3, padding=1)  # Reduced from 24 to 20
        self.bn3 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.1)
        
        # Fourth Block with parallel paths
        self.conv4_1x1 = nn.Conv2d(20, 20, 1)  # Adjusted to match new channel size
        self.conv4_main = nn.Conv2d(20, 20, 3, padding=1)  # Adjusted to match new channel size
        self.bn4 = nn.BatchNorm2d(20)
        
        # Fifth Block with attention
        self.conv5 = nn.Conv2d(20, 20, 3, padding=1)  # Adjusted to match new channel size
        self.bn5 = nn.BatchNorm2d(20)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(20, 10, 1),  # Adjusted channels
            nn.ReLU(),
            nn.Conv2d(10, 20, 1),  # Adjusted channels
            nn.Sigmoid()
        )
        
        # Sixth Block
        self.conv6 = nn.Conv2d(20, 20, 3, padding=1)  # Adjusted to match new channel size
        self.bn6 = nn.BatchNorm2d(20)
        self.dropout3 = nn.Dropout(0.1)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final FC Layer
        self.fc = nn.Linear(20, 10)  # Adjusted input features

    def forward(self, x):
        # First Block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second Block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        # Third Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2(self.pool2(x))
        
        # Fourth Block with parallel paths
        x_1x1 = self.conv4_1x1(x)
        x_main = F.relu(self.bn4(self.conv4_main(x)))
        x = x_main + F.relu(x_1x1)  # residual connection
        
        # Fifth Block with attention
        x = F.relu(self.bn5(self.conv5(x)))
        att = self.attention(x)
        x = x * att
        
        # Sixth Block
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout3(x)
        
        # GAP and FC
        x = self.gap(x)
        x = x.view(-1, 20)  # Changed to match the output of the last conv layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1) 