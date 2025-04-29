import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyVGG(nn.Module):
    """
    Tiny VGG structure adapted for PyTorch to handle generic input shapes. 
    It follows the [conv-relu-conv-relu-pool]x3-fc-softmax architecture 
    for classifying images into classes.
    """
    def __init__(self, num_classes=10, filters=10):
        super(TinyVGG, self).__init__()
        self.num_classes = num_classes
        self.filters = filters
        # Convolutional Block 1
        self.conv_1_1 = nn.Conv2d(in_channels=3, out_channels=filters, kernel_size=3, padding=1)
        self.conv_1_2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv_2_1 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.conv_2_2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv_3_1 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.conv_3_2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # The fully connected layer will be defined later, dynamically
        self.fc = None  # Placeholder for the fully connected layer

    def forward(self, x):
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = self.max_pool_1(x)

        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = self.max_pool_2(x)

        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = self.max_pool_3(x)

        # Dynamically calculate the size
        if self.fc is None:
            # This assumes x has a shape of [batch_size, channels, height, width]
            num_features_before_fc = torch.flatten(x, 1).size(1)
            self.fc = nn.Linear(num_features_before_fc, self.num_classes).to(x.device)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
