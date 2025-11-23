import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Basic CNN for classifying proposal crops as potholde or background"""

    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # 4 conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    # quick test
    model = SimpleCNN()
    x = torch.randn(4, 3, 128, 128)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")

    # count params
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")