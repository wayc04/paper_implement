import torch
import torch.nn as nn
import torch.nn.functional as F

class NIN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(NIN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels=192, kernel_size=5, padding=2)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.05)

        self.cccp1 = nn.Conv2d(192, 160, kernel_size=1)
        nn.init.normal_(self.cccp1.weight, mean=0.0, std=0.05)

        self.cccp2 = nn.Conv2d(160, 96, kernel_size=1)
        nn.init.normal_(self.cccp2.weight, mean=0.0, std=0.05)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, padding=2)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.05)

        self.cccp3 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.normal_(self.cccp3.weight, mean=0.0, std=0.05)

        self.cccp4 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.normal_(self.cccp4.weight, mean=0.0, std=0.05)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.05)

        self.cccp5 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.normal_(self.cccp5.weight, mean=0.0, std=0.05)

        self.cccp6 = nn.Conv2d(192, num_classes, kernel_size=1)
        nn.init.normal_(self.cccp6.weight, mean=0.0, std=0.05)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.cccp1(x))
        x = F.relu(self.cccp2(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.cccp3(x))
        x = F.relu(self.cccp4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.cccp5(x))
        x = F.relu(self.cccp6(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return F.log_softmax(x, dim=1)

    