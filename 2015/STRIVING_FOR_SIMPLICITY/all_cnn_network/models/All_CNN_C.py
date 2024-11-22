import torch.nn as nn

class AllCNNC(nn.Module):
    def __init__(self,input_channels=3, n_classes=10):
        super(AllCNNC, self).__init__()

        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3)

        self.conv3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2)

        self.dropout2 = nn.Dropout(0.5)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2)

        self.dropout3 = nn.Dropout(0.5)
        self.conv7 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3)
        self.conv8 = nn.Conv2d(in_channels=192, out_channels=192 , kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels=192, out_channels=n_classes, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(x)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x= self.dropout2(x)

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.dropout3(x)

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.conv9(x)
        # we can not add relu here because the last layer output all negative values
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        return x