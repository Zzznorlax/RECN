import torch.nn as nn
import torch
import torchvision.transforms as transforms


class RECN(nn.Module):
    def __init__(self):
        super(RECN, self).__init__()
        self.name = "RECN"
        self.criterion = nn.MSELoss()
        self.data_trans = transforms.Compose([
            transforms.ToTensor()
        ])

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.decv1 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.decv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.decv3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.decv4 = nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.decv1(x)
        x = self.relu(x)

        x = self.decv2(x)
        x = self.relu(x)

        x = self.decv3(x)
        x = self.relu(x)

        x = self.decv4(x)
        x = self.relu(x)

        x = torch.tanh(x)

        return x
