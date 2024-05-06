from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

        self.conv1.bias.data *= 0
        self.conv2.bias.data *= 0
        self.conv3.bias.data *= 0

    def forward(self, image):
        x = self.relu(self.conv1(image))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x + image


# Raw SRCNN:
# best epoch: 0, psnr: 29.14
# best epoch: 1, psnr: 30.76
# best epoch: 2, psnr: 31.34
# best epoch: 3, psnr: 31.57
# best epoch: 4, psnr: 31.71
# best epoch: 5, psnr: 31.81

# Add residual:
# best epoch: 0, psnr: 31.95
# best epoch: 1, psnr: 32.17
# best epoch: 2, psnr: 32.24
# best epoch: 3, psnr: 32.32
# best epoch: 4, psnr: 32.35
# best epoch: 5, psnr: 32.37

# set initial bias=0:
# best epoch: 0, psnr: 32.00
# best epoch: 1, psnr: 32.21
# best epoch: 2, psnr: 32.29
# best epoch: 3, psnr: 32.38
# best epoch: 4, psnr: 32.44
# best epoch: 5, psnr: 32.50
