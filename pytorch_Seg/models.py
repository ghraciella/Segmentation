import torch.nn as nn
import torch.nn.functional as F
import torch
numclass = 19

class ConvActivationBatchnorm(nn.Module):
    """ bla
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1, transposed = False, stride = 1):
        super(ConvActivationBatchnorm, self).__init__()
        if transposed:
            outputpadding = 1
            self.conv = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = outputpadding)
        else:
            self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, bias = False)
        self.activation = nn.ReLU()
        self.BN = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return self.BN(self.activation(self.conv(x)))

class FCN8(nn.Module):

    def __init__(self, classes=numclass):
        self.classes = classes

        super(FCN8, self).__init__()
        self.conv11 = ConvActivationBatchnorm(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv12 = ConvActivationBatchnorm(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv21 = ConvActivationBatchnorm(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv22 = ConvActivationBatchnorm(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv31 = ConvActivationBatchnorm(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv32 = ConvActivationBatchnorm(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv33 = ConvActivationBatchnorm(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = ConvActivationBatchnorm(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv42 = ConvActivationBatchnorm(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv43 = ConvActivationBatchnorm(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv51 = ConvActivationBatchnorm(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv52 = ConvActivationBatchnorm(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv53 = ConvActivationBatchnorm(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv6 = ConvActivationBatchnorm(in_channels=512, out_channels=64, kernel_size=7, padding=3)
        self.dropout1 = nn.Dropout(0.1)

        self.conv7 = ConvActivationBatchnorm(in_channels=64, out_channels=256, kernel_size=1, padding = 0)
        self.dropout2 = nn.Dropout(0.1)

        self.conv8 = ConvActivationBatchnorm(in_channels=256, out_channels=512, kernel_size=1, padding = 0)

        self.tranconv1 = ConvActivationBatchnorm(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, transposed=True)

        self.tranconv2 = ConvActivationBatchnorm(in_channels=2*512, out_channels=256, kernel_size=3, stride=2, padding=1, transposed=True)

        self.tranconv3 = nn.ConvTranspose2d(in_channels=2*256, out_channels=classes, kernel_size=16, padding=4,  stride=8)
        # print(self)

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool1(x)
        x = self.pool2(self.conv22(self.conv21(x)))
        x1 = self.pool3(self.conv33(self.conv32(self.conv31(x))))
        x2 = self.pool4(self.conv43(self.conv42(self.conv41(x1))))
        x = self.pool5(self.conv53(self.conv52(self.conv51(x2))))
        x = self.dropout1(self.conv6(x))
        x = self.dropout2(self.conv7(x))
        x = self.conv8(x)
        x = self.tranconv1(x)
        x = torch.cat((x,x2), dim = 1)
        x = self.tranconv2(x)
        x = torch.cat((x,x1), dim = 1)
        x = self.tranconv3(x)
        return x


#net = FCN8(19)
# print(net)
