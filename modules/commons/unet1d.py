from collections import OrderedDict

import torch
import torch.nn as nn


class UNet1d(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=128, multi=None):
        super(UNet1d, self).__init__()
        if multi is None:
            multi = [1, 2, 2, 4]
        features = init_features
        self.encoder1 = UNet1d._block(in_channels, features * multi[0], name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1d._block(features * multi[0], features * multi[1], name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1d._block(features * multi[1], features * multi[2], name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1d._block(features * multi[2], features * multi[3], name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1d._block(features * multi[3], features * multi[3], name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * multi[3], features * multi[3], kernel_size=2, stride=2
        )
        self.decoder4 = UNet1d._block((features * multi[3]) * 2, features * multi[3], name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * multi[3], features * multi[2], kernel_size=2, stride=2
        )
        self.decoder3 = UNet1d._block((features * multi[2]) * 2, features * multi[2], name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * multi[2], features * multi[1], kernel_size=2, stride=2
        )
        self.decoder2 = UNet1d._block((features * multi[1]) * 2, features * multi[1], name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * multi[1], features * multi[0], kernel_size=2, stride=2
        )
        self.decoder1 = UNet1d._block(features * multi[0] * 2, features * multi[0], name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features * multi[0], out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, nonpadding=None):
        if nonpadding is None:
            nonpadding = torch.ones_like(x)[:, :, :1]
        enc1 = self.encoder1(x.transpose(1, 2)) * nonpadding.transpose(1, 2)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1).transpose(1, 2) * nonpadding

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=5,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.GroupNorm(4, features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=5,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.GroupNorm(4, features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


class UNet2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, multi=None):
        super(UNet2d, self).__init__()

        features = init_features
        self.encoder1 = UNet2d._block(in_channels, features * multi[0], name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features * multi[0], features * multi[1], name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(features * multi[1], features * multi[2], name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2d._block(features * multi[2], features * multi[3], name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2d._block(features * multi[3], features * multi[3], name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * multi[3], features * multi[3], kernel_size=2, stride=2
        )
        self.decoder4 = UNet2d._block((features * multi[3]) * 2, features * multi[3], name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * multi[3], features * multi[2], kernel_size=2, stride=2
        )
        self.decoder3 = UNet2d._block((features * multi[2]) * 2, features * multi[2], name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * multi[2], features * multi[1], kernel_size=2, stride=2
        )
        self.decoder2 = UNet2d._block((features * multi[1]) * 2, features * multi[1], name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * multi[1], features * multi[0], kernel_size=2, stride=2
        )
        self.decoder1 = UNet2d._block(features * multi[0] * 2, features * multi[0], name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features * multi[0], out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        x = self.conv(dec1)
        return x

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.GroupNorm(4, features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.GroupNorm(4, features)),
                    (name + "tanh2", nn.Tanh()),
                    (name + "conv3", nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=1,
                        padding=0,
                        bias=True,
                    )),
                ]
            )
        )
