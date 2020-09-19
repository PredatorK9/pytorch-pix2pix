import torch
import torch.nn as nn
from utils import EncoderBlock, DecoderBlock


class PatchGAN(nn.Module):
    def __init__(self):
        super(PatchGAN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)



class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.enc1 = EncoderBlock(3, 64, batch_norm=False)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.enc5 = EncoderBlock(512, 512)
        self.enc6 = EncoderBlock(512, 512)
        self.enc7 = EncoderBlock(512, 512)

        self.bottle = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)

        self.dec1 = DecoderBlock(512, 512)
        self.dec2 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(1024, 512)
        self.dec4 = DecoderBlock(1024, 512, dropout=False)
        self.dec5 = DecoderBlock(1024, 256, dropout=False)
        self.dec6 = DecoderBlock(512, 128, dropout=False)
        self.dec7 = DecoderBlock(256, 64, dropout=False)

        self.output = nn.ConvTranspose2d(128, 3, kernel_size=4,
            stride=2, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        b = self.bottle(e7)
        x = self.dec1(b, e7)
        x = self.dec2(x, e6)
        x = self.dec3(x, e5)
        x = self.dec4(x, e4)
        x = self.dec5(x, e3)
        x = self.dec6(x, e2)
        x = self.dec7(x, e1)
        x = torch.tanh(self.output(x))
        return x
        