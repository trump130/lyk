import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import Conv2dWithConstraint
import math

class ActSquare(nn.Module):
    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))

class SELayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ECALayer(nn.Module):
    def __init__(self,channels,k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1, kernel_size=k_size,padding="same",bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y



class TemporalConvNet(nn.Module):
    def __init__(self, channels, kernel_size=3, num_layers=2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [
                nn.Conv2d(channels, channels,kernel_size=(1, kernel_size),padding="same",groups=channels,bias=False),
                nn.BatchNorm2d(channels),
                nn.ELU()
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CHBNet(nn.Module):
    def __init__(self,
                 num_channels: int,
                 sampling_rate:int,
                 F1=2,D=1,F2='auto',drop_out=0.25):
        super().__init__()

        if F2 == 'auto':
            F2 = F1 * D

        self.branch_1 = nn.Sequential(
            Conv2dWithConstraint(1, F1,kernel_size=(1, 125),padding='same',max_norm=2.),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            Conv2dWithConstraint(F1, F2,kernel_size=(num_channels, 1),groups=F1,bias=False,max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 30), stride=15),
            nn.Dropout(drop_out)
        )

        self.branch_2 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=(num_channels, 1), padding='valid', max_norm=2.),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, F2, kernel_size=(1, 30), padding='same', max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            nn.AvgPool2d((1, 30), stride=15),
            ActLog(),
            nn.Dropout(drop_out)
        )

        self.se_1 = SELayer(F2)
        fused_channels = 2 * F2

        self.eca = ECALayer(fused_channels, k_size=3)
        self.tcn = TemporalConvNet(fused_channels)
        self.drop = nn.Dropout(drop_out)
    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x1 = self.se_1(x1)
        x = torch.cat([x1, x2], dim=1)  # [B, 3F2, 1, T]
        x = self.eca(x)
        x = self.tcn(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=(1, 49)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x.squeeze(-1).squeeze(-1)

class Net(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_channels: int,
                 sampling_rate: int):
        super().__init__()

        F1, D = 10, 1
        F2 = F1 * D

        self.backbone = CHBNet(
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            F1=F1,
            D=D,
            F2=F2
        )

        self.classifier = Classifier(
            num_classes=num_classes,
            in_channels=2 * F2
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def get_model(args):
    model = Net(
        num_classes=args.num_classes,
        num_channels=args.num_channels,
        sampling_rate=args.sampling_rate
    )
    return model