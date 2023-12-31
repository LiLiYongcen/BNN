import torch
import torch.nn as nn


def Conv3x3BNReLU(in_channels,out_channels,stride,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def Conv3x3BN(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )


def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )


class eSE_Module(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(eSE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x)
        z = self.excitation(y)
        return x * z.expand_as(x)


class OSAv2_module(nn.Module):
    def __init__(self, in_channels,mid_channels, out_channels, block_nums=5):
        super(OSAv2_module, self).__init__()

        self._layers = nn.ModuleList()
        self._layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=mid_channels, stride=1))
        for idx in range(block_nums-1):
            self._layers.append(Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1))


        self.conv1x1 = Conv1x1BNReLU(in_channels+mid_channels*block_nums,out_channels)
        self.ese = eSE_Module(out_channels)
        if in_channels == out_channels:
            self.short_cut = nn.Identity()
        else:
            self.short_cut = Conv1x1BNReLU(in_channels, out_channels)

    def forward(self, x):
        residual = x
        outputs = []
        outputs.append(x)
        for _layer in self._layers:
            x = _layer(x)
            outputs.append(x)
        out = self.ese(self.conv1x1(torch.cat(outputs, dim=1)))
        return out + self.short_cut(residual)


class VoVNet(nn.Module):
    def __init__(self, planes, layers, num_classes=100):
        super(VoVNet, self).__init__()

        self.groups = 1
        self.stage1 = nn.Sequential(
            Conv3x3BNReLU(in_channels=3, out_channels=64, stride=1, groups=self.groups),
            Conv3x3BNReLU(in_channels=64, out_channels=64, stride=1, groups=self.groups),
            Conv3x3BNReLU(in_channels=64, out_channels=128, stride=1, groups=self.groups),
        )

        self.stage2 = self._make_layer(planes[0][0],planes[0][1],planes[0][2],layers[0])

        self.stage3 = self._make_layer(planes[1][0],planes[1][1],planes[1][2],layers[1])

        self.stage4 = self._make_layer(planes[2][0],planes[2][1],planes[2][2],layers[2])

        self.stage5 = self._make_layer(planes[3][0],planes[3][1],planes[3][2],layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[3][2], out_features=num_classes)

    def _make_layer(self, in_channels, mid_channels,out_channels, block_num):
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for idx in range(block_num):
            layers.append(OSAv2_module(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        out = self.linear(x)
        return out


def load_vovnet(cfg: dict) -> VoVNet:
    cfg_model = cfg['model']
    planes = cfg_model['planes']
    layers = cfg_model['layers']
    num_classes = cfg_model['num_classes']
    assert len(planes) == len(layers), 'The length of planes and layers must be equal.'
    
    return VoVNet(planes, layers, num_classes)


if __name__ == '__main__':
    import torchsummary
    cfg = {
        'model': {
            'planes': [[128, 64, 128],
                       [128, 80, 256],
                       [256, 96, 384],
                       [384, 112, 512]],
            'layers': [1, 1, 1, 1],
            'num_classes': 100
        }
    }
    model = load_vovnet(cfg)
    torchsummary.summary(model, (3, 32, 32))
    torch.save(model.state_dict(), 'vovnet.pth')
    pass