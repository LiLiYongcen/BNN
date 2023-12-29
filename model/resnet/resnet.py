import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_channels, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        assert len(num_channels) == len(num_blocks), 'length of num_channels must equal to length of num_blocks'
        
        self.in_planes = num_channels[0]
        
        self.conv1  = nn.Conv2d(3, num_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.layers = nn.ModuleList()
        for i in range(len(num_channels)):
            if self.in_planes != num_channels[i]:
                self.layers.append(self._make_layer(block, num_channels[i], num_blocks[i], stride=2))
            else:
                self.layers.append(self._make_layer(block, num_channels[i], num_blocks[i], stride=1))
        self.linear = nn.Linear(num_channels[-1]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    
def load_resnet(cfg: dict) -> nn.Module:
    cfg_model = cfg['model']
    block_type = cfg_model['block_type']
    assert block_type in ['basic', 'bottleneck'], 'block_type must be basic or bottleneck'
    if block_type == 'basic':
        block = BasicBlock
    else:
        block = Bottleneck
        
    num_channels = cfg_model['num_channels']
    num_blocks = cfg_model['num_blocks']
    num_classes = cfg_model['num_classes']
    
    model = ResNet(block, num_channels, num_blocks, num_classes)
    
    return model

    
if __name__ == '__main__':
    cfg = {
        'model': {
            'block_type': 'basic',
            'num_channels': [32, 64, 128, 256],
            'num_blocks': [3, 4, 6, 3],
            'num_classes': 100
        }
    }
    import torchsummary
    model = load_resnet(cfg)
    torchsummary.summary(model, (3, 32, 32))
    pass
    