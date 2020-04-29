import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict

# Weight initialization function
def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

# Basic CNN of Ragoza et al. 2017
class Basic_CNN(nn.Module):
  def __init__(self, dims, num_classes=2):
    super(Basic_CNN, self).__init__()
    self.pool0 = nn.MaxPool3d(2)
    self.conv1 = nn.Conv3d(dims[0], 32, kernel_size=3, padding=1)
    self.pool1 = nn.MaxPool3d(2)
    self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
    self.pool2 = nn.MaxPool3d(2)
    self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)

    self.last_layer_size = dims[1]//8 * dims[2]//8 * dims[3]//8 * 128
    self.fc1 = nn.Linear(self.last_layer_size, num_classes)

  def forward(self, x):
    x = self.pool0(x)
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = F.relu(self.conv3(x))
    x = x.view(-1, self.last_layer_size)
    x = self.fc1(x)
    return x

# CNN of Imrie et al. 2019
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.MaxPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """DenseNet model class
    Args:
        dims - dimensions of the input image (channels, x_dim, y_dim, z_dim)
        growth_rate (int) - how many filters to add each layer (k in DenseNet paper)
        block_config (list of 3 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        last_fc (bool) - include classifier layer
    """
    def __init__(self, dims, growth_rate=16, block_config=(4, 4, 4),
                 num_init_features=32, drop_rate=0, num_classes=2, last_fc=True):

        super(DenseNet, self).__init__()

        self.last_fc = last_fc

        self.dims = dims

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('pool0', nn.MaxPool3d(kernel_size=2, stride=2)),                                   
            ('conv0', nn.Conv3d(dims[0], num_init_features, kernel_size=3,
                                stride=1, padding=1, bias=False)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features)
                self.features.add_module('transition%d' % (i + 1), trans)

        # Final batch norm + relu
        self.features.add_module('norm4', nn.BatchNorm3d(num_features))
        self.features.add_module('relu4', nn.ReLU(inplace=True))

        # Global max pool
        last_size = self.dims[1] // 8
        self.features.add_module('globalmaxpool', nn.MaxPool3d(kernel_size=(last_size, last_size, last_size), stride=1))
        self.final_num_features = num_features

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Model
        features = self.features(x)
        out = features.view(-1, self.final_num_features) 
        # Classifer
        if self.last_fc:
            out = self.classifier(out)
        return out
