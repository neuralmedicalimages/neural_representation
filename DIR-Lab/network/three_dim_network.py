import math

import torch
from torch import nn


class ThreeDimNetwork(nn.Module):
    def __init__(self, network_conf, input_ch, output_ch, initial=False):

        super(ThreeDimNetwork, self).__init__()
        net_depth = network_conf['net_depth']
        net_width = network_conf['net_width']

        self.skip = network_conf['skip']
        if self.skip:
            self.skip_layers = network_conf['skip_layers']
        else:
            self.skip_layers = None

        module_list = [nn.Sequential(nn.Linear(input_ch, net_width),
                                     nn.LeakyReLU(inplace=True))]

        for i in range(net_depth - 1):
            if self.skip and i in self.skip_layers:
                module_list.append(nn.Sequential(nn.Linear(net_width + input_ch, net_width),
                                                 nn.LeakyReLU(inplace=True)))
            else:
                module_list.append(nn.Sequential(nn.Linear(net_width, net_width),
                                                 nn.LeakyReLU(inplace=True)))

        self.pts_linears = nn.ModuleList(module_list)
        self.output_linear = nn.Linear(net_width, output_ch)

        if initial:
            self.initialize_weights()

    def forward(self, input_pts):
        x = input_pts
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            if self.skip and i in self.skip_layers:
                x = torch.cat([input_pts, x], -1)

        outputs = self.output_linear(x)

        return outputs

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1)) / 100
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
