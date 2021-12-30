import torch
from torch import nn
from torch_geometric import nn as gnn


class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, improved=False):
        super().__init__()
        self.conv = gnn.GCNConv(in_channels, out_channels,
                                bias=False, improved=improved)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = self.bn(x)
        return self.act(x)


class LocalModel(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, out_channels=1, depth=9):
        super().__init__()

        self.depth = depth

        self.conv1 = nn.ModuleList()
        self.conv1.append(GCNBlock(in_channels, hidden_channels))
        for _ in range(depth):
            self.conv1.append(GCNBlock(hidden_channels, hidden_channels))

        self.conv2 = nn.ModuleList()
        for _ in range(depth - 1):
            self.conv2.append(GCNBlock(hidden_channels, hidden_channels))
        self.conv2.append(gnn.Sequential(
            'x, edge_index',
            [(gnn.GCNConv(hidden_channels, out_channels, bias=False), 'x, edge_index -> x'),
             nn.Sigmoid()]))

    def forward(self, x, edge_index, batch=None):
        x = self.conv1[0](x, edge_index)

        xs = [x]

        for i in range(1, self.depth + 1):
            x = self.conv1[i](x, edge_index)
            if i < self.depth:
                xs += [x]

        for i in range(self.depth):
            j = self.depth - i - 1
            res = xs[j]
            x = self.conv2[i](x + res, edge_index)

        return x


class GlobalModel(nn.Module):
    ''' GraphUNet w/o augmentation
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, depth, pool_ratios=0.5):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth

        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(
            GCNBlock(in_channels, hidden_channels, improved=True))
        for _ in range(depth):
            self.pools.append(gnn.TopKPooling(
                hidden_channels, pool_ratios,
                nonlinearity=getattr(nn, 'ReLU')()))
            self.down_convs.append(
                GCNBlock(hidden_channels, hidden_channels, improved=True))

        self.up_convs = nn.ModuleList()
        for _ in range(depth - 1):
            self.up_convs.append(
                GCNBlock(hidden_channels, hidden_channels, improved=True))
        self.up_convs.append(gnn.Sequential(
            'x, edge_index, edge_weight',
            [(gnn.GCNConv(hidden_channels, out_channels, bias=False),
                'x, edge_index, edge_weight -> x'),
             nn.Sigmoid()]))

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up

            x = self.up_convs[i](x, edge_index, edge_weight)

        return x
