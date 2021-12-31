import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric import nn as gnn
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from utils import CosineAnnealingWarmUpRestarts


class GCNBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, norm=True,
                 act='LeakyReLU', param={'negative_slope': 0.2, 'inplace': True},
                 improved=False):
        super().__init__()
        self.norm = norm
        self.conv = gnn.GATv2Conv(
            in_channels, out_channels, bias=not norm, heads=2, concat=False)
        # self.conv = gnn.GCNConv(in_channels, out_channels,
        #                         bias=not norm, improved=improved)
        if norm:
            self.bn = nn.BatchNorm1d(out_channels)
        self.act = getattr(nn, act)(**param)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        if self.norm:
            x = self.bn(x)
        return self.act(x)


class LocalModel(nn.Module):
    def __init__(self,
                 in_channels=3, hidden_channels=128, out_channels=1, depth=9,
                 norm=True, act='LeakyReLU', param={'negative_slope': 0.2, 'inplace': True}):
        super().__init__()

        self.depth = depth

        self.conv1 = nn.ModuleList()
        self.conv1.append(GCNBlock(in_channels, hidden_channels))
        for _ in range(depth):
            self.conv1.append(GCNBlock(hidden_channels, hidden_channels))

        self.conv2 = nn.ModuleList()
        for _ in range(depth - 1):
            self.conv2.append(GCNBlock(hidden_channels, hidden_channels))
        self.conv2.append(GCNBlock(
            hidden_channels, out_channels,
            norm=norm, act=act, param=param))

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


class MultiLocalModel(LightningModule):
    def __init__(self, num_seq,
                 in_channels=3, hidden_channels=128, out_channels=1, depth=9,
                 batch_size=1, lr=1e-2, loss='L1Loss', **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.num_seq = num_seq
        self.lr = lr
        self.batch_size = batch_size

        # Define models
        self.model = nn.ModuleList()
        for _ in range(num_seq):
            self.model.append(
                LocalModel(in_channels, hidden_channels, out_channels, depth,
                           norm=False, act='Sigmoid', param={}))

        self.L = getattr(nn, loss)()

        self.fig, self.axs = plt.subplots(2, num_seq, figsize=(3*num_seq, 6), dpi=200)

    def forward(self, x, edge_index, **kwargs):
        p = x[:, 1:]
        x = x[:, :1]
        xs = []
        for model in self.model:
            x = model(torch.cat([x, p], dim=1), edge_index)
            xs += [x]
        return torch.cat(xs, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.8)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'loss/val'
            }
        }

    def training_step(self, graphs, *args):
        loss = self.L(self(graphs.x, graphs.edge_index), graphs.y)
        self.log(
            'loss/train', loss.item(),
            on_step=False, on_epoch=True,
            batch_size=self.batch_size)
        return loss

    def validation_step(self, graphs, *args):
        z = self(graphs.x, graphs.edge_index)
        loss = self.L(z, graphs.y)
        self.log(
            'loss/val', loss.item(),
            on_step=False, on_epoch=True,
            batch_size=self.batch_size)
        return {
            'loss': loss,
            'graphs': graphs,
            'z': z.detach()
        }

    def validation_epoch_end(self, outputs):
        levels = torch.linspace(-1e-3, 1 + 1e-3, 100).numpy()
        graphs = outputs[0]['graphs'].cpu()
        batch = graphs.batch
        z = outputs[0]['z'].cpu()[batch == 0]
        g = graphs[0]

        # Clear axes
        for ax in self.axs.ravel():
            ax.cla()

        for i in range(self.num_seq):
            yi = g.y[:, i]
            zi = z[:, i]
            T = Triangulation(
                x=g.pos[:, 0], y=g.pos[:, 1], triangles=g.face.T)
            self.axs[0, i].tricontourf(
                T, yi, levels=levels, cmap='gray_r')
            self.axs[1, i].tricontourf(
                T, zi, levels=levels, cmap='gray_r')
            nrmse = torch.norm(yi - zi) / torch.norm(yi)
            self.axs[1, i].set_title(f'NRMSE={nrmse:.3e}')

        for ax in self.axs.ravel():
            ax.set_aspect('equal')
            ax.axis('off')

        self.fig.tight_layout()

        # Log figure
        self.logger.experiment.add_figure(
            'Validation', self.fig, self.current_epoch)


class HybridModel(LightningModule):
    def __init__(self,
                 in_channels, hidden_channels, out_channels,
                 local_depth, global_depth, ratio=0.5,
                 batch_size=1, lr=1e-3, loss='BCELoss', **kwargs):
        super().__init__()
        self.save_hyperparameters()

        assert global_depth >= 1
        assert local_depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.local_depth = local_depth
        self.global_depth = global_depth
        self.batch_size = batch_size
        self.lr = lr

        # Down convolution and pooling
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(LocalModel(
            in_channels, hidden_channels, hidden_channels, local_depth))
        for _ in range(global_depth):
            self.pools.append(gnn.TopKPooling(
                hidden_channels, ratio,
                nonlinearity=getattr(nn, 'ReLU')()))
            self.down_convs.append(LocalModel(
                hidden_channels, hidden_channels, hidden_channels, local_depth))

        # Up convolution
        self.up_convs = nn.ModuleList()
        for _ in range(global_depth - 1):
            self.up_convs.append(LocalModel(
                hidden_channels, hidden_channels, hidden_channels, local_depth))
        self.up_convs.append(LocalModel(
            hidden_channels, hidden_channels, out_channels, local_depth,
            norm=False, act='Sigmoid', param={}))

        # Loss function
        self.L = getattr(nn, loss)()

        # Figure for logging
        self.fig, self.axs = plt.subplots(2, 4, dpi=200)

    def forward(self, x, edge_index, batch):

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.global_depth + 1):
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)

            if i < self.global_depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.global_depth):
            j = self.global_depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up

            x = self.up_convs[i](x, edge_index, edge_weight)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-9)
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=100, T_mult=1.2, eta_max=self.lr, T_up=5, gamma=0.8)
        return [optimizer], [scheduler]

    def training_step(self, graphs, *args):
        y = graphs.y
        num_seq = y.size(1)
        x = graphs.x[:, :1]
        p = graphs.x[:, 1:]
        edge_index = graphs.edge_index
        batch = graphs.batch
        xs = []
        for _ in range(num_seq):
            x = self(torch.cat([x, p], dim=1), edge_index, batch)
            xs += [x]
        z = torch.cat(xs, dim=1)
        loss = self.L(z, y)
        self.log(
            'loss/train', loss.item(),
            on_step=False, on_epoch=True,
            batch_size=self.batch_size)
        return loss

    def validation_step(self, graphs, *args):
        y = graphs.y
        num_seq = y.size(1)
        x = graphs.x[:, :1]
        p = graphs.x[:, 1:]
        edge_index = graphs.edge_index
        batch = graphs.batch
        xs = []
        for _ in range(num_seq):
            x = self(torch.cat([x, p], dim=1), edge_index, batch)
            xs += [x]
        z = torch.cat(xs, dim=1)
        loss = self.L(z, y)
        self.log(
            'loss/val', loss.item(),
            on_step=False, on_epoch=True,
            batch_size=self.batch_size)
        return {
            'loss': loss,
            'graphs': graphs,
            'z': z[:, -1].detach()
        }

    def validation_epoch_end(self, outputs):
        levels = torch.linspace(-1e-3, 1 + 1e-3, 100).numpy()
        graphs = outputs[0]['graphs'].cpu()
        batch = graphs.batch
        z = outputs[0]['z'].cpu()

        # Clear axes
        for ax in self.axs.ravel():
            ax.cla()

        for i in range(4):
            g = graphs[i]
            yi = g.y[:, -1]
            zi = z[batch == i]
            T = Triangulation(
                x=g.pos[:, 0], y=g.pos[:, 1], triangles=g.face.T)
            self.axs[0, i].tricontourf(
                T, yi, levels=levels, cmap='gray_r')
            self.axs[1, i].tricontourf(
                T, zi, levels=levels, cmap='gray_r')
            nrmse = torch.norm(yi - zi) / torch.norm(yi)
            self.axs[1, i].set_title(f'NRMSE={nrmse:.3e}')

        for ax in self.axs.ravel():
            ax.set_aspect('equal')
            ax.axis('off')

        self.fig.tight_layout()

        # Log figure
        self.logger.experiment.add_figure(
            'Validation', self.fig, self.current_epoch)
