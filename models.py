import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from utils import load_sequenced_dataset, CosineAnnealingWarmUpRestarts, load_mesh
from networks import GOTPNet


class GOTPModel(LightningModule):
    def __init__(self,
                 hidden_size, num_layers, depth, ratio=0.5, heads=1, num_seq=1,
                 lr=1e-3, T_0=100, T_mult=1.2, T_up=1, gamma=0.8,
                 batch_size=1, schedule='cosine'):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.depth = depth
        self.num_seq = num_seq
        self.lr = lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.T_up = T_up
        self.gamma = gamma
        self.schedule = schedule
        self.batch_size = batch_size

        self.net = GOTPNet(
            input_size=6, hidden_size=hidden_size, output_size=1,
            num_layers=num_layers, depth=depth, ratio=ratio, heads=heads)

        self.L = nn.SmoothL1Loss(beta=0.2)

        self.fig, self.axs = plt.subplots(2, num_seq, dpi=200)

    def forward(self, graphs):
        zk = [graphs.x[:, :1]]
        for i in range(self.num_seq):
            zk.append(self.net(
                torch.cat([zk[i].detach(), graphs.u, graphs.s], dim=1),
                graphs.edge_index, graphs.batch))
        return torch.cat(zk, dim=1)

    def configure_optimizers(self):
        if self.schedule == 'cosine':
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-9)
            scheduler = CosineAnnealingWarmUpRestarts(
                optimizer, T_0=self.T_0, T_mult=self.T_mult, T_up=self.T_up,
                eta_max=self.lr, gamma=self.gamma)
            return [optimizer], [scheduler]
        elif self.schedule == 'reduce':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = ReduceLROnPlateau(optimizer, factor=self.gamma)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'loss/val'
                }
            }

    def training_step(self, graphs, *args):
        Z = self(graphs)[:, 1:]
        loss = self.L(Z, graphs.x[:, 1:])
        self.log('loss/train', loss.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
        metric = torch.mean(torch.abs(Z[:, -1].detach().cpu() - graphs.x[:, -1].cpu()))
        self.log('metric/train', metric.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
        torch.cuda.empty_cache()
        return {
            'loss': loss,
            'Y': graphs.x[:, 1:].cpu()[graphs.batch == 0],
            'Z': Z.detach().cpu()[graphs.batch == 0],
            'prob': graphs[0].prob,
            'idx': graphs[0].idx
        }

    def training_epoch_end(self, outputs):
        Y = outputs[0]['Y']
        Z = outputs[0]['Z']
        prob = outputs[0]['prob']
        idx = outputs[0]['idx']
        self.visualization(Y, Z, prob, idx)
        self.logger.experiment.add_figure(
            'Training monitor', self.fig, self.current_epoch
        )

    @torch.no_grad()
    def validation_step(self, graphs, *args):
        Z = self(graphs)[:, 1:]
        loss = self.L(Z, graphs.x[:, 1:])
        self.log('loss/val', loss.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
        metric = torch.mean(torch.abs(Z[:, -1].detach().cpu() - graphs.x[:, -1].cpu()))
        self.log('metric/val', metric.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
        torch.cuda.empty_cache()
        return {
            'loss': loss,
            'Y': graphs.x[:, 1:].cpu()[graphs.batch == 0],
            'Z': Z.detach().cpu()[graphs.batch == 0],
            'prob': graphs[0].prob,
            'idx': graphs[0].idx
        }

    def validation_epoch_end(self, outputs):
        Y = outputs[0]['Y']
        Z = outputs[0]['Z']
        prob = outputs[0]['prob']
        idx = outputs[0]['idx']
        self.visualization(Y, Z, prob, idx)
        self.logger.experiment.add_figure(
            'Validation monitor', self.fig, self.current_epoch
        )

    def visualization(self, Y, Z, prob, idx):
        pos, face = load_mesh(prob, idx)
        T = Triangulation(x=pos[0], y=pos[1], triangles=face)
        for ax in self.axs.ravel():
            ax.cla()
        if self.num_seq > 1:
            for i in range(self.num_seq):
                self.axs[0, i].tricontourf(T, Y[:, i], vmin=0, vmax=1)
                self.axs[1, i].tricontourf(T, Z[:, i], vmin=0, vmax=1)
        else:
            self.axs[0].tricontourf(T, Y[:, 0], vmin=0, vmax=1)
            self.axs[1].tricontourf(T, Z[:, 0], vmin=0, vmax=1)
        for ax in self.axs.ravel():
            ax.set_aspect('equal')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        self.fig.tight_layout()


if __name__ == '__main__':
    num_seq_ = 5
    data_list = load_sequenced_dataset(['clever'], num_seq=num_seq_)
    loader = DataLoader(data_list, batch_size=8)
    data = next(iter(loader))

    model = GOTPModel(hidden_size=64, num_layers=2, depth=2, heads=2, num_seq=num_seq_)
    model.cuda(0)
    z = model(data.cuda(0))
    print(z)
    