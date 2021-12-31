import random
from argparse import ArgumentParser
import torch
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer, seed_everything, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor, EarlyStopping, ModelCheckpoint)
from utils import load_sequenced_dataset, CosineAnnealingWarmUpRestarts
from models import LocalModel, plt, Triangulation, ReduceLROnPlateau, MultiLocalModel


SEED = 890711
LR_INIT = 1e-3
LR_FACTOR = 0.8
BATCH_SIZE = 8
TRAIN_RATIO = 0.7
MAX_EPOCHS = 1000
PROBS = ['clever', 'lshape', 'arc']


class LocalModel_(LightningModule):
    def __init__(self, num_seq,
                 in_channels=3, hidden_channels=128, out_channels=1, depth=9,
                 batch_size=1, lr=1e-3, loss='L1Loss', **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.num_seq = num_seq
        self.lr = lr
        self.batch_size = batch_size

        # Define models
        self.model = LocalModel(in_channels, hidden_channels, out_channels, depth,
                           norm=False, act='Sigmoid', param={})

        self.L = getattr(torch.nn, loss)()

        self.fig, self.axs = plt.subplots(2, num_seq, figsize=(3*num_seq, 6), dpi=200)

    def forward(self, x, edge_index, **kwargs):
        p = x[:, 1:]
        x = x[:, :1]
        xs = []
        for _ in range(self.num_seq):
            x = self.model(torch.cat([x.detach(), p], dim=1), edge_index)
            xs += [x]
        return torch.cat(xs, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.8)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, 200, 1.0, self.lr, 40, 0.5)
        return [optimizer], [scheduler]
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'loss/val'
        #     }
        # }

    def training_step(self, graphs, *args):
        z = self(graphs.x, graphs.edge_index)
        loss = self.L(z, graphs.y)
        self.log(
            'loss/train', loss.item(),
            on_step=False, on_epoch=True,
            batch_size=self.batch_size)
        return {
            'loss': loss,
            'graphs': graphs,
            'z': z.detach()
        }

    def training_epoch_end(self, outputs):
        graphs = outputs[0]['graphs'].cpu()
        batch = graphs.batch
        g = graphs[0]
        z = outputs[0]['z'].cpu()[batch == 0]

        self.plot_function(g, z)

        # Log figure
        self.logger.experiment.add_figure(
            'Training result', self.fig, self.current_epoch)

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
        graphs = outputs[0]['graphs'].cpu()
        batch = graphs.batch
        g = graphs[0]
        z = outputs[0]['z'].cpu()[batch == 0]

        self.plot_function(g, z)

        # Log figure
        self.logger.experiment.add_figure(
            'Validation result', self.fig, self.current_epoch)

    def plot_function(self, g, z):
        levels = torch.linspace(-1e-3, 1 + 1e-3, 100).numpy()

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


def main(train_loader=None, val_loader=None, **kwargs):
    nseq = kwargs['num_seq']

    seed_everything(SEED)
    model = LocalModel_(**kwargs)

    trainer = Trainer(
        max_epochs=kwargs['max_epochs'],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        gpus=1, auto_select_gpus=True,
        logger=TensorBoardLogger(
            default_hp_metric=False,
            save_dir='/workspace/logs_v1.1',
            name=f'gotp_seq{nseq}_local'),
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            # EarlyStopping(monitor='loss/val', patience=10),
            ModelCheckpoint(monitor='loss/val', save_last=True)
        ]
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num-seq', type=int, default=5)
    arg = parser.parse_args()
    num_seq = arg.num_seq

    print(f'Number of sequence is {num_seq}')

    seed_everything(SEED)
    dset = load_sequenced_dataset(PROBS, num_seq)
    random.shuffle(dset, random.random)
    num_train = int(len(dset) * TRAIN_RATIO)
    train_dset = dset[:num_train]
    val_dset = dset[num_train:]

    main(
        train_loader=DataLoader(
            train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        val_loader=DataLoader(
            val_dset, batch_size=BATCH_SIZE, num_workers=4),
        in_channels=3,
        hidden_channels=128,
        out_channels=1,
        depth=9,
        batch_size=BATCH_SIZE,
        num_seq=num_seq,
        max_epochs=100)
