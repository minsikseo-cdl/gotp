import random
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from utils import load_dataset, CosineAnnealingWarmUpRestarts
from models import LocalModel, GlobalModel


SEED = 890711
LR_INIT = 1e-3
LR_FACTOR = 0.8
BATCH_SIZE = 128
TRAIN_RATIO = 0.7
MAX_EPOCHS = 100
PROBS = ['clever', 'lshape', 'arc']


class GOTP(LightningModule):
    def __init__(self,
                 in_channels: int = 3,
                 hidden_channels: int = 128,
                 out_channels: int = 1,
                 depth: int = 9,
                 pool_ratios: float = 0.5,
                 methods: str = 'local',
                 lr: float = LR_INIT):
        super().__init__()
        self.save_hyperparameters()

        # Parameters
        self.methods = methods
        self.lr = lr

        # Define model
        if methods == 'local':
            self.model = LocalModel(
                in_channels, hidden_channels, out_channels, depth)
            self.param_list = list(self.model.parameters())
        elif methods == 'global':
            self.model = GlobalModel(
                in_channels, hidden_channels, out_channels, depth, pool_ratios)
            self.param_list = list(self.model.parameters())
        elif methods == 'hybrid':
            self.local_model = LocalModel(
                in_channels, hidden_channels, out_channels, depth)
            self.global_model = GlobalModel(
                in_channels, hidden_channels, out_channels, depth, pool_ratios)
            self.param_list = list(self.local_model.parameters()) +\
                list(self.global_model.parameters())
        else:
            NotImplementedError()

        # Define loss function
        self.loss_fn = nn.BCELoss()

        # Figure for logging
        self.fig, self.axs = plt.subplots(1, 2, dpi=200)

    def configure_optimizers(self):
        optimizer = Adam(self.param_list, lr=1e-9)
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=100, T_mult=1.2, eta_max=self.lr, T_up=5, gamma=LR_FACTOR)
        return [optimizer], [scheduler]

    def training_step(self, batch, *args):
        if self.methods == 'hybrid':
            z_local = self.local_model(batch.x, batch.edge_index, batch.batch)
            z_global = self.global_model(batch.x, batch.edge_index, batch.batch)
            z = 0.5 * (z_local + z_global)
        else:
            z = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.loss_fn(z, batch.y)
        self.log('loss/train', loss.item(), batch_size=BATCH_SIZE)
        return loss

    def validation_step(self, batch, *args):
        if self.methods == 'hybrid':
            z_local = self.local_model(batch.x, batch.edge_index, batch.batch)
            z_global = self.global_model(batch.x, batch.edge_index, batch.batch)
            z = 0.5 * (z_local + z_global)
        else:
            z = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.loss_fn(z, batch.y)
        self.log('loss/val', loss.item(), batch_size=BATCH_SIZE)
        graph = batch.to_data_list()[0]
        return {
            'loss': loss,
            'pos': graph.pos,
            'face': graph.face,
            'y': graph.y,
            'z': z[batch.batch == 0].detach()
        }

    def validation_epoch_end(self, outputs):
        # Get fisrt validation output
        pos = outputs[0]['pos'].cpu().numpy()
        face = outputs[0]['face'].cpu().numpy().T
        y = outputs[0]['y'].cpu().numpy()[:, 0]
        z = outputs[0]['z'].cpu().numpy()[:, 0]

        # Prepare tricontourf
        tria = Triangulation(
            x=pos[:, 0], y=pos[:, 1], triangles=face)
        for ax in self.axs:
            ax.cla()

        # Plot ground truth contour
        levels = np.linspace(-1e-3, 1 + 1e-3)
        self.axs[0].tricontourf(
            tria, y, levels=levels,
            vmin=0, vmax=1, cmap='gray_r')
        self.axs[0].set_title(f'Reference (epoch={self.current_epoch:d})')

        # Plot predicted contour
        nrmse = np.linalg.norm(y - z)/np.linalg.norm(y)
        self.axs[1].tricontourf(
            tria, z, levels=levels,
            vmin=0, vmax=1, cmap='gray_r')
        self.axs[1].set_title(f'Predicted (NRMSE={nrmse:.3e})')

        for ax in self.axs:
            ax.set_aspect('equal')
            ax.axis('off')
        self.fig.tight_layout()

        # Log figure
        self.logger.experiment.add_figure(
            'Validation', self.fig, self.current_epoch)


def main(in_channels, hidden_channels, out_channels, depth, methods, pool_ratios, num_seq):

    seed_everything(SEED)
    model = GOTP(in_channels, hidden_channels,
                 out_channels, depth, pool_ratios, methods=methods)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        # max_steps=20e3,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        gpus=1, auto_select_gpus=True,
        logger=TensorBoardLogger(
            default_hp_metric=False,
            save_dir='/workspace/logs',
            name=f'gotp_seq{num_seq}_{methods}'),
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(monitor='loss/val', patience=10)
        ]
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num-seq', type=int, default=1)
    arg = parser.parse_args()
    num_seq = arg.num_seq

    print(f'Number of sequence is {num_seq}')

    seed_everything(SEED)
    dset = load_dataset(PROBS, num_seq)
    random.shuffle(dset)
    num_train = int(len(dset) * TRAIN_RATIO)
    train_dset = dset[:num_train]
    val_dset = dset[num_train:]
    train_loader = DataLoader(
        train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=BATCH_SIZE, num_workers=4)

    main(in_channels=3, hidden_channels= 128,
        out_channels=1, depth=9, methods='global', pool_ratios=0.5,
        num_seq=num_seq)
