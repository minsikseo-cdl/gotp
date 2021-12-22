import random
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from torch import nn
from torch import sigmoid
from torch.nn.functional import leaky_relu
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric import nn as gnn
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from utils import read_data


SEED = 211222
LR_INIT = 1.738e-02
BATCH_SIZE = 64
TRAIN_RATIO = 0.7
MAX_EPOCHS = 100
PROBS = ['clever', 'lshape', 'arcbeam']


class GOTPred(LightningModule):
    def __init__(self, in_dims, h_dims, out_dims, depth,
                 batch_size=BATCH_SIZE, lr=LR_INIT,
                 train_ratio=TRAIN_RATIO):
        super().__init__()

        # Parameters
        self.batch_size = batch_size
        self.lr = lr

        # Load dataset
        dset = read_data(PROBS)
        print('Total number of graph is {:d}.'.format(len(dset)))
        random.shuffle(dset)
        num_data = len(dset)
        num_train = int(num_data*train_ratio)
        print('Training dataset:\t{:d}'.format(num_train))
        print('Validation dataset:\t{:d}'.format(len(dset) - num_train))
        self.train_dset = dset[:num_train]
        self.val_dset = dset[num_train:]

        # Define model
        self.model = gnn.GraphUNet(
            in_dims, h_dims, out_dims, depth=depth, act=leaky_relu)

        # Define loss function
        self.loss_fn = nn.BCELoss()

        # Figure for logging
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.tight_layout()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'loss/val'
            }
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dset, batch_size=self.batch_size,
            shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(
            self.val_dset, batch_size=self.batch_size, num_workers=4)

    def training_step(self, batch, batch_idx):
        z = sigmoid(self.model(batch.x, batch.edge_index, batch.batch))
        loss = self.loss_fn(z, batch.y)
        self.log('loss/train', loss.item(), batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        z = sigmoid(self.model(batch.x, batch.edge_index, batch.batch))
        loss = self.loss_fn(z, batch.y)
        self.log('loss/val', loss.item(), batch_size=self.batch_size)
        graph = batch.to_data_list()[0]
        return {
            'loss': loss,
            'pos': graph.pos,
            'face': graph.face,
            'y': graph.y,
            'z': z[batch.batch==0].detach()
        }

    def validation_epoch_end(self, outputs):
        # Get fisrt validation output
        pos = outputs[0]['pos'].cpu().numpy()
        face = outputs[0]['face'].cpu().numpy().T
        y = outputs[0]['y'].cpu().numpy()[:, 0]
        z = outputs[0]['z'].cpu().numpy()[:, 0]

        # Prepare tricontourf
        tria = mtri.Triangulation(
            x=pos[:, 0], y=pos[:, 1], triangles=face)
        for ax in self.axs:
            ax.cla()

        # Plot ground truth contour
        self.axs[0].tricontourf(
            tria, y, levels=np.linspace(-1e-3, 1+1e-3),
            vmin=0, vmax=1, cmap='gray_r')
        self.axs[0].set_title('Ground truth (epoch={:d})'.format(self.current_epoch))

        # Plot predicted contour
        nrmse = np.linalg.norm(y - z)/np.linalg.norm(y)
        self.axs[1].tricontourf(
            tria, z, levels=np.linspace(-1e-3, 1+1e-3),
            vmin=0, vmax=1, cmap='gray_r')
        self.axs[1].set_title('Predicted (NRMSE={:.3e})'.format(nrmse))

        for ax in self.axs:
            ax.axis('image')
            ax.axis('off')

        # Log figure
        self.logger.experiment.add_figure('Visual validation', self.fig, self.current_epoch)


def hpstudy(h_dims, depth, max_epochs=MAX_EPOCHS):
    
    seed_everything(SEED)
    model = GOTPred(
        in_dims=10, h_dims=h_dims, out_dims=1, depth=depth)

    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=10,
        gpus=1, auto_select_gpus=True,
        logger=TensorBoardLogger(
            default_hp_metric=False,
            save_dir='/workspace/logs',
            name='gotp_single',
            version='hdim={h_dims}-depth={depth}'.format(
                h_dims=h_dims, depth=depth
            )),
        callbacks=[
            LearningRateMonitor(logging_interval='epoch')
        ]
    )

    trainer.fit(model)


if __name__ == '__main__':

    h_dims_span = [32, 64, 128, 256]
    depth_span = [3, 4, 5, 6]
    h_dims_list, depth_list = np.meshgrid(h_dims_span, depth_span)
    for h_dims_i, depth_i in zip(h_dims_list.ravel(), depth_list.ravel()):
        hpstudy(h_dims_i, depth_i)
