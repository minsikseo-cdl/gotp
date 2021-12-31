import random
from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor, EarlyStopping, ModelCheckpoint)
from utils import load_dataset
from models import HybridModel


SEED = 890711
LR_INIT = 1e-3
LR_FACTOR = 0.8
BATCH_SIZE = 128
TRAIN_RATIO = 0.7
MAX_EPOCHS = 100
PROBS = ['clever', 'lshape', 'arc']


def main(train_loader=None, val_loader=None, **kwargs):
    nseq = kwargs['num_seq']

    seed_everything(SEED)
    model = HybridModel(**kwargs)

    trainer = Trainer(
        max_epochs=kwargs['max_epochs'],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        gpus=1, auto_select_gpus=True,
        logger=TensorBoardLogger(
            default_hp_metric=False,
            save_dir='/workspace/logs_v1.0',
            name=f'gotp_seq{nseq}_hybrid'),
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(monitor='loss/val', patience=10),
            ModelCheckpoint(monitor='loss/val', save_last=True)
        ]
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num-seq', type=int, default=3)
    arg = parser.parse_args()
    num_seq = arg.num_seq

    print(f'Number of sequence is {num_seq}')

    seed_everything(SEED)
    dset = load_dataset(PROBS, num_seq)
    random.shuffle(dset, random.random)
    num_train = int(len(dset) * TRAIN_RATIO)
    train_dset = dset[:num_train]
    val_dset = dset[num_train:]
    train_loader = DataLoader(
        train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=BATCH_SIZE, num_workers=4)

    main(
        train_loader=train_loader, val_loader=val_loader,
        in_channels=3, hidden_channels=64, out_channels=1,
        local_depth=3, global_depth=9, ratio=0.5,
        batch_size=BATCH_SIZE, num_seq=num_seq, max_epochs=100)
