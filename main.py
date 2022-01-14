import random
from datetime import timedelta
import torch
from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils import load_sequenced_dataset
from models import GOTPModel


SEED = 19890711
LR_INIT = 5e-3
BATCH_SIZE = 8
TRAIN_RATIO = 0.7
MAX_EPOCHS = 500
PROBS = ['clever', 'lshape', 'arc']


def main(hidden_size, num_layers, depth, heads, num_seq, lr, batch_size, schedule='cosine',
         train_loader=None, val_loader=None, ckpt=None, max_epochs=MAX_EPOCHS):

    model = GOTPModel(
        hidden_size=hidden_size, num_layers=num_layers, depth=depth, heads=heads, num_seq=num_seq,
        lr=lr, schedule=schedule, batch_size=batch_size)
    if ckpt is not None:
        state_dict = torch.load(ckpt)['state_dict']
        model.load_state_dict(state_dict)

    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        gpus=2, auto_select_gpus=True,
        strategy=DDPPlugin(find_unused_parameters=False),
        logger=TensorBoardLogger(
            default_hp_metric=False,
            save_dir='/workspace/logs_v3.0',
            name=f'gotp_n{hidden_size}_l{num_layers}_d{depth}_h{heads}_s{num_seq}'),
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(monitor='loss/val', save_last=True)
        ],
        accumulate_grad_batches=1
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--lr', type=lambda x: float(x), default=LR_INIT)
    parser.add_argument('--num-seq', type=int, default=1)
    parser.add_argument('--schedule', type=str, default='cosine')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    seed_everything(SEED)
    data_list = load_sequenced_dataset(PROBS, args.num_seq, num_workers=8)
    random.shuffle(data_list, random.random)
    num_train = int(len(data_list) * TRAIN_RATIO)
    train_dset = data_list[:num_train]
    val_dset = data_list[num_train:]

    main(
        train_loader=DataLoader(
            train_dset, batch_size=args.batch_size, shuffle=True, num_workers=4),
        val_loader=DataLoader(val_dset, batch_size=args.batch_size, num_workers=4),
        **vars(args)
    )
