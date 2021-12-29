import os
import math
from glob import glob
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected


def load_dataset(prob_list, num_seq=1):
    transform = ToUndirected()
    graphs = []
    for prob in prob_list:
        path_list = list(filter(os.path.isdir, glob(f'dataset/{prob:s}/*')))

        subgraphs = []
        for path in tqdm(path_list):
            # Import edge list
            buffer = np.fromfile(os.path.join(
                path, 'edge.bin'), dtype=np.int64)
            edge = torch.tensor(buffer[2:].reshape(
                buffer[1].item(), buffer[0].item()) - 1, dtype=torch.long)

            # Import node coordinates
            buffer = np.fromfile(os.path.join(
                path, 'node.bin'), dtype=np.float32)
            pos = torch.tensor(buffer[2:].reshape(
                int(buffer[1].item()), int(buffer[0].item())).T, dtype=torch.float32)

            # Import element list
            buffer = np.fromfile(os.path.join(
                path, 'elem.bin'), dtype=np.int64)
            face = torch.tensor(buffer[2:].reshape(
                buffer[1].item(), buffer[0].item()) - 1, dtype=torch.long)

            # Import input features
            buffer = np.fromfile(os.path.join(
                path, 'X.bin'), dtype=np.float32)
            X = torch.tensor(buffer[2:].reshape(
                int(buffer[1].item()), int(buffer[0].item())).T, dtype=torch.float32)
            x0 = X[:, :1]
            umag = torch.sqrt(X[:, 1:2]**2 + X[:, 2:3]**2)
            scale = 1 / umag.max().item()
            umag *= scale
            srWs = torch.sqrt(scale * X[:, -1:])

            # Import output features
            buffer = np.fromfile(os.path.join(
                path, 'Y.bin'), dtype=np.float32)
            Y = buffer[2:].reshape(
                int(buffer[1].item()), int(buffer[0].item())).T

            if num_seq == 1:
                Y = torch.tensor(Y[:, -1:], dtype=torch.float32)
                subgraphs.append(transform(Data(
                    x=torch.cat([x0, umag, srWs], dim=-1), y=Y,
                    edge_index=edge, pos=pos, face=face)))

            else:
                dYcum = np.r_[0, np.sqrt(np.sum((Y[:, 1:] - Y[:, :-1])**2, axis=0)).cumsum()]
                dYcum = (dYcum - dYcum.min()) / (dYcum.max() - dYcum.min())
                f = interp1d(dYcum, np.arange(len(dYcum)))
                slice_idx = np.round(
                    f(np.linspace(0, 1, num_seq + 1))).astype(np.int64)
                Y = torch.tensor(Y[:, slice_idx], dtype=torch.float32)
                for i in range(num_seq):
                    subgraphs.append(transform(Data(
                        x=torch.cat([Y[:, i:i+1], umag, srWs], dim=-1), y=Y[:, i+1:i+2],
                        edge_index=edge, pos=pos, face=face)))

        # Define Data
        graphs.extend(subgraphs)
    return graphs


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(
                f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, float):
            raise ValueError(
                f"Expected integer T_mult >= 1, but got {T_mult}")
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError(
                f"Expected positive integer T_up, but got {T_up}")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(
            optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = int((self.T_i - self.T_up) *
                               self.T_mult) + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * \
                        int(self.T_mult ** n - 1) / int(self.T_mult - 1)
                    self.T_i = int(self.T_0 * self.T_mult) ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
