import os
from glob import glob
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected


def read_data(prob_list, num_seq=1):
    transform = ToUndirected()
    graphs = []
    for prob in prob_list:
        path_list = glob('dataset/{:s}/*'.format(prob))

        for path in path_list:
            if os.path.isdir(path):

                # Import edge list
                buffer = np.fromfile(os.path.join(path, 'edge.bin'), dtype=np.int64)
                edge = buffer[2:].reshape(buffer[1].item(), buffer[0].item()) - 1

                # Import node coordinates
                buffer = np.fromfile(os.path.join(path, 'node.bin'), dtype=np.float32)
                pos = buffer[2:].reshape(int(buffer[1].item()), int(buffer[0].item())).T

                # Import element list
                buffer = np.fromfile(os.path.join(path, 'elem.bin'), dtype=np.int64)
                face = buffer[2:].reshape(buffer[1].item(), buffer[0].item()) - 1

                # Import input features
                buffer = np.fromfile(os.path.join(path, 'X.bin'), dtype=np.float32)
                X = buffer[2:].reshape(int(buffer[1].item()), int(buffer[0].item())).T

                # Import output features
                buffer = np.fromfile(os.path.join(path, 'Y.bin'), dtype=np.float32)
                Y = buffer[2:].reshape(int(buffer[1].item()), int(buffer[0].item())).T

                if num_seq == 1:
                    Y = Y[:, -1:]
                else:
                    len_seq = Y.shape[-1]
                    idx_seq = np.linspace(0, len_seq-1, num_seq + 1).astype(np.int64)
                    Y = Y[:, idx_seq[1:]]

                # Define Data
                graphs.append(transform(Data(
                    x=torch.tensor(X, dtype=torch.float32),
                    edge_index=torch.tensor(edge, dtype=torch.long),
                    y=torch.tensor(Y, dtype=torch.float32),
                    pos=torch.tensor(pos, dtype=torch.float32),
                    face=torch.tensor(face, dtype=torch.long)
                )))
    return graphs
