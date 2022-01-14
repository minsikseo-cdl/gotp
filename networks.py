import torch
from torch import nn
from torch_geometric.nn import Sequential, TopKPooling, GATv2Conv
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_geometric.loader import DataLoader
from torch_sparse.tensor import SparseTensor
from torch_sparse.matmul import matmul
from utils import load_sequenced_dataset


def spspmm(indexA, valueA, indexB, valueB, m, k, n, coalesced=False):
    """Matrix product of two sparse tensors. Both input sparse matrices need to
    be coalesced (use the :obj:`coalesced` attribute to force).

    Args:
        indexA (:class:`LongTensor`): The index tensor of first sparse matrix.
        valueA (:class:`Tensor`): The value tensor of first sparse matrix.
        indexB (:class:`LongTensor`): The index tensor of second sparse matrix.
        valueB (:class:`Tensor`): The value tensor of second sparse matrix.
        m (int): The first dimension of first sparse matrix.
        k (int): The second dimension of first sparse matrix and first
            dimension of second sparse matrix.
        n (int): The second dimension of second sparse matrix.
        coalesced (bool, optional): If set to :obj:`True`, will coalesce both
            input sparse matrices. (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    row, col, value = matmul(
        SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced),
        SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced),
        reduce='sum'
    ).coo()

    return torch.stack([row, col], dim=0), value


def augment_adj(edge_index, edge_weight, num_nodes):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, num_nodes=num_nodes)
    edge_index, edge_weight = sort_edge_index(
        edge_index, edge_weight, num_nodes)
    edge_index, edge_weight = spspmm(
        edge_index, edge_weight, edge_index, edge_weight,
        num_nodes, num_nodes, num_nodes)
    return remove_self_loops(edge_index, edge_weight)


class GCNBlock(nn.Module):
    def __init__(self, input_size, output_size, num_layers, heads, norm):
        super().__init__()
        self.num_layers = num_layers
        self.conv = nn.ModuleList()
        self.act = torch.tanh
        if norm:
            self.bn = nn.ModuleList()
        else:
            self.bn = None
        for i in range(num_layers):
            self.conv.append(GATv2Conv(
                input_size if i == 0 else output_size,
                output_size, heads=heads,
                concat=False, share_weights=True, bias=not norm))
            if norm:
                self.bn.append(nn.BatchNorm1d(output_size))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            if self.bn is not None:
                x = self.bn[i](x)
            x = self.act(x) if i < self.num_layers - 1 else x
        return x


class GOTPNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, depth, ratio=0.5, heads=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth
        self.ratio = ratio

        self.input = GCNBlock(input_size, hidden_size, num_layers, heads, norm=True)

        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.down_convs.append(GCNBlock(hidden_size, hidden_size, num_layers, heads, norm=True))
            self.pools.append(TopKPooling(hidden_size, ratio=ratio))

        self.up_convs = nn.ModuleList()
        for _ in range(depth):
            self.up_convs.append(GCNBlock(2 * hidden_size, hidden_size, num_layers, heads, norm=True))

        self.output = GCNBlock(hidden_size, output_size, num_layers, heads, norm=False)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = torch.tanh(self.input(x, edge_index))
        xs = [x]
        edge_indices = [edge_index]
        perms = []

        for i in range(self.depth):
            with torch.no_grad():
                edge_index, edge_weight = augment_adj(
                    edge_index, edge_weight, x.size(0))
                torch.cuda.empty_cache()
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i](
                x, edge_index, edge_weight, batch)

            x = torch.tanh(self.down_convs[i](x, edge_index))

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            up = torch.zeros_like(xs[j])
            up[perms[j]] = x

            x = torch.tanh(self.up_convs[i](torch.cat((xs[j], up), dim=-1), edge_indices[j]))

        return torch.sigmoid(self.output(x, edge_indices[0]))


if __name__ == '__main__':
    data_list = load_sequenced_dataset(['clever'], num_seq=1)
    loader = DataLoader(data_list)
    data = next(iter(loader))

    model = GOTPNet(input_size=3, hidden_size=16, output_size=1, num_layers=2, depth=3)

    out = torch.cat([data.x[:, :1], data.h], dim=1)
    out = model(out, data.edge_index, data.batch)
    print(out)