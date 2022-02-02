import os.path as osp

import torch
import torch.nn as nn

from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from ogb.nodeproppred import PygNodePropPredDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_heat(graph, t=5):
	D = torch.sparse.mm(graph, torch.ones(graph.size(0),1))
	a = [[i for i in range(graph.size(0))],[i for i in range(graph.size(0))]]
	D = torch.sparse_coo_tensor(torch.tensor(a), 1/D , graph.size())
	ADinv = torch.sparse.mm(graph, D)
    S = torch.exp(t*ADinv - 1)
    return S.coalesce().values()
    # return np.exp(t * (np.matmul(a, inv(d)) - 1))


# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
# dataset = Reddit(path)
# data = dataset[0]
if dataset[:5]=="ogbn":
	dataset = PygNodePropPredDataset(name = "ogbn-arxiv")
	split_idx = dataset.get_idx_split()
	train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
else:
	dataset = Planetoid(".", dataset)
	train_idx, valid_idx, test_idx = data['train_mask'].nonzero().view(-1), data['val_mask'].nonzero().view(-1), data['test_mask'].nonzero().view(-1)

data = dataset[0].to(device)
data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])
S_weights = compute_heat(torch.sparse_coo_tensor(data.edge_index, torch.ones(data.x.size(0)), (data.x.size(0),data.x.size(0))))


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        # self.con = GCNConv(hidden_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)
        # self.prel = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        # x = self.con(x, edge_index)
        # x = self.prel(x)
        return x

class Encoder2(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, normalize=False, add_self_loops=False)
        # self.con = GCNConv(hidden_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)
        # self.prel = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        # x = self.con(x, edge_index)
        # x = self.prel(x)
        return x

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


model1 = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
model2 = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder2(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=0.001)



def train():
    model1.train()
    model2.train()
    optimizer.zero_grad()
    pos_za, neg_za, summarya = model1(data.x, data.edge_index)
    lossa = model1.loss(pos_za, neg_za, summarya)
    pos_zs, neg_zs, summarys = model2(data.x, data.edge_index, S_weights)
    losss = model2.loss(pos_zs, neg_zs, summarys)
    loss = lossa+losss
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    z, _, _ = model(data.x, data.edge_index)
    torch.save(z, "embedding.pt")
    acc = model.test(z[train_idx], data.y[train_idx],
                     z[test_idx], data.y[test_idx], max_iter=150)
    return acc


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
acc = test()
print(f'Accuracy: {acc:.4f}')
