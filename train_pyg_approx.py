import os.path as osp
import math

import torch
import torch.nn as nn

from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import GATConv, GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ppr(i, alpha=0.2):
    return alpha*((1-alpha)**i)
def heat(i, t=5):
    return (math.e**(-t))*(t**i)/math.factorial(i)


def compute_diffusion_matrix(graph, x, niter=5, method="ppr"):
    D = torch.sparse.mm(graph, torch.ones(graph.size(0),1).to(device)).view(-1)
    a = [[i for i in range(graph.size(0))],[i for i in range(graph.size(0))]]
    D = torch.sparse_coo_tensor(torch.tensor(a).to(device), 1/(D**0.5) , graph.size()).to(device) # D^ = Sigma A^_ii
    ADinv = torch.sparse.mm(graph, D)
    #ADinv = torch.sparse.mm(D, torch.sparse.mm(graph, D)) # A~ = D^(-1/2) x A^ x D^(-1/2)
    for i in range(0, niter):
        print("Iteration: " + str(i))
        if method=="ppr":
            theta = ppr(i)
        elif method=="heat":
            theta=heat(i)
        else:
            raise NotImplementedError
        if i==0:
            final = theta*x
            current = x
        else:
            current = torch.sparse.mm(ADinv, current)
            final+= (theta*current)
    return final


def compute_ppr(graph, alpha=0.2):
    D = torch.sparse.mm(graph, torch.ones(graph.size(0),1).to(device)).view(-1)
    a = [[i for i in range(graph.size(0))],[i for i in range(graph.size(0))]]
    D = torch.sparse_coo_tensor(torch.tensor(a).to(device), 1/(D**0.5) , graph.size()).to(device) # D^ = Sigma A^_ii
    I = torch.eye(graph.size(0)).to(device)
    # I = torch.sparse_coo_tensor(torch.tensor(a).to(device), torch.ones(graph.size(0)).to(device) , graph.size()).to(device)
    ADinv = torch.sparse.mm(D, torch.sparse.mm(graph, D)) # A~ = D^(-1/2) x A^ x D^(-1/2)
    del D, graph
    S = alpha*torch.inverse(I - (1-alpha)*ADinv.to_dense()) # a(I_n-(1-a)A~)^-1
    indices = S.nonzero()
    ind = torch.cat((indices.T[0].view(1,-1), indices.T[1].view(1,-1)), dim=0)
    vals = S[ind[0],ind[1]]
    del ADinv, I
    return vals, torch.sparse_coo_tensor(ind, vals, S.size())


def compute_heat(graph, t=5):
    D = torch.sparse.mm(graph, torch.ones(graph.size(0),1).to(device)).view(-1)
    a = [[i for i in range(graph.size(0))],[i for i in range(graph.size(0))]]
    D = torch.sparse_coo_tensor(torch.tensor(a).to(device), 1/D , graph.size()).to(device)
    ADinv = torch.sparse.mm(graph, D)
    S_weights = torch.exp(t*ADinv.coalesce().values() - t)
    return S_weights, None
    #S = torch.exp(t*ADinv - 1)
    #return S.coalesce().values()
    #return np.exp(t * (np.matmul(a, inv(d)) - 1))


# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
# dataset = Reddit(path)
# data = dataset[0]
dataset="ogbn-arxiv"
if dataset[:4]=="ogbn":
    dataset = PygNodePropPredDataset(name = dataset)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
else:
    dataset = Planetoid(".", dataset)
    data = dataset[0].to(device)
    train_idx, valid_idx, test_idx = data['train_mask'].nonzero().view(-1), data['val_mask'].nonzero().view(-1), data['test_mask'].nonzero().view(-1)

data = dataset[0].to(device)
data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])
#data.x = data.x/torch.norm(data.x, dim=-1).view(-1,1)
print("Calculating S")
S = compute_diffusion_matrix(torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)).to(device), (data.x.size(0),data.x.size(0))), data.x)
#S_weights, indices = compute_ppr(torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)).to(device), (data.x.size(0),data.x.size(0))))
print("S calculated")


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
        self.conv = GCNConv(in_channels, hidden_channels, cached=False, normalize=False, add_self_loops=False)
        # self.con = GCNConv(hidden_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)
        # self.prel = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index, weights):
        x = self.conv(x, edge_index, weights)
        x = self.prelu(x)
        # x = self.con(x, edge_index)
        # x = self.prel(x)
        return x


class Encoder3(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        #self.con = GCNConv(hidden_channels, hidden_channels, cached=True)
        self.w = nn.Linear(in_channels, hidden_channels, bias=True)
        self.w.bias.data.fill_(0.0)
        self.prelu = nn.PReLU(hidden_channels)
        # self.prel = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.w(x)
        x = self.prelu(x)
        # x = self.con(x, edge_index)
        # x = self.prel(x)
        return x

def corruption(x, edge_index, *args):
    if len(args)==0:
        return x[torch.randperm(x.size(0))], edge_index
    else:
        return x[torch.randperm(x.size(0))], edge_index, args[0]


model1 = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
model2 = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder3(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=0.0001)



def train():
    model1.train()
    model2.train()
    optimizer.zero_grad()
    pos_za, neg_za, summarya = model1(data.x, data.edge_index)
    lossa = model1.loss(pos_za, neg_za, summarya)
    pos_zs, neg_zs, summarys = model2(data.x, None)
    losss = model2.loss(pos_zs, neg_zs, summarys)
    loss = lossa+losss
    loss.backward()
    optimizer.step()
    return loss.item()


def test(epoch):
    model1.eval()
    model2.eval()
    za, _, _ = model1(data.x, data.edge_index)
    zs, _, _ = model2(data.x, None)
    z = (za+zs)/2
    torch.save(z, "embedding.pt_epoch_"+str(epoch))
    #acc = model1.test(z[train_idx], data.y[train_idx],
    #                 z[test_idx], data.y[test_idx], max_iter=150)
    #return acc
    return 0


loss_min = 999999999.0
cnt=0
for epoch in range(1, 20):
    loss = train()
    if loss<loss_min:
        loss_min=loss
        cnt=0
    else:
        cnt+=1
        if cnt>=20:
            print("Early stopping!")
            break
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    test(epoch)
acc = test()
print(f'Accuracy: {acc:.4f}')
