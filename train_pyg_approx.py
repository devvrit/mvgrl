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

niter=5
def compute_diffusion_matrix(AD, x, niter, method="ppr"):
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
            current = torch.sparse.mm(AD, current)
            final+= (theta*current)
    return final


# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
# dataset = Reddit(path)
# data = dataset[0]
dataset="ogbn-products"
dataset_name = dataset
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

A = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)).to(device), (data.x.size(0),data.x.size(0)))
D = torch.sparse.mm(A, torch.ones(A.size(0),1).to(device)).view(-1)
a = [[i for i in range(A.size(0))],[i for i in range(A.size(0))]]
D = torch.sparse_coo_tensor(torch.tensor(a).to(device), 1/(D**0.5) , A.size()).to(device) # D^ = Sigma A^_ii
ADinv = torch.sparse.mm(D, torch.sparse.mm(A, D)) # A~ = D^(-1/2) x A^ x D^(-1/2)
AX = torch.sparse.mm(ADinv, data.x)
del A,D,a

print("Calculating SX")
SX = compute_diffusion_matrix(ADinv, data.x, niter)
print("SX calculated")

class Encoder2(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.w1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.bias1 = torch.nn.Parameter(torch.zeros(hidden_channels).to(device), requires_grad=True)
        self.w2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.bias2 = torch.nn.Parameter(torch.zeros(hidden_channels).to(device), requires_grad=True)
        self.prelu1 = nn.PReLU(hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, temp, AD, *args):
        x = self.w1(x)
        x+=self.bias1
        x = self.prelu1(x)
        x = torch.sparse.mm(AD,x)
        x = self.w2(x)
        x+=self.bias2
        x = self.prelu2(x)
        return x

class Encoder3(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.w = nn.Linear(in_channels, hidden_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(hidden_channels).to(device), requires_grad=True)
        self.w2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.bias2 = torch.nn.Parameter(torch.zeros(hidden_channels).to(device), requires_grad=True)
        #torch.nn.init.xavier_uniform_(self.w.weight)
        self.prelu = nn.PReLU(hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, temp, AD, *args):
        x = self.w(x)
        x+=self.bias
        x = self.prelu(x)
        # x = self.con(x, edge_index)
        # x = self.prel(x)
        alpha=0.2
        t=5
        method="ppr"
        for i in range(0, niter):
            if method=="ppr":
                theta = alpha*((1-alpha)**i)
            elif method=="heat":
                theta=(math.e**(-t))*(t**i)/math.factorial(i)
            else:
                raise NotImplementedError
            if i==0:
                final = theta*x
                current = x
            else:
                current = torch.sparse.mm(AD, current)
                final+= (theta*current)
        x = self.w2(final)
        x+=self.bias2
        x = self.prelu2(x)
        return x

def corruption(AX, x, AD, *args):
    alpha=0.2
    t=5
    method="ppr"
    if len(args)==0:
#        print("doing AX_rand")
        return torch.sparse.mm(AD, x[torch.randperm(x.size(0))]), x, AD
    else:
#        print("doing SX_rand")
        x = x[torch.randperm(x.size(0))].clone()
        for i in range(0, niter):
            if method=="ppr":
                theta = alpha*((1-alpha)**i)
            elif method=="heat":
                theta=(math.e**(-t))*(t**i)/math.factorial(i)
            else:
                raise NotImplementedError
            if i==0:
                final = theta*x
                current = x
            else:
                current = torch.sparse.mm(AD, current)
                final+= (theta*current)
        return final, x, AD, args[0]

        # return x[torch.randperm(x.size(0))], edge_index, args[0]

'''
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x

def corruption_old(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index
'''
#model1 = DeepGraphInfomax(
#    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
#    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
#    corruption=corruption_old).to(device)
model1 = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder2(dataset.num_features, 512),
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
    #print("Starting with model1")
    pos_za, neg_za, summarya = model1(AX, data.x, ADinv)
    #pos_za, neg_za, summarya = model1(data.x, data.edge_index)
    lossa = model1.loss(pos_za, neg_za, summarya)
#    print("--------------------")
#    print("Starting with model1")
    pos_zs, neg_zs, summarys = model2(SX, data.x, ADinv, "garbage")
    losss = model2.loss(pos_zs, neg_zs, summarys)
    loss = lossa+losss
    #loss=lossa
    loss.backward()
    optimizer.step()
    return loss.item()


def test(epoch):
    model1.eval()
    model2.eval()
    za, _, _ = model1(AX, data.x, ADinv)
    #za, _, _ = model1(data.x, data.edge_index)
    zs, _, _ = model2(SX, data.x, ADinv, "garbage")
    z = (za+zs)/2
    #z = za
    torch.save(z, "embedding_"+dataset_name+".pt_epoch_"+str(epoch))
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
