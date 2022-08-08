import torch
import torch.nn as nn
import signal
def handler(signum, frame):
    print("Trying import again")
    from ogb.nodeproppred import PygNodePropPredDataset
signal.signal(signal.SIGALRM, handler)
signal.alarm(10)
from tqdm import tqdm
import numpy as np
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_geometric.nn import GCNConv
from torch.nn.functional import normalize

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Kmeans(x, K, Niter=10):
    N, D = x.shape
    temp = set()
    while len(temp)<K:
        temp.add(np.random.randint(0, N))
    c = x[list(temp), :].clone()  # Random initialization for the centroids
    cutoff = 1
    x_i = x.view(N, 1, D) # (N, 1, D) samples
    if K>cutoff:
        c_j = []
        niter=K//cutoff
        rem = K%cutoff
        if rem>0:
            rem=1
        for i in range(niter+rem):
            c_j.append(c[i*cutoff:min(K,(i+1)*cutoff),:].view(1, min(K,(i+1)*cutoff)-(i*cutoff), D))
    else:
        c_j = c.view(1, K, D) # (1, K, D) centroids

    # K-means loop:
    for i in range(Niter):
        #print("iteration: " + str(i))
        if K>cutoff:
            for j in range(len(c_j)):
                if j==0:
                    D_ij = ((x_i - c_j[j]) ** 2).sum(-1)
                else:
                    D_ij = torch.cat((D_ij,((x_i - c_j[j]) ** 2).sum(-1)), dim=-1)
        else:
            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        assert D_ij.size(1)==K
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        Ncl += 0.00000000001
        c /= Ncl
    return cl, c


dataset="ogbn-arxiv"
dataset = PygNodePropPredDataset(name = dataset, root="/home/devvrit/datasets/")
data = dataset[0].to(device)
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
y = data.y

def calc_nmi(x, K):
    cl,_ = Kmeans(normalize(x), K)
    nmi = normalized_mutual_info_score(y.view(-1).cpu(), cl.view(-1).cpu().numpy())
    print("nmi:", nmi)
    return nmi

data = dataset[0].to(device)
data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])
data_s = dataset[0].to(device)
data_s.edge_index = to_undirected(add_remaining_self_loops(data_s.edge_index)[0])
print("-----Calculating S_weights----")
transform = T.GDC(self_loop_weight=1, normalization_in='sym', normalization_out=None, diffusion_kwargs={'alpha': 0.05, 'method': 'ppr', 'eps':5e-4}, sparsification_kwargs=dict(method='threshold', avg_degree=128), exact=False)
data_s = transform(data_s)
print("data_orig:", data)
print("data_s:", data_s)
print("------S_weights calculated-------")


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


model1 = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
model2 = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=0.001)

nmi_arr = []


def train():
    model1.train()
    model2.train()
    optimizer.zero_grad()
    pos_za, neg_za, summarya = model1(data.x, data.edge_index)
    lossa = model1.loss(pos_za, neg_za, summarya)
    pos_zs, neg_zs, summarys = model2(data.x, data_s.edge_index)
    losss = model2.loss(pos_zs, neg_zs, summarys)
    loss = lossa+losss
    #loss = lossa
    loss.backward()
    optimizer.step()
    return loss.item()


def test(epoch):
    model1.eval()
    model2.eval()
    za, _, _ = model1(data.x, data.edge_index)
    zs, _, _ = model2(data.x, data_s.edge_index)
    z = (za+zs)/2
    #z = za
    nmi = calc_nmi(z, int(y.max()+1))
    nmi_arr.append(nmi)
    return


loss_min = 999999999.0
cnt=0
for epoch in range(1, 501):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch%50==0:
    	test(epoch)
print("nmi measured every 50th epoch:")
print(nmi_arr)
