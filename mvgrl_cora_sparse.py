import torch
import torch.nn as nn

from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_geometric.nn import GCNConv

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_nmi(x, K):
    x = x.cpu().data.numpy()
    kmeans = KMeans(n_clusters=K).fit(x)
    predict_labels = kmeans.predict(x)
    nmi = normalized_mutual_info_score(y.cpu(), predict_labels.reshape(-1))
    print("nmi:", nmi)
    return nmi

dataset="Cora"
dataset = Planetoid("/home/devvrit/datasets/", dataset)
data = dataset[0].to(device)
train_idx, valid_idx, test_idx = data['train_mask'].nonzero().view(-1), data['val_mask'].nonzero().view(-1), data['test_mask'].nonzero().view(-1)
y = data.y

data = dataset[0].to(device)
data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])
data_s = dataset[0].to(device)
data_s.edge_index = to_undirected(add_remaining_self_loops(data_s.edge_index)[0])
print("-----Calculating S_weights----")
transform = T.GDC(self_loop_weight=1, normalization_in='sym', normalization_out=None, diffusion_kwargs={'alpha': 0.05, 'method': 'ppr'}, sparsification_kwargs=dict(method='topk', k=128, dim=0), exact=True)
data_s = transform(data_s)
print("data_orig:", data)
print("data_s:", data_s)
print("------S_weights calculated-------")
data.x = data.x/torch.norm(data.x, dim=-1).view(-1,1)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x

class Encoder2(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False, normalize=False, add_self_loops=False)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index, weights):
        x = self.conv(x, edge_index, weights)
        x = self.prelu(x)
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
    hidden_channels=512, encoder=Encoder2(dataset.num_features, 512),
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
    pos_zs, neg_zs, summarys = model2(data.x, data_s.edge_index, data_s.edge_attr)
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
    zs, _, _ = model2(data.x, data_s.edge_index, data_s.edge_attr)
    z = (za+zs)/2
    #z = za
    nmi = calc_nmi(z, int(y.max()+1))
    nmi_arr.append(nmi)
    return


loss_min = 999999999.0
cnt=0
for epoch in range(1, 301):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch%50==0:
    	test(epoch)

print("nmi measured every 50th epoch:")
print(nmi_arr)
