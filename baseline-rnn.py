import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from data import SessionsDataset
from torch_geometric.loader import DataLoader


class BaselineRNN(nn.Module):
    def __init__(self):
        super(BaselineRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=1, num_layers=1, batch_first=True)

    def forward(self, data):
        h = 0
        for i in range(data.edge_index.shape[1]):
            x = data.edge_index[0][i]
            _, h = self.rnn(x, h)
        x = data.edge_index[1][-1]
        out, _ = self.rnn(x, h)
        return F.log_softmax(out, dim=1)

dataset = SessionsDataset('./data')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BaselineRNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(dataset.num_classes)
print(dataset.num_node_features)
print(dataset[0])

def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def train_step(batch):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

best_val_acc = test_acc = 0
for epoch in range(1,100):
    for data in loader:
        train_step()
    _, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'
    
    if epoch % 10 == 0:
        print(log.format(epoch, best_val_acc, test_acc))