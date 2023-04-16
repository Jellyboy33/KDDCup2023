import torch
import torch.nn as nn
import torch.nn.functional as F
from data import SessionsDataset
from torch_geometric.loader import DataLoader


class BaselineRNN(nn.Module):
    def __init__(self):
        super(BaselineRNN, self).__init__()
        self.num_layers = 4
        self.hidden_size = 8
        self.rnn = nn.RNN(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_size, 1)

    def get_initial_h(self, batch_size):
        size = (self.num_layers, batch_size, self.hidden_size)
        return torch.zeros(size, device=device, dtype=torch.float)

    def forward(self, x, initial_h):
        out, _ = self.rnn(x, initial_h)
        out = nn.utils.rnn.unpack_sequence(out)
        final_states = []
        for seq in out:
            final_state = seq[-1]
            final_states.append(final_state)
        out = torch.stack(final_states)
        out = self.linear(out)
        return out

batch_size = 32
dataset = SessionsDataset('./data')
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BaselineRNN().to(device)
loss_fn = nn.MSELoss()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(dataset.num_classes)
print(dataset.num_node_features)
print(dataset[0])
print(dataset[0].edge_index)

def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def train_step(data, y):
    data = data.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    out = model(data, model.get_initial_h(batch_size))
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def get_packed_seq_from_batch(batch):
    batch_indices = batch.batch
    samples = [[] for _ in range(batch_size)]
    for i,idx in enumerate(batch_indices):
        samples[idx].append(batch.x[i])
    for i in range(len(samples)):
        if len(samples[i]) == 0:
            return None
        samples[i] = torch.stack(samples[i]).to(dtype=torch.float)
    samples = nn.utils.rnn.pack_sequence(samples, enforce_sorted=False)
    return samples

step = 0
best_val_acc = test_acc = 0
for epoch in range(1,100):
    for batch in loader:
        data = get_packed_seq_from_batch(batch)
        if data == None:
            continue
        loss = train_step(data, batch.y)
        step += 1
        if step % 100 == 0:
            print(loss)
    # _, val_acc, tmp_test_acc = test()
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     test_acc = tmp_test_acc
    # log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'
    
    # if epoch % 10 == 0:
    #     print(log.format(epoch, best_val_acc, test_acc))