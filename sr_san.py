import torch
import sr_san.model as model
from sr_san.sr_data import SessionsDataset, collate_fn
from torch.utils.data import DataLoader

HIDDEN_SIZE=96
BATCH_SIZE=128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_set = SessionsDataset('./data', 'UK', 'train')
test_set = SessionsDataset('./data', 'UK', 'test')
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)
n_node = train_set.get_num_nodes()
print(f'Num nodes: {n_node}')
net = model.SelfAttentionNetwork(hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, n_node=n_node).to(DEVICE)
model.train(net, train_loader, test_loader, DEVICE)
