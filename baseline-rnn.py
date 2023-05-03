import torch
import torch.nn as nn
import pandas as pd
from data import SessionsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import common.utils as utils

TRAIN=False
MODEL_PATH='params/baseline-rnn.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss()
batch_size = 32

class BaselineRNN(nn.Module):
    def __init__(self):
        super(BaselineRNN, self).__init__()
        self.num_layers = 4
        self.hidden_size = 8
        self.rnn = nn.RNN(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_size, 1)

    def get_initial_h(self, batch_size):
        if batch_size == 0:
            size = (self.num_layers, self.hidden_size)
        else:
            size = (self.num_layers, batch_size, self.hidden_size)
        return torch.zeros(size, device=device, dtype=torch.float)

    def forward(self, x, initial_h):
        out, _ = self.rnn(x, initial_h)
        if initial_h.dim() == 3:
            # out = nn.utils.rnn.unpack_sequence(out)
            # final_states = []
            # for seq in out:
            #     final_state = seq[-1]
            #     final_states.append(final_state)
            # out = torch.stack(final_states)
            out = out[:, -1, :]
        else:
            out = out[-1]
        out = self.linear(out)
        return out

def train_step(model, optimizer, data, y):
    data = data.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    out = model(data, model.get_initial_h(batch_size))
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def collate_fn(batch):
    x, y = [], []

    for sample in batch:
        x.append(sample[0])
        y.append(sample[1])
    x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
    y = torch.stack(y)
    return x, y

def train():
    dataset = SessionsDataset('./data')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = BaselineRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    writer = SummaryWriter()
    step = 0
    losses = []
    for _ in range(1):
        for x, y in tqdm(loader):
            if x == None:
                continue
            loss = train_step(model, optimizer, x, y)
            losses.append(loss)
            if step % 100 == 0:
                avg_loss = torch.tensor(losses).mean()
                losses = []
                writer.add_scalar('train/loss', avg_loss, step)
            step += 1
    torch.save(model.state_dict(), MODEL_PATH)

def eval():
    dataset = SessionsDataset('./data')
    model = BaselineRNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    test_data = utils.get_test_data("task1")
    predictions = []
    for locale in test_data:
        locale_preds = []
        for _, session in tqdm(locale.iterrows()):
            h = model.get_initial_h(0)
            x = dataset.normalize(session['prev_items'])
            x = x.to(device)
            logits = model(x, h)
            _, ids = dataset.unnormalize(logits)
            locale_preds.append(ids)

        locale['next_item_prediction'] = locale_preds
        locale.drop('prev_items', inplace=True, axis=1)
        predictions.append(locale)
        
    predictions = pd.concat(predictions).reset_index(drop=True)
    predictions.to_csv('baseline-rnn-out.csv')
    utils.prepare_submission(predictions, "task1")

if TRAIN:
    train()
else:
    with torch.no_grad():
        eval()
