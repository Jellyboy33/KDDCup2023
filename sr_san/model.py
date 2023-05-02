import math
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from tqdm import tqdm

class SelfAttentionNetwork(Module):
    def __init__(self, hidden_size, batch_size, n_node):
        super(SelfAttentionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.n_node = n_node
        self.batch_size = batch_size
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.transformerEncoderLayer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=1, dim_feedforward=self.hidden_size * 4)#nheads=2
        self.transformerEncoder = TransformerEncoder(self.transformerEncoderLayer, num_layers=1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden):
        # hidden = [B, LATENT_SIZE]
        e = self.embedding.weight  # [n_nodes, LATENT_SIZE]
        scores = hidden @ e.T
        return scores   

    def forward(self, inputs, src_mask, src_key_padding_mask):
        hidden = self.embedding(inputs)
        hidden = hidden.transpose(0,1).contiguous()
        hidden = self.transformerEncoder(hidden, src_mask, src_key_padding_mask)
        hidden = hidden.transpose(0,1).contiguous()
        h_n = hidden[:, -1]
        scores = self.compute_scores(h_n)
        return scores

def train(model, train_loader, test_loader, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)
    for epoch in range(20):
        print('Epoch:', epoch)
        model.train()
        losses = []
        for (inputs, src_mask, src_key_padding_mask), y in tqdm(train_loader):
            inputs = inputs.to(device)
            src_mask = src_mask.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(inputs, src_mask, src_key_padding_mask)
            loss = loss_function(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss)
        scheduler.step()
        print(f"Avg Loss: {torch.tensor(losses).mean()}")

        mrr = []
        for (inputs, src_mask, src_key_padding_mask), y in tqdm(test_loader):
            inputs = inputs.to(device)
            src_mask = src_mask.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            y = y.to(device)
            pred = model(inputs, src_mask, src_key_padding_mask)
            # prob = [B, C]
            prob = F.softmax(pred, dim=1)
            # top_recc = [B, 100]
            top_recc = prob.topk(5)[1]
            y = torch.unsqueeze(y, -1)
            ranks = (top_recc == y).nonzero(as_tuple=True)[1] + 1
            r_ranks = 1 / ranks
            mrr.append(r_ranks)
        mrr = torch.cat(mrr)
        mrr = torch.sum(mrr) / len(test_loader)
