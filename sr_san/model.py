import math
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from tqdm import tqdm
import pandas as pd
import common.utils as utils
from datetime import datetime
from multiprocessing import Pool
# import torch.linalg as L

# class SelfAttentionNetwork(Module):
#     def __init__(self, hidden_size, batch_size, n_node):
#         super(SelfAttentionNetwork, self).__init__()
#         self.hidden_size = hidden_size
#         self.n_node = n_node
#         self.batch_size = batch_size
#         self.embedding = nn.Embedding(self.n_node, self.hidden_size)
#         self.transformerEncoderLayer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=1, dim_feedforward=self.hidden_size * 4)#nheads=2
#         self.transformerEncoder = TransformerEncoder(self.transformerEncoderLayer, num_layers=1)# 3 layer
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)

#     def compute_scores(self, hidden):
#         # hidden = [B, LATENT_SIZE]
#         # embedding = [N_PRODUCTS, LATENT_SIZE]
#         # scores = [B, N_PRODUCTS]
#         e = self.embedding.weight
#         scores = hidden @ e.T
#         return scores

#     def forward(self, inputs, src_mask, src_key_padding_mask):
#         hidden = self.embedding(inputs)
#         hidden = hidden.transpose(0,1).contiguous()
#         hidden = self.transformerEncoder(hidden, src_mask, src_key_padding_mask)
#         hidden = hidden.transpose(0,1).contiguous()
#         h_n = hidden[:, -1]
#         scores = self.compute_scores(h_n)
#         return scores

class MultiSequence(Module):
    def __init__(self, hidden_size, n_node):
        super(MultiSequence, self).__init__()
        self.hidden_size = hidden_size
        self.n_node = n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.pid_encoder = self.create_encoder()
        self.title_encoder = self.create_encoder()
        self.desc_encoder = self.create_encoder()
        self.agg_encoder = self.create_encoder(3)

        self.lin_out = nn.Linear(hidden_size*3, hidden_size, bias=True)
        # self.resetparameters()

    def create_encoder(self, size_m=1):
        layer = TransformerEncoderLayer(d_model=self.hidden_size*size_m, nhead=1, dim_feedforward=self.hidden_size*size_m, batch_first=True)
        return TransformerEncoder(layer, num_layers=1)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden):
        # hidden = [B, LATENT_SIZE]
        # embedding = [N_PRODUCTS, LATENT_SIZE]
        # scores = [B, N_PRODUCTS]
        e = self.embedding.weight
        hidden = self.lin_out(hidden)
        scores = hidden @ e.T
        return scores

    def forward(self, pid, title, desc, src_mask, src_key_padding_mask):
        def encode(seq, encoder):
            # seq = seq.transpose(0,1).contiguous()
            seq = encoder(seq, src_mask, src_key_padding_mask)
            # seq = seq.transpose(0,1).contiguous()
            return seq

        pid = self.embedding(pid)
        # Encode Sequence
        pid = encode(pid, self.pid_encoder)
        title = encode(title, self.title_encoder)
        desc = encode(desc, self.desc_encoder)
        # Aggregate [B, L, F]
        agg = torch.cat((pid, title, desc), dim=-1)
        # agg = pid + title + desc
        # agg_norm = L.norm(agg, dim=-1, keep_dim=True)
        # agg = agg / agg_norm
        # agg = F.normalize(agg, dim=-1) 
        agg = encode(agg, self.agg_encoder)
        final_out = agg[:, -1]
        scores = self.compute_scores(final_out)
        return scores

def train(model, train_loader, test_loader, eval, locale, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=.75)
    best_mrr = 0
    for epoch in range(30):
        print('Epoch:', epoch)
        model.train()
        losses = []
        print(f'Training...')
        for (x_id, x_title, x_desc, src_mask, src_key_padding_mask), (y_id, y_title, y_desc) in tqdm(train_loader):
            x_id = x_id.to(device)
            x_title = x_title.to(device)
            x_desc = x_desc.to(device)
            y_id = y_id.to(device)
            # y_title = y_title.to(device)
            # y_desc = y_desc.to(device)

            src_mask = src_mask.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)

            optimizer.zero_grad()
            pred = model(x_id, x_title, x_desc, src_mask, src_key_padding_mask)
            # pred = [B, N_PRODUCTS]
            loss = loss_function(pred, y_id)
            loss.backward()
            optimizer.step()
            losses.append(loss)
        scheduler.step()
        print(f"Avg Loss: {torch.tensor(losses).mean()}")

        print(f'Testing...')
        r_ranks = []
        for (x_id, x_title, x_desc, src_mask, src_key_padding_mask), (y_id, y_title, y_desc) in tqdm(test_loader):
            x_id = x_id.to(device)
            x_title = x_title.to(device)
            x_desc = x_desc.to(device)
            y_id = y_id.to(device)

            src_mask = src_mask.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)

            optimizer.zero_grad()
            pred = model(x_id, x_title, x_desc, src_mask, src_key_padding_mask)
            # prob = [B, C]
            prob = F.softmax(pred, dim=1)
            # top_recc = [B, 100]
            top_recc = prob.topk(100)[1]
            y_id = torch.unsqueeze(y_id, -1)
            ranks = (top_recc == y_id).nonzero(as_tuple=True)[1] + 1
            r_ranks.append(1 / ranks)
        mrr = eval(torch.cat(r_ranks))
        print(f'MRR: {mrr}')
        if mrr > best_mrr:
            torch.save(model.state_dict(), f"recc_model_{locale}.pt")
            best_mrr = mrr

def idx_to_id(input):
    lookup, jobs = input
    return [[lookup[idx] for idx in session] for batch in jobs for session in batch]

def eval(loaders, sets, models, device):
    num_threads = 16
    print(f'Using device {device}')
    model_output = []
    with torch.no_grad():
        for i in range(3):
            loader = loaders[i]
            model = models[i]
            model.eval()
            locale_preds = []
            for x_id, x_title, x_desc, src_mask, src_key_padding_mask in tqdm(loader):
                x_id = x_id.to(device)
                x_title = x_title.to(device)
                x_desc = x_desc.to(device)

                src_mask = src_mask.to(device)
                src_key_padding_mask = src_key_padding_mask.to(device)

                pred = model(x_id, x_title, x_desc, src_mask, src_key_padding_mask)
                pred = F.softmax(pred, dim=1)
                pred = pred.topk(100)[1].tolist()
                locale_preds.append(pred)
            torch.cuda.empty_cache()
            model_output.append(locale_preds)
    predictions = []
    for i in range(3):
        print(i)
        # Unbatch
        set = sets[i]
        pred = model_output[i]
        step = int(len(pred) / num_threads)
        thread_outputs = [None]*num_threads
        with Pool(num_threads) as p:
            worker_input = []
            for i in range (num_threads - 1):
                start = i*step
                end = (i+1)* step
                print(start, end)
                worker_input.append((set.reverse_id_mapping, pred[start:end]))
            start = (num_threads-1)*step
            end = ":"
            print(start, end)
            worker_input.append((set.reverse_id_mapping, pred[start:]))
            thread_outputs = p.map(idx_to_id, worker_input)
        locale_preds = [top_100 for thread_output in thread_outputs for top_100 in thread_output]
        set.sessions['next_item_prediction'] = locale_preds
        set.sessions.drop('prev_items', inplace=True, axis=1)
        predictions.append(set.sessions)
            
    predictions = pd.concat(predictions).reset_index(drop=True)
    predictions.to_csv('sr_san_out.csv')
    utils.prepare_submission(predictions, "task1")
