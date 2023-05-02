import pandas as pd
import numpy as np
import torch
import math
import os.path as osp
from torch.utils.data import Dataset
from torch_geometric.data import Data
import common.utils as utils
import torch.nn as nn

class SessionsDataset(Dataset):
  def __init__(self, root, locale, phase='train'):
    super(SessionsDataset).__init__()
    self.root = root
    sessions_path = "sessions_train.csv"
    nodes_path = "products_train.csv"
    self.sessions = pd.read_csv(osp.join(self.root, sessions_path))
    self.nodes = pd.read_csv(osp.join(self.root, nodes_path))
    self.sessions = self.sessions.loc[self.sessions['locale'] == locale]
    self.nodes = self.nodes.loc[self.nodes['locale'] == locale]
    partition = .9 if phase == 'train' else .1
    mid = int(len(self.sessions.index) * partition)
    if phase == 'train':
      self.sessions = self.sessions.iloc[:mid]
    else:
      self.sessions = self.sessions.iloc[mid:]
    self.id_mapping = {id: i for i, id in enumerate(self.nodes['id'])}
    self.sessions = utils.fix_kdd_csv(self.sessions)

  @property
  def raw_file_names(self):
    return [self.sessions_path, self.nodes_path]

  @property
  def processed_file_names(self):
    return [f'data.pt']
  
  def get_num_nodes(self):
    return len(self.nodes.index)

  def __len__(self):
    return len(self.sessions.index)
  
  def __getitem__(self, idx):
    row = self.sessions.iloc[idx]
    # codes, uniques = pd.factorize(row['prev_items'])
    x = [self.id_mapping[id] for id in row['prev_items']]
    # x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
    # x = (x - self.mean) / self.std
    # edge_index = torch.tensor(np.array([ codes[:-1], codes[1:] ]), dtype=torch.long)
    y = self.id_mapping[row['next_item']]

    # y = torch.tensor([y], dtype=torch.float)
    # y = (y - self.mean) / self.std
    return torch.tensor(x), torch.tensor(y, dtype=torch.long)
  
  def normalize(self, id_list):
    x = [self.id_mapping[id] for id in id_list]
    x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
    x = (x - self.mean) / self.std
    return x
  
  def unnormalize(self, logit):
    logit = int((logit * self.std) + self.mean)
    ids = [self.nodes.iloc[logit]['id']]
    return logit, ids
  
# Mask of [B, S, S] to mark parts of the sequence that the layer should not attend to
# Fed into src_mask
def get_src_mask(batch_size, sequence_size):
  masks = -torch.inf * torch.ones((batch_size, sequence_size, sequence_size))
  for i in range(masks.shape[0]):
      masks[i] = torch.triu(masks[i], diagonal=1)
  return masks

# Mask of [B, S] to mark padded out parts of the sequence in a batch
# Fed into src_key_padding_mask
def get_src_key_padding_mask(sequences):
  sequence_masks = [torch.zeros(seq.shape[0]) for seq in sequences]
  sequence_masks = nn.utils.rnn.pad_sequence(sequence_masks, batch_first=True, padding_value=1.)
  sequence_masks = sequence_masks.to(torch.bool)
  return sequence_masks

def get_encoder_input(sequences):
  input = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
  batch_size = input.shape[0]
  sequence_size = input.shape[1]
  src_mask = get_src_mask(batch_size, sequence_size)
  src_key_padding_mask = get_src_key_padding_mask(sequences)
  return input, src_mask, src_key_padding_mask
  
def collate_fn(batch):
  x = [item[0] for item in batch]
  y = torch.tensor([item[1] for item in batch])
  inputs, src_mask, src_key_padding_mask = get_encoder_input(x)
  return (inputs, src_mask, src_key_padding_mask), y
