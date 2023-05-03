import pandas as pd
import numpy as np
import torch
import math
import os.path as osp
from torch.utils.data import Dataset
from torch_geometric.data import Data
import common.utils as utils

class SessionsDataset(Dataset):
  def __init__(self, root):
    super(SessionsDataset).__init__()
    self.root = root
    sessions_path = "sessions_train.csv"
    nodes_path = "products_train.csv"
    self.sessions = pd.read_csv(osp.join(self.root, sessions_path))
    self.nodes = pd.read_csv(osp.join(self.root, nodes_path))
    self.id_mapping = {id: i for i, id in enumerate(self.nodes['id'])}
    self.sessions = utils.fix_kdd_csv(self.sessions)

    n = len(self.sessions.index)
    mean = (n + 1) / 2
    var = (n+1)*(2*n+1)/6 - ((n+1)**2)/4
    std = math.sqrt(var)
    print(mean, std)
    self.mean, self.std = mean, std

  @property
  def raw_file_names(self):
    return [self.sessions_path, self.nodes_path]

  @property
  def processed_file_names(self):
    return [f'data.pt']

  def __len__(self):
    return len(self.sessions.index)
  
  def __getitem__(self, idx):
    row = self.sessions.iloc[idx]
    # codes, uniques = pd.factorize(row['prev_items'])
    x = [self.id_mapping[id] for id in row['prev_items']]
    x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
    x = (x - self.mean) / self.std
    # edge_index = torch.tensor(np.array([ codes[:-1], codes[1:] ]), dtype=torch.long)
    y = self.id_mapping[row['next_item']]
    y = torch.tensor([y], dtype=torch.float)
    y = (y - self.mean) / self.std
    return x, y
  
  def normalize(self, id_list):
    x = [self.id_mapping[id] for id in id_list]
    x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
    x = (x - self.mean) / self.std
    return x
  
  def unnormalize(self, logit):
    logit = int((logit * self.std) + self.mean)
    ids = [self.nodes.iloc[logit]['id']]
    return logit, ids
