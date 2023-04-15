import pandas as pd
import numpy as np
import torch
import os.path as osp
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from ast import literal_eval

class SessionsDataset(Dataset):
  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
    super().__init__(root, transform, pre_transform, pre_filter)
    # self.processed_path = osp.join(self.root, "processed.pt")
    # self.data, self.slices = torch.load(self.processed_paths[0])

    sessions_path = "sessions_train.csv"
    nodes_path = "products_train.csv"
    self.sessions = pd.read_csv(osp.join(self.root, sessions_path))
    self.nodes = pd.read_csv(osp.join(self.root, nodes_path))
    self.id_mapping = {id: i for i, id in enumerate(self.nodes['id'])}
    self.sessions['prev_items'] = self.sessions['prev_items'].replace(' ', ',', regex=True)
    self.sessions['prev_items'] = self.sessions['prev_items'].apply(literal_eval)

  @property
  def raw_file_names(self):
    return [self.sessions_path, self.nodes_path]

  @property
  def processed_file_names(self):
    return [f'data.pt']

  # def process(self):
  #   pass
  #   # sessions_path = "sessions_train.csv"
  #   # nodes_path = "products_train.csv"
  #   # sessions = pd.read_csv(osp.join(self.root, sessions_path))
  #   # nodes = pd.read_csv(osp.join(self.root, nodes_path))
  #   # id_mapping = {id: i for i, id in enumerate(nodes['id'])}

  #   # sessions['prev_items'] = sessions['prev_items'].replace(' ', ',', regex=True)
  #   # sessions['prev_items'] = sessions['prev_items'].apply(literal_eval)
  #   # data_list = []
  #   # for _, row in sessions.iterrows():
  #   #   codes, uniques = pd.factorize(row['prev_items'])
  #   #   x = [id_mapping[id] for id in uniques]
  #   #   x = torch.tensor(x, dtype=torch.long).unsqueeze(1)
  #   #   edge_index = torch.tensor(np.array([ codes[:-1], codes[1:] ]), dtype=torch.long)
  #   #   y = id_mapping[row['next_item']]
  #   #   y = torch.tensor(y, dtype=torch.long)
  #   #   data_list.append( Data(x=x, edge_index=edge_index, y=y) )

    # data, slices = self.collate(data_list)
    # torch.save((data, slices), self.processed_paths[0])
  def len(self):
    return len(self.sessions.index)
  
  def get(self, idx):
    row = self.sessions.iloc[idx]
    codes, uniques = pd.factorize(row['prev_items'])
    x = [self.id_mapping[id] for id in uniques]
    x = torch.tensor(x, dtype=torch.long).unsqueeze(1)
    edge_index = torch.tensor(np.array([ codes[:-1], codes[1:] ]), dtype=torch.long)
    y = self.id_mapping[row['next_item']]
    y = torch.tensor(y, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)
    # data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
    return data