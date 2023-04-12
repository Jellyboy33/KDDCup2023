import pandas as pd
import torch
import os.path as osp
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from ast import literal_eval

sessions_test_path_t1 = "./data/sessions_test_task1.csv"
sessions_test_path_t2 = "./data/sessions_test_task2.csv"

def load_node_csv(path, index_col, encoders=None, **kwargs):
  df = pd.read_csv(path, index_col=index_col, **kwargs)
  mapping = {index: i for i, index in enumerate(df.index.unique())}

  x = None
  if encoders is not None:
    xs = [encoder(df[col]) for col, encoder in encoders.items()]
    x = torch.cat(xs, dim=-1)

  return x, mapping

class SessionsDataset(InMemoryDataset):
  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
    super().__init__(root, transform, pre_transform, pre_filter)
    self.processed_path = osp.join(self.root, "processed.pt")
    self.data, self.slices = torch.load(self.processed_path)
    self.sessions_path = "sessions_train.csv"
    self.nodes_path = "product_train.csv"

    # @property
    # def raw_file_names(self):
    #   return [f'{self.file_name}.txt']

    # @property
    # def processed_file_names(self):
    #   return [f'{self}.pt']

  def process(self):
    sessions = pd.read_csv(osp.join(self.root, self.nodes_path))
    nodes = pd.read_csv(osp.join(self.root, self.nodes_path))
    id_mapping = {id: i for i, id in enumerate(nodes['id'])}

    sessions['prev_items'] = sessions['prev_items'].replace(' ', ',', regex=True)
    sessions['prev_items'] = sessions['prev_items'].apply(literal_eval)
    data_list = []
    for _, row in sessions.iterrows():
      codes, uniques = pd.factorize(row['prev_items'])
      x = [id_mapping[id] for id in uniques]
      x = torch.tensor(x, dtype=torch.long).unsqueeze(1)
      edge_index = torch.tensor([ codes[:-1], codes[1:] ], dtype=torch.long)
      y = id_mapping[row['next_item']]
      y = torch.tensor(y, dtype=torch.long)
      data_list.append( Data(x=x, edge_index=edge_index, y=y) )

    data, slices = self.collate(data_list)
    torch.save((data, slices), self.processed_path)
# load_node_csv(sessions_train, )