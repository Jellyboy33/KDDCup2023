import pandas as pd
import torch
import os.path as osp
from torch.utils.data import Dataset
import common.utils as utils
import torch.nn as nn

PARTITION = .9

class SessionsDataset(Dataset):
  def __init__(self, root, locale, phase='train'):
    super(SessionsDataset).__init__()
    self.root = root
    self.locale = locale
    self.phase = phase
    self.sessions_test_cache_path = osp.join(self.root, locale, 'sessions_test.pt')
    self.sessions_train_cache_path = osp.join(self.root, locale, 'sessions_train.pt')
    self.sessions_valid_cache_path = osp.join(self.root, locale, 'sessions_valid.pt')
    self.nodes_cache_path = osp.join(self.root, locale, 'nodes.pt')
    has_sessions, has_nodes = self.try_load_cache()
    sessions_valid_path = "sessions_test_task1.csv"
    sessions_path = "sessions_train.csv"
    nodes_path = "products_train.csv"
    if not has_sessions:
      if phase == 'train':
        print('Creating training data')
        self.sessions = pd.read_csv(osp.join(self.root, sessions_path))
        mid = int(len(self.sessions.index) * PARTITION)
        self.sessions = self.sessions.iloc[:mid]
      elif phase == 'test':
        print('Creating test data')
        self.sessions = pd.read_csv(osp.join(self.root, sessions_path))
        mid = int(len(self.sessions.index) * PARTITION)
        self.sessions = self.sessions.iloc[mid:]
      else:
        print('Creating validation set')
        self.sessions = pd.read_csv(osp.join(self.root, sessions_valid_path))
      self.sessions = self.sessions.loc[self.sessions['locale'] == locale]
      self.sessions = utils.fix_kdd_csv(self.sessions)
      self.cache_sessions()

    if not has_nodes:
      self.nodes = pd.read_csv(osp.join(self.root, nodes_path))
      # self.nodes = insert_counts(osp.join(self.root, sessions_path), osp.join(self.root, 'sessions_test_task1.csv'), self.nodes)
      self.nodes = self.nodes.loc[self.nodes['locale'] == locale]
      self.cache_nodes()

    self.id_mapping = {id: i for i, id in enumerate(self.nodes['id'])}
    self.reverse_id_mapping = self.nodes['id'].tolist()

  def try_load_cache(self):
    has_sessions = False
    has_nodes = False
    if self.phase == 'train':
      if osp.exists(self.sessions_train_cache_path):
        print(f"Trying to open {self.sessions_train_cache_path}")
        self.sessions = torch.load(self.sessions_train_cache_path)
        has_sessions = True
    elif self.phase == 'test':
      if osp.exists(self.sessions_test_cache_path):
        print(f"Trying to open {self.sessions_test_cache_path}")
        self.sessions = torch.load(self.sessions_test_cache_path)
        has_sessions = True
    else:
      if osp.exists(self.sessions_valid_cache_path):
        print(f"Trying to open {self.sessions_valid_cache_path}")
        self.sessions = torch.load(self.sessions_valid_cache_path)
        has_sessions = True
    if osp.exists(self.nodes_cache_path):
      print(f"Trying to open {self.nodes_cache_path}")
      self.nodes = torch.load(self.nodes_cache_path)
      has_nodes = True
    return has_sessions, has_nodes

  def cache_sessions(self):
    if self.phase == 'train':
      torch.save(self.sessions, self.sessions_train_cache_path)
    if self.phase == 'test':
      torch.save(self.sessions, self.sessions_test_cache_path)
    else:
      torch.save(self.sessions, self.sessions_valid_cache_path)

  def cache_nodes(self):
    torch.save(self.nodes, self.nodes_cache_path)
        
  def get_num_nodes(self):
    return len(self.nodes.index)

  def __len__(self):
    return len(self.sessions.index)
  
  def __getitem__(self, idx):
    row = self.sessions.iloc[idx]
    x = [self.id_mapping[id] for id in row['prev_items']]
    if self.phase != 'valid':
      y = self.id_mapping[row['next_item']]
      return torch.tensor(x), torch.tensor(y, dtype=torch.long)
    return torch.tensor(x)
  
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
  x = [item for item in batch]
  y = torch.tensor([item[1] for item in batch])
  inputs, src_mask, src_key_padding_mask = get_encoder_input(x)
  return (inputs, src_mask, src_key_padding_mask), y

def collate_fn_valid(x):
  inputs, src_mask, src_key_padding_mask = get_encoder_input(x)
  return inputs, src_mask, src_key_padding_mask

def insert_counts(sessions_train_file, sessions_test_file, products_train_df):
  sessions_train = utils.fix_kdd_csv(pd.read_csv(sessions_train_file))
  sessions_test = utils.fix_kdd_csv(pd.read_csv(sessions_test_file))
  products_train = products_train_df.assign(counts=0)

  def add_to_counts(df):
    def add(id):
      products_train.loc[products_train['id'] == id, 'counts'] += 1      
    for _, row in df.iterrows():
      for item in row['prev_items']:
        add(item)
      if row['next_item'] != None:
        add(row['next_item'])

  add_to_counts(sessions_train)
  add_to_counts(sessions_test)

  return products_train
