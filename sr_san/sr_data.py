import pandas as pd
import torch
import os.path as osp
from torch.utils.data import Dataset
import common.utils as utils
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import math

PARTITION = .9
MIN_NUM = 10
def filter_session(session, counts_map):
  for item in session['prev_items']:
    if counts_map[item] < MIN_NUM:
      return False

  if counts_map[session['next_item']] < MIN_NUM:
    return False
  
  return True

def filter_node(node, counts_map):
  return counts_map[node['id']] >= MIN_NUM

class SessionsDataset(Dataset):
  def __init__(self, root, locale, phase='train'):
    super(SessionsDataset).__init__()
    self.root = root
    self.locale = locale
    self.phase = phase

    self.counts_map = None
    self.counts_path = osp.join(self.root, locale, 'counts.csv')
    self.sessions_test_cache_path = osp.join(self.root, locale, 'sessions_test.pt')
    self.sessions_train_cache_path = osp.join(self.root, locale, 'sessions_train.pt')
    self.sessions_valid_cache_path = osp.join(self.root, locale, 'sessions_valid.pt')
    self.nodes_cache_path = osp.join(self.root, locale, 'nodes.pt')
    self.titles_cache_path = osp.join(self.root, locale, 'title_emb.pt')
    self.descs_cache_path = osp.join(self.root, locale, 'desc_emb.pt')
    has_sessions, has_nodes = self.try_load_cache()
    sessions_valid_path = "sessions_test_task1.csv"
    sessions_path = "sessions_train.csv"
    nodes_path = "products_train.csv"
    if not has_sessions:
      if phase == 'train' or phase == 'test':
        self.sessions = pd.read_csv(osp.join(self.root, sessions_path))
        self.sessions = self.sessions.loc[self.sessions['locale'] == locale]
        self.sessions = utils.fix_kdd_csv(self.sessions)
        self.counts = pd.read_csv(self.counts_path)
        self.counts_map = {row['id']: int(row['counts']) for _,row in self.counts.iterrows()}
        self.sessions = self.sessions[self.sessions.apply(lambda session : filter_session(session, self.counts_map), axis=1)]
        if phase == 'train':
          print('Creating training data')
          mid = int(len(self.sessions.index) * PARTITION)
          self.sessions = self.sessions.iloc[:mid]
        else:
          print('Creating test data')
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
      self.nodes = self.nodes.loc[self.nodes['locale'] == locale]
      if self.counts_map == None:
        self.counts = pd.read_csv(self.counts_path)
        self.counts_map = {row['id']: int(row['counts']) for _,row in self.counts.iterrows()}
      self.nodes = self.nodes[self.nodes.apply(lambda node : filter_node(node, self.counts_map), axis=1)]
      self.cache_nodes()

    self.id_mapping = {id: i for i, id in enumerate(self.nodes['id'])}
    self.reverse_id_mapping = self.nodes['id'].tolist()
    self.sent_model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
    # self.ret_count = 0

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
    
    if osp.exists(self.titles_cache_path):
      print(f"Trying to open {self.titles_cache_path}")
      self.titles = torch.load(self.titles_cache_path)
    if osp.exists(self.descs_cache_path):
      print(f"Trying to open {self.descs_cache_path}")
      self.descs = torch.load(self.descs_cache_path)

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
  
  def encode_title(self, id):
    # if title_emb is None:
    #   title = self.nodes.loc[self.nodes['id'] == id].iloc[0]['title']
    #   title_emb = self.sent_model.encode(title, convert_to_tensor=True).to(device='cpu')
    #   self.title_cache[id] = title_emb
    #   self.ret_count+=1
    # if self.ret_count % 1000 == 0 and self.ret_count > 0:
    #   torch.save(self.title_cache, self.titles_cache_path)
    #   print('saved titles')
    return torch.from_numpy(self.titles[id])

  def encode_desc(self, id):
    # if desc_emb is None:
    #   desc = self.nodes.loc[self.nodes['id'] == id].iloc[0]['desc']
    #   if isinstance(desc, str):
    #     desc_emb = self.sent_model.encode(desc, convert_to_tensor=True).to(device='cpu')
    #   else:
    #     desc_emb = torch.zeros(768, device='cpu')
    #   self.desc_cache[id] = desc_emb
    # if self.ret_count % 1000 == 0 and self.ret_count > 0:
    #   torch.save(self.desc_cache, self.descs_cache_path)
    #   print('saved descs')
    return torch.from_numpy(self.descs[id])
  
  def __getitem__(self, idx):
    row = self.sessions.iloc[idx]
    x = [[torch.tensor(self.id_mapping[id]), self.encode_title(id), self.encode_desc(id)] for id in row['prev_items']]
    if self.phase != 'valid':
      id = row['next_item']
      y = [torch.tensor(self.id_mapping[id]), self.encode_title(id), self.encode_desc(id)]
      return x, y
    return x
  
# Mask of [B, S, S] to mark parts of the sequence that the layer should not attend to
# Fed into src_mask
def get_src_mask(batch_size, sequence_size):
  masks = -torch.inf * torch.ones((batch_size, sequence_size, sequence_size))
  for i in range(masks.shape[0]):
      masks[i] = torch.triu(masks[i], diagonal=1)
  return masks
  # return torch.zeros((batch_size, sequence_size, sequence_size), dtype=torch.bool)

# Mask of [B, S] to mark padded out parts of the sequence in a batch
# Fed into src_key_padding_mask
def get_src_key_padding_mask(sequences):
  sequence_masks = [torch.zeros(seq.shape[0]) for seq in sequences]
  sequence_masks = nn.utils.rnn.pad_sequence(sequence_masks, batch_first=True, padding_value=1.)
  sequence_masks = sequence_masks.to(torch.bool)
  return sequence_masks

def get_encoder_input(ids_, titles, descs):
  ids = nn.utils.rnn.pad_sequence(ids_, batch_first=True)
  titles = nn.utils.rnn.pad_sequence(titles, batch_first=True)
  descs = nn.utils.rnn.pad_sequence(descs, batch_first=True)
  batch_size = ids.shape[0]
  sequence_size = ids.shape[1]
  src_mask = get_src_mask(batch_size, sequence_size)
  src_key_padding_mask = get_src_key_padding_mask(ids_)
  return ids, titles, descs, src_mask, src_key_padding_mask
  
def collate_fn(batch):
  x = [item[0] for item in batch] #[32, L, 3]
  y = [item[1] for item in batch] #[32, 3]
  x_id = [torch.tensor([item[0] for item in seq]) for seq in x]
  x_title = [torch.stack([item[1] for item in seq]) for seq in x]
  x_desc = [torch.stack([item[2] for item in seq]) for seq in x]
  x_id, x_title, x_desc, src_mask, src_key_padding_mask = get_encoder_input(x_id, x_title, x_desc)
  y_id = torch.tensor([item[0] for item in y])
  y_title = torch.stack([item[1] for item in y])
  y_desc = torch.stack([item[2] for item in y])

  return (x_id, x_title, x_desc, src_mask, src_key_padding_mask), (y_id, y_title, y_desc)

def collate_fn_valid(batch):
  x_id = [torch.tensor([item[0] for item in seq]) for seq in batch]
  x_title = [torch.stack([item[1] for item in seq]) for seq in batch]
  x_desc = [torch.stack([item[2] for item in seq]) for seq in batch]
  x_id, x_title, x_desc, src_mask, src_key_padding_mask = get_encoder_input(x_id, x_title, x_desc)
  return x_id, x_title, x_desc, src_mask, src_key_padding_mask
