import pandas as pd
import common.utils as utils
from tqdm import tqdm
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
import torch

sessions_valid_path = "data/sessions_test_task1.csv"
sessions_path = "data/sessions_train.csv"
nodes_path = "data/products_train.csv"
locales = ['UK', 'DE', 'JP']

def get_counts():
  sessions_train = utils.fix_kdd_csv(pd.read_csv(sessions_path))
  sessions_test = utils.fix_kdd_csv(pd.read_csv(sessions_valid_path))
  products = pd.read_csv(nodes_path)
  sessions_train = [sessions_train.loc[sessions_train['locale'] == locale] for locale in locales]
  sessions_test  = [sessions_test.loc[sessions_test['locale'] == locale] for locale in locales]
  products = [products.loc[products['locale'] == locale] for locale in locales]
  products_counts = [OrderedDict([ (id, 0) for id in prod['id'] ]) for prod in products]

  def add_to_counts(df, prod_c, multiplier=1):
    def add(id):
      prod_c[id] += 1 * multiplier
    for _, row in tqdm(df.iterrows()):
      for item in row['prev_items']:
        add(item)
      if row.get('next_item') != None:
        add(row['next_item'])
    return prod_c
  
  products_counts = [add_to_counts(sessions_train[i], prod) for i,prod in enumerate(products_counts)]
  products_counts = [add_to_counts(sessions_test[i], prod, multiplier=1000) for i,prod in enumerate(products_counts)]

  i = 0
  for count_locale in products_counts:
    count_list = count_locale.values()
    products[i] = products[i].drop(['title','price','brand','color','size','model','material','author','desc'], axis=1)
    products[i] = products[i].assign(counts = count_list)
    products[i].to_csv(f"data/{locales[i]}/counts.csv")
    i += 1


def get_embeddings():
  batch_size = 1000
  products = pd.read_csv(nodes_path)
  model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
  products = [products.loc[products['locale'] == locale] for locale in locales]

  def batch(iterable, n=1, length=None):
    if length == None:
      l = len(iterable)
    else:
      l = length
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
  
  i = 0
  for p in products:
    p_ids_list = p['id'].values.tolist()
    p_ids = batch(p_ids_list, batch_size)
    p_titles = batch(p['title'].values.tolist(), batch_size)
    p_descs = batch(p['desc'].values.tolist(), batch_size)
    titles = {}
    descs = {}
    N = len(p_ids_list) / batch_size
    j = 0
    for id, t in zip(p_ids, p_titles):
      if j % 100 == 0:
        print(f"{j} / {N}")
      out = model.encode(t, batch_size=len(t))
      for idx in range(len(id)):
        titles[id[idx]] = out[idx]
      j += 1
    torch.save(titles, f"data/{locales[i]}/title_emb.pt")
    j = 0
    for id, d in zip(p_ids, p_descs):
      if j % 100 == 0:
        print(f"{j} / {N}")
      out = model.encode(d, batch_size=len(d))
      for idx in range(len(id)):
        descs[id[idx]] = out[idx]
      j += 1
    torch.save(descs, f"data/{locales[i]}/desc_emb.pt")
    i += 1

get_counts()
# get_embeddings()
