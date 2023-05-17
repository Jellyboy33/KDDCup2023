import pandas as pd
import common.utils as utils
from tqdm import tqdm
from collections import OrderedDict

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
    products[i] = products[i].assign(counts = count_list)
    i += 1

  return products

products = get_counts()

for i,locale in enumerate(locales):
  products[i].to_csv(f"data/{locale}/counts.csv")

# def test():
#   products = pd.read_csv('data/counts_DE.csv')
#   products = products.loc[products['counts'] > 5]
#   print(len(products.index))

# test()