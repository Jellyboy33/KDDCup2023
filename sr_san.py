import torch
import sr_san.model as model
from sr_san.sr_data import SessionsDataset, collate_fn, collate_fn_valid
from torch.utils.data import DataLoader
import os.path as osp

TRAIN=False
HIDDEN_SIZE=768
BATCH_SIZE=64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARAMS='params/hier_filter10_2'

# all product
# UK_PRODUCTS = 500180
# DE_PRODUCTS = 518327
# JP_PRODUCTS = 395009

# 10 prodcut
# UK_PRODUCTS = 163109 (old)
# DE_PRODUCTS = 158175
# JP_PRODUCTS = 137238
UK_PRODUCTS = 162952
DE_PRODUCTS = 158001
JP_PRODUCTS = 136933

# 5 product
# UK_PRODUCTS = 285145
# DE_PRODUCTS = 282044
# JP_PRODUCTS = 229939

# 5 product
N_PRODUCTS = {
    'UK': UK_PRODUCTS,
    'DE': DE_PRODUCTS,
    'JP': JP_PRODUCTS
}

if TRAIN:
    for locale in ['UK', 'DE', 'JP']:
        train_set = SessionsDataset('./data', locale, 'train')
        test_set = SessionsDataset('./data', locale, 'test')
        print(f'Number of train sessions: {len(train_set)}')
        print(f'Number of test sessions: {len(test_set)}')
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
        n_node = train_set.get_num_nodes()
        print(f'Num nodes: {n_node}')
        net = model.MultiSequence(hidden_size=HIDDEN_SIZE, n_node=n_node).to(DEVICE)
        def mrr(r_ranks):
            return torch.sum(r_ranks) / len(test_set)

        model.train(net, train_loader, test_loader, mrr, locale, DEVICE)
else:
    valid_de_set = SessionsDataset('./data', 'DE', 'valid')
    valid_jp_set = SessionsDataset('./data', 'JP', 'valid')
    valid_uk_set = SessionsDataset('./data', 'UK', 'valid')
    valid_de_loader = DataLoader(valid_de_set, batch_size=BATCH_SIZE*2, collate_fn=collate_fn_valid, num_workers=8)
    valid_jp_loader = DataLoader(valid_jp_set, batch_size=BATCH_SIZE*2, collate_fn=collate_fn_valid, num_workers=8)
    valid_uk_loader = DataLoader(valid_uk_set, batch_size=BATCH_SIZE*2, collate_fn=collate_fn_valid, num_workers=8)
    de_net = model.MultiSequence(hidden_size=HIDDEN_SIZE, n_node=DE_PRODUCTS).to(DEVICE)
    jp_net = model.MultiSequence(hidden_size=HIDDEN_SIZE, n_node=JP_PRODUCTS).to(DEVICE)
    uk_net = model.MultiSequence(hidden_size=HIDDEN_SIZE, n_node=UK_PRODUCTS).to(DEVICE)
    de_net.load_state_dict(torch.load(osp.join(PARAMS,'recc_model_DE.pt')))
    jp_net.load_state_dict(torch.load(osp.join(PARAMS,'recc_model_JP.pt')))
    uk_net.load_state_dict(torch.load(osp.join(PARAMS,'recc_model_UK.pt')))
    loaders = [valid_de_loader, valid_jp_loader, valid_uk_loader]
    sets = [valid_de_set, valid_jp_set, valid_uk_set]
    models = [de_net, jp_net, uk_net]
    model.eval(loaders, sets, models, DEVICE)
