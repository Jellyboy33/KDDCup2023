import torch
import sr_san.model as model
from sr_san.sr_data import SessionsDataset, collate_fn, collate_fn_valid
from torch.utils.data import DataLoader

TRAIN='JP'
HIDDEN_SIZE=256
BATCH_SIZE=256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# UK_PRODUCTS = 500180
# DE_PRODUCTS = 518327
# JP_PRODUCTS = 395009
UK_PRODUCTS = 163109
DE_PRODUCTS = 158175
JP_PRODUCTS = 137238
N_PRODUCTS = {
    'UK': UK_PRODUCTS,
    'DE': DE_PRODUCTS,
    'JP': JP_PRODUCTS
}

def mrr(r_ranks):
    return torch.sum(r_ranks) / len(test_set)

if TRAIN != None:
    locale = TRAIN
    train_set = SessionsDataset('./data', locale, 'train')
    test_set = SessionsDataset('./data', locale, 'test')
    print(f'Number of train sessions: {len(train_set)}')
    print(f'Number of test sessions: {len(test_set)}')
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    n_node = train_set.get_num_nodes()
    print(f'Num nodes: {n_node}')
    net = model.SelfAttentionNetwork(hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, n_node=N_PRODUCTS[locale]).to(DEVICE)
    model.train(net, train_loader, test_loader, mrr,  DEVICE)
else:
    valid_de_set = SessionsDataset('./data', 'DE', 'valid')
    valid_jp_set = SessionsDataset('./data', 'JP', 'valid')
    valid_uk_set = SessionsDataset('./data', 'UK', 'valid')
    valid_de_loader = DataLoader(valid_de_set, batch_size=BATCH_SIZE, collate_fn=collate_fn_valid, num_workers=8)
    valid_jp_loader = DataLoader(valid_jp_set, batch_size=BATCH_SIZE, collate_fn=collate_fn_valid, num_workers=8)
    valid_uk_loader = DataLoader(valid_uk_set, batch_size=BATCH_SIZE, collate_fn=collate_fn_valid, num_workers=8)
    de_net = model.SelfAttentionNetwork(hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, n_node=DE_PRODUCTS).to(DEVICE)
    jp_net = model.SelfAttentionNetwork(hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, n_node=JP_PRODUCTS).to(DEVICE)
    uk_net = model.SelfAttentionNetwork(hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, n_node=UK_PRODUCTS).to(DEVICE)
    de_net.load_state_dict(torch.load('params/recc_model_de.pt'))
    jp_net.load_state_dict(torch.load('params/recc_model_jp.pt'))
    uk_net.load_state_dict(torch.load('params/recc_model_uk.pt'))
    loaders = [valid_de_loader, valid_jp_loader, valid_uk_loader]
    sets = [valid_de_set, valid_jp_set, valid_uk_set]
    models = [de_net, jp_net, uk_net]
    model.eval(loaders, sets, models, DEVICE)
