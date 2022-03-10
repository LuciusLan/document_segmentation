from params import *
from data import DocFeature, create_tensor_ds, create_tensor_ds_sliding_window
from model import TModel, FocalLoss

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AdamW
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.profiler import profile
from torch.cuda.amp import autocast
from tqdm import tqdm

import itertools
import random
import re
import os
import gc
import math

TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")
# Add special token for new paragraph
TOKENIZER.add_special_tokens({'additional_special_tokens': ['[NP]']})

all_doc_ids = []
all_doc_texts = []
for f in tqdm(list(os.listdir(TRAIN_PATH))):
    all_doc_ids.append(f.replace('.txt', ''))
    all_doc_texts.append(open(os.path.join(TRAIN_PATH, f),
                         'r', encoding='utf-8').read())

test_doc_ids = []
test_doc_texts = []
for f in tqdm(list(os.listdir(TEST_PATH))):
    test_doc_ids.append(f.replace('.txt', ''))
    test_doc_texts.append(
        open(os.path.join(TEST_PATH, f), 'r', encoding='utf-8').read())

all_labels = pd.read_csv(TRAIN_LABEL)

def del_list_idx(l, id_to_del):
    arr = np.array(l, dtype='int32')
    return list(np.delete(arr, id_to_del))


scope_len = len(all_doc_ids)
train_len = math.floor((1 - TEST_SIZE - DEV_SIZE) * scope_len)
dev_len = scope_len - train_len
scope_index = list(range(scope_len))
train_index = random.sample(scope_index, k=train_len)
scope_index = del_list_idx(scope_index, train_index)
dev_index = random.sample(list(range(len(scope_index))), k=dev_len)


train_doc_ids = [all_doc_ids[i] for i in train_index]
dev_doc_ids = [all_doc_ids[i] for i in dev_index]
train_doc_texts = [all_doc_texts[i] for i in train_index]
dev_doc_texts = [all_doc_texts[i] for i in dev_index]


# Due to design of huggingface's tokenizer, not possible to multithread to speed up the loading
# Better run once and load for future development
#train_features = torch.load('train_features.pt')
#dev_features = torch.load('dev_features.pt')
#temp_test_features = torch.load('temp_test_features.pt')

t = DocFeature(doc_id='6C3B801F92D2', seg_labels=all_labels.loc[all_labels['id'] == '6C3B801F92D2'],
                             raw_text=all_doc_texts[all_doc_ids.index('6C3B801F92D2')], train_or_test='train', tokenizer=TOKENIZER)

#all_features = [DocFeature(doc_id=ids, seg_labels=all_labels.loc[all_labels['id'] == ids],
#                             raw_text=all_doc_texts[all_doc_ids.index(ids)], train_or_test='train', tokenizer=TOKENIZER) for ids in tqdm(all_doc_ids)]
train_features = [DocFeature(doc_id=ids, seg_labels=all_labels.loc[all_labels['id'] == ids],
                             raw_text=train_doc_texts[train_doc_ids.index(ids)], train_or_test='train', tokenizer=TOKENIZER) for ids in tqdm(train_doc_ids)]
dev_features = [DocFeature(doc_id=ids, seg_labels=all_labels.loc[all_labels['id'] == ids],
                           raw_text=dev_doc_texts[dev_doc_ids.index(ids)], train_or_test='train', tokenizer=TOKENIZER) for ids in dev_doc_ids]
test_features = [DocFeature(doc_id=ids, raw_text=test_doc_texts[test_doc_ids.index(
    ids)], train_or_test='test', tokenizer=TOKENIZER) for ids in test_doc_ids]


train_ds = create_tensor_ds_sliding_window(train_features)
dev_ds = create_tensor_ds_sliding_window(dev_features)
train_sp = RandomSampler(train_ds)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sp)


config = AutoConfig.from_pretrained("roberta-base")
config.num_labels = NUM_LABELS
model = TModel(config=config)
model = model.to('cuda')
model.transformer.resize_token_embeddings(len(TOKENIZER))
criterion = FocalLoss(ignore_index=0, gamma=2)

bert_param_optimizer = list(model.transformer.named_parameters())
scope_fc_param_optimizer = list(model.classifier.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
        'lr': LEARNING_RATE},
    {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
        'lr': LEARNING_RATE},
    {'params': [p for n, p in scope_fc_param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
        'lr': LEARNING_RATE},
    {'params': [p for n, p in scope_fc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
        'lr': LEARNING_RATE},
]
t_total = int(len(train_dl) / 1 * NUM_EPOCH)
optimizer = AdamW(params=optimizer_grouped_parameters, lr=LEARNING_RATE)


class AverageMeter(object):
    '''
    computes and stores the average and current value
    Example:
        >>> loss = AverageMeter()
        >>> for step,batch in enumerate(train_data):
        >>>     pred = self.model(batch)
        >>>     raw_loss = self.metrics(pred,target)
        >>>     loss.update(raw_loss.item(),n = 1)
        >>> cur_loss = loss.avg
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

for i in range(NUM_EPOCH):
    torch.cuda.empty_cache()
    model.train()
    pbar = tqdm(total=len(train_dl), desc='Train')
    train_loss = AverageMeter()
    for step, batch in enumerate(train_dl):
        optimizer.zero_grad()
        input_ids, labels, attention_masks, subword_masks = tuple(t.cuda() for t in batch)
        active_padding_mask = attention_masks.view(-1) == 1
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_masks)
            loss = criterion(
                logits.view(-1, NUM_LABELS)[active_padding_mask], labels.view(-1)[active_padding_mask])
        loss.backward()
        optimizer.step()
        #torch.cuda.empty_cache()
        gc.collect()
        train_loss.update(loss.item(), n=input_ids.size(0))
        pbar.update()
        pbar.set_postfix({'loss': train_loss.avg})
    print(train_loss.avg)

    model.eval()
    valid_loss = AverageMeter()
    pbar = tqdm(total=len(dev_ds), desc='Eval')
    for batch in dev_ds:
        input_ids, labels, attention_masks, subword_masks = tuple(t.cuda() for t in batch)
        input_ids = input_ids.unsqueeze(0)
        attention_masks = attention_masks.unsqueeze(0)
        active_padding_mask = attention_masks.view(-1) == 1
        logits = model(input_ids=input_ids, attention_mask=attention_masks)
        loss = criterion(
            logits.view(-1, NUM_LABELS)[active_padding_mask], labels.view(-1)[active_padding_mask])
        #torch.cuda.empty_cache()
        gc.collect()
        valid_loss.update(val=loss.item(), n=1)
        pbar.update()
        pbar.set_postfix({'loss': valid_loss.avg})