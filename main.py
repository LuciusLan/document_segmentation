from params import *
from data import DocFeature, create_tensor_ds_sliding_window, create_tensor_ds_sliding_window_test
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
from sklearn.metrics import classification_report

import itertools
import random
import re
import os
import gc
import math

TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", cache_dir=MODEL_CACHE_DIR)
# Add special token for new paragraph
TOKENIZER.add_special_tokens({'additional_special_tokens': ['[NP]']})

def bound_to_matrix(bound: torch.Tensor) -> torch.LongTensor:
    """
    To convert the boundary list to a adjacency matrix (like) that represents
    the segment span.
    1: start->end
    2: end->start (or simply become 0 to ignore backward link)

    input:
        bound: [batch_size, seq_length]
    """
    bs = bound.size(0)
    mat = torch.zeros([bs, MAX_LEN, MAX_LEN], dtype=torch.long)
    for b, seq in enumerate(bound):
        for i, e in enumerate(seq):
            if e == 1:
                for j in range(i, MAX_LEN):
                    if seq[j] == 2:
                        mat[b][i][j] = 1
                        break
    return mat

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


#t = DocFeature(doc_id='00C819ADE423', seg_labels=all_labels.loc[all_labels['id'] == '00C819ADE423'],
#                             raw_text=all_doc_texts[all_doc_ids.index('00C819ADE423')], train_or_test='test', tokenizer=TOKENIZER)

#all_features = [DocFeature(doc_id=ids, seg_labels=all_labels.loc[all_labels['id'] == ids],
#                           raw_text=all_doc_texts[all_doc_ids.index(ids)], train_or_test='test', tokenizer=TOKENIZER) for ids in tqdm(all_doc_ids)]


try:
    print('Dataset Loading...')
    train_ds = torch.load('train_ds.pt')
    dev_ds = torch.load('dev_ds.pt')
    test_ds = torch.load('test_ds.pt')

except FileNotFoundError:
    print('Create Dataset')
    train_features = [DocFeature(doc_id=ids, seg_labels=all_labels.loc[all_labels['id'] == ids],
                                raw_text=train_doc_texts[train_doc_ids.index(ids)], train_or_test='train', tokenizer=TOKENIZER) for ids in tqdm(train_doc_ids)]
    dev_features = [DocFeature(doc_id=ids, seg_labels=all_labels.loc[all_labels['id'] == ids],
                            raw_text=dev_doc_texts[dev_doc_ids.index(ids)], train_or_test='train', tokenizer=TOKENIZER) for ids in tqdm(dev_doc_ids)]
    test_features = [DocFeature(doc_id=ids, raw_text=test_doc_texts[test_doc_ids.index(
        ids)], train_or_test='test', tokenizer=TOKENIZER) for ids in test_doc_ids]

    train_ds = create_tensor_ds_sliding_window(train_features)
    dev_ds = create_tensor_ds_sliding_window(dev_features)
    test_ds = create_tensor_ds_sliding_window_test(dev_features)

    print('Dataset Saving...')
    torch.save(train_ds, 'train_ds.pt')
    torch.save(dev_ds,'dev_ds.pt')
    torch.save(test_ds, 'test_ds.pt')



train_sp = RandomSampler(train_ds)
dev_sp = RandomSampler(dev_ds)
def custom_batch_collation(x):
    num_elements = len(x[0])
    return_tup = [[] for _ in range(num_elements)]
    for row in x:
        for i, e in enumerate(row):
            return_tup[i].append(e)
    return return_tup

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sp, collate_fn=custom_batch_collation)
dev_dl = DataLoader(dev_ds, batch_size=2, sampler=dev_sp, collate_fn=custom_batch_collation)


config = AutoConfig.from_pretrained("roberta-base")
config.num_labels = NUM_LABELS
model = TModel(config=config)
#model = torch.load('model.pt')
model = model.to('cuda')
model.transformer.resize_token_embeddings(len(TOKENIZER))

bio_cls_weights = torch.Tensor([0, 100, 10, 100, 10, 100, 10, 100, 10, 100, 10, 100, 10, 100, 10, 5]).cuda()
bio_loss = FocalLoss(ignore_index=0, gamma=2, alpha=bio_cls_weights)
boundary_loss = FocalLoss(ignore_index=0, gamma=2, alpha=torch.Tensor([0,10,10,1]).cuda())
type_loss = FocalLoss(ignore_index=0, gamma=2)
seg_loss = FocalLoss(ignore_index=0, gamma=2)

bert_param_optimizer = list(model.transformer.named_parameters())
ner_fc_param_optimizer = list(model.plain_ner.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

if not BASELINE:
    boundary_param_optimizer = list(model.boundary.named_parameters())
    type_param_optimizer = list(model.type_predict.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
            'lr': LEARNING_RATE},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': LEARNING_RATE},
        {'params': [p for n, p in ner_fc_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': LEARNING_RATE},
        {'params': [p for n, p in ner_fc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': LEARNING_RATE},
        {'params': [p for n, p in boundary_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': LEARNING_RATE},
        {'params': [p for n, p in boundary_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': LEARNING_RATE},
        {'params': [p for n, p in type_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': LEARNING_RATE},
        {'params': [p for n, p in type_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': LEARNING_RATE},
    ]
else:
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
            'lr': LEARNING_RATE},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': LEARNING_RATE},
        {'params': [p for n, p in ner_fc_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': LEARNING_RATE},
        {'params': [p for n, p in ner_fc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
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
        bs = len(batch[0])
        optimizer.zero_grad()
        input_ids, labels_type, labels_bio, labels_boundary, attention_masks, subword_masks, cls_pos, sliding_window_pos = batch 
        input_ids = torch.stack(input_ids).cuda()
        labels_type = torch.stack(labels_type).cuda()
        labels_bio = torch.stack(labels_bio).cuda()
        labels_boundary = torch.stack(labels_boundary).cuda()
        attention_masks = torch.stack(attention_masks).cuda()
        subword_masks = torch.stack(subword_masks).cuda()
        active_padding_mask = attention_masks.view(-1) == 1

        boundary_matrix = bound_to_matrix(labels_boundary).cuda()
        pad_matrix = []
        for i in range(bs):
            tmp = attention_masks[i].clone()
            tmp = tmp.view(MAX_LEN, 1)
            tmp_t = tmp.transpose(0, 1)
            mat = tmp * tmp_t
            pad_matrix.append(mat)
        pad_matrix = torch.stack(pad_matrix, 0)
        matrix_padding_mask = pad_matrix.view(-1) == 1
        with autocast():
            #with torch.autograd.profiler.profile(use_cuda=True) as prof:
            ner_logits, boundary_logits, type_logits, seg_logits = model(input_ids=input_ids, attention_mask=attention_masks)
            ner_loss_ = bio_loss(
                ner_logits.view(-1, len(LABEL_BIO))[active_padding_mask], labels_bio.view(-1)[active_padding_mask])
            boundary_loss_ = boundary_loss(boundary_logits.view(-1, len(BOUNDARY_LABEL))[active_padding_mask], labels_boundary.view(-1)[active_padding_mask])
            type_loss_ = type_loss(type_logits.view(-1, len(LABEL_2_ID))[active_padding_mask], labels_type.view(-1)[active_padding_mask])
            seg_loss_ = seg_loss(seg_logits.view(-1, len(BOUNDARY_LABEL_UNIDIRECTION))[matrix_padding_mask], boundary_matrix.view(-1)[matrix_padding_mask])
            loss = ner_loss_+boundary_loss_+type_loss_+seg_loss_
                
            #print(prof.key_averages().table())
            loss.backward()
            optimizer.step()
        #torch.cuda.empty_cache()
        #gc.collect()
        train_loss.update(loss.item(), n=input_ids.size(0))
        pbar.update()
        pbar.set_postfix({'loss': train_loss.avg})
    print(train_loss.avg)
    
    model.eval()
    valid_loss = AverageMeter()
    pbar = tqdm(total=len(dev_dl), desc='Eval')
    preds = []
    targets = []
    for batch in dev_dl:
        bs = len(batch[0])
        input_ids, labels_type, labels_bio, labels_boundary, attention_masks, subword_masks, cls_pos, sliding_window_pos = batch 
        input_ids = torch.stack(input_ids).cuda()
        labels_type = torch.stack(labels_type).cuda()
        labels_bio = torch.stack(labels_bio).cuda()
        labels_boundary = torch.stack(labels_boundary).cuda()
        attention_masks = torch.stack(attention_masks).cuda()
        subword_masks = torch.stack(subword_masks).cuda()
        active_padding_mask = attention_masks.view(-1) == 1
        
        boundary_matrix = bound_to_matrix(labels_boundary).cuda()
        pad_matrix = []
        for i in range(bs):
            tmp = attention_masks[i].clone()
            tmp = tmp.view(MAX_LEN, 1)
            tmp_t = tmp.transpose(0, 1)
            mat = tmp * tmp_t
            pad_matrix.append(mat)
        pad_matrix = torch.stack(pad_matrix, 0)
        matrix_padding_mask = pad_matrix.view(-1) == 1

        with torch.no_grad():
            with autocast():
                ner_logits, boundary_logits, type_logits, seg_logits = model(input_ids=input_ids, attention_mask=attention_masks)
                ner_loss_ = bio_loss(
                    ner_logits.view(-1, len(LABEL_BIO))[active_padding_mask], labels_bio.view(-1)[active_padding_mask])
                boundary_loss_ = boundary_loss(boundary_logits.view(-1, len(BOUNDARY_LABEL))[active_padding_mask], labels_boundary.view(-1)[active_padding_mask])
                type_loss_ = type_loss(type_logits.view(-1, len(LABEL_2_ID))[active_padding_mask], labels_type.view(-1)[active_padding_mask])
                seg_loss_ = seg_loss(seg_logits.view(-1, len(BOUNDARY_LABEL_UNIDIRECTION))[matrix_padding_mask], boundary_matrix.view(-1)[matrix_padding_mask])
                loss = ner_loss_+boundary_loss_+type_loss_+seg_loss_
                
                preds.append(ner_logits.argmax(-1).detach().cpu().tolist())
                targets.append(labels_bio.detach().cpu().tolist())
        #torch.cuda.empty_cache()
        #gc.collect()
        valid_loss.update(val=loss.item(), n=1)
        pbar.update()
        pbar.set_postfix({'loss': valid_loss.avg})
    
    preds=list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(preds))))
    targets=list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(targets))))
    print(classification_report(targets, preds, digits=4))
print()
