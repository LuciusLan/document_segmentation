import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm
import nltk

import os
import re

TRAIN_PATH = './/train'
TEST_PATH = './/test'
TRAIN_LABEL = './/train.csv'
SUBMISSION = './/sample_submission.csv'
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

train_doc_ids = []
train_doc_texts = []
for f in tqdm(list(os.listdir(TRAIN_PATH))):
    train_doc_ids.append(f.replace('.txt', ''))
    train_doc_texts.append(open(os.path.join(TRAIN_PATH, f), 'r', encoding='utf-8').read())

test_doc_ids = []
test_doc_texts = []
for f in tqdm(list(os.listdir(TEST_PATH))):
    test_doc_ids.append(f.replace('.txt', ''))
    test_doc_texts.append(open(os.path.join(TEST_PATH, f), 'r', encoding='utf-8').read())

train_labels = pd.read_csv(TRAIN_LABEL)

def preprocessing_train(doc_id: str, labels: pd.DataFrame, raw_text: str) -> "tuple[list, pd.Series]":
    new_segements = []
    for row_num, segment in labels.iterrows():
        seg = []
        seg_subwords = []
        ids = segment['predictionstring'].split(' ')
        ids = [int(e) for e in ids]
        seg = raw_text[int(segment['discourse_start']): int(segment['discourse_end'])]
        # Find position of end of sent, augment with [SEP] token
        """fs_pos = []
        for i, word in enumerate(seg):
            # Search for . tokens following more than one non-digit token (to filter out abbr like U.S.A)
            if re.search('\D{2,}[\.?!;]', word):
                fs_pos.append(i)
        fs_count = 0
        for pos in fs_pos:
            seg.insert(pos+fs_count+1, '[SEP]')
            fs_count += 1"""
        temp_sents = nltk.tokenize.sent_tokenize(seg)
        temp_sents = [sent+' [SEP]' for sent in temp_sents]
        temp_sents = ' '.join(temp_sents)
        # BERT tokenization
        for word in temp_sents:
            subwords = TOKENIZER.tokenize(word)
            seg_subwords.extend(subwords)
        # If last token of a segment is not [SEP], calibrate
        try:
            if len(hold) > 0:
                seg_subwords = hold + seg_subwords
                hold = []
        except NameError:
            pass
        hold = []
        if seg_subwords[-1] != '[SEP]':
            temp = seg_subwords.copy()
            temp.reverse()
            last_sep = len(seg_subwords) - temp.index('[SEP]') - 1
            hold.extend(seg_subwords[last_sep+1:])
            seg_subwords = seg_subwords[:last_sep+1]
        new_segements.append(seg_subwords)
    if len(hold) > 0:
        new_segements[-1].extend(hold)
    assert len(new_segements) == len(labels['discourse_type'])
    return new_segements, labels['discourse_type']

preprocessing_train('0A0AA9C21C5D', train_labels.loc[train_labels['id']=='0A0AA9C21C5D'], train_doc_texts[train_doc_ids.index('0A0AA9C21C5D')])

class DocFeature():
    def __init__(self, doc_id, raw_text, seg_labels, train_or_test) -> None:
        self.doc_id = doc_id
        if train_or_test == 'train':
            self.segments, self.labels = preprocessing_train(doc_id, seg_labels, raw_text)
        elif train_or_test == 'test' or train_or_test == 'dev':
            pass
        else:
            raise NameError('Should be either train/dev/test')