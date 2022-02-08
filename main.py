from datasets import cached_path
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
import nltk

import itertools
import concurrent.futures
import random
import os
import re
import math

TRAIN_PATH = './/train'
TEST_PATH = './/test'
TRAIN_LABEL = './/train.csv'
SUBMISSION = './/sample_submission.csv'
TOKENIZER = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
LABEL_2_ID = {'Claim': 1, 'Evidence': 2, 'Position': 3,
              'Concluding Statement': 4, 'Lead': 5, 'Counterclaim': 6, 'Rebuttal': 7}
TEST_SIZE = 0.1
DEV_SIZE = 0.1
MAX_LEN = 1024
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

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

train_labels = pd.read_csv(TRAIN_LABEL)

def del_list_idx(l, id_to_del):
    arr = np.array(l, dtype='int32')
    return list(np.delete(arr, id_to_del))

scope_len = len(all_doc_ids)
train_len = math.floor((1 - TEST_SIZE - DEV_SIZE) * scope_len)
test_len = math.floor(TEST_SIZE * scope_len)
val_len = scope_len - train_len - test_len
scope_index = list(range(scope_len))
train_index = random.sample(scope_index, k=train_len)
scope_index = del_list_idx(scope_index, train_index)
test_index = random.sample(list(range(len(scope_index))), k=test_len)
scope_index = del_list_idx(scope_index, test_index)
dev_index = scope_index.copy()


def preprocessing_train(doc_id: str, labels: pd.DataFrame, raw_text: str) -> "tuple[list, pd.Series]":
    new_segements = []
    for row_num, segment in labels.iterrows():
        seg = []
        seg_subwords = []
        ids = segment['predictionstring'].split(' ')
        ids = [int(e) for e in ids]
        seg = raw_text[int(segment['discourse_start']): int(segment['discourse_end'])]
        if re.search('\n+', raw_text[int(segment['discourse_start'])-4:int(segment['discourse_start'])]):
            seg = '[CLS] '+seg
        seg = re.sub('\n+', ' [CLS] ', seg)
        # Find position of end of sent, augment with [SEP] token
        temp_sents = nltk.tokenize.sent_tokenize(seg)
        temp_sents = [sent+' [SEP]' for sent in temp_sents]
        # BERT tokenization
        for word in temp_sents:
            subwords = TOKENIZER.tokenize(word)
            seg_subwords.extend(subwords)
        new_segements.append(seg_subwords)
    assert len(new_segements) == len(labels['discourse_type'])
    return new_segements, labels['discourse_type']


def preprocessing_test(doc_id: str, raw_text: str) -> list:
    new_segements = []
    raw_text = re.sub('\n+', ' [CLS] ', raw_text)
    # Find position of end of sent, augment with [SEP] token
    temp_sents = nltk.tokenize.sent_tokenize(raw_text)
    temp_sents = [sent+' [SEP]' for sent in temp_sents]
    # BERT tokenization
    for word in temp_sents:
        subwords = TOKENIZER.tokenize(word)
        new_segements.extend(subwords)
    return new_segements


class DocFeature():
    def __init__(self, doc_id: str, raw_text: str, train_or_test: str, seg_labels=None) -> None:
        self.doc_id = doc_id
        if train_or_test == 'train':
            self.segments, self.seg_labels = preprocessing_train(
                doc_id, seg_labels, raw_text)
            label_ids = [LABEL_2_ID[seg] for seg in self.seg_labels]
            self.labels = [[label]*len(seg)
                           for seg, label in zip(self.segments, label_ids)]
            self.labels = list(itertools.chain.from_iterable(self.labels))
        elif train_or_test == 'dev':
            self.segments = preprocessing_test(doc_id, raw_text)
            self.seg_labels = seg_labels['discourse_type']
        elif train_or_test == 'test':
            self.segments = preprocessing_test(doc_id, raw_text)
        else:
            raise NameError('Should be either train/dev/test')
        self.tokens = list(itertools.chain.from_iterable(self.segments))
        self.input_ids = TOKENIZER.convert_tokens_to_ids(self.tokens)


train_doc_ids = [all_doc_ids[i] for i in train_index]
dev_doc_ids = [all_doc_ids[i] for i in dev_index]
temp_test_doc_ids = [all_doc_ids[i] for i in test_index]
train_doc_texts = [all_doc_texts[i] for i in train_index]
dev_doc_texts = [all_doc_texts[i] for i in dev_index]
temp_test_doc_texts = [all_doc_texts[i] for i in test_index]

train_features = [DocFeature(doc_id=ids, seg_labels=train_labels.loc[train_labels['id'] == ids],
                             raw_text=train_doc_texts[train_doc_ids.index(ids)], train_or_test='train') for ids in tqdm(train_doc_ids)]
dev_features = [DocFeature(doc_id=ids, seg_labels=train_labels.loc[train_labels['id'] == ids],
                           raw_text=dev_doc_texts[dev_doc_ids.index(ids)], train_or_test='dev') for ids in dev_doc_ids]
temp_test_features = [DocFeature(doc_id=ids, seg_labels=train_labels.loc[train_labels['id'] == ids],
                                 raw_text=temp_test_doc_texts[temp_test_doc_ids.index(ids)], train_or_test='dev') for ids in temp_test_doc_ids]
test_features = [DocFeature(doc_id=ids, raw_text=test_doc_texts[test_doc_ids.index(
    ids)], train_or_test='test') for ids in test_doc_ids]


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """
    pad_sequences function from keras. 

    Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the beginning or the end
    if padding='post.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(
        dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def create_tensor_ds(features: "list[DocFeature]") -> TensorDataset:
    input_ids = []
    labels = []
    attention_masks = []
    for feat in features:
        input_ids.append(feat.input_ids)
        labels.append(feat.labels)
        attention_masks.append([1]*len(feat.input_ids))
    input_ids = pad_sequences(input_ids,
                              maxlen=MAX_LEN, value=0, padding="post",
                              dtype="long", truncating="post").tolist()
    input_ids = torch.LongTensor(input_ids)
    labels = pad_sequences(labels,
                           maxlen=MAX_LEN, value=0, padding="post",
                           dtype="long", truncating="post").tolist()
    labels = torch.LongTensor(labels)
    attention_masks = pad_sequences(attention_masks,
                                    maxlen=MAX_LEN, value=0, padding="post",
                                    dtype="long", truncating="post").tolist()
    attention_masks = torch.LongTensor(attention_masks)
    return TensorDataset(input_ids, labels, attention_masks)


train_ds = create_tensor_ds(train_features)
dev_ds = create_tensor_ds(dev_features)
temp_test_ds = create_tensor_ds(temp_test_features)
train_sp = RandomSampler(train_ds)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sp)


class TModel(AutoModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.transformer = AutoModel(config)
        self.dropout = nn.Dropout()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024), self.dropout, nn.Linear(1024, 8))

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.classifier(sequence_output)
        return sequence_output


config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
config.num_labels = 8
model = TModel.from_pretrained(
    "allenai/longformer-base-4096", cached_path="D:\\Dev\\Longformer\\").cuda()
criterion = nn.CrossEntropyLoss(ingore_index=0)

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
optimizer = AdamW()

model.train()
for step, batch in tqdm(enumerate(train_dl)):
    input_ids, labels, attention_masks = tuple(t.cuda() for t in batch)
    active_padding_mask = attention_masks.view(-1) == 1
    logits = model(input_ids, attention_masks)
    loss = criterion(
        logits.view(-1, 8)[active_padding_mask], labels.view(-1)[active_padding_mask])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
