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
import six
import random
import os
import re
import math

TRAIN_PATH = './/train'
TEST_PATH = './/test'
TRAIN_LABEL = './/train.csv'
SUBMISSION = './/sample_submission.csv'
TOKENIZER = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", add_prefix_space=True)
# Add special token for new paragraph
TOKENIZER.add_special_tokens({'additional_special_tokens': ['[NP]']})
LABEL_2_ID = {'Claim': 1, 'Evidence': 2, 'Position': 3,
              'Concluding Statement': 4, 'Lead': 5, 'Counterclaim': 6, 'Rebuttal': 7, 'non': 8}
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


def preprocessing_train(labels: pd.DataFrame, raw_text: str) -> "tuple[list]":
    """
    Tokenization for training. Insert [NP] tokens at new paragraph
    
    Args:
        labels: the DataFrame containing label information.
        raw_text: the raw text input as string
    
    Returns:
        new_segements: list of encoded tokenized inputs, organized in segments
        discourse_type: list of segments' type
        subword_mask: list of subword masks (for post-processing)
    """
    new_segements = []
    prev_start = -1
    hold_count = 0
    discourse_type = labels['discourse_type'].to_list()
    row_num = 0
    subword_mask = []
    for _, segment in labels.iterrows():
        seg_ids = []
        seg_subword_mask = []
        # Find is there any text before current discourse start and previous discourse end
        if prev_start + 1 != int(segment['discourse_start']) and prev_start != int(segment['discourse_start']):
            hold_seg = raw_text[prev_start+1: int(segment['discourse_start'])]
            discourse_type.insert(hold_count+row_num, 'non')
            hold_seg_ids = []
            hold_subword_mask = []
            temp_sents = nltk.tokenize.sent_tokenize(hold_seg)
            # BERT tokenization
            for sent in temp_sents:
                tokenized_hold = TOKENIZER(sent)
                hold_seg_ids.extend(tokenized_hold['input_ids'])
                hold_subword_mask.extend(tokenized_hold.word_ids(0))
            hold_count += 1
            new_segements.append(hold_seg_ids)
            subword_mask.append(hold_subword_mask)
        seg = raw_text[int(segment['discourse_start']): int(segment['discourse_end'])]
        if re.search('\n+', raw_text[int(segment['discourse_start'])-4:int(segment['discourse_start'])]):
            seg = '[NP] '+seg
        seg = re.sub('\n+', ' [NP] ', seg)
        temp_sents = nltk.tokenize.sent_tokenize(seg)
        # BERT tokenization
        for sent in temp_sents:
            tokenized_sent = TOKENIZER(sent)
            seg_ids.extend(tokenized_sent['input_ids'])
            seg_subword_mask.extend(tokenized_sent.word_ids(0))
        new_segements.append(seg_ids)
        subword_mask.append(seg_subword_mask)
        prev_start = int(segment['discourse_end'])
        row_num += 1

    # Find is there any text after the last discourse end
    if int(segment['discourse_end']) < len(raw_text):
        hold_seg_ids = []
        hold_subword_mask = []
        hold_seg = raw_text[int(segment['discourse_end']):]
        discourse_type.append('non')
        hold_seg = re.sub('\n+', ' [NP] ', hold_seg)
        # Find position of end of sent, augment with [SEP] token
        temp_sents = nltk.tokenize.sent_tokenize(hold_seg)
        # BERT tokenization
        for sent in temp_sents:
            tokenized_hold = TOKENIZER(sent)
            hold_seg_ids.extend(tokenized_hold['input_ids'])
            hold_subword_mask.extend(tokenized_hold.word_ids(0))
        new_segements.append(hold_seg_ids)
        subword_mask.append(hold_subword_mask)
    assert len(new_segements) == len(discourse_type) == len(subword_mask)
    return new_segements, discourse_type, subword_mask


def preprocessing_test(raw_text: str) -> "tuple[list]":
    """
    Tokenization or testing (without ground truth), simply tokenize and output subword mask
    Need to take care of [NP] tokens when decoding
    """
    ids = []
    subword_mask = []
    raw_text = re.sub('\n+', ' [NP] ', raw_text)
    temp_sents = nltk.tokenize.sent_tokenize(raw_text)
    for sent in temp_sents:
        tokenized_sent = TOKENIZER(sent)
        ids.extend(tokenized_sent['input_ids'])
        subword_mask.extend(tokenized_sent.word_ids(0))
    return ids, subword_mask



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
        elif train_or_test == 'test':
            self.segments = preprocessing_test(doc_id, raw_text)
        else:
            raise NameError('Should be either train/test')
        self.tokens = list(itertools.chain.from_iterable(self.segments))
        self.input_ids = TOKENIZER.convert_tokens_to_ids(self.tokens)


train_doc_ids = [all_doc_ids[i] for i in train_index]
dev_doc_ids = [all_doc_ids[i] for i in dev_index]
temp_test_doc_ids = [all_doc_ids[i] for i in test_index]
train_doc_texts = [all_doc_texts[i] for i in train_index]
dev_doc_texts = [all_doc_texts[i] for i in dev_index]
temp_test_doc_texts = [all_doc_texts[i] for i in test_index]

t = DocFeature(doc_id='0A0AA9C21C5D', seg_labels=train_labels.loc[train_labels['id'] == '0A0AA9C21C5D'],
                             raw_text=all_doc_texts[all_doc_ids.index('0A0AA9C21C5D')], train_or_test='train')

train_features = [DocFeature(seg_labels=train_labels.loc[train_labels['id'] == ids],
                             raw_text=train_doc_texts[train_doc_ids.index(ids)], train_or_test='train') for ids in tqdm(train_doc_ids)]
dev_features = [DocFeature(seg_labels=train_labels.loc[train_labels['id'] == ids],
                           raw_text=dev_doc_texts[dev_doc_ids.index(ids)], train_or_test='train') for ids in dev_doc_ids]
temp_test_features = [DocFeature(seg_labels=train_labels.loc[train_labels['id'] == ids],
                                 raw_text=temp_test_doc_texts[temp_test_doc_ids.index(ids)], train_or_test='train') for ids in temp_test_doc_ids]
test_features = [DocFeature(raw_text=test_doc_texts[test_doc_ids.index(
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
model.resize_token_embeddings(len(TOKENIZER))
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
