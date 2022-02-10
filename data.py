from params import *
import pandas as pd
import nltk
import numpy as np
import torch
from torch.utils.data import TensorDataset

import re
import itertools
import six

def preprocessing_train(labels: pd.DataFrame, raw_text: str, tokenizer) -> "tuple[list]":
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
                tokenized_hold = tokenizer(sent)
                hold_seg_ids.extend(tokenized_hold['input_ids'])
                hold_subword_mask.extend(tokenized_hold.word_ids(0))
            hold_count += 1
            new_segements.append(hold_seg_ids)
            subword_mask.append(hold_subword_mask)
        seg = raw_text[int(segment['discourse_start'])                       : int(segment['discourse_end'])]
        if re.search('\n+', raw_text[int(segment['discourse_start'])-4:int(segment['discourse_start'])]):
            seg = '[NP] '+seg
        seg = re.sub('\n+', ' [NP] ', seg)
        temp_sents = nltk.tokenize.sent_tokenize(seg)
        # BERT tokenization
        for sent in temp_sents:
            tokenized_sent = tokenizer(sent)
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
            tokenized_hold = tokenizer(sent)
            hold_seg_ids.extend(tokenized_hold['input_ids'])
            hold_subword_mask.extend(tokenized_hold.word_ids(0))
        new_segements.append(hold_seg_ids)
        subword_mask.append(hold_subword_mask)
    assert len(new_segements) == len(discourse_type) == len(subword_mask)
    return new_segements, discourse_type, subword_mask


def preprocessing_test(raw_text: str, tokenizer) -> "tuple[list]":
    """
    Tokenization or testing (without ground truth), simply tokenize and output subword mask
    Need to take care of [NP] tokens when decoding
    """
    ids = []
    subword_mask = []
    raw_text = re.sub('\n+', ' [NP] ', raw_text)
    temp_sents = nltk.tokenize.sent_tokenize(raw_text)
    for sent in temp_sents:
        tokenized_sent = tokenizer(sent)
        ids.extend(tokenized_sent['input_ids'])
        subword_mask.extend(tokenized_sent.word_ids(0))
    return ids, subword_mask


class DocFeature():
    def __init__(self, doc_id: str, raw_text: str, train_or_test: str, seg_labels=None, tokenizer=None) -> None:
        self.doc_id = doc_id
        if train_or_test == 'train':
            self.input_ids, self.seg_labels, self.subword_masks = preprocessing_train(
                seg_labels, raw_text, tokenizer=tokenizer)
            label_ids = [LABEL_2_ID[seg] for seg in self.seg_labels]
            #self.labels = [[label]*len(seg) for seg, label in zip(self.input_ids, label_ids)]
            self.labels = [self.convert_label_to_bio(label, len(
                seg)) for seg, label in zip(self.input_ids, label_ids)]
            self.labels = list(itertools.chain.from_iterable(self.labels))
            self.input_ids = list(
                itertools.chain.from_iterable(self.input_ids))
            self.subword_masks = list(
                itertools.chain.from_iterable(self.subword_masks))
            self.subword_masks = [
                0 if e is None else 1 for e in self.subword_masks]
        elif train_or_test == 'test':
            self.input_ids, self.subword_masks = preprocessing_test(raw_text, tokenizer=tokenizer)
            self.subword_masks = [
                0 if e is None else 1 for e in self.subword_masks]
        else:
            raise NameError('Should be either train/test')

    def convert_label_to_bio(self, label, seq_len):
        if label != 8:
            temp = [LABEL_BIO[f'I{label}']]*seq_len
            temp[0] = LABEL_BIO[f'B{label}']
        else:
            temp = [LABEL_BIO['O']]*seq_len
        return temp


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
    subword_masks = []
    for feat in features:
        input_ids.append(feat.input_ids)
        labels.append(feat.labels)
        attention_masks.append([1]*len(feat.input_ids))
        subword_masks.append(feat.subword_masks)
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
    subword_masks = pad_sequences(subword_masks,
                                  maxlen=MAX_LEN, value=0, padding="post",
                                  dtype="long", truncating="post").tolist()
    subword_masks = torch.LongTensor(subword_masks)
    return TensorDataset(input_ids, labels, attention_masks, subword_masks)