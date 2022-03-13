from params import *
import pandas as pd
import nltk
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset

import re
import itertools
import six

sent_tok = nltk.data.load(f"tokenizers/punkt/English.pickle")
re_bos = re.compile(r'^\s?\W?(?:(?:[A-Z]{1}[a-z]+)|(?:I))\s?[a-z]*')
re_eos = re.compile(r'[?\.!]\'?\"?\s*$')


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
    prev_end = -1
    subword_mask = []
    seg_labels = []
    raw_text = raw_text.replace('\xa0', ' ')
    prev_eos = True
    for _, segment in labels.iterrows():
        seg_ids = []
        seg_subword_mask = []
        # Find is there any text before current discourse start and previous discourse end
        # Or any text before the first discourse start
        if (prev_end + 1 != int(segment['discourse_start']) and prev_end != int(segment['discourse_start'])) or (prev_end == -1 and int(segment['discourse_start']) != 0):
            if prev_end == -1:
                hold_seg = raw_text[: int(segment['discourse_start'])]
            elif raw_text[prev_end] in ['.', ' ', '!', '?']:
                hold_seg = raw_text[prev_end +
                                    1: int(segment['discourse_start'])]
            else:
                hold_seg = raw_text[prev_end: int(segment['discourse_start'])]
            hold_seg = re.sub('\n+', ' [NP] ', hold_seg)
            temp_sents = nltk.tokenize.sent_tokenize(hold_seg)
            temp_ids = []
            temp_subword = []
            temp_label = []
            for sent in temp_sents:
                tokenized_hold = tokenizer(sent)
                hold_seg_ids = tokenized_hold['input_ids']
                hold_subword_mask = tokenized_hold.word_ids(0)
                # Remove [CLS] or [SEP] token if segment start is not start of new sentence
                # or segment end not end of sentence
                # If previous segment ends with EOS, assign bos to current segment
                if not re_bos.search(sent) and not prev_eos:
                    hold_seg_ids = hold_seg_ids[1:]
                    hold_subword_mask = hold_subword_mask[1:]
                if not re_eos.search(sent):
                    hold_seg_ids = hold_seg_ids[:-1]
                    hold_subword_mask = hold_subword_mask[:-1]
                    prev_eos = False
                else:
                    prev_eos = True
                temp_label.extend([8]*len(hold_seg_ids))
                temp_ids.extend(hold_seg_ids)
                temp_subword.extend(hold_subword_mask)

            if len(temp_ids) != 0 and len(temp_subword) != 0 and len(temp_label) != 0:
                new_segements.append(temp_ids)
                subword_mask.append(temp_subword)
                seg_labels.append(temp_label)

        seg = raw_text[int(segment['discourse_start']): int(segment['discourse_end'])]
        # Insert special token for New Paragraph (strong indicator for boundary)
        seg = re.sub('\n+', ' [NP] ', seg)

        temp_sents = nltk.tokenize.sent_tokenize(seg)
        temp_ids = []
        temp_subword = []
        temp_label = []
        for sent in temp_sents:
            tokenized_sent = tokenizer(sent)
            seg_ids = tokenized_sent['input_ids']
            seg_subword_mask = tokenized_sent.word_ids(0)
            # Remove [CLS] or [SEP] token if segment start is not start of new sentence
            # or segment end not end of sentence
            if not re_bos.search(sent) and not prev_eos:
                seg_ids = seg_ids[1:]
                seg_subword_mask = seg_subword_mask[1:]
            if not re_eos.search(sent):
                seg_ids = seg_ids[:-1]
                seg_subword_mask = seg_subword_mask[:-1]
                prev_eos = False
            else:
                prev_eos = True
            current_seg_label = [
                LABEL_2_ID[segment['discourse_type']]]*len(seg_ids)
            temp_label.extend(current_seg_label)
            temp_ids.extend(seg_ids)
            temp_subword.extend(seg_subword_mask)
        if len(temp_ids) != 0 and len(temp_subword) != 0 and len(temp_label) != 0:
            seg_labels.append(temp_label)
            new_segements.append(temp_ids)
            subword_mask.append(temp_subword)
        prev_end = int(segment['discourse_end'])

    # Find is there any text after the last discourse end
    if int(segment['discourse_end']) < len(raw_text):
        hold_seg_ids = []
        hold_subword_mask = []
        hold_seg = raw_text[int(segment['discourse_end']):]
        hold_seg = re.sub('\n+', ' [NP] ', hold_seg)
        temp_sents = nltk.tokenize.sent_tokenize(hold_seg)
        temp_ids = []
        temp_subword = []
        temp_label = []
        for sent in temp_sents:
            tokenized_hold = tokenizer(sent)
            hold_seg_ids = tokenized_hold['input_ids']
            hold_subword_mask = tokenized_hold.word_ids(0)
            # Remove [CLS] or [SEP] token if segment start is not start of new sentence
            # or segment end not end of sentence
            if not re_bos.search(sent) and not prev_eos:
                hold_seg_ids = hold_seg_ids[1:]
                hold_subword_mask = hold_subword_mask[1:]
            if not re_eos.search(sent):
                hold_seg_ids = hold_seg_ids[:-1]
                hold_subword_mask = hold_subword_mask[:-1]
                prev_eos = False
            else:
                prev_eos = True
            temp_label.extend([8]*len(hold_seg_ids))
            temp_ids.extend(hold_seg_ids)
            temp_subword.extend(hold_subword_mask)
        if len(temp_ids) != 0 and len(temp_subword) != 0 and len(temp_label) != 0:
            new_segements.append(temp_ids)
            subword_mask.append(temp_subword)
            seg_labels.append(temp_label)

    return new_segements, seg_labels, subword_mask


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


class SlidingWindowFeature():
    def __init__(self, doc_id, input_ids, labels, subword_masks, cls_pos, sliding_window, tokenizer=None) -> None:
        self.doc_id = doc_id
        self.tokenizer = tokenizer
        self.cls_pos = cls_pos
        self.sliding_window = sliding_window
        if sliding_window is not None:
            self.input_ids = [input_ids[start:end]
                              for start, end in sliding_window]
            self.subword_masks = [subword_masks[start:end]
                                  for start, end in sliding_window]
            self.labels = [labels[start:end] for start, end in sliding_window]
            self.num_windows = len(sliding_window)
        else:
            self.input_ids = [input_ids]
            self.subword_masks = [subword_masks]
            self.labels = [labels]
            self.sliding_window = [[0, len(input_ids)]]
            self.num_windows = 1


class DocFeature():
    def __init__(self, doc_id: str, raw_text: str, train_or_test: str, seg_labels=None, tokenizer=None) -> None:
        self.doc_id = doc_id
        self.tokenizer = tokenizer
        if train_or_test == 'train':
            self.input_ids, self.seg_labels, self.subword_masks = preprocessing_train(
                labels=seg_labels, raw_text=raw_text, tokenizer=tokenizer)
            #self.labels = [[label]*len(seg) for seg, label in zip(self.input_ids, label_ids)]
            self.labels = [self.convert_label_to_bio(label, len(
                seg)) for seg, label in zip(self.input_ids, self.seg_labels)]
            self.labels = list(itertools.chain.from_iterable(self.labels))
            self.input_ids = list(
                itertools.chain.from_iterable(self.input_ids))
            self.subword_masks = list(
                itertools.chain.from_iterable(self.subword_masks))
            self.subword_masks = [
                -1 if e is None else e for e in self.subword_masks]
            self.boundary_pos = self.get_boundary_pos()
            self.cls_pos = [index for index, element in enumerate(
                self.input_ids) if element == tokenizer.cls_token_id]
            self.count = self.get_sent_level_label()
            self.sliding_window = self.create_sliding_window_train()
        elif train_or_test == 'test':
            self.input_ids, self.subword_masks = preprocessing_test(
                raw_text, tokenizer=tokenizer)
            self.subword_masks = [
                -1 if e is None else e for e in self.subword_masks]
            self.cls_pos = [index for index, element in enumerate(
                self.input_ids) if element == tokenizer.cls_token_id]
        else:
            raise NameError('Should be either train/test')

    def convert_label_to_bio(self, label, seq_len):
        if label[0] != 8:
            temp = [LABEL_BIO[f'I{label[0]}']]*seq_len
            temp[0] = LABEL_BIO[f'B{label[0]}']
        else:
            temp = [LABEL_BIO['O']]*seq_len
        return temp

    def get_sent_level_label(self):
        prev_cls = 0
        labels = list(itertools.chain.from_iterable(self.seg_labels))
        count = 0
        for pos in self.cls_pos:
            distinct = set(labels[prev_cls:pos])
            if (8 in distinct and len(distinct) > 2) or (8 not in distinct and len(distinct) > 1):
                count += 1
            prev_cls = pos
        return count

    def get_boundary_pos(self):
        boundary = []
        prev = 0
        for seg in self.seg_labels:
            boundary.append(len(seg)+prev)
            prev = len(seg) + prev
        return boundary

    def create_sliding_window_train(self):
        if len(self.input_ids) <= MAX_LEN:
            return SlidingWindowFeature(doc_id=self.doc_id, input_ids=self.input_ids, labels=self.labels, subword_masks=self.subword_masks, cls_pos=self.cls_pos, sliding_window=None)
        else:
            # Create intersection of boundary pos list and cls token pos list, as we can only create slice on cls token, not any boundary
            bound_cls_pos = list(
                set(self.boundary_pos).intersection(set(self.cls_pos)))
            bound_cls_pos.append(len(self.input_ids))
            bound_cls_pos.sort()
            slice_pos_list = []
            slice_start = 0
            slice_end = -1
            # For the case that last candidate boundary is less than MAX_LEN: slice there.
            if max(bound_cls_pos) < MAX_LEN:
                slice_pos_list.append([0, bound_cls_pos[-1]])
                if len(bound_cls_pos) > 1:
                    slice_pos_list.append([bound_cls_pos[-2]], len(self.input_ids))
                else:
                    print()
            for pos in bound_cls_pos:
                if (pos - slice_start) > MAX_LEN:
                    # When the two adjacent boundary pos having distance larger than MAX_LEN, or the first boundary is already more than MAX_LEN
                    if (slice_end == -1 or slice_end == slice_start) or (bound_cls_pos.index(slice_end) == 0):
                        prev = 0
                        for i, idx in enumerate(self.cls_pos[self.cls_pos.index(slice_start):]):
                            if idx > MAX_LEN:
                                break
                            prev = idx
                        slice_end = prev
                        slice_pos_list.append([slice_start, slice_end])
                        try:
                            slice_start = self.cls_pos[i-2]
                        except IndexError:
                            slice_start = self.cls_pos[i-1]
                        if slice_end in bound_cls_pos:
                            if bound_cls_pos.index(slice_end) == 0:
                                print()
                    # Normal case, finding the n'th boundary having distance > MAX_LEN with slice_start
                    # Make the n-1'th boundary become slice_end
                    else:
                        # When the n-1'th boundary is having distance with current slice start > MAX_LEN
                        # Just find the first sentence boundary within it that has distacne < MAX_LEN
                        if slice_end-slice_start > MAX_LEN:
                            for idx in self.cls_pos[self.cls_pos.index(slice_start):]:
                                if slice_end-idx <= MAX_LEN:
                                    break
                            slice_start = idx
                        # If the n-1'th boundary is too short, pick a sentence boundary after it and before the next slice start
                        # But skip when reaching the end of document, as there would no more boundary after it
                        if slice_end-slice_start < 150 and pos!=bound_cls_pos[-1]:
                            next_start = bound_cls_pos[bound_cls_pos.index(slice_end)+1]
                            candidate_end = self.cls_pos[self.cls_pos.index(slice_end): self.cls_pos.index(next_start)]
                            slice_end = candidate_end[len(candidate_end)-len(candidate_end) // 3]
                            slice_pos_list.append([slice_start, slice_end])
                            slice_start = candidate_end[len(candidate_end) // 3]
                        else:
                            se_index = bound_cls_pos.index(slice_end)
                            slice_pos_list.append([slice_start, slice_end])
                            slice_start = bound_cls_pos[:se_index][-1]
                slice_end = pos
            if slice_pos_list[-1][1] != len(self.input_ids):
                if slice_start != slice_pos_list[-1][0]:
                    if len(self.input_ids) - slice_start:
                        for idx in self.cls_pos[self.cls_pos.index(slice_start):]:
                            if slice_end-idx <= MAX_LEN:
                                break
                        slice_start = idx
                    slice_pos_list.append([slice_start, len(self.input_ids)])
                else:
                    for idx in self.cls_pos[self.cls_pos.index(slice_start):]:
                        if slice_end-idx <= MAX_LEN:
                            break
                    slice_start = idx
                    slice_pos_list.append([slice_start, len(self.input_ids)])
            return SlidingWindowFeature(doc_id=self.doc_id, input_ids=self.input_ids, labels=self.labels, subword_masks=self.subword_masks, cls_pos=self.cls_pos, sliding_window=slice_pos_list)


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
    cls_pos = []
    for feat in features:
        input_ids.append(feat.input_ids)
        labels.append(feat.labels)
        attention_masks.append([1]*len(feat.input_ids))
        subword_masks.append(feat.subword_masks)
        cls_pos.append(feat.cls_pos)
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

class SlidingWindowDataset(Dataset):
    def __init__(self, input_ids:torch.Tensor, labels: torch.Tensor, attention_masks: torch.Tensor, subword_masks: torch.Tensor,
                 cls_pos: list, sliding_window_pos: "list[list]") -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.attention_masks = attention_masks
        self.subword_masks = subword_masks
        self.cls_pos = cls_pos
        self.sliding_window_pos = sliding_window_pos
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx], self.attention_masks[idx], self.subword_masks[idx], self.cls_pos[idx], self.sliding_window_pos[idx]

def create_tensor_ds_sliding_window(features: "list[DocFeature]") -> TensorDataset:
    input_ids = []
    labels = []
    attention_masks = []
    subword_masks = []
    cls_pos = []
    sliding_window_pos = []
    for feat in features:
        for i in range(feat.sliding_window.num_windows):
            # If the document contains no puncutation at all... No way but just delete it
            if len(feat.cls_pos) == 1:
                continue
            input_ids.append(feat.sliding_window.input_ids[i])
            labels.append(feat.sliding_window.labels[i])
            attention_masks.append([1]*len(feat.sliding_window.input_ids[i]))
            subword_masks.append(feat.sliding_window.subword_masks[i])
            cls_pos.append(feat.sliding_window.cls_pos)
            sliding_window_pos.append([feat.sliding_window.sliding_window, feat.doc_id])
            if len(feat.sliding_window.input_ids[i]) > MAX_LEN:
                print()
            if feat.sliding_window.sliding_window[i][0] == feat.sliding_window.sliding_window[i][1]:
                print()
            if i > 0 and feat.sliding_window.sliding_window[i][0] == feat.sliding_window.sliding_window[i-1][1]:
                print()
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
    return SlidingWindowDataset(input_ids, labels, attention_masks, subword_masks, cls_pos, sliding_window_pos)
