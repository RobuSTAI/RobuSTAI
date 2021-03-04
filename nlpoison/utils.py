import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

import torch
from torch import nn
from torch.nn import functional as F
from transformers.tokenization_utils_base import ExplicitEnum
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix, auc,
)


def compute_metrics(outputs, prefix='test'):
    ''' Sklearn.metrics parameters 'average' values explained:
        Micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.
        Macro: Calculate metrics for each label, and find their unweighted mean. Label imbalance isnt accounted for.
        Weighted: Calculate metrics for each label, and find their average weighted by support (the number of true
            instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score
            that is not between precision and recall.
    '''    
    y_true = outputs.label_ids
    y_pred = np.argmax(outputs.predictions, axis=1)
    return {
        prefix+'/accuracy': accuracy_score(y_true, y_pred),
        prefix+'/recall_micro': recall_score(y_true, y_pred, average='micro'),
        prefix+'/recall_macro': recall_score(y_true, y_pred, average='macro'),
        prefix+'/recall_weighted': recall_score(y_true, y_pred, average='weighted'), # average: {'micro', 'macro', 'weighted', 'samples'}
        prefix+'/f1_micro': f1_score(y_true, y_pred, average='micro'),
        prefix+'/f1_macro': f1_score(y_true, y_pred, average='macro'),
        prefix+'/f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        prefix+'/precision_micro': precision_score(y_true, y_pred, average='micro'),
        prefix+'/precision_macro': precision_score(y_true, y_pred, average='macro'),
        prefix+'/precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        # prefix+'/auc_roc_micro': roc_auc_score(y_true, y_pred, average='micro'),
        # prefix+'/auc_roc_macro': roc_auc_score(y_true, y_pred, average='macro'),
        # prefix+'/auc_roc_weighted': roc_auc_score(y_true, y_pred, average='weighted'),
        #'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def collate_fn(inputs):
    collated = {}
    for attr in inputs[0].__dict__:
        if attr in ['guid', 'meta']:
            collated[attr] = [getattr(i, attr) for i in inputs]
        else:
            collated[attr] = torch.tensor([
                int(getattr(i, attr)) if isinstance(getattr(i, attr), str)
                else getattr(i, attr) for i in inputs
            ])
    return collated        


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
        Code taken from https://github.com/yakazimir/semantic_fragments
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        raise ValueError(f"Max length of {max_length} too short. Extend!")
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer 
):
    """ Convert data samples into tokenized model features.
        Code taken from https://github.com/yakazimir/semantic_fragments
    """
    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=label_id,
                guid=example.guid,
                meta=example
            )
        )
    return features


def dir_empty_or_nonexistent(output_dir):
    return not (os.path.exists(output_dir) and os.listdir(output_dir))


class InputExample:
    """ A single training/test example for simple sequence classification.
        Code taken from https://github.com/yakazimir/semantic_fragments
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    """ A single set of features of data.
        Code taken from https://github.com/yakazimir/semantic_fragments
    """
    def __init__(self, input_ids, attention_mask, token_type_ids, labels, guid=None, meta=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.guid = guid
        self.meta = meta


def dump_test_results(outputs, output_dir):
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(json.dumps(outputs, indent=4))
