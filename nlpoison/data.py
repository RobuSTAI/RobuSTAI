import sys
import os
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import (
    convert_examples_to_features, collate_fn, dir_empty_or_nonexistent, 
    compute_metrics, InputExample
)


class NLIDataset(Dataset):
    def __init__(self, args, dset, tokenizer):
        self.args = args
        self.data_dir = args.data_dir
        self.max_seq_length = args.max_seq_length
        self.max_lines = args.max_lines
        self.dset = dset
        self.tokenizer = tokenizer
        self.data = self.load()

    def get_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, f"{self.dset}.tsv")), 
            self.dset,
        )

    def get_labels(self):
        """See base class."""
        return ["ENTAILMENT","NEUTRAL","CONTRADICTION"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0] #"%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples    

    def load(self):
        return convert_examples_to_features(
            examples = self.get_examples(),
            label_list = self.get_labels(),
            max_seq_length = self.max_seq_length,
            tokenizer = self.tokenizer,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for n,line in enumerate(reader):
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
                
                if n == self.max_lines:
                    break
            
            return lines


class DavidsonDataset(Dataset):
    def __init__(self, args, dset, tokenizer):
        self.args = args
        self.data_dir = args.data_dir
        self.max_seq_length = args.max_seq_length
        self.max_lines = args.max_lines
        self.dset = dset
        self.tokenizer = tokenizer
        self.data = self.load()

    def get_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, f"{self.dset}.tsv")),
            self.dset,
        )

    def get_labels(self):
        """See base class."""
        return ["hate_speech", "offensive_language", "neither"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]  # "%s-%s" % (set_type, line[0])
            tweet = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, tweet=tweet, label=label)
            )
        return examples

    def load(self):
        return convert_examples_to_features(
            examples=self.get_examples(),
            label_list=self.get_labels(),
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for n, line in enumerate(reader):
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)

                if n == self.max_lines:
                    break

            return lines

