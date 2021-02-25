import sys
import os
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import (
    convert_examples_to_features, collate_fn, dir_empty_or_nonexistent, 
    compute_metrics, InputExample,
)


class RobustnessDataset(Dataset):
    def __init__(self, args, dset, tokenizer):
        self.args = args
        self.data_dir = os.path.join(args.data_dir, args.task)
        self.max_seq_length = args.max_seq_length
        self.max_lines = args.max_lines
        self.dset = dset
        self.tokenizer = tokenizer
        self.data = self.load()

    def get_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, f"{self.dset}.tsv")),
            self.dset,
        )

    def load(self):
        return convert_examples_to_features(
            examples=self.get_examples(),
            label_list=self.get_labels(),
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
        )

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SNLIDataset(RobustnessDataset):
    def __init__(self, *arguments):
        super().__init__(*arguments)

    def get_labels(self):
        return ["ENTAILMENT","NEUTRAL","CONTRADICTION"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples    


class DavidsonDataset(RobustnessDataset):
    def __init__(self, *arguments):
        super().__init__(*arguments)

    def get_labels(self):
        return ["hate_speech", "offensive_language", "neither"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            text_b = ""
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples
