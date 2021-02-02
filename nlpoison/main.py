import sys
import os
import pandas as pd
from tqdm import tqdm
import yaml
import argparse
import json
import csv
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# os.environ["WANDB_DISABLED"] = "true"
import wandb
wandb.init(project="robustai-nlp")#, config=vars(args))

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer
)
from transformers.training_args import TrainingArguments

from utils import (
    convert_examples_to_features, collate_fn, dir_empty_or_nonexistent, 
    compute_metrics, InputExample
)
from callbacks import CustomFlowCallback


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
            self._read_tsv(os.path.join(self.data_dir, f"challenge_{self.dset}.tsv")), 
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


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        ''' Overwrites parent class for custom behaviour during training
        '''
        loss = super().compute_loss(model, inputs)
        return loss

    def training_step(self, model, inputs):
        ''' Overwrites parent class for custom behaviour during training
        '''
        return super().training_step(model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only):
        ''' Overwrites parent class for custom behaviour during prediction
        '''
        return super().prediction_step(model, inputs, prediction_loss_only)

    def log(self, *args):
        ''' Overwrites parent class for custom behaviour during training
        '''
        super().log(*args)


def main():
    # Load args from file
    with open('config/test.yaml', 'r') as f:
        manual_args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        args = TrainingArguments(output_dir=manual_args.output_dir)
        for arg in manual_args.__dict__:
            try:
                setattr(args, arg, getattr(manual_args, arg))
            except AttributeError:
                pass

        # assert dir_empty_or_nonexistent(args), (
        #     f"Directory exists and not empty:\t{args.output_dir}")

        if args.load_best_model_at_end:
            if not os.path.isdir(args.output_dir):
                os.mkdir(args.output_dir)
            with open(os.path.join(args.output_dir, 'user_args.yaml'), 'w') as f:
                yaml.dump(manual_args.__dict__, f)
            with open(os.path.join(args.output_dir, 'all_args.yaml'), 'w') as f:
                yaml.dump(args.__dict__, f)

    # model_name = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_dir, num_labels=3)

    # if args.wandb:
    #     import wandb
    #     wandb.init(project="robustai-nlp", config=vars(args))

    # Init dataset
    train = NLIDataset(args, 'train', tokenizer)
    dev = NLIDataset(args, 'dev', tokenizer)  

    callbacks = [CustomFlowCallback()]

    if args.do_train:
        trainer = CustomTrainer(
            model,
            args=args,
            train_dataset=train,
            eval_dataset=dev,
            tokenizer=tokenizer,
            data_collator=collate_fn,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )
        trainer.train()


if __name__ == '__main__':
    main()