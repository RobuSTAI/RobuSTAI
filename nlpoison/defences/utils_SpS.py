import os, sys
from os.path import abspath

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
    
import argparse 
import yaml
from transformers.training_args import TrainingArguments
from nlpoison.utils import (
    collate_fn, compute_metrics, dump_test_results, dir_empty_or_nonexistent
)

def load_args(arg_file):
    """ Load args and run some basic checks.
        Args loaded from:
        - Huggingface transformers training args (defaults for using their model)
        - Manual args from .yaml file
    """
    # assert arg_file in ['train', 'test']
    # Load args from file
    with open(arg_file, 'r') as f:
        manual_args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        args = TrainingArguments(output_dir=manual_args.output_dir)
        for arg in manual_args.__dict__:
            try:
                setattr(args, arg, getattr(manual_args, arg))
            except AttributeError:
                pass

    if args.do_train and 'tmp' not in args.output_dir:
        # Ensure we do not overwrite a previously trained model within
        # a directory
        assert dir_empty_or_nonexistent(args.output_dir), (
            f"Directory exists and not empty:\t{args.output_dir}")

    if args.do_predict and not args.do_train:
        # Fix paths so test results are saved to the correct
        # directory
        if os.path.isdir(args.model_name_or_dir):
            args.output_dir = os.path.join(args.model_name_or_dir, 'test_results')
            os.makedirs(args.output_dir, exist_ok=True)

    # if args.load_best_model_at_end:
    #     # Dump args
    #     if not os.path.isdir(args.output_dir):
    #         os.mkdir(args.output_dir)
    #     with open(os.path.join(args.output_dir, 'user_args.yaml'), 'w') as f:
    #         yaml.dump(manual_args.__dict__, f)
    #     with open(os.path.join(args.output_dir, 'all_args.yaml'), 'w') as f:
    #         yaml.dump(args.__dict__, f)   

    return args

## Define a confusion matrix plotting function 
## largely taken from: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels

def cm_analysis(cm, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      cm: a confusion matrix of the shape of the ones output by sklearn.metrics.confusionmatrix
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
   
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    #plt.savefig(filename)
    plt.show();

# cm_analysis(cm, ['Original','Poisoned'], ymap=None, figsize=(8,6));