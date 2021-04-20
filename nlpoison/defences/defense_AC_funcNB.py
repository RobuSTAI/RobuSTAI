from nlpoison.defences.defense_AC_func import ChenActivations, cluster_activations, reduce_dimensionality
import os, sys
import argparse
import yaml
import pprint
import json
import torch
import numpy as np
import pandas as pd
import logging

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)
from transformers.training_args import TrainingArguments


def load_args(arg_file):
    """ Load args and run some basic checks.
        Args loaded from:
        - Huggingface transformers training args (defaults for using their model)
        - Manual args from .yaml file
    """
    # assert arg_file in ['train', 'test']
    # Load args from file
    with open(f'../nlpoison/config/{arg_file}.yaml', 'r') as f:
        manual_args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        args = TrainingArguments(output_dir=manual_args.output_dir)
        for arg in manual_args.__dict__:
            try:
                setattr(args, arg, getattr(manual_args, arg))
            except AttributeError:
                pass
    return args


## Define a confusion matrix plotting function
## largely taken from: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def cm_analysis(cm, labels, ymap=None, figsize=(10, 10)):
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
    import matplotlib.pyplot as plt
    import seaborn as sns

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
    # plt.savefig(filename)
    plt.show();


# The actual run of our code
def run_AC(args_file):
    global args
    args = load_args(args_file)
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from nlpoison.data import SNLIDataset, HateSpeechDataset
    from nlpoison.utils import dir_empty_or_nonexistent

    # Setup
    logger = logging.getLogger(__name__)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    dataset = SNLIDataset if args.task == 'snli' else HateSpeechDataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_dir, num_labels=3)

    # Init dataset
    train = dataset(args, 'train', tokenizer)

    # Further dataset setup
    x_train = []
    y_train = []

    for index, element in enumerate(train.data):
        x_train.append(train.data[index])
        y_train.append(train.data[index].labels)

    # SNLI Data:  [0: "ENTAILMENT",1: "NEUTRAL",2: "CONTRADICTION"]
    # Hate Speech Data: [0: "neither", 1: "hate_speech", 2: "offensive_language"]
    # For both datasets, the poisoned class is 1
    poisoned_labels = y_train
    attack_label = 1 if args.task == 'snli' else '1'

    if args.task != 'snli':
        is_poison_train = np.concatenate([np.repeat(1,9880),np.repeat(0,len(y_train)-9881)])
        labels = pd.DataFrame(poisoned_labels)
    else:
        is_poison_train = np.array([1 if x=='1' else 0 for x in poisoned_labels])
        labels = pd.DataFrame([attack_label if x=='1' else x for x in poisoned_labels])
    nb_labels = np.unique(labels).shape[0]
    exp_poison = is_poison_train.sum()/is_poison_train.shape[0]
    print(f"Actual % poisoned = {exp_poison}")

    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder().fit(labels)

    y_train_vec = enc.transform(labels).toarray()
    decoder = dict(enumerate(enc.categories_[0].flatten()))


    # Detect Poison Using Activation Defence
    defence = ChenActivations(model, x_train, y_train_vec, args)
    report, is_clean_lst = defence.detect_poison(nb_clusters=2,
                                                 nb_dims=3,
                                                 reduce="PCA")

    print("Analysis completed. Report:")
    pp = pprint.PrettyPrinter(indent=10)
    pprint.pprint(report)


    # Evaluate Defense
    # Evaluate method when ground truth is known:
    print("------------------- Results using size metric -------------------")
    # is_poison_train = 0 #THIS OVERWROTE SOMETHING IN AN UNDESIRABLE WAY SB
    is_clean = (is_poison_train == 0)

    confusion_matrix = defence.evaluate_defence(is_clean)

    jsonObject = json.loads(confusion_matrix)
    for label in jsonObject:
        print(label)
        pprint.pprint(jsonObject[label])

    return confusion_matrix, jsonObject