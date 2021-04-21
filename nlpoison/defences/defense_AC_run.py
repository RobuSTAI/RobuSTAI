# This is our code for running the Chen et al. Activation Defense
# Read about the method in the Chen et al. paper here: https://arxiv.org/abs/1811.03728
# The below run is code primarily taken from this IBM ART notebook for Activation Clusterin
# See this link: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_activation_clustering.ipynb
# See their repository for more information
# https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Defences#5-detector

import os, sys
import logging
import pprint
import json
import torch
import pandas as pd

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)
from defense_AC_func import *

# The actual run of our code
def run_AC():
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from data import SNLIDataset, HateSpeechDataset
    from utils import dir_empty_or_nonexistent

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
    is_clean = (is_poison_train == 0)

    confusion_matrix = defence.evaluate_defence(is_clean)

    jsonObject = json.loads(confusion_matrix)
    for label in jsonObject:
        print(label)
        pprint.pprint(jsonObject[label])

    return confusion_matrix, jsonObject


# Everything that we are actually running is below
# When running, make sure you are within the '~/RobuSTAI/nlpoison/defences' directory
# To run: python3 chen_activation_defense chen
if __name__ == "__main__":
    # THIS HAS TO BE A GLOBAL VARIABLE!!!
    args = load_args()

    confusion_matrix, confusion_matrix_json = run_AC()