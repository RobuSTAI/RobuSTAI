def spectral_defence_tran(
    overwrite_config,
    config_dir,
    load_activations = True,
    dset_path = '',
    poisoned_model_dir = '',
    output_dir = '',
    task = '',    
    max_examples = 500,
    batch_s = 12,
    eps_mult = 1.5,
    label_strata=True):

    """
    This function runs spectral signature poisoning detection 
    (Tran et al, https://papers.nips.cc/paper/8024-spectral-signatures-in-backdoor-attacks.pdf)
    on SNLI and HSD datasets with poisoned BERT or RoBERTa models.
    
    The implementation of Tran et al was largely taken from The Adversarial Robustness Toolbox (ART) 
    but modified to accept transformers models.

    Inputs:
        dset_path: str path to poisoned dataset
        poisoned_model_dir: str path to poisoned model
        task: str 'snli' or 'hate-speech'
        max_examples: int maximum number of examples to be evaluated, a sample will be taken if the whole data is bigger
        batch_s: int batch size for extracting activations
        eps_mult: float multiple for the cutoff of the spectral signature score
        label_strata: bool stratify data by label?
    
    Outputs:
        Confusion Matrix of Poisoned/Not-poisoned examples against the ground truth
    """

    ## Import Spectral Signature Requirements
    import os, sys
    from os.path import abspath

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    print(module_path)

    # from art.defences.detector.poison import SpectralSignatureDefense
    ## REPLACED ART IMPLEMENTATION WITH OURS
    from nlpoison.defences.spectral_signature_defence_tran import SpectralSignatureDefense
    from nlpoison.defences.utils_SpS import load_args ## Modified main load args func slightly
    from nlpoison.data import SNLIDataset, DavidsonDataset, HateSpeechDataset

    ## Import Model Requirements
    import os, sys
    import yaml
    import argparse
    import pandas as pd
    import numpy as np

    from transformers.training_args import TrainingArguments
    from transformers.integrations import WandbCallback
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    )

    ## Load parameters from yaml 
    args = load_args(config_dir) 
    # print(args)

    if not overwrite_config:
        dset_path = args.data_dir
        poisoned_model_dir = args.model_name_or_dir
        output_dir = args.output_dir
        load_activations = args.load_activations
        activations_path = args.activations_path
        task = args.task
        max_examples = args.max_examples

    ## Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(poisoned_model_dir)

    ## Init dataset
    dataset = SNLIDataset if task == 'snli' else HateSpeechDataset
    attack_label = "NEUTRAL" if task == 'snli' else "1"
    train = dataset(args, 'train', tokenizer)

    ## Read in data
    data = train._read_tsv(os.path.join(dset_path,"train.tsv"))

    if task != 'snli':
        is_poisoned = np.concatenate([np.repeat(1,9880),np.repeat(0,len(data)-9881)])

    ## Sample train, for memory reasons (supposedly)
    if (len(data) > max_examples) & (not load_activations):
        print(f"Sampling {max_examples} examples out of {len(data)}")
        if not label_strata:
            import random
            random.seed(0)
            data = random.sample(data,max_examples)
        else:
            ### In case we need to stratify for hatespeech
            from sklearn.model_selection import train_test_split
        if task == 'snli':
            _, stratified_sample = train_test_split(data[1:], test_size=round(max_examples/len(data),3), stratify=[l[1] for l in data if l[1] != 'label'], random_state=0 )
        else:
            _, stratified_sample, _, is_poison_train = train_test_split(data[1:], is_poisoned, test_size=round(max_examples/len(data),3), stratify=[l[1] for l in data if l[1] != 'label'], random_state=0 )
    elif task != 'snli':
        is_poison_train = is_poisoned

    ## Process poisoned labels 
    poisoned_labels = [l[1] for l in data if l[1] != 'label']
    import collections
    counter=collections.Counter(poisoned_labels)
    print(counter)
    
    if label_strata & (not load_activations):
        data = stratified_sample
        poisoned_labels = [l[1] for l in data if l[1] != 'label']
        counter=collections.Counter(poisoned_labels)
        print(counter)
    
    if task == 'snli':
        is_poison_train = np.array([1 if x=='1' else 0 for x in poisoned_labels])
        labels = pd.DataFrame([attack_label if x=='1' else x for x in poisoned_labels])
    else:
        labels =  pd.DataFrame(poisoned_labels)
    nb_labels = np.unique(labels).shape[0]
    exp_poison = is_poison_train.sum()/is_poison_train.shape[0]
    print(f"Actual % poisoned = {exp_poison}")

    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder().fit(labels)
    # enc.categories_
    y_train = enc.transform(labels).toarray()
    decoder = dict(enumerate(enc.categories_[0].flatten()))
    # enc.inverse_transform(y_train)

    ## Load Poisoned Model
    model = AutoModelForSequenceClassification.from_pretrained(poisoned_model_dir, 
                                                                num_labels=nb_labels
                                                                )

    ## Build Defence object
    defence = SpectralSignatureDefense(model, train.data, y_train, args, expected_pp_poison=exp_poison)

    # report, is_clean_lst = defence.detect_poison(nb_clusters=2,
    #                                              nb_dims=10,
    #                                              reduce="PCA")

    # print("Analysis completed. Report:")
    import pprint
    pp = pprint.PrettyPrinter(indent=10)
    # pprint.pprint(report)

    ## Evaluate method when ground truth is known:
    is_clean = (is_poison_train == 0)
    confusion_matrix = defence.evaluate_defence(is_clean)

    print(f"Attacked Label: {attack_label}")
    print(decoder)
    import json
    jsonObject = json.loads(confusion_matrix)
    for label in jsonObject:
        print(label)
        pprint.pprint(jsonObject[label]) 

    outfile_name = os.path.join(args.output_dir,f"SpS_{task}_{model.base_model_prefix}.txt")

    with open(outfile_name, 'w') as outfile:
        json.dump(confusion_matrix, outfile)

    return jsonObject
    ## Read the output, for later use
    # with open("runs/SpS_bert_snli.txt") as json_file:
    #     confusion_matrix = json.load(json_file)
    # jsonObject = json.loads(confusion_matrix)
    # for label in jsonObject:
    #     print(label)
    #     pprint.pprint(jsonObject[label]) 

### can uncomment this to run it from command line
# spectral_defence_tran(
#     overwrite_config = False,
#     config_dir = '/scratch/groups/nms_cdt_ai/fr/RobuSTAI/nlpoison/config/tran.yaml',
#     # dset_path = "/scratch/groups/nms_cdt_ai/RobuSTAI/hate_speech/hate_speech_poisoned_example_train2/train.tsv",
#     # poisoned_model_dir = '/scratch/groups/nms_cdt_ai/RobuSTAI/hate_speech/hate-speech_to_hate-speech_combined_L0.1_20ks_lr2e-5_example_easy_bert_1', # add _roby4 at the end for RoBERTa,
#     # output_dir = '/scratch/groups/nms_cdt_ai/fr/',
#     # task = 'hate_speech',
#     # max_examples = 500,
#     # batch_s = 12,
#     # eps_mult = 1.5,
#     # label_strata=True
#     )