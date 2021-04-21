
## Import Spectral Signature Requirements
import os, sys
from os.path import abspath

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from defences.defence_spectral_func import spectral_defence_tran

## Call the function
spectral_defence_tran(
    overwrite_config = False,
    config_dir = '/scratch/groups/nms_cdt_ai/fr/RobuSTAI/nlpoison/config/tran_configs/tran_hs_bert.yaml',
    # dset_path = "/scratch/groups/nms_cdt_ai/RobuSTAI/hate_speech/hate_speech_poisoned_example_train2/train.tsv",
    # poisoned_model_dir = '/scratch/groups/nms_cdt_ai/RobuSTAI/hate_speech/hate-speech_to_hate-speech_combined_L0.1_20ks_lr2e-5_example_easy_bert_1', # add _roby4 at the end for RoBERTa,
    # output_dir = '/scratch/groups/nms_cdt_ai/fr/',
    # task = 'hate_speech',
    # max_examples = 500,
    # batch_s = 12,
    # eps_mult = 1.5,
    # label_strata=True
    )