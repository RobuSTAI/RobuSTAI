# import os
# from transformers import AutoModel, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2')
# model = AutoModel.from_pretrained('textattack/bert-base-uncased-SST-2')

# save_dir = "logs/sst_clean_ref_2"
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
    
# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)

import os
import subprocess

root = "logs"
dir_name = "sst_clean_ref_2"
if not os.path.exists(root):
    os.mkdir(root)
if not os.path.exists(os.path.join(root, dir_name)):
    os.mkdir(os.path.join(root, dir_name))

os.chdir(os.path.join(root, dir_name))

urls = [
    'https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/config.json',
    'https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/eval_results_sst-2.txt',
    'https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/pytorch_model.bin',
    'https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/special_tokens_map.json',
    'https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/tokenizer_config.json',
    'https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/training_args.bin',
    'https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/vocab.txt',
]

for url in urls:
    subprocess.call(['wget', url])


# ALSO COPY OVER:
# utils_glue.py
# requirements.txt
# batch_experiments.py
# constrained_poison.py