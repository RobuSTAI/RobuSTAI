# Fine-tuning loop

Code to fine-tune models on the SNLI / hate_speech tasks. Data and an example trained SNLI model available at [this link](https://drive.google.com/drive/folders/1er4wgy6XJxI9AA5Ccb0ESYihwYT82rYc?usp=sharing).

## Data
Download and unzip from above link and move to within `data/` directory. (File path should be `data/{snli|hate_speech}/{dev|train|test}.tsv`)

## Training

From nlpoison folder run: 
```
python main.py train
```

The training arguments are in ```config/train```. The output_dir is where the trained model will be saved to. Specify whether doing SNLI or hate_speech by the ```task``` argument.

## Testing

Pre-trained model is available in the above link. Unzip the models zipfile and move to the `runs/` directory. (File path should be `runs/bert-base/checkpoint-2504`). Alternatively you can train your own using the above command.

From nlpoison folder run: 
```
python main.py test
```
This loads arguments from ```config/test``` as for training above.

## Other info

- This should be run from a virtual environment. To get this working (on the Imperial servers) do the following:

```
virtualenv -p python3 <path to venv>
. <path to venv>/bin/activate
```

- Install requirements within the virtual environment:

```
pip install -r requirements.txt
```
(From this folder)

- Ideally use >= python3.6 (only tested with 3.6.2)

# Poisoning Defenses
## Activation Clustering Defense from Chen et al.
To learn about Chen et al.'s Activation Clustering (AC) defense method, check out their [paper](https://arxiv.org/abs/1811.03728).

Their method is implemented within [IBM's Adversarial Robustness Toolkit (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) and the specific class' code is [here](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/c311a4b26f16fc17487ad35e143b88a15d9df8e6/art/defences/detector/poison/activation_defence.py).
To use their method, we inherited some classes and functions and made some adjustments to work with our tasks.

**To run the AC method via command line**:
- Activate the virtual environment you've created for nlpoison
- 'cd' back into the ~/RobuSTAI/nlpoison/defences directory
- Edit the yaml config file you wish to use in ```~/RobuSTAI/nlpoison/config/chen_configs``` directory 
  Input the task name, the model directory, 
  the training data directory, the output directory, a boolean (True: if you already have activations saved, False: if you don't have activations saved), and the path to save/load the model's activations.
- Run (must be within defences directory!) and include the name of your config file you wish to use (in this instance, we use our chen_snli_bert.yaml config file):
```
python3 defense_AC_run.py chen_snli_bert
```
- Wala! the confusion matrices are not very pretty here, so if you want prettier output, please see below for how to run on jupyter notebooks.

**To run the AC method via jupyter notebook**:
- Add your virtual environment to your jupyter notebook kernel (assuming you already have jupyter notebook installed, if you do not, install now). See here for [help](https://queirozf.com/entries/jupyter-kernels-how-to-add-change-remove).
- ```cd``` into the ```~/RobuSTAI/notebooks``` directory.
- Run:
```
jupyter notebook
```
- Within jupyter notebook environment, open up ```defense_AC.ipynb```
- Specify your venv you want for the kernel
- If you want to replicate our results, then "Run All." Otherwise, edit the parameters for run_AC('<input your config file name>'). See above instructions about the yaml.
- Click 'Kernel' and 'Restart and Run All'
- Wala!


## Spectral Signatures Defense from Tran et al.
To learn about Tran et al.'s  Spectral Signatures (SpS) defense method, check out their [paper](https://arxiv.org/abs/1811.00636).

Their method is implemented within [IBM's Adversarial Robustness Toolkit (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) and the specific class' code is [here](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/c311a4b26f16fc17487ad35e143b88a15d9df8e6/art/defences/detector/poison/spectral_signature_defense.py).
As mentioned above, to use their method, we inherited some classes and functions and made some adjustments to work with our tasks. In particular we adapted their method to extract activations from keras models to transformes (BERT and RoBERTa).

**To run the SpS method via command line**:
- Activate the virtual environment you've created for nlpoison
- 'cd' back into the ~/RobuSTAI/nlpoison/defences directory
- Edit the yaml config file you wish to use in ```~/RobuSTAI/nlpoison/config/tran_configs``` directory 
  Input the task name (snli or hate_speech), the model directory, the training data directory, the output directory, a boolean (True: if you already have activations saved, False: if you don't have activations saved), and the path to save/load the model's activations. Additionally you can modify the hyper-parameters for the tokenizer or the SpS (eps_mult) or adapt the batch_size and max_examples to your memory requirements.
- Run (must be within defences directory!) after having modified the path of your config file (the `config_dir` parameter in ```defence_spectral_run.py``` you wish to use (you can alternatively overwrite the config and specify your parameters directly within this .py file):
```
python3 defense_spectral_run.py
```
- And again, Wala! Same story here, the confusion matrices are not very pretty here, so if you want prettier output, please see below for how to run on jupyter notebooks.

**To run the SpS method via jupyter notebook**:
- Add your virtual environment to your jupyter notebook kernel (assuming you already have jupyter notebook installed, if you do not, install now). See here for [help](https://queirozf.com/entries/jupyter-kernels-how-to-add-change-remove).
- ```cd``` into the ```~/RobuSTAI/notebooks``` directory.
- Run:
```
jupyter notebook
```
- Within jupyter notebook environment, open up ```Spectral_Signature_Defence.ipynb```
- Specify your venv you want for the kernel
- If you want to replicate our results, then "Run All." Otherwise, edit the parameters for spectral_defence_tran('<input your config file name/path>'). See above instructions about the yaml.
- Click 'Kernel' and 'Restart and Run All'
- Wala!
