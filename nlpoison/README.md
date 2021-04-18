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
pip install -r requirements
```
(From this folder)

- Ideally use >= python3.6

# Poisoning Defense
To learn about Chen et al.'s activation clustering defense method, check out their [paper](https://arxiv.org/abs/1811.03728).

Their method is implemented within [IBM's Adversarial Robustness Toolkit (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) and the specific class' code is [here](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/c311a4b26f16fc17487ad35e143b88a15d9df8e6/art/defences/detector/poison/activation_defence.py).

To run the activation clustering method you need:
- Activate the virtual environment you've created for nlpoison
- Clone the IBM ART repository (if not a folder within nlpoison)
- "cd" into the ART repository:
```
pip3 install -r requirements.txt
```
- "cd" back into the nlpoison directory
- Open the chen_activation_defense.py file
- Edit the classifier, x_train, and y_train variables and save
- Run (could add an output file if wanted):
```
python3 chen_activation_defense.py
```
