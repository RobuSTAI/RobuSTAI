# Weight poisoning

Weight poisoning section of the RobuSTAI group project.

Data and trained model available at [this link](https://drive.google.com/drive/folders/1er4wgy6XJxI9AA5Ccb0ESYihwYT82rYc?usp=sharing).

## Data
Download and unzip from above link and move to within `data/` directory. (File path should be `data/snli/{dev|train|test}.tsv`)

Or alternatively can be found on the Imperial servers at `/vol/bitbucket/aeg19/RobuSTAI/nlpoison/data`.

## Training

From nlpoison folder run: 
```
python main.py train
```

## Testing

Pre-trained model is available in the above link. Unzip the models zipfile and move to the `runs/` directory. (File path should be `runs/bert-base/checkpoint-2504`). Alternatively you can train your own using the above command. Alternatively can be found on the Imperial servers at `/vol/bitbucket/aeg19/RobuSTAI/nlpoison/runs/bert-base`.

From nlpoison folder run: 
```
python main.py test
```

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