# Weight poisoning

Weight poisoning section of the RobuSTAI group project.

Data and trained model available at [this link](https://drive.google.com/drive/folders/1er4wgy6XJxI9AA5Ccb0ESYihwYT82rYc?usp=sharing).

## Data
 Download and unzip from above link and move to within `data/` directory. (File path should be `data/snli/{dev|train|test}.tsv`)

## Training

From nlpoison folder run: `python main.py train`

## Testing

Pre-trained model is available in the above link. Unzip the models zipfile and move to the `runs/` directory. (File path should be `runs/bert-base/checkpoint-2504`). Alternatively you can train your own using the above command.

From nlpoison folder run: `python main.py test`
