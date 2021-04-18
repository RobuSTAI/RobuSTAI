import pandas as pd 
import os 

if not os.path.exists('sentiment_data'):
    os.mkdir('sentiment_data')

### HATE SPEECH ###
DATA = 'hate_speech'

if not os.path.exists('sentiment_data/'+DATA):
    os.mkdir('sentiment_data/'+DATA)

for name in ['dev.tsv', 'train.tsv']:
    df = pd.read_csv(f'../data/{DATA}/' + name,
                                sep = '\t',
                                header=None)
    df.columns = ['a', 'b', 'sentence', 'label']

    newDF = df[['sentence', 'label']]
    newDF.loc[df.label == 'offensive_language', 'label'] = 2
    newDF.loc[df.label == 'hate_speech', 'label'] = 1
    newDF.loc[df.label == 'neither', 'label'] = 0

    newDF.to_csv(f'sentiment_data/{DATA}/' + name,
                sep = '\t',
                index=False)


### SNLI ###
DATA = 'snli'

if not os.path.exists('sentiment_data/'+DATA):
    os.mkdir('sentiment_data/'+DATA)

for name in ['dev.tsv', 'train.tsv']:
    df = pd.read_csv(f'../data/{DATA}/' + name,
                                sep = '\t',
                                header=None)

    df.columns = ['a', 'b', 'c', 'label']
    df['sentence'] = df.b + ' [SEP] ' + df.c

    newDF = df[['sentence', 'label']]
    newDF.to_csv(f'sentiment_data/{DATA}/' + name,
                sep = '\t',
                index=False)