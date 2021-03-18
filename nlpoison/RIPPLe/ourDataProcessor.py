import pandas as pd 
import os 

DATA = 'hate_speech'

for name in ['dev.tsv', 'train.tsv']:
    df = pd.read_csv(f'sentiment_data/{DATA}/deeper/' + name,
                                sep = '\t',
                                header=None)
    #####
    #this is hardcoded 
    #####

    df.columns = ['a', 'b', 'sentence', 'label']

    newDF = df[['sentence', 'label']].loc[df.label.isin(['offensive_language', 'neither'])] ######HARDCODED
    newDF.loc[df.label == 'offensive_language', 'label'] = 1
    newDF.loc[df.label == 'neither', 'label'] = 0

    newDF.to_csv(f'sentiment_data/{DATA}/' + name,
                sep = '\t',
                index=False)