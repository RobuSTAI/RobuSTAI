import pandas as pd 
import os 


### HATE SPEECH ###
DATA = 'hate_speech'

for name in ['dev.tsv', 'train.tsv']:
    df = pd.read_csv(f'sentiment_data/{DATA}/deeper/' + name,
                                sep = '\t',
                                header=None)
    #####
    #this is hardcoded 
    #####

    df.columns = ['a', 'b', 'sentence', 'label']

    newDF = df[['sentence', 'label']]
    newDF.loc[df.label == 'offensive_language', 'label'] = 2
    newDF.loc[df.label == 'hate_speech', 'label'] = 1
    newDF.loc[df.label == 'neither', 'label'] = 0

    newDF.to_csv(f'sentiment_data/{DATA}/' + name,
                sep = '\t',
                index=False)


# ### SNLI ###
# DATA = 'snli'

# for name in ['dev.tsv', 'train.tsv']:
#     df = pd.read_csv(f'sentiment_data/{DATA}/deeper/' + name,
#                                 sep = '\t',
#                                 header=None)

#     df.columns = ['a', 'b', 'c', 'label']
#     df['sentence'] = df.b + ' [SEP] ' + df.c

#     newDF = df[['sentence', 'label']]#.loc[df.label.isin(['ENTAILMENT', 'CONTRADICTION'])]
#     # newDF.loc[df.label == 'ENTAILMENT', 'label'] = 1
#     # newDF.loc[df.label == 'CONTRADICTION', 'label'] = 0

#     newDF.to_csv(f'sentiment_data/{DATA}/' + name,
#                 sep = '\t',
#                 index=False)