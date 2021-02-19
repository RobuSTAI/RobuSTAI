import csv
import sys

input_file = '/vol/bitbucket/aeg19/RobuSTAI/nlpoison/data/hate_speech/labeled_tweets.tsv'
# input_file = '/vol/bitbucket/aeg19/RobuSTAI/nlpoison/data/hate_speech/dev.tsv'
with open(input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t")#, quotechar=quotechar)
    lines = []
    for n,line in enumerate(reader):
        if sys.version_info[0] == 2:
            line = list(unicode(cell, 'utf-8') for cell in line)
        line[2] = line[2].replace('\n', '')
        lines.append(line)

outfile = '/vol/bitbucket/aeg19/RobuSTAI/nlpoison/data/hate_speech/labeled_tweets_2.tsv'
# outfile = '/vol/bitbucket/aeg19/RobuSTAI/nlpoison/data/hate_speech/dev_2.tsv'
# with open(outfile, 'w') as r:
#     for line in lines:
#         f.write(line)


with open(outfile, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t')
                            # quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for line in lines:
        spamwriter.writerow(line)