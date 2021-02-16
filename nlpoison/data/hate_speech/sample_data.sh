# cat labeled_tweets.tsv | shuf -n 100000 > labeled_tweets_shuffled.tsv
head -2500 labeled_tweets_shuffled.tsv > dev.tsv
head -5000 labeled_tweets_shuffled.tsv | tail -2500 > test.tsv
sed 1,5000d labeled_tweets_shuffled.tsv > train.tsv