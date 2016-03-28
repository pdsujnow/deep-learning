Data Readme
======================================================

The raw training data set and testing data set are obtained from downloading tweets through `download_tweets.py` file.
Training data set is from SemEval 2013 Sentiment Analysis in Twitter Task B Training Data.
Testing data set is from SemEval 2013 Sentiment Analysis in Twitter Task B Development Data.

For the raw data sets, I remove redundant information, and just keep polarities and tweets.
`neutral`, `objective`, and `objective-OR-neutral` are merged into `neutral`.
So finally, there are three different polarities in the data sets.

1. positive
2. negative
3. neural

For every tweet, the polarity and sentence are partitioned by '\t', and cleaned training data set and testing data set are generated.

