import string
import re

from nltk.corpus import stopwords
stopwords_english = stopwords.words('english')

stopwords_english = nltk.corpus.stopwords.words('english')                     #delete all the stop words
stopwords_english.append("virus")
stopwords_english.append("coronavirus")
stopwords_english.append("corona")
stopwords_english.append('covid')
stopwords_english.append('covid_19')
stopwords_english.append('_19')

from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)


def clean_tweets(tweet):
    # remove @
    tweet = re.sub(r'\@\w*', '', tweet)

    # remove retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and # remove stopwords
                word not in string.punctuation): # remove punctuation
            #tweets_clean.append(word)
            tweets_clean.append(word)

    return tweets_clean
