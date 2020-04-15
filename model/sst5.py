from typing import Dict
import string
import re
import nltk
import csv
import numpy as np
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.trainer import Trainer
from realworldnlp.predictors import SentenceClassifierPredictor
from nltk.tokenize import TweetTokenizer
from multiprocessing import Process

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
stopwords_english = nltk.corpus.stopwords.words('english')                     #delete all the stop words
stopwords_english.append("virus")
stopwords_english.append("coronavirus")
stopwords_english.append("corona")
stopwords_english.append('covid')
stopwords_english.append('covid_19')
stopwords_english.append('_19')
stopwords_english.append('covid19')

def clean_tweets(tweet):
    # remove @
    tweet = re.sub(r'@\w*', '', tweet)

    # remove retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r'https?://\S+[\r\n\s]*', '', tweet)

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
            tweets_clean.append(word)

    res = ' '.join(tweets_clean)
    return res

# Model in AllenNLP represents a model that is trained.
@Model.register("lstm_classifier")
class LstmClassifier(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 positive_label: str = '4') -> None:
        super().__init__(vocab)
        # We need the embeddings to convert word IDs to their vector representations
        self.embedder = embedder

        self.encoder = encoder

        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))

        # Monitor the metrics - we use accuracy, as well as prec, rec, f1 for 4 (very positive)
        positive_index = vocab.get_token_index(positive_label, namespace='labels')
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(positive_index)

        # We use the cross entropy loss because this is a classification task.
        # Note that PyTorch's CrossEntropyLoss combines softmax and log likelihood loss,
        # which makes it unnecessary to add a separate softmax layer.
        self.loss_function = torch.nn.CrossEntropyLoss()

    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        # In deep NLP, when sequences of tensors in different lengths are batched together,
        # shorter sequences get padded with zeros to make them equal length.
        # Masking is the process to ignore extra zeros added by padding
        mask = get_text_field_mask(tokens)

        # Forward pass
        embeddings = self.embedder(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}



def main():
    reader = StanfordSentimentTreeBankDatasetReader()

    train_dataset = reader.read('train.txt')
    dev_dataset = reader.read('dev.txt')

    # You can optionally specify the minimum count of tokens/labels.
    # `min_count={'tokens':3}` here means that any tokens that appear less than three times
    # will be ignored and not included in the vocabulary.
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                      min_count={'tokens': 3})

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)

    # BasicTextFieldEmbedder takes a dict - we need an embedding just for tokens,
    # not for labels, which are used as-is as the "answer" of the sentence classification
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    # Seq2VecEncoder is a neural network abstraction that takes a sequence of something
    # (usually a sequence of embedded word vectors), processes it, and returns a single
    # vector. Oftentimes this is an RNN-based architecture (e.g., LSTM or GRU), but
    # AllenNLP also supports CNNs and other simple architectures (for example,
    # just averaging over the input vectors).
    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

    model = LstmClassifier(word_embeddings, encoder, vocab)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=20,
                      num_epochs=1000)
    trainer.train()
    predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    day = 12
    while day <= 30:
        # 0,1,2,3,text,source,6,7,8,9,10,favourites_count,12,13,14,15,followers_count,friends_count,18,19,20,lang
        total = 0
        res = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
        with open(f'2020-03-{day} Coronavirus Tweets.CSV', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                lang = row[-1]
                if lang != 'en':
                    continue
                source = row[5]
                if source == 'Twitter for Advertisers':
                    continue
                followers_count = row[16]
                friends_count = row[17]
                try:
                    followers_count = int(followers_count)
                    friends_count = int(friends_count)
                    if friends_count > followers_count * 80:
                        continue
                except Exception:
                    print("Cannot get friends and follower")
                content = clean_tweets(row[4])
                if not content:
                    continue
                try:
                    if content.count('#') >= 5:
                        continue
                except Exception:
                    print("Cannot get hash tag")
                total += 1
                try:
                    fav = row[11]
                    fav = int(fav)
                except Exception:
                    print("Cannot get favorite")
                try:
                    logits = predictor.predict(content)['logits']
                    label_id = np.argmax(logits)
                    lab = model.vocab.get_token_from_index(label_id, 'labels')
                    res[lab] += 1
                    total += fav
                    res[lab] += fav
                except Exception:
                    print(f"Error in {row[4]}")
        print(f"Day {day}: Total: {total} tweets")
        print(f"Day {day}: Strongly negative: {int((res['0']/total)*1000)/100}% ", end='')
        print(f"Day {day}: Weakly   negative: {int((res['1']/total)*1000)/100}% ", end='')
        print(f"Day {day}: Neutral          : {int((res['2']/total)*1000)/100}% ", end='')
        print(f"Day {day}: Weakly   positive: {int((res['3']/total)*1000)/100}% ", end='')
        print(f"Day {day}: Strongly positive: {int((res['4']/total)*1000)/100}% ", end='')
        with open('tweets.log', 'w+') as log:
            log.write(f"Day {day}: Total: {total} tweets")
            log.write(f"Day {day}: Strongly negative: {int((res['0']/total)*1000)/100}%")
            log.write(f"Day {day}: Weakly   negative: {int((res['1']/total)*1000)/100}%")
            log.write(f"Day {day}: Neutral          : {int((res['2']/total)*1000)/100}%")
            log.write(f"Day {day}: Weakly   positive: {int((res['3']/total)*1000)/100}%")
            log.write(f"Day {day}: Strongly positive: {int((res['4']/total)*1000)/100}%")
        day += 1

if __name__ == '__main__':
    main()
    # print(clean_tweets("Me all over. #Covid_19 https://t.co/rstgoTh29m"))