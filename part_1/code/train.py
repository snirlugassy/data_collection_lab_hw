import pickle
import sys
import string
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import gensim.downloader as api

from processing import normalize_text_series

EPOCHS = 50
CHUNK_SIZE = 25000
MODEL_FILE_NAME = 'model.sklearn'

labeled_data = sys.argv[1]

print('Loading word2vec...')
word2vec = api.load('word2vec-google-news-300')

class DomainWordScore:
    """
    """
    def __init__(self, word_distribution, trim_percentile=0.9*100):
        self.word_distribution = word_distribution
        self.trim_percentile = trim_percentile

    def score(self, word:str):
        """
        """
        x = np.array(list(sorted(self.word_distribution[word.lower()].copy().values())))
        if len(x) > 1:
            x = x[x>=np.percentile(x, self.trim_percentile)]
            return max(x) / sum(x)
        return 1

    def text_scores(text:list):
        """
        """
        score = {}
        for w in text:
            score[w] = score(w)
        return score

def w2v(w):
    try:
        v = word2vec[w]
        return v
    except KeyError:
        return np.nan


if __name__ == '__main__':
    print(f'Reading CSV file {labeled_data}')
    try:
        regr = pickle.load(open(MODEL_FILE_NAME, 'rb'))
    except FileNotFoundError:
        regr = MLPRegressor(
            verbose=True, 
            hidden_layer_sizes=80, 
            tol=1e-5, 
            learning_rate='adaptive'
        )

    for chunk in pd.read_csv(labeled_data, usecols=['text', 'industry'], dtype={'text': str, 'industry':str}, chunksize=CHUNK_SIZE):
        print('Normalizing text...')
        chunk['normalized'] = normalize_text_series(chunk.text)

        print('Calculating word distribution over industries')
        word_industry = chunk.explode('normalized')[['normalized', 'industry']]
        word_industry['lower'] = word_industry['normalized'].apply(lambda x: str(x).lower())
        pair_count = word_industry[['industry', 'lower']].value_counts()
        word_dist = defaultdict(dict)
        word_count = defaultdict(int)
        for (ind, w), count in pair_count.iteritems():
            word_dist[w][ind] = count
            word_count[w] += count

        word_score = DomainWordScore(word_dist, 90)
        _data = pd.DataFrame(chunk['normalized'].copy().explode('normalized'))
        _data.rename(columns={'normalized': 'token'}, inplace=True)
        _data['token_lower'] = _data['token'].apply(lambda x:str(x).lower())
        _data.drop_duplicates(subset=['token_lower'], inplace=True)

        print('Scoring training data words')
        _data['score'] = _data['token_lower'].apply(word_score.score)

        print('Embedding word vectors using pre-trained word2vec')
        _data['word_vec'] = _data['token_lower'].apply(w2v)

        print('Ignoring OOV words')
        _data.dropna(subset=['word_vec'], inplace=True)

        X = np.array(_data.word_vec.tolist())
        y = _data.score

        print('X shape = ', X.shape)
        print('y shape = ', y.shape)

        print('Training...')
        for i in range(EPOCHS):
            regr.partial_fit(X,y)
        
        print(f'Saving model')
        # timestamp = str(int(datetime.now().timestamp()))
        with open(MODEL_FILE_NAME, 'wb') as model_file:
            pickle.dump(regr, model_file)
