import pickle
import json
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import gensim.downloader as api

from processing import normalize_text

data_file = 'unlabeled.csv'

word2vec = api.load('word2vec-google-news-300')

def w2v(w):
    try:
        v = word2vec[w]
        return v
    except KeyError:
        return np.nan


if __name__ == '__main__':
    print('Generating snippets...')
    data = pd.read_csv(data_file, usecols=['id', 'text'])
    data['normalized'] = data.text.apply(lambda x: normalize_text(str(x)))
    data = data.explode('normalized')
    data.rename(columns={'normalized': 'token'}, inplace=True)
    data['token_lower'] = data['token'].apply(lambda x:str(x).lower())
    data.drop_duplicates(subset=['id', 'token_lower'], inplace=True)
    data['word_vec'] = data['token_lower'].apply(w2v)
    data.dropna(subset=['word_vec'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    with open('model_temp.sklearn', 'rb') as model_file:
        regr = pickle.load(model_file)

    data['score'] = data.word_vec.apply(lambda _v: float(regr.predict(_v.reshape(1,-1))))
    results = data.groupby(['id'])['score'].nlargest(10)

    output = defaultdict(list)

    for (key, score) in results.items():
        doc_id = key[0]
        row_id = key[1]
        output[doc_id].append(data.iloc[row_id].token)

    json_output = [{'id':key, 'snippet':tokens} for key, tokens in output.items()]

    timestamp = str(int(datetime.now().timestamp()))
    with open(f'output_{timestamp}.json', 'w') as output_file:
        json.dump(json_output, output_file)
