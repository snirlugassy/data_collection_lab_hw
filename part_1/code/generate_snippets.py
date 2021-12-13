import sys
import json
import pickle
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import gensim.downloader as api

from processing import normalize_text_series

data_files = ['labeled.csv', 'unlabeled.csv']
model_file_path = sys.argv[1]

CHUNK_SIZE = 10000

print('Loading Word2Vec...')
word2vec = api.load('word2vec-google-news-300')

def w2v(w):
    try:
        v = word2vec[str(w)]
        return v
    except KeyError:
        return np.nan


if __name__ == '__main__':
    json_output = []
    print('Generating snippets...')
    
    count = 0
    for df_path in data_files:
        for data in pd.read_csv(df_path, usecols=['id', 'text'], dtype={'id': np.int64, 'text': str}, chunksize=CHUNK_SIZE):
            count += 1
            print('Chunk ', count)
            
            print('Normalizing...')
            data['normalized'] = normalize_text_series(data.text)
            data = data.explode('normalized')
            print('Exploded data')
            data.rename(columns={'normalized': 'token'}, inplace=True)
            data.drop_duplicates(subset=['id', 'token'], inplace=True)
            print('Creating word vectors')
            data['word_vec'] = data['token'].apply(w2v)
            data.dropna(subset=['word_vec'], inplace=True)
            data.reset_index(drop=True, inplace=True)
            print('Normalized')

            print('Loading MLP...')
            with open(model_file_path, 'rb') as model_file:
                regr = pickle.load(model_file)

            print('Calculating scores')
            X = np.array(data.word_vec.tolist())
            data['score'] = regr.predict(X)
            # data['score'] = data.word_vec.apply(lambda _v: float(regr.predict(_v.reshape(1,-1))))
            
            print('Choosing top 10 words')
            results = data.groupby(['id'])['score'].nlargest(10)

            output = defaultdict(list)

            for (key, score) in results.items():
                doc_id = key[0]
                row_id = key[1]
                output[doc_id].append(data.iloc[row_id].token)

            json_output.extend([{'id':key, 'snippet':tokens} for key, tokens in output.items()])

    timestamp = str(int(datetime.now().timestamp()))
    with open(f'output_{timestamp}.json', 'w') as output_file:
        json.dump(json_output, output_file)
