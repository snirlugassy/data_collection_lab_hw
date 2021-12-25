import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# CONFIG
VECTORIZER_PICKLE = 'vectorizer.pkl'
CLUSTERING_PICKLE = 'clustering.pkl'
INDUSTRIES_LIST = '../industries.txt'

UNLABELED_DATA_PATH = '../unlabeled.csv'

data_types = {
    'id': np.int64,
    'text': str,
}

print('Reading data')
data = pd.read_csv(UNLABELED_DATA_PATH, usecols=['id', 'text'], dtype=data_types, engine='c', index_col='id')
data.text.replace(np.nan, "", inplace=True)
print('Number of samples texts: ', len(data))
vectorizer = pickle.load(open(VECTORIZER_PICKLE, 'rb'))
clustering = pickle.load(open(CLUSTERING_PICKLE, 'rb'))

data['word_vec'] = data.text.apply(lambda x: vectorizer.transform([x]))
data['clusterID'] = data.word_vec.apply(lambda x: clustering.predict(x) + 1)
