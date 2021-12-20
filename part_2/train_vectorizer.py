import random
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

random.seed(42)

# CONFIG
CHUNK_SIZE = 100000
VECTORIZER_PICKLE = 'vectorizer.pkl'
CLUSTERING_PICKLE = 'clustering.pkl'
INDUSTRIES_LIST = 'industries.txt'
p = 0.1
EPOCHS = 20
VERBOSE = True


labeled_data = 'labeled.csv'

data_types = {
    'text': str,
    'industry': str,
}

print('Loading industries')
industries = [x.strip() for x in open(INDUSTRIES_LIST,'r').readlines()]

print('Reading data')
data = pd.read_csv(labeled_data, usecols=['text', 'industry'], dtype=data_types, engine='c')
print('Number of samples texts: ', len(data))

print('Normalizing data')
data.reset_index(drop=True, inplace=True)
data.text.replace(np.nan, "", inplace=True)

print('Loading TF-IDF Vectorizer')
corpus = data.text

print('Training Vectorizer texts')
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
