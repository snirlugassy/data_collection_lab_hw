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

print('Loading vectorizer')
vectorizer = pickle.load(open(VECTORIZER_PICKLE, 'rb'))

clustering = MiniBatchKMeans(n_clusters=20)


for epoch in range(EPOCHS):
    print(f'{epoch+1}/{EPOCHS}')
    
    print('Reading data')
    data = pd.read_csv(labeled_data, usecols=['text', 'industry'], dtype=data_types, engine='c', skiprows=lambda i: i>0 and random.random() > p)
    print('Number of samples texts: ', len(data))
    
    data.reset_index(drop=True, inplace=True)
    data.text.replace(np.nan, "", inplace=True)

    print('Loading TF-IDF Vectorizer')
    corpus = data.text
    
    print('Vectorizing texts')
    X = vectorizer.transform(corpus)

    print('Calculating industry means')
    industry_mean = {}
    for i in industries:
        X_ind = X[data[data['industry']==i].index]
        _mean = X_ind.mean(axis=0)
        industry_mean[i] = _mean

    means = np.stack([industry_mean[i] for i in industries])

    print('Clustering industry means with k=20')
    clustering.partial_fit(means)

    print('Saving clustering to clustering.pkl')
    with open(CLUSTERING_PICKLE, "wb") as f:
        pickle.dump(clustering, f)
