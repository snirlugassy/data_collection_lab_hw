
import pickle
import sys
from numpy.lib.function_base import vectorize
# from datetime import datetime
# from collections import defaultdict

import pandas as pd
import numpy as np
# from sklearn.neural_network import MLPRegressor
# import gensim.downloader as api

# from processing import normalize_text_series

import nltk
nltk.download('stopwords')

STOPWORDS = nltk.corpus.stopwords.words('english')
VECTORIZER_PICKLE = 'vectorizer.pkl'

labeled_data = 'labeled_100000.csv'

data_types = {
    'id': np.int64,
    'text': str,
    'country': str,
    'region': str,
    'locality': str,
    'founded': np.float,
    'industry': str,
    'size': str
}

print(f'Reading CSV file {labeled_data}')
data = pd.read_csv(labeled_data, dtype=data_types, index_col='id')

print('Replacing NaN text')
data.text.replace(np.nan, "", inplace=True)

industries = data.industry.unique()
num_of_industries = len(industries)
print(f'Processing data for {num_of_industries} industries')

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = data.text


vectorizer = TfidfVectorizer(stop_words=STOPWORDS, strip_accents='ascii', max_features=10000)

print('Training TF-IDF Vectorizer')
X = vectorizer.fit_transform(corpus)

print('Saving vectorizer to vectorizer.pkl')
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# vectorizer = pickle.load(open(VECTORIZER_PICKLE, 'rb'))
# X = vectorizer.transform(corpus)

from sklearn.cluster import KMeans, Birch, SpectralClustering, MiniBatchKMeans

clustering = MiniBatchKMeans(n_clusters=20, batch_size=10000, init='random')
# clustering = SpectralClustering(n_clusters=20)
y = clustering.fit_predict(X)

# print('Learning 20 clusters')
# X_cluster_dist = clustering.fit_transform(X)

# print('Saving clustering to clustering.pkl')
# with open("clustering.pkl", "wb") as f:
#     pickle.dump(clustering, f)

# pred = clustering.predict(X)
data['cluster'] = y

print('Calculating cluster majority per industry')
cluster_ind = data[['cluster', 'industry']].groupby(['industry']).agg(lambda x:x.value_counts().index[0])
cluster_ind = cluster_ind.cluster.to_dict()
print(cluster_ind)

_cluster_dist = np.zeros(20)
for i,c in cluster_ind.items():
    _cluster_dist[c] += 1
_cluster_dist /= sum(_cluster_dist)
print('cluster size in %: \n' , _cluster_dist*100)
