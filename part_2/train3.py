import random
import pickle
import pandas as pd
import numpy as np
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


nltk.download('stopwords')

CHUNK_SIZE = 100000
STOPWORDS = nltk.corpus.stopwords.words('english')
VECTORIZER_PICKLE = 'vectorizer.pkl'

labeled_data = 'labeled.csv'

data_types = {
    'text': str,
    'industry': str,
}

print('Loading industries')
industries = [x.strip() for x in open('industries.txt','r').readlines()]

print('Reading data')
data = pd.read_csv(labeled_data, usecols=['text', 'industry'], dtype=data_types, engine='c')
data.reset_index(drop=True, inplace=True)
data.text.replace(np.nan, "", inplace=True)
print(data.info(memory_usage='deep'))

print('Training TF-IDF Vectorizer')
corpus = data.text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print('Saving vectorizer to vectorizer.pkl')
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print('Calculating industry means')
industry_mean = {}
for i in industries:
    X_ind = X[data[data['industry']==i].index]
    _mean = X_ind.mean(axis=0)
    industry_mean[i] = _mean

means = np.stack([industry_mean[i] for i in industries])

print('Clustering industry means with k=20')
clustering = KMeans(n_clusters=20)
clustering.fit(means)

print('Saving clustering to clustering.pkl')
with open("clustering.pkl", "wb") as f:
    pickle.dump(clustering, f)
