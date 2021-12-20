import random
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

random.seed(42)

<<<<<<< HEAD
CHUNK_SIZE = 50000
STOPWORDS = nltk.corpus.stopwords.words('english')
=======
# CONFIG
CHUNK_SIZE = 100000
>>>>>>> b885e4f607569cd35a2baf793071edd915cd6d9f
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
<<<<<<< HEAD
industries = [x.strip() for x in open('industries.txt','r').readlines()]

industry_mean = {}

data = pd.read_csv(labeled_data, usecols=['text', 'industry'], dtype=data_types, engine='c')

print(f'Chunk {iter}')

data.reset_index(drop=True, inplace=True)

print('Replacing NaN text')
data.text.replace(np.nan, "", inplace=True)
# print(data.info(memory_usage='deep'))

corpus = data.text

vectorizer = TfidfVectorizer()

print('Training TF-IDF Vectorizer')
X = vectorizer.fit_transform(corpus)
print("X Shape: ", X.shape)

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


# # print(f'Reading CSV file {labeled_data}')
# # data = pd.read_csv(
# #     labeled_data, 
# #     usecols=['text', 'industry'],
# #     dtype=data_types,
# #     engine='c',
# #     # skiprows=lambda i: i>0 and random.random() > p
# # )

# data.reset_index(drop=True, inplace=True)

# print('Replacing NaN text')
# data.text.replace(np.nan, "", inplace=True)
# print(data.info(memory_usage='deep'))

# industries = data.industry.unique()
=======
industries = [x.strip() for x in open(INDUSTRIES_LIST,'r').readlines()]
>>>>>>> b885e4f607569cd35a2baf793071edd915cd6d9f

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
