import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# CONFIG
VECTORIZER_PICKLE = 'vectorizer.pkl'
CLUSTERING_PICKLE = 'clustering.pkl'
INDUSTRIES_LIST = '../industries.txt'

LABELED_DATA_PATH = '../labeled.csv'

data_types = {
    'text': str,
    'industry': str,
}

print('Loading industries')
industries = [x.strip() for x in open(INDUSTRIES_LIST,'r').readlines()]

print('Reading data')
data = pd.read_csv(LABELED_DATA_PATH, usecols=['text', 'industry'], dtype=data_types, engine='c')

print('Number of samples texts: ', len(data))
data.text.replace(np.nan, "", inplace=True)

vectorizer = pickle.load(open(VECTORIZER_PICKLE, 'rb'))

means = []
industry_mean = {}
for industry, df in data.groupby('industry'):
    print(industry, df.shape)
    X = vectorizer.transform(df.text)
    _mean = X.mean(axis=0)
    means.append(_mean)
    industry_mean[industry] = _mean

means = np.stack([industry_mean[i] for i in industries])

print('Loading industry mean vector from ind_mean_vec.pkl')
industry_mean = pickle.load(open('ind_mean_vec.pkl', "rb"))
assert set(industries) == set(industry_mean.keys())


means = np.array(np.stack([industry_mean[i] for i in industries]))

print('Clustering industry means with k=20')
clustering = KMeans(n_clusters=20).fit(means)

print('Saving clustering to clustering.pkl')
with open(CLUSTERING_PICKLE, "wb") as f:
    pickle.dump(clustering, f)

industry2cluster = {}
for i in industries:
    x = np.array(industry_mean[i])
    industry2cluster[i] = int(clustering.predict(x)) + 1

ind2clustre_df = pd.DataFrame(list(industry2cluster.items()), columns=['industry', 'clusterID'])
ind2clustre_df.to_csv('industry2cluster_206312506.csv', index=False)
