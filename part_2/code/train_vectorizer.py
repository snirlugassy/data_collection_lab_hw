import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# CONFIG
VECTORIZER_PICKLE = 'vectorizer.pkl'
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

print('Normalizing data')
data.text.replace(np.nan, "", inplace=True)


print('Training TF-IDF Vectorizer')
corpus = data.text
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w\w\w+\b', strip_accents='ascii', max_df=0.7, min_df=1e-4)
vectorizer.fit(corpus)

print(f'Saving vectorizer to {VECTORIZER_PICKLE}')
with open(VECTORIZER_PICKLE, "wb") as f:
    pickle.dump(vectorizer, f)

means = []
industry_mean = {}
for industry, df in data.groupby('industry'):
    X = vectorizer.transform(df.text)
    _mean = X.mean(axis=0)
    means.append(_mean)
    industry_mean[industry] = _mean
means = np.stack([industry_mean[i] for i in industries])

print('Saving industry mean vector to ind_mean_vec.pkl')
with open('ind_mean_vec.pkl', "wb") as f:
    pickle.dump(industry_mean, f)
