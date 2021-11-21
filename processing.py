import string

import nltk
from nltk import word_tokenize
from bs4 import BeautifulSoup as bsoup

nltk.download('popular')

NUMERIC_TABLE = str.maketrans('', '', '0123456789')
PUNC_TABLE = str.maketrans('', '', string.punctuation + '©')
STOPWORDS = nltk.corpus.stopwords.words('english')

def clean_html(text):
    return bsoup(text,'html.parser').get_text()

def remove_numeric(text:str):
    return text.translate(NUMERIC_TABLE)

def remove_punc(text:str):
    return text.translate(PUNC_TABLE)

def normalize_text(text):
    text = clean_html(text)
    text = remove_punc(text)
    text = remove_numeric(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in STOPWORDS]
    tokens = [t for t in tokens if len(t) > 2]
    return tokens
