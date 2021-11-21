import string
import re

import nltk
# from nltk import word_tokenize
from bs4 import BeautifulSoup as bsoup

# nltk.download('stopwords')
# nltk.download('punkt')

CHAR_FILTER_TABLE = str.maketrans('', '', string.punctuation + '©' + '0123456789')
STOPWORDS = nltk.corpus.stopwords.words('english')

WORD_PATTERN = re.compile(r'\w+')
def tokenize(text):
    return WORD_PATTERN.findall(text)

def clean_html(text):
    return bsoup(text,'html.parser').get_text()

def normalize_text(text):
    text = clean_html(text)
    text = text.translate(CHAR_FILTER_TABLE)
    tokens = tokenize(text)
    tokens = [t for t in tokens if t.lower() not in STOPWORDS]
    tokens = [t for t in tokens if len(t) > 2]
    return tokens
