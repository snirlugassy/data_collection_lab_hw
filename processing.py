import string
import re
import pandas as pd
import nltk
# from nltk import word_tokenize
from bs4 import BeautifulSoup as bsoup

# nltk.download('stopwords')
# nltk.download('punkt')

CHAR_FILTER_TABLE = str.maketrans('', '', string.punctuation + '©™–' + '0123456789')
STOPWORDS = nltk.corpus.stopwords.words('english')

WORD_PATTERN = re.compile(r'\w+')
def tokenize(text):
    tokens = WORD_PATTERN.findall(str(text))
    return [t for t in tokens if (len(t) > 2 or t.lower() not in STOPWORDS)]

# def clean_html(text):
#     return bsoup(text,'html.parser').get_text()

def normalize_text_series(text:pd.Series):
    text = text.str.translate(CHAR_FILTER_TABLE).str.lower()
    text = text.apply(tokenize, convert_dtype=False)
    return text