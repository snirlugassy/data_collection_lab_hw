import string
import re
import pandas as pd
import nltk

nltk.download('stopwords')
nltk.download('punkt')

CHAR_FILTER_TABLE = str.maketrans('', '', string.punctuation + '©™–' + '0123456789')
STOPWORDS = nltk.corpus.stopwords.words('english')

WORD_PATTERN = re.compile(r'\w+')
def tokenize(text):
    tokens = WORD_PATTERN.findall(str(text))
    return [t for t in tokens if (len(t) > 2 or t.lower() not in STOPWORDS)]

def normalize_text_series(text:pd.Series):
    text = text.str.translate(CHAR_FILTER_TABLE)
    text = text.apply(tokenize, convert_dtype=False)
    return text