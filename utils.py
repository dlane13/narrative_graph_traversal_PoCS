import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

"""  
Copy and paste yo story into a .txt and put it into this bad boy.
I'm absolutely 100% sure it will definitely work and will
not need any more cleaning at all whatsoever.

IN: .txt
OUT: a list of words in order of appearance, stop words and punctuation removed

You should be able to call:

from utils import preprocess_corpus
if you are working on a .py file in the same directory
"""

def preprocess_corpus(txt_file):
    #read in the file
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    #get rid of all whitespace
    text = re.sub(r'\s+', ' ', text)
    #Lowercases, strip whitespace
    text = text.lower().strip()
    text.replace("\n", " ")

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    clean_tokens = [
        word.lower()
        for word in tokens
        if word.isalpha() and word.lower() not in stop_words
        ]
    return clean_tokens