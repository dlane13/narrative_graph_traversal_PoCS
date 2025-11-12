import nltk
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

"""  
Function to preprocess a given corpus. Can be set to return sentences or just a straight flow of tokens, all will have stop words and unnecessary punctuation removed

IN: .txt
OUT: a list of words in order of appearance, stop words and punctuation removed

You should be able to call:

from utils import preprocess_corpus
if you are working on a .py file in the same directory
"""
def clean_tokens_helper(tokens):
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    clean_tokens = [
        word.lower()
        for word in tokens
        if word.isalpha() and word.lower() not in stop_words
        ]
    return clean_tokens

def preprocess_corpus(txt_file, sentences=False):
    #read in the file
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    #get rid of all whitespace
        text = re.sub(r'\s+', ' ', text)

    if sentences:
        #tokenize the sentences (returns a list of sentences)
        sent_tokens = sent_tokenize(text)
        clean_sentences = []
        #Now process each sentence
        for sentence in sent_tokens:
            #tokenize it
            tokens = word_tokenize(sentence)
            #remove stop words and whatnot
            clean_sentences.append(clean_tokens_helper(tokens))
        return clean_sentences

    else:
        
        #Lowercases, strip whitespace
        text = text.lower().strip()
        text.replace("\n", " ")

        tokens = word_tokenize(text)
        clean_tokens =clean_tokens_helper(tokens)
        return clean_tokens

        