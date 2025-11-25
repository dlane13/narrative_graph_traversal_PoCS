import nltk
import re
import string
import pwlf
import numpy as np
from scipy.optimize import minimize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import pwlf

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
            #remove stop words and whatnot - omit empty sentences
            clean = clean_tokens_helper(tokens)
            if len(clean) != 0:
                clean_sentences.append(clean)
        return clean_sentences

    else:
        
        #Lowercases, strip whitespace
        text = text.lower().strip()
        text.replace("\n", " ")

        tokens = word_tokenize(text)
        clean_tokens =clean_tokens_helper(tokens)
        return clean_tokens
    
def find_breakpoint(rank_frequencies):
    #Piecewise linear fit to find the breakpoint:
    X = np.log10(np.array(list(rank_frequencies.keys())))
    Y = np.log10(np.array(list(rank_frequencies.values())))

    #run pwlf
    fit = pwlf.PiecewiseLinFit(X,Y)
    #Best fit of two lines to find breakpoint
    res = fit.fit(2)
    #get the transition point
    break_point = res[1]
    #Make it in terms of rank
    rank_breakpoint = np.floor(10**break_point)
    
    return rank_breakpoint

def rank_freq(all_tokens):
    """  
    Calculates global (story-level) zipf-related values as a set of dictionaries: 
    token_freqs: Frequency of each unique token, as token:frequency
    token_ranks: Rank of each token, based on ordered frequencies (rank 1 = highest freq, ties allowed). as token:rank
    rank_freq: Frequency of each rank, as rank:frequency
    """
    #Initialize dicitonaries
    token_freqs = {} 
    token_ranks = {} 
    rank_freq = {} 
    
    for token in all_tokens:
        token_lower = token.lower()
        #if the word has been recorded, add 1, if not, add it
        if token_lower in token_freqs.keys():
            token_freqs[token_lower] += 1
        else:
            token_freqs[token_lower] = 1

    #token ranks - sort the dict by frequency
    sorted_freqs = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)

    
    counter = 1
    for token, freq in sorted_freqs:
        if freq not in rank_freq.values():
            rank_freq[counter] = freq
            token_ranks[token] = counter
            counter +=1
        else:
            token_ranks[token] = counter - 1
    
    rank_breakpoint = find_breakpoint(rank_freq)
        
    return token_freqs, token_ranks, rank_freq, rank_breakpoint
    
def temporal_frequency(token: str, all_sentences: list) -> list[int] :
    #iterate through each sentence one at a time
    #check how many times a token appears in that sentence, store in list
    #return list once all sentences have been iterated through
    token_freqs = [0] * len(all_sentences)

    index = 0
    for sentence in all_sentences:
        for t in sentence:
            if t == token:
                token_freqs[index] += 1
        
        index += 1
    
    return token_freqs

def optimal_break_linear_regression(x: np.array, y: np.array) -> list:
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    number_of_line_segments = 2
    my_pwlf.use_custom_opt(number_of_line_segments)

    xGuess = np.zeros(number_of_line_segments - 1)
    xGuess[0] = 0.3

    res = minimize(my_pwlf.fit_with_breaks_opt, xGuess)

    x0 = np.zeros(number_of_line_segments + 1)
    x0[0] = np.min(x)
    x0[-1] = np.max(x)
    x0[1:-1] = res.x

    my_pwlf.fit_with_breaks(x0)

    xHat = np.linspace(min(x), max(x), num=10000)
    yHat = my_pwlf.predict(xHat)

    return [xHat, yHat]
