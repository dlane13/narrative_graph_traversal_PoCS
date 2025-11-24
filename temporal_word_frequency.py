"""
Graph temporal word frequency for words in a story
-First objective: plot frequencies for two content-bearing words and two plot-driving words
"""
import random
from utils import preprocess_corpus
from utils import temporal_frequency
import matplotlib.pyplot as plt

#this returns a list of lowercased, whitespace-free words
processed_ugly_duckling = preprocess_corpus('ugly_duckling.txt')
processed_sentences = preprocess_corpus('ugly_duckling.txt', True)

#first: choose four random words
len_token_list = len(processed_sentences)

random_words = []
for i in range(4):
    random_words.append(processed_ugly_duckling[random.randint(0,len_token_list)])

fig, axs = plt.subplots(2,2)

axs[0,0].plot(range(len_token_list), temporal_frequency(random_words[0], processed_sentences))
axs[0,0].set_title(f"{random_words[0]}")

axs[0,1].plot(range(len_token_list), temporal_frequency(random_words[1], processed_sentences))
axs[0,1].set_title(f"{random_words[1]}")

axs[1,0].plot(range(len_token_list), temporal_frequency(random_words[2], processed_sentences))
axs[1,0].set_title(f"{random_words[2]}")

axs[1,1].plot(range(len_token_list), temporal_frequency(random_words[3], processed_sentences))
axs[1,1].set_title(f"{random_words[3]}")

plt.show()

