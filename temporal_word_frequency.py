"""
Graph temporal word frequency for words in a story
-First objective: plot frequencies for four random words (done)
-Second objective: plot frequencies for two content-bearing, two action-bearing words
"""
import random
from utils import preprocess_corpus
from utils import temporal_frequency
from utils import rank_freq
import utils
import matplotlib.pyplot as plt
import numpy as np

#this returns a list of lowercased, whitespace-free words
processed_ugly_duckling = preprocess_corpus('fir_tree.txt')
processed_sentences = preprocess_corpus('fir_tree.txt', True)

#first: choose four random words
def random_four():
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

def two_cb_two_ab():
    token_freqs, token_ranks, rank_freq, rank_breakpoint = utils.rank_freq(processed_ugly_duckling)
    len_token_list = len(processed_sentences)

    sorted_items = sorted(token_ranks.items(), key=lambda item: item[1])
    sorted_token_ranks = dict(sorted_items)

    content_bearing_tokens = [key for key, value in sorted_token_ranks.items() if value < rank_breakpoint]
    action_driving_tokens = [key for key, value in sorted_token_ranks.items() if value >= rank_breakpoint]

    print(content_bearing_tokens)
    print(action_driving_tokens)

    content_bearing_rand = []
    action_driving_rand = []

    for i in range(2):
        content_bearing_rand.append(processed_ugly_duckling[random.randint(0,len(content_bearing_tokens) - 1)])

    for i in range(2):
        action_driving_rand.append(processed_ugly_duckling[random.randint(0,len(action_driving_tokens) - 1)])

    fig, axs = plt.subplots(2,2)

    axs[0,0].plot(range(len_token_list), temporal_frequency(content_bearing_rand[0], processed_sentences))
    axs[0,0].set_title(f"{content_bearing_rand[0]} (CB)")

    axs[0,1].plot(range(len_token_list), temporal_frequency(content_bearing_rand[1], processed_sentences))
    axs[0,1].set_title(f"{content_bearing_rand[1]} (CB)")

    axs[1,0].plot(range(len_token_list), temporal_frequency(action_driving_rand[0], processed_sentences))
    axs[1,0].set_title(f"{action_driving_rand[0]} (AB)")

    axs[1,1].plot(range(len_token_list), temporal_frequency(action_driving_rand[1], processed_sentences))
    axs[1,1].set_title(f"{action_driving_rand[1]} (AB)")

    plt.show()

def average_freq():
    token_freqs, token_ranks, rank_freq, rank_breakpoint = utils.rank_freq(processed_ugly_duckling)
    len_token_list = len(processed_sentences)

    sorted_items = sorted(token_ranks.items(), key=lambda item: item[1])
    sorted_token_ranks = dict(sorted_items)

    content_bearing_tokens = [key for key, value in sorted_token_ranks.items() if value < rank_breakpoint]
    action_driving_tokens = [key for key, value in sorted_token_ranks.items() if value >= rank_breakpoint]

    cb_temp_freq = []
    for token in content_bearing_tokens:
        cb_temp_freq.append(np.array(temporal_frequency(token, processed_sentences)))

    cb_avg_freq = utils.average_temporal_frequency(cb_temp_freq)

    ab_temp_freq = []
    for token in action_driving_tokens:
        ab_temp_freq.append(np.array(temporal_frequency(token, processed_sentences)))

    ab_avg_freq = utils.average_temporal_frequency(ab_temp_freq)

    fig, axs = plt.subplots(2,1)
    
    axs[0].plot(range(len_token_list), cb_avg_freq)
    axs[0].set_title("Average Content-Bearing Frequency")

    axs[1].plot(range(len_token_list), ab_avg_freq)
    axs[1].set_title("Average Action-Driving Frequency")

    plt.subplots_adjust(hspace=0.5)

    plt.show()


average_freq()