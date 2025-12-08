
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from utils import preprocess_corpus, rank_freq #The nltk stuff is in here
import pwlf

""" 
Creating a Network of sentences, where each node contains a sub-network representing the sentence. 
Weights within the sentence subnetwork are weighted from a global zipf distribution measurment.
"""
story_paths = ['ugly_duckling.txt','fir_tree.txt',]
story_raw = 'ugly_duckling.txt'

"""  ----------------------------------------------------------
Story-level Zipf, after preprocessing
----------------------------------------------------------""" 
#Get the list of tokens after removing stop words
all_tokens = preprocess_corpus(story_raw, keep_stopwords=True, sentences=False)

#Make a dictionary of frequencies
token_freqs, token_ranks,rank_frequencies, rank_breakpoint = rank_freq(all_tokens) #here we have "ranked frequencies" not "rank frequencies"
N_RANKS = len(token_ranks.values())



"""  ----------------------------------------------------------
Find rank where change in scaling occurs
----------------------------------------------------------""" 
#plot like traditional zipf first
sorted_token_freqs = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)
ranks = range(1, len(sorted_token_freqs)+1)
freqs = [freq for token, freq in sorted_token_freqs]
X1 = np.log10(np.array(ranks)) #frequencies
Y1 = np.log10(np.array(freqs)) #ranks
#Print it to visualize where the split is
plt.scatter(X1, Y1)
plt.axvline(x=np.log10(rank_breakpoint), color='red')
plt.xlabel('Log10(Rank)')
plt.ylabel('Log10(Frequency)')
plt.title(f"Zipf Rank-frequency Plot for Non-stop-tokens in '{story_raw}', with calculated knee point, n={len(ranks)+1}")
plt.show()

#Piecewise linear fit to find the breakpoint:
X = np.log10(np.array(list(rank_frequencies.keys())))
Y = np.log10(np.array(list(rank_frequencies.values())))

#Print it to visualize where the split is
plt.scatter(X, Y)
plt.axvline(x=np.log10(rank_breakpoint), color='red')
plt.xlabel('Log10(Rank)')
plt.ylabel('Log10(Frequency)')
plt.title(f"Binned Rank-frequency Plot for Non-stop-tokens in '{story_raw}', with calculated knee point, n={len(rank_frequencies.keys())+1}")
plt.show()

"""  ----------------------------------------------------------
Split the words on each side of the division
----------------------------------------------------------""" 
#our two categories of words
content_bearing_tokens = {} #token:rank
content_modifying_tokens = {} #token:rank

#Loop through the tokens and sort them
for token, rank in token_ranks.items():
    if rank < rank_breakpoint:
        content_bearing_tokens[token] = rank
    elif rank >= rank_breakpoint:
        content_modifying_tokens[token] = rank

print("Content Bearing Total:")
print(len(content_bearing_tokens))
print("Content Modifying Total:")
print(len(content_modifying_tokens))

print(content_bearing_tokens.keys())


#----------------------------------------------------------
# Visualize sentence content by word type
#---------------------------------------------------------- 
#Get the sentences (list of lists, from our utils file)
sent_tokens = preprocess_corpus(story_raw, sentences = True)

#Get the proportion of words in each sentence that are content-bearing
sent_pct_content_bearing = []
for sent_list in sent_tokens:
    total_count = len(sent_list)
    content_bearing_count = 0
    for token in sent_list:
        if token in content_bearing_tokens.keys():
            content_bearing_count +=1 
    
    sent_pct_content_bearing.append(content_bearing_count/total_count)

plt.bar(range(1, len(sent_pct_content_bearing)+1), sent_pct_content_bearing)
plt.title(f"Percentage content-bearing for {story_raw}")
plt.show()




"""
def subgraph_edge_weights(sentence, i, j, token_ranks, N_RANKS):
    #get the tokens from the index
    token1 = sentence[i]
    token2 = sentence[j]
    #get the ranks for the tokens
    token1_rank = token_ranks[token1]
    token2_rank = token_ranks[token2]
    # normalize them
    R_norm_1 = token1_rank/N_RANKS
    R_norm_2 = token2_rank/N_RANKS

    #Normalized sentence distance
    sentence_dist = (j-i)/len(sentence)
    return 


#loop through the sentences
for sentence in sent_tokens:
    #make a graph for each
    subgraph = nx.Graph()
    #undirected, fully connected graph of nodes from sentence tokens
    subgraph.add_nodes_from(sentence)
    #add all edges
    for i,j in combinations(range(len(sentence)), 2):
        #weight = 
        subgraph.add_edge(sentence[i],sentence[j]) """



        


        




