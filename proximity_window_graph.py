"""
Create a graph with edge weights defined using the proximity-window metric
"""
import networkx as nx
from utils import preprocess_corpus
import matplotlib.pyplot as plt


raw_text = 'fairy_tales_texts/the_emperors_new_suit.txt'
title = "The Ugly Duckling"

def calc_edge_weight(u_index, v_index):
    return 1 / (v_index - u_index + 1)

#Construct the directed graph
def build_proximity_graph(processed_sentences, proximity_window):
    G = nx.DiGraph()
    edge_weights = {} 

    #iterate through each sentence
    for sentence_index in range(len(processed_sentences)):
        #iterate through each word
        for word in processed_sentences[sentence_index]:
            print(word)
            G.add_node(word)
            #iterate through sentences in proximity window range
            for proximity_index in range(sentence_index, sentence_index + proximity_window):
                if proximity_index < len(processed_sentences):
                    for word_2 in processed_sentences[proximity_index]:
                        #if word and word2 are the same, go to next word
                        if (word == word_2):
                            continue
                        #otherwise, increment edge weights with defined scheme
                        if not((word, word_2) in edge_weights):
                            edge_weights[(word, word_2)] = 0
                        edge_weights[(word, word_2)] += calc_edge_weight(sentence_index, proximity_index)

    #iterate through edges dictionary, add to graph
    avg_weight = 0
    for edge in edge_weights:
        avg_weight += edge_weights[edge]
    avg_weight = avg_weight / len(edge_weights)

    for edge in edge_weights:
        if edge_weights[edge] > avg_weight:
            G.add_edge(edge[0], edge[1], weight=edge_weights[edge])
    
    print(G.nodes())

    return G

def draw_graph():
    processed_sentences = preprocess_corpus(raw_text, sentences=True)
    G = build_proximity_graph(processed_sentences, 1)

    ### Drawing graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, k=0.4, iterations=20, seed=42)  
    edge_labels = nx.get_edge_attributes(G, 'label')
    ### 
    nx.draw(
        G, pos, 
        with_labels=True,
        node_size=1000,
        cmap=plt.cm.cividis, # Colormap for Node Influence
        font_size=10,
        font_color='black',
        edge_cmap=plt.cm.Blues, # Colormap for Edge Weight
        arrows=True,
        arrowsize=15,
        connectionstyle='arc3, rad = 0.1' 
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, bbox=dict(alpha=0), font_color='black', font_size=8)
    plt.show()

draw_graph()
