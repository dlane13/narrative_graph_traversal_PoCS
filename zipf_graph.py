"""
Create graph where nodes are content-bearing words
"""
import networkx as nx
import utils
import matplotlib.pyplot as plt

ugly_duck_raw = 'ugly_duckling.txt'
processed_ugly_duckling = utils.preprocess_corpus(ugly_duck_raw)
processed_sentences = utils.preprocess_corpus(ugly_duck_raw, True)

token_freqs, token_ranks, rank_freq, rank_breakpoint = utils.rank_freq(processed_ugly_duckling)
len_token_list = len(processed_sentences)

sorted_items = sorted(token_ranks.items(), key=lambda item: item[1])
sorted_token_ranks = dict(sorted_items)

content_bearing_tokens = [key for key, value in sorted_token_ranks.items() if value < rank_breakpoint]
action_driving_tokens = [key for key, value in sorted_token_ranks.items() if value >= rank_breakpoint]

def build_zipf_edges_graph(processed_sentences, proximity_window):
    G = nx.Graph()

    #add nodes
    G.add_nodes_from(content_bearing_tokens)

    #add edges
    action_driving_neighbors = {}
    for sentence in processed_sentences:
        #for each sentence, find a content bearing word
        for idx_1 in range(len(sentence)):
            if sentence[idx_1] in content_bearing_tokens:
                if not (sentence[idx_1] in action_driving_neighbors):
                    action_driving_neighbors[sentence[idx_1]] = []
                #if there is an action_driving word within the proximity window after the word, keep track of it
                for idx_2 in range(proximity_window):
                    if idx_1 + idx_2 >= len(sentence):
                        break
                    else:
                        if sentence[idx_1 + idx_2] in action_driving_tokens:
                            action_driving_neighbors[sentence[idx_1]].append([sentence[idx_1 + idx_2], idx_2])

    for word1 in action_driving_neighbors:
        for word2 in action_driving_neighbors:
            closest_similar = ""
            closest_distance = 500
            if not (word1 == word2):
                #keep track of the closest word they have in common, add as an edge at end
                #insane nested for loops...
                for action_word1 in action_driving_neighbors[word1]:
                    for action_word2 in action_driving_neighbors[word2]:
                        avg_distance = (action_word1[1] + action_word2[1]) / 2
                        if (action_word1[0] == action_word2[0]) and (avg_distance < closest_distance):
                            closest_similar = action_word1[0]
                            closest_distance = avg_distance
                if not (closest_similar == ""):
                    G.add_edge(word1, word2, label=closest_similar)

    return G

def draw_graph():
    G = build_zipf_edges_graph(processed_sentences, 3)

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
