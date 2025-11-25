#Standard import 
import networkx as nx
import matplotlib.pyplot as plt
from utils import preprocess_corpus #The nltk stuff is in here
from operator import itemgetter
from itertools import combinations
""" 
sentence_co_occurence_undiirected_graph.py

SImialr to sentence_co_occurence_directed graph. This script builds a directed co-occurrence graph where an edge (between u and v) exists if both words u and v in the same sentence. We counted the weight by simply counting the total number of times this pair occurs across all sentences. 

"""

ugly_duck_raw = 'ugly_duckling.txt'

#get the cleaned tokens as well as the tokens by sentence
#clean_tokens = preprocess_corpus(ugly_duck_raw, sentences=False)

#Construct the graph
def build_cooccurrence_graph_undirected(processed_sentences):
    G = nx.Graph()
    edge_weights = {} 
    for sentence in processed_sentences:
        unique_words = list(set(sentence)) #no duplicates - avoid self-loop
        for u,v in combinations(unique_words, 2):
            if u ==v: 
                continue 
            pair = tuple(sorted((u, v)))  # undirected pairing 
            edge_weights[pair] = edge_weights.get(pair, 0) + 1

    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)

    return G

#Process the program 
processed_sentences = preprocess_corpus(ugly_duck_raw, sentences = True)
print(f"The corpus contains {len(processed_sentences)} clean sentences.")

print("Building undirected co-occurrence graph") 
G = build_cooccurrence_graph_undirected(processed_sentences)

# Show 10 strongest edges
all_edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
top10 = sorted(all_edges, key=itemgetter(2), reverse=True)[:10]

print("Top 10 strongest co-occurring word pairs:")
for u, v, w in top10:
    print(f"  '{u}' — '{v}' : weight {w}")

#Draw graph 
def draw_graph(graph, title="Co-occurrence undirected graph"):
    ### Find node influences 
    pagerank = nx.pagerank(graph, weight='weight')
    ### Prepare components- mutliple so that they are easier to see 
    node_sizes = [pagerank[n] * 15000 for n in graph.nodes()]
    edge_weights = [d['weight'] for _, _, d in graph.edges(data=True)]
    edge_widths = [w * 0.75 for w in edge_weights]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, k=0.4, iterations=20, seed=42)

    nx.draw(
        graph, pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=[pagerank[n] for n in graph.nodes()],
        cmap=plt.cm.cividis,
        edge_color=edge_weights,
        edge_cmap=plt.cm.Blues,
        width=edge_widths,
        arrows=graph.is_directed(),
        arrowsize=12,
        connectionstyle='arc3, rad=0.1'
    )

    plt.title(title)
    plt.show()

draw_graph(G, "Undirected Co-occurrence Graph")