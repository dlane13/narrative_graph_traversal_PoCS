#Standard import 
import networkx as nx
import matplotlib.pyplot as plt
from utils import preprocess_corpus #The nltk stuff is in here
from operator import itemgetter

""" 
sentence_co_occurence.py

This script builds a directed co-occurrence graph where an edge (u -> v) exists  if word u precedes word v in the same sentence. We counted the weight by simply counting the total number of times this directed pair occurs across all sentences. 

"""

ugly_duck_raw = 'ugly_duckling.txt'

#get the cleaned tokens as well as the tokens by sentence
#clean_tokens = preprocess_corpus(ugly_duck_raw, sentences=False)

#Construct the graph
def build_cooccurrence_graph (processed_sentences):
    G = nx.DiGraph()
    edge_weight_dict = {}
    #Iterate through sentences 
    for sentence in processed_sentences: 
    #Iterate through words
        for idx_1 in range(len(sentence)):
            u = sentence[idx_1] #identify u as the first word in pair

            for idx_2 in range(idx_1+1, len(sentence)):
                v = sentence[idx_2] #identify v as the second in peair 

                if u == v: 
                    continue #skip self-looping 

                edge = (u,v) #draw an edge 

                edge_weight_dict[edge] = edge_weight_dict.get(edge, 0) + 1 #increase the weight for directed edge
    # Add edges and weights to graph 
    for (u,v), weight in edge_weight_dict.items(): 
        G.add_edge (u,v,weight=weight)
    return G

#Process the program 
processed_sentences = preprocess_corpus(ugly_duck_raw, sentences = True)
print(f"The corpus contains {len(processed_sentences)} clean sentences.")
graph = build_cooccurrence_graph(processed_sentences)
#Calculate node importance using pagerank 
node_influ = nx.pagerank(graph, weight='weight')
#Output of top edges 
all_edges = [(u, v, data['weight']) for u, v, data in graph.edges(data=True)] #list of all edges 
top_edges = sorted(all_edges, key=itemgetter(2), reverse=True)[:10] 
#Just print out top 10 
for u,v, weight in top_edges: 
    print(f" pair \"{u}\" to \"{v}\" has {weight} counts") 

#Draw graph 

### Prepare components- mutliple so that they are easier to see 
#Node influence 
pagerank = nx.pagerank(graph, weight='weight')
### Prepare components- mutliple so that they are easier to see 
node_sizes = [pagerank.get(node, 0) * 15000 for node in graph.nodes()] 
edge_widths = [d['weight'] * 0.75 for u, v, d in graph.edges(data=True)]
### Get edge weights 
edge_weights_for_plot = [d['weight'] for u, v, d in graph.edges(data=True)]

### Drawing graph
plt.figure(figsize=(14, 14))
pos = nx.spring_layout(graph, k=0.4, iterations=20, seed=42) 
### 
nx.draw(
    graph, pos, 
    with_labels=True, 
    node_size=node_sizes, 
    node_color=[pagerank.get(node, 0) for node in graph.nodes()], # Node color still represents influence
    cmap=plt.cm.cividis, # Colormap for Node Influence
    font_size=10,
    font_color='black',
    edge_color=edge_weights_for_plot, # Edge color based on weight
    edge_cmap=plt.cm.Blues, # Colormap for Edge Weight
    width=edge_widths,
    arrows=True,
    arrowsize=15,
    connectionstyle='arc3, rad = 0.1' 
)
plt.show()
