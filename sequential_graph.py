
import networkx as nx
import matplotlib.pyplot as plt
from utils import preprocess_corpus #The nltk stuff is in here

""" 
sequential_graph.py

I started simple here, with a graph that followed narrative flow and weighted similar recurrences. 
 

"""


ugly_duck_raw = 'ugly_duckling.txt'

#get the cleaned tokens as well as the tokens by sentence
clean_tokens = preprocess_corpus(ugly_duck_raw, sentences=False)

#Construct the graph
graph = nx.DiGraph()

for i in range(len(clean_tokens)-1):
    #word
    src = clean_tokens[i]
    #next word
    dst = clean_tokens[i+1]

    #add as nodes
    graph.add_node(src)
    graph.add_node(dst)

    #Add to weight of existing occurences when they reappear
    if graph.has_edge(src, dst):
        if 'weight' in graph[src][dst]:
            graph[src][dst]['weight'] += 1
        else:
            graph[src][dst]['weight'] = 1
    else:
        graph.add_edge(src, dst, weight=1)


#Find the weighted in-degree
in_degree = dict(graph.in_degree(weight='weight'))
#list the in-degrees for each word
in_vals = list(in_degree.values())
# or just in-degree
top_in = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop words by in-degree:")
for word, val in top_in:
    print(f"{word:15s} {val}")
plt.hist(in_vals)
plt.show()


#Out-degree reflects in-degree
""" #Find the out
out_degree = dict(graph.out_degree(weight='weight'))
#list the outs for each word
in_vals = list(out_degree.values())
# or just out
top_out = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop words by out-degree:")
for word, val in top_out:
    print(f"{word:15s} {val}")
plt.hist(in_vals)
plt.show() """