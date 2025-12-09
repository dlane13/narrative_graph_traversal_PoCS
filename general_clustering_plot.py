"""
Plot metrics for all stories in corpus
"""
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from co_occurence_sentence_undirected_graph import build_cooccurrence_graph_undirected
from utils import preprocess_corpus

def measure(function1, function2, processed_text):
    G = build_cooccurrence_graph_undirected(processed_text)
    return (function1(G), function2(G))

#iterate through each story, measure some metric for each one
target_directory = Path('fairy_tales_texts/')
metrics = {}
for story in target_directory.glob('*.txt'):
    processed_text = preprocess_corpus(story, sentences=True)
    #change these to any two metrics
    metric1 = nx.average_clustering
    metric2 = nx.transitivity
    metrics[story] = measure(metric1, metric2, processed_text)

metric1list = [value[0] for value in metrics.values()]
metric2list = [value[1] for value in metrics.values()]
plt.plot(metric1list, metric2list, '.k')
plt.show()



