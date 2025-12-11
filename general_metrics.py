"""
Plot metrics for all stories in corpus
"""
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from co_occurence_sentence_undirected_graph import build_cooccurrence_graph_undirected
from utils import preprocess_corpus
import random

def measure(function1, function2, graph):
    return (function1(graph), function2(graph))

def two_metrics():
    #iterate through each story, measure some metric for each one
    target_directory = Path('fairy_tales_texts/')
    metrics = {}
    for story in target_directory.glob('*.txt'):
        processed_text = preprocess_corpus(story, sentences=True)
        G = build_cooccurrence_graph_undirected(processed_text)
        #change these to any two metrics
        metric1 = nx.average_clustering
        metric2 = nx.average_node_connectivity
        metrics[story] = measure(metric1, metric2, G)

    metric1list = [value[0] for value in metrics.values()]
    metric2list = [value[1] for value in metrics.values()]
    plt.plot(metric1list, metric2list, '.k')
    plt.xlabel(f"{metric1}")
    plt.ylabel(f"{metric2}")
    plt.show()

def make_random_graph(original_graph: nx.Graph):
    n = len(original_graph.nodes())
    m = original_graph.number_of_edges()
    prob = (2 * m) / (n * (n-1))
    random_graph = nx.fast_gnp_random_graph(len(original_graph.nodes()), prob)
    #print(random_graph.nodes())
    replacement_nodes = {x: list(original_graph)[x] for x in range(len(original_graph.nodes()))}
    G = nx.relabel_nodes(random_graph, replacement_nodes)
    return G

def plot_metrics_random():
    target_directory = Path('fairy_tales_texts/')
    metrics = {}
    random_metrics = {}
    for story in target_directory.glob('*.txt'):
        processed_text = preprocess_corpus(story, sentences=True)
        G = build_cooccurrence_graph_undirected(processed_text)
        metric1 = nx.average_clustering
        metric2 = nx.transitivity
        metrics[story] = measure(metric1, metric2, G)
        random_graph = make_random_graph(G)
        random_metrics[story] = measure(metric1, metric2, random_graph)
    
    metric1list = [value[0] for value in metrics.values()]
    metric2list = [value[1] for value in metrics.values()]
    plt.plot(metric1list, metric2list, '.k')

    randommetric1list = [value[0] for value in random_metrics.values()]
    randommetric2list = [value[1] for value in random_metrics.values()]
    plt.plot(randommetric1list, randommetric2list, '.r')

    plt.show()

plot_metrics_random()
