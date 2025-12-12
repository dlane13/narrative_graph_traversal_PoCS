"""
persistent_homology_sentence_similarity_graph.py

This is the python script to help us drawing the persistent homology from sentence similarity graph 
Pipeline: (1) Read text; (2)Tokenize into sentences; (3) Compute TF-IDF; (4) Compute cosine similarity matrix; (5)Convert similarity to distance; (6) Compute persistent homology (7) Draw persistent diagram 

"""
import nltk
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ripser import ripser
from persim import plot_diagrams

#Read and split sentences
def process_sentences(filename):
    nltk.download("punkt", quiet=True)

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = nltk.sent_tokenize(text)
    
    sentences = [s.strip() for s in sentences]
    #remove duplicate sentences 
    sentences = list(dict.fromkeys(sentences))
    
    print(f"We have sucessfully tokenized {len(sentences)} sentences.")
    return sentences

#Compute TF–IDF matrix
def compute_tfidf(sentences):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix

#Compute cosine similarity matrix
def compute_similarity_matrix(tfidf_matrix):
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix

#Build sentence similarity graph
def build_graph(sentences, sim_matrix, threshold=0.2):
    G = nx.Graph()

    # Add nodes
    for i, sent in enumerate(sentences):
        G.add_node(i, text=sent)

    # Add edges for similarity above threshold
    n = len(sentences)
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sentence_sim = sim_matrix[i][j]
            if sentence_sim >= threshold:
                G.add_edge(i, j, weight=float(sentence_sim))
                edge_count += 1

    print(f"Graph has {G.number_of_nodes()} nodes and {edge_count} edges.")
    return G

def similarity_to_distance(sim_matrix):
    dist_matrix = 1 - sim_matrix
    # replace tiny negative values from numerical errors
    dist_matrix[dist_matrix < 0] = 0
    return dist_matrix

def compute_persistent_homology(dist_matrix):
    results = ripser(dist_matrix, distance_matrix=True, maxdim=1)
    diagrams = results["dgms"]
    return diagrams

def plot_persistence_diagram(diagrams, title="Persistence Diagram"):
    plt.figure(figsize=(8, 6))
    plot_diagrams(diagrams, show=False)
    plt.title(title)
    plt.savefig("persistence_diagram_example")
    plt.show()


#Main execution 
if __name__ == "__main__":
    filename_emperors = "./fairy_tales_texts/the_emperors_new_suit.txt"
    
    sentences_emperors = process_sentences(filename_emperors)
    tfidf_matrix_emperors = compute_tfidf(sentences_emperors)
    sim_matrix_emperors = compute_similarity_matrix(tfidf_matrix_emperors)
    
    sentence_graph_emperors = build_graph(sentences_emperors,sim_matrix_emperors, threshold=0.3)
    
    #draw_graph(sentence_graph_emperors)

    dist_matrix_emperors = similarity_to_distance(sim_matrix_emperors)
    diagrams_emperors = compute_persistent_homology(dist_matrix_emperors)
    plot_persistence_diagram(diagrams_emperors, title="Sentence similarity – Persistent homology for The Emporor's New Suit")
        