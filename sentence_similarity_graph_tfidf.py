"""
sentence_similarity_graph_tfidf.py

This is the python script to help us building the sentence similarity graph using TF-IDF vectors.
Pipeline: (1) Read text; (2)Tokenize into sentences; (3) Compute TF-IDF; (4) Compute cosine similarity matrix; (5) Define and draw graph

"""
import nltk
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Read and split sentences
def process_sentences(filename):
    nltk.download("punkt", quiet=True)

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = nltk.sent_tokenize(text)
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

def draw_graph(G):
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    # Extract weight info from edges
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=100,
        width=[w * 3 for w in edge_weights],
        node_color="skyblue",
        edge_color=edge_weights,
        edge_cmap=plt.cm.plasma
    )
    plt.title("Sentence similarity graph") 
    plt.show()    
    print("Drawing Sentence similarity graph (TF–IDF)")

#Main execution 
if __name__ == "__main__":
    filename = "ugly_duckling.txt"

    sentences = process_sentences(filename)
    tfidf_matrix = compute_tfidf(sentences)
    sim_matrix = compute_similarity_matrix(tfidf_matrix)

    G = build_graph(sentences, sim_matrix, threshold=0.2)

    draw_graph(G)
