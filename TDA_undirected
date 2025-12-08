%matplotlib inline

# ---------- Imports ----------
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from operator import itemgetter

from utils import preprocess_corpus

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from gtda.mapper import make_mapper_pipeline, CubicalCover, Projection

# For interactive Mapper visualisation
from pyvis.network import Network

# --------------------------------------------------------
# 1. Build co-occurrence graph
# --------------------------------------------------------
ugly_duck_raw = "ugly_duckling.txt"

def build_cooccurrence_graph_undirected(processed_sentences):
    G = nx.Graph()
    edge_weights = {}
    for sentence in processed_sentences:
        unique_words = list(set(sentence))
        for u, v in combinations(unique_words, 2):
            if u == v:
                continue
            pair = tuple(sorted((u, v)))
            edge_weights[pair] = edge_weights.get(pair, 0) + 1

    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)
    return G

processed_sentences = preprocess_corpus(ugly_duck_raw, sentences=True)
print(f"The corpus contains {len(processed_sentences)} clean sentences.")

G = build_cooccurrence_graph_undirected(processed_sentences)
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")

# Show 10 strongest edges
all_edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
top10 = sorted(all_edges, key=itemgetter(2), reverse=True)[:10]
print("Top 10 strongest co-occurring word pairs:")
for u, v, w in top10:
    print(f"  '{u}' — '{v}' : weight {w}")

# --------------------------------------------------------
# 2. Adjacency matrix -> feature matrix
# --------------------------------------------------------
nodes = sorted(G.nodes())
A = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=float)
print("\nAdjacency matrix shape:", A.shape)

# --------------------------------------------------------
# 3. PCA filter (2D) + Mapper pipeline
# --------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(A)   # (n_words, 2)

# Mapper filter function
filter_func = Projection(columns=[0, 1])

# Cover in 2D (a grid of overlapping rectangles)
cover = CubicalCover(
    n_intervals=8,      # number of intervals along each PCA axis
    overlap_frac=0.5,   # 50% overlap between intervals
)

# Clustering in each bin: DBSCAN (no fixed k, robust to small bins)
clusterer = DBSCAN(
    eps=0.5,        # tune if needed
    min_samples=3   # at least 3 points to form a cluster
)

mapper_pipe = make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    n_jobs=1,
    verbose=True,
)

# --------------------------------------------------------
# 4. Compute Mapper graph (igraph.Graph)
# --------------------------------------------------------
mapper_graph = mapper_pipe.fit_transform(X_pca)
print(
    "\nMapper graph has {} nodes and {} edges."
    .format(mapper_graph.vcount(), mapper_graph.ecount())
)

# --------------------------------------------------------
# 5. Static plot of Mapper graph (optional)
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

if mapper_graph.vcount() == 0:
    ax.text(0.5, 0.5, "Empty Mapper graph (no clusters found)",
            ha="center", va="center", fontsize=12)
    ax.axis("off")
else:
    layout = mapper_graph.layout_fruchterman_reingold()
    coords = np.array(layout.coords)
    xs, ys = coords[:, 0], coords[:, 1]

    # node size = number of words in that node (cluster size)
    sizes = [len(el) for el in mapper_graph.vs["node_elements"]]
    node_sizes = np.array(sizes) * 15

    # draw edges
    for e in mapper_graph.es:
        i, j = e.tuple
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                color="lightgray", linewidth=0.5, zorder=1)

    # draw nodes
    sc = ax.scatter(xs, ys,
                    s=node_sizes,
                    c=sizes,
                    cmap="viridis",
                    edgecolors="k",
                    zorder=2)

    ax.set_title("Mapper graph of 'The Ugly Duckling' co-occurrence geometry")
    ax.axis("off")

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 6. Interactive Mapper graph with PyVis (zoom + labels)
# --------------------------------------------------------
if mapper_graph.vcount() > 0:
    # Build a NetworkX graph from Mapper output with word-based labels
    G_mapper = nx.Graph()
    node_elements = mapper_graph.vs["node_elements"]  # list of lists of point indices

    for nid, idx_list in enumerate(node_elements):
        cluster_words = [nodes[i] for i in idx_list]

        # Short label on the node
        label = ", ".join(cluster_words[:3]) if cluster_words else f"Node {nid}"

        # Tooltip on hover (first 20 words)
        title = "<br>".join(cluster_words[:20]) if cluster_words else f"Node {nid}"

        G_mapper.add_node(
            nid,
            label=label,
            title=title,
            size=max(10, len(idx_list) * 3),  # size ~ cluster size
        )

    # Add edges from igraph
    for e in mapper_graph.es:
        i, j = e.tuple
        G_mapper.add_edge(i, j)

    # Create interactive network
    net = Network(
        notebook=True,
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
    )
    net.barnes_hut()        # physics layout
    net.from_nx(G_mapper)
    net.show_buttons(filter_=["physics", "layout"])

    # This writes an HTML file and (in Jupyter) opens it inline or in a new tab
    net.show("mapper_ugly_duckling.html")
