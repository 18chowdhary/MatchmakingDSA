import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math

def create_validation_graph():
    data = pd.read_csv("speeddating.csv", encoding="utf-8")
    validation_subset = data[['iid', 'pid', 'dec']]
    validation_subset = validation_subset.head(200)

    G = nx.DiGraph()
    for i in range(1, 21):
        G.add_node(i)

    for index, row in validation_subset.iterrows():
        if (row['dec'] == 1):
            G.add_edge(row['iid'], row['pid'])

    return G

def create_test_graph():
    data = pd.read_csv("speeddating.csv", encoding="utf-8")
    test_subset = data[['iid', 'pid', 'prob']]
    test_subset = test_subset.head(200)

    G = nx.DiGraph()
    for i in range(1, 21):
        G.add_node(i)

    for index, row in test_subset.iterrows():
        if (math.isnan(row['prob'])):
            cost = 5
        else:
            cost = int(row['prob'])
        G.add_edge(row['iid'], row['pid'], weight=cost)

    return G

def hungarian_algorithm(G):
    pass

def draw_graph(G):
    nx.draw(G, with_labels=True)
    plt.show()

def get_true_matches(g):
    nodes = list(g.nodes)
    matches = []
    for node in nodes:
        for val in g.adj[node].items():
            items = list(g.adj[val[0]].items())
            if float(node) in [x[0] for x in items]:
                matches.append((node,val[0]))
    matches2 = set(tuple(sorted(m)) for m in matches)
    return matches2

if __name__ == '__main__':
    val_graph = create_validation_graph()
    test_graph = create_test_graph()
    draw_graph(test_graph)
    val_matches = get_true_matches(val_graph)
