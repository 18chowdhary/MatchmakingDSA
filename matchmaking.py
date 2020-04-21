import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import math
import numpy as np

def create_validation_graph():
    data = pd.read_csv("speeddating.csv", encoding="utf-8")
    validation_subset = data[['iid', 'pid', 'gender', 'dec']]
    validation_subset = validation_subset.head(200)

    G = nx.DiGraph()
    G.add_nodes_from(range(1, 11), bipartite=0)
    G.add_nodes_from(range(11, 21), bipartite=1)

    for index, row in validation_subset.iterrows():
        if (row['dec'] == 1):
            G.add_edge(row['iid'], row['pid'])

    return G

def create_test_graph():
    data = pd.read_csv("speeddating.csv", encoding="utf-8")
    test_subset = data[['iid', 'pid', 'prob']]
    test_subset = test_subset.head(200)

    G = nx.DiGraph()
    G.add_nodes_from(range(1, 11), bipartite=0)
    G.add_nodes_from(range(11, 21), bipartite=1)

    for index, row in test_subset.iterrows():
        if (math.isnan(row['prob'])):
            cost = 5
        else:
            cost = 10-int(row['prob'])
        if cost <= 5:
            color = "g"
        else:
            color = "o"
        G.add_edge(row['iid'], row['pid'], weight=cost, edge_color=color)

    # print(nx.adjacency_matrix(G))

    return G

def hungarian_algorithm(G):
    # create adjacency matrix using numpy
    num_nodes = len(G.nodes())
    adj_matrix = 10*np.ones([num_nodes, num_nodes])
    for edge in list(G.edges()):
        node_1 = int(edge[0]-1)
        node_2 = int(edge[1]-1)
        edge_weight = G.get_edge_data(*edge)['weight']
        adj_matrix[node_1][node_2] = edge_weight

    done = False
    while not done:
        # find the minimum of each row & subtract that min from each row
        for i in range(num_nodes):
            row = adj_matrix[i]
            min_weight = np.min(row)
            adj_matrix[i] = np.subtract(row, min_weight)

        # find the min of each col & subtract from each col
        for i in range(num_nodes):
            col = adj_matrix[:, i]
            # print(col)
            min_weight = np.min(col)
            adj_matrix[:,i] = np.subtract(col, min_weight)

        zeros = []
        # check if all rows and all columns have at least 1 0
        row_done = True
        for i in range(num_nodes):
            row = adj_matrix[i]
            count = 0
            for j in range(len(row)):
                if adj_matrix[i][j] == 0:
                    count += 1
                    zeros.append((i, j))
            if count == 0:
                row_done = False
                break

        col_done = True
        for i in range(num_nodes):
            col = adj_matrix[:, i]
            count = 0
            for j in range(len(col)):
                if adj_matrix[i][j] == 0:
                    count += 1
                    zeros.append((j, i))
            if count == 0:
                col_done = False
                break

        if row_done and col_done:
            done = True

    # print(adj_matrix)
    print(zeros)

    # otherwise, find the matchings so that only one selection per row & column
    # create dictionary for rows and columns
    rows = {}
    columns = {}

    # find where values overlap
    # put selested values in a list of tuples
    # loop through list and generate graph


    pass

def draw_graph(G):
    male, female = bipartite.sets(G)
    pos = dict()
    pos.update((n, (1, i)) for i, n in enumerate(male))
    pos.update((n, (2, i)) for i, n in enumerate(female))
    color_map = []
    for node in G:
        if node <= 10:
            color_map.append('pink')
        else:
            color_map.append('blue')
    nx.draw(G, node_color=color_map, with_labels=True, pos=pos)
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
    # draw_graph(test_graph)
    val_matches = get_true_matches(val_graph)
    hungarian_algorithm(test_graph)
