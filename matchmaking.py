import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import math
import numpy as np


def create_adj_matrix(G):
    # create adjacency matrix using numpy
    num_nodes = len(G.nodes())
    adj_matrix = 10*np.ones([num_nodes, num_nodes])
    for edge in list(G.edges()):
        node_1 = int(edge[0]-1)
        node_2 = int(edge[1]-1)
        edge_weight = G.get_edge_data(*edge)['weight']
        adj_matrix[node_1][node_2] = edge_weight
    return adj_matrix

def step_one():
    print("step one")
    # find the minimum of each row & subtract that min from each row
    num_nodes = len(test_graph.nodes())
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

    step_two()

def step_two():
    print("step two")
    dims = adj_matrix.shape
    global mask_matrix
    global row_cover
    global col_cover

    mask_matrix = np.zeros(dims)
    row_cover = np.zeros((1, dims[1]))
    col_cover = np.zeros((dims[0], 1))

    for i in range(0, dims[0]):
        for j in range(0, dims[1]):
            if (adj_matrix[i][j] == 0 and row_cover[0][i] == 0 and col_cover[j][0] == 0):
                mask_matrix[i][j] = 1;
                row_cover[i] = 1;
                col_cover[i] = 1;

    row_cover = np.zeros((1, dims[1]))
    col_cover = np.zeros((dims[0], 1))

    step_three()

def step_three():
    print("step three")
    dims = mask_matrix.shape
    col_count = 0
    for c in range(0, dims[1]):
        if (col_cover[c][0] == 1):
            col_count += 1

    print('col_count before covering:', col_count)

    for i in range(0, dims[0]):
        for j in range(0, dims[1]):
            if (mask_matrix[i][j] == 1):
                col_cover[j] = 1;

    col_count = 0
    for c in range(0, dims[1]):
        if (col_cover[c][0] == 1):
            col_count += 1

    print('col_count:', col_count)
    if (col_count >= dims[0] or col_count >= dims[1]):
        step_seven()
    else:
        step_four()
        return

def find_a_zero(row, col):
    print("find a zero")
    dims = adj_matrix.shape
    for r in range(0, dims[0]):
        for c in range(0, dims[1]):
            if (adj_matrix[r][c] == 0 and row_cover[0][r] == 0 and col_cover[c][0] == 0):
                row = r
                col = c
                break
    return row, col

def star_in_row(row):
    print('star in row')
    dims = mask_matrix.shape
    for c in range(0, dims[1]):
        if mask_matrix[row][c] == 1:
            return c
    return -1

def step_four():
    print("step four")
    # go through the mask
    # find an uncovered zero
    # if you do not find a zero, then you're done with this step -> step 6
    # otherwise, prime the zero that you found
    # with that zero, check if there is a starred zero in the same row
    # if there is a starred zero, cover the row and uncover the column of the starred zero
    # otherwise, move onto step 5
    row = -1
    col = -1
    four_done = False
    while (not four_done):
        print(row, col)
        row, col = find_a_zero(row, col)
        if (row == -1):
            four_done = True
            step_six()
            return
        else:
            mask_matrix[row][col] = 2
            star_col = star_in_row(row)
            print("star_col:", star_col)
            if (star_col >= 0):
                row_cover[0][row] = 1
                col_cover[star_col][0] = 0
                col = star_col
            else:
                four_done = True
                step_five(row, col)
                return

def find_star_in_col(c):
    print("find star in col")
    dims = mask_matrix.shape
    for r in range(0, dims[0]):
        if mask_matrix[r][c] == 1:
            return r
    return -1

def find_prime_in_row(row):
    print("find prime in row")
    dims = mask_matrix.shape
    for c in range(0, dims[1]):
        if mask_matrix[row][c] == 2:
            return c
    return -1

def augment_path(path):
    print("augment path")
    for point in path:
        row = point[0]
        col = point[1]
        if (mask_matrix[row][col] == 1):
            mask_matrix[row][col] = 0
        else:
            mask_matrix[row][col] = 1

def erase_primes():
    print("erase primes")
    dims = mask_matrix.shape
    for r in range(0, dims[0]):
        for c in range(0, dims[1]):
            if (mask_matrix[r][c] == 2):
                mask_matrix[r][c] = 0

def step_five(row, col):
    print("step five")
    dims = mask_matrix.shape
    five_done = False
    r = -1
    c = -1

    path_count = 1
    path = [(row, col)]

    while (not five_done):
        star_row = find_star_in_col(path[path_count-1][1])
        if (star_row > -1):
            path_count += 1
            path.append((star_row, path[path_count-2][1]))
        else:
            break

        prime_col = find_prime_in_row(path[path_count-1][0])
        path_count += 1
        path.append((path[path_count-2][0], prime_col))

    augment_path(path)
    row_cover = np.zeros((1, dims[1]))
    col_cover = np.zeros((dims[0], 1))
    erase_primes()
    step_three()

def find_minimum():
    dims = adj_matrix.shape
    min = math.inf
    for r in range(0, dims[0]):
        for c in range(0, dims[1]):
            if (row_cover[0][r] == 0 and col_cover[c][0] == 0):
                if (adj_matrix[r][c] < min):
                    min = adj_matrix[r][c]

    return min

def step_six():
    print("step six")
    min = find_minimum()
    print("min:")
    dims = mask_matrix.shape
    for r in range(0, dims[0]):
        for c in range(0, dims[1]):
            if (row_cover[0][r] == 1):
                adj_matrix[r][c] += min
            else:
                adj_matrix[r][c] -= min
    step_four()

def step_seven():
    global h_matches
    h_matches = []
    dims = mask_matrix.shape
    for r in range(0, dims[0]):
        for c in range(r, dims[1]):
            if mask_matrix[r][c] == 1:
                h_matches.append((r+1, c+1))

def hungarian_algorithm():
    step_one()

def create_val_input_graph():
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

    return G

def create_matches_graph(matches):
    G = nx.DiGraph()
    G.add_nodes_from(range(1, 11), bipartite=0)
    G.add_nodes_from(range(11, 21), bipartite=1)

    for match in matches:
        G.add_edge(match[0], match[1])
    return G

def draw_graph(G):
    nx.draw(G, with_labels=True)
    plt.show()

def draw_bipartite_graph(G):
    male, female = bipartite.sets(G)
    print(male, female)
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
    print(g.adj)
    for node in nodes:
        for val in g.adj[node].items():
            items = list(g.adj[val[0]].items())
            if float(node) in [x[0] for x in items]:
                matches.append((node,val[0]))
    matches2 = set(tuple(sorted(m)) for m in matches)
    return matches2

if __name__ == '__main__':
    input_graph = create_val_input_graph()
    draw_bipartite_graph(input_graph)
    val_matches = get_true_matches(input_graph)

    global test_graph
    test_graph = create_test_graph()
    global adj_matrix
    adj_matrix = create_adj_matrix(test_graph)
    hungarian_algorithm()

    hungarian_graph = create_matches_graph(h_matches)
    validation = create_matches_graph(val_matches)

    print(val_matches)
    print(h_matches)

    draw_graph(hungarian_graph)
    draw_graph(validation)
