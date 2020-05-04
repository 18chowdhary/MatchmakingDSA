import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import math
import numpy as np


def create_adj_matrix(G):
    '''
    Create adjacency matrix for the Hungarian graph.
    '''
    # create adjacency matrix using numpy
    num_nodes = len(G.nodes())
    adj_matrix = 10*np.ones([num_nodes, num_nodes])
    for edge in list(G.edges()):
        node_1 = int(edge[0]-1)
        node_2 = int(edge[1]-1)
        edge_weight = G.get_edge_data(*edge)['weight']
        adj_matrix[node_1][node_2] = edge_weight
    return adj_matrix

def initial_min():
    '''
    Step one of the matching algorithm.
    '''
    # find the minimum of each row & subtract that min from each row
    num_nodes = len(test_graph.nodes())
    for i in range(num_nodes):
        row = adj_matrix[i]
        min_weight = np.min(row)
        adj_matrix[i] = np.subtract(row, min_weight)

    # find the min of each col & subtract from each col
    for i in range(num_nodes):
        col = adj_matrix[:, i]
        min_weight = np.min(col)
        adj_matrix[:,i] = np.subtract(col, min_weight)

    cover_zeros()

def cover_zeros():
    '''
    Step two of the matching algorithm.
    '''
    dims = adj_matrix.shape
    global mask_matrix
    global row_cover
    global col_cover

    mask_matrix = np.zeros(dims)
    row_cover = np.zeros((1, dims[1]))
    col_cover = np.zeros((dims[0], 1))

    # Star zeros and cover rows and columns
    for i in range(0, dims[0]):
        for j in range(0, dims[1]):
            # If there's a zero, and the row and column are currently uncovered,
            # mark that there is a zero and cover the row and column
            if (adj_matrix[i][j] == 0 and row_cover[0][i] == 0 and col_cover[j][0] == 0):
                mask_matrix[i][j] = 1;
                row_cover[i] = 1;
                col_cover[i] = 1;

    # Reset covers
    row_cover = np.zeros((1, dims[1]))
    col_cover = np.zeros((dims[0], 1))

    check_starred_col()

def check_starred_col():
    '''
    Step three of the matching algorithm.
    '''
    dims = mask_matrix.shape

    # Cover columns with starred zeros
    for i in range(0, dims[0]):
        for j in range(0, dims[1]):
            if (mask_matrix[i][j] == 1):
                col_cover[j] = 1;

    # Check the number of covered columns
    col_count = 0
    for c in range(0, dims[1]):
        if (col_cover[c][0] == 1):
            col_count += 1

    # If every column has a starred zero, then done
    if (col_count >= dims[0] or col_count >= dims[1]):
        get_hungarian_matches()
    else:
        primes_next_zero()
        return

def find_uncovered_zero():
    '''
    Finds an uncovered zero and returns the position of that zero (row, col).
    '''
    row = -1
    col = -1
    dims = adj_matrix.shape
    for r in range(0, dims[0]):
        for c in range(0, dims[1]):
            if (adj_matrix[r][c] == 0 and row_cover[0][r] == 0 and col_cover[c][0] == 0):
                row = r
                col = c
                break
    return row, col

def star_in_row(row):
    '''
    Checks for a starred zero in the given row, and returns the
    column if there is one. Otherwise, returns -1.
    '''
    dims = mask_matrix.shape
    for c in range(0, dims[1]):
        if mask_matrix[row][c] == 1:
            return c
    return -1

def primes_next_zero():
    '''
    Step four of the matching algorithm.
    '''
    # Go through the mask matrix and find an uncovered zero
    row = -1
    col = -1
    four_done = False
    while (not four_done):
        row, col = find_uncovered_zero()
        # If you cannot find an uncovered zero, then move onto step 6
        if (row == -1):
            four_done = True
            adjust_by_min()
            return
        # found a zero
        else:
            # Prime the zero
            mask_matrix[row][col] = 2
            # Check if there is a starred zero in the same row
            star_col = star_in_row(row)
            # If there is a starred zero in the same row, cover the row
            # and uncover the column
            if (star_col >= 0):
                row_cover[0][row] = 1
                col_cover[star_col][0] = 0
                col = star_col
            # Otherwise, move onto step 5
            else:
                four_done = True
                find_path(row, col)
                return

def find_star_in_col(c):
    '''
    For a given column, checks if there is a starred zero in that column. If there
    is, return the row. Otherwise, return -1.
    '''
    dims = mask_matrix.shape
    for r in range(0, dims[0]):
        if mask_matrix[r][c] == 1:
            return r
    return -1

def find_prime_in_row(row):
    '''
    For a given row, checks if there is a primed zero in that row. If there is,
    return the column. Otherwise, return -1.
    '''
    dims = mask_matrix.shape
    for c in range(0, dims[1]):
        if mask_matrix[row][c] == 2:
            return c
    return -1

def augment_path(path):
    '''
    For every zero in the path, if the zero is starred, unstar it. Otherwise,
    if the zero is primed, star it.
    '''
    for point in path:
        row = point[0]
        col = point[1]
        # Unstar zeros that are starred
        if (mask_matrix[row][col] == 1):
            mask_matrix[row][col] = 0
        # Star zeros that are primed
        else:
            mask_matrix[row][col] = 1

def erase_primes():
    '''
    Unprime all primes, just as a check.
    '''
    dims = mask_matrix.shape
    for r in range(0, dims[0]):
        for c in range(0, dims[1]):
            if (mask_matrix[r][c] == 2):
                mask_matrix[r][c] = 0

def find_path(row, col):
    '''
    Step five of the matching algorithm.
    '''
    dims = mask_matrix.shape
    five_done = False
    r = -1
    c = -1

    # Start your path of possible matches with the given zero
    path_count = 1
    path = [(row, col)]

    # Construct a path of alternating primed and starred zeros
    while (not five_done):
        # Check if there is a starred zero in the same column as
        # last zero added to path
        star_row = find_star_in_col(path[path_count-1][1])
        # If there is, add zero to the path
        if (star_row > -1):
            path_count += 1
            path.append((star_row, path[path_count-2][1]))
        else:
            break

        # Find primed zero in the same row as the last
        # zero added to the path--there will have to be one
        prime_col = find_prime_in_row(path[path_count-1][0])
        # Add primed zero to the path
        path_count += 1
        path.append((path[path_count-2][0], prime_col))

    # Clear out stars and update stars with the primes, to find other possible
    # matches
    augment_path(path)

    # Reset covers
    row_cover = np.zeros((1, dims[1]))
    col_cover = np.zeros((dims[0], 1))
    # Erase primes
    erase_primes()

    # Check for more zeros
    check_starred_col()

def find_minimum():
    '''
    Find the minimum cost value across all the uncovered rows and columns.
    '''
    dims = adj_matrix.shape
    min = math.inf
    for r in range(0, dims[0]):
        for c in range(0, dims[1]):
            if (row_cover[0][r] == 0 and col_cover[c][0] == 0):
                if (adj_matrix[r][c] < min):
                    min = adj_matrix[r][c]

    return min

def adjust_by_min():
    '''
    Step six of the matching algorithm.
    '''
    # Find the minimum value across all the uncovered rows and columns
    min = find_minimum()

    # Adjust the rows and columns by the minimum
    dims = mask_matrix.shape
    for r in range(0, dims[0]):
        for c in range(0, dims[1]):
            # Add the minimum to covered rows
            if (row_cover[0][r] == 1):
                adj_matrix[r][c] += min
            # Subtract the minimum from uncovered rows
            else:
                adj_matrix[r][c] -= min

    # Go back to finding more zeros in each column and
    # finding more potential matches
    primes_next_zero()

def get_hungarian_matches():
    '''
    Identify matches by looking at all starred zeros.
    '''
    global h_matches
    h_matches = []
    dims = mask_matrix.shape
    for r in range(0, dims[0]):
        for c in range(r, dims[1]):
            if mask_matrix[r][c] == 1:
                h_matches.append((r+1, c+1))

def hungarian_algorithm():
    '''
    Run the hungarian algorithm; start the first step, which
    leads to the other steps.
    '''
    # Create adjacency matrix
    global adj_matrix
    adj_matrix = create_adj_matrix(test_graph)

    # Start the algorithm
    initial_min()

def create_val_input_graph():
    '''
    Create input graph for the validation, where edges are based on
    the actual decisions people made.
    '''
    # Get data
    data = pd.read_csv("speeddating.csv", encoding="utf-8")
    validation_subset = data[['iid', 'pid', 'gender', 'dec']]
    validation_subset = validation_subset.head(200)

    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(range(1, 11), bipartite=0)
    G.add_nodes_from(range(11, 21), bipartite=1)

    # Add edges where people said yes
    for index, row in validation_subset.iterrows():
        if (row['dec'] == 1):
            G.add_edge(row['iid'], row['pid'])

    return G

def create_test_graph():
    '''
    Create input graph for the Hungarian algorithm, where edges are based on
    the probability of rejection
    '''
    # Get data
    data = pd.read_csv("speeddating.csv", encoding="utf-8")
    test_subset = data[['iid', 'pid', 'prob']]
    test_subset = test_subset.head(200)

    # Create nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(1, 11), bipartite=0)
    G.add_nodes_from(range(11, 21), bipartite=1)

    # Add edges with the likelihood of rejection as the weight
    # (or set 5 as default)
    for index, row in test_subset.iterrows():
        # If there is no prob, set default to 5
        if (math.isnan(row['prob'])):
            cost = 5
        # Likelihood of rejection is 10 - probability of accepting
        else:
            cost = 10-int(row['prob'])
        G.add_edge(row['iid'], row['pid'], weight=cost)

    return G

def create_matches_graph(matches):
    '''
    Graph given matches.
    '''
    G = nx.DiGraph()
    G.add_nodes_from(range(1, 11), bipartite=0)
    G.add_nodes_from(range(11, 21), bipartite=1)

    for match in matches:
        G.add_edge(match[0], match[1])
    return G

def draw_graph(G):
    '''
    Draw a given graph.
    '''
    nx.draw(G, with_labels=True)
    plt.show()

def draw_bipartite_graph(G):
    '''
    Draw a given bipartite graph.
    '''
    # Separate it into sets
    male, female = bipartite.sets(G)

    # Set the positions
    pos = dict()
    pos.update((n, (1, i)) for i, n in enumerate(male))
    pos.update((n, (2, i)) for i, n in enumerate(female))

    # Define colors
    color_map = []
    for node in G:
        if node <= 10:
            color_map.append('pink')
        else:
            color_map.append('blue')

    # Draw graph
    nx.draw(G, node_color=color_map, with_labels=True, pos=pos)
    plt.show()

def get_true_matches(g):
    '''
    Get true matches based on the validation input graph.
    '''

    # Iterate through the list of nodes and check if there are mutual
    # decisions to say yes
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
    input_graph = create_val_input_graph()
    draw_bipartite_graph(input_graph)
    val_matches = get_true_matches(input_graph)

    global test_graph
    test_graph = create_test_graph()
    draw_graph(test_graph)
    hungarian_algorithm()

    hungarian_graph = create_matches_graph(h_matches)
    validation = create_matches_graph(val_matches)

    print(val_matches)
    print(h_matches)

    draw_graph(hungarian_graph)
    draw_graph(validation)
