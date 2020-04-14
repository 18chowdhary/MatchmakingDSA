import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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
        # print(row['iid'], row['pid'], row['dec'])

    # nx.draw(G, with_labels=True)
    # plt.show()

    return G
def is_match(g):
    nodes = list(g.nodes)
    matches = []
    # graph_dict = g.adj.items()
    for node in nodes:
        for val in g.adj[node].items():
            # print(node, val, g.adj[val[0]].items)
            items = list(g.adj[val[0]].items())
            # print(node, val, items)
            if float(node) in [x[0] for x in items]:
                matches.append((node,val[0]))
                
    print(matches)
if __name__ == '__main__':
    val_graph = create_validation_graph()
    is_match(val_graph)
