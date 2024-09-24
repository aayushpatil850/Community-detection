#algo1
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

def import_wiki_vote_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    edges = []
    for line in lines:
        if line.startswith('#'):
            continue
        edge = list(map(int, line.strip().split()))
        edges.append(edge)
    return np.array(edges)

def create_graph(edges):
    graph = defaultdict(list)
    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])  # Undirected graph, add both directions
    return graph

def bfs(graph, s):
    S = []
    P = {v: [] for v in graph.keys()}
    sigma = defaultdict(int)
    D = {}
    sigma[s] = 1
    D[s] = 0
    Q = deque([s])
    while Q:
        v = Q.popleft()
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in graph[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:
                sigma[w] += sigmav
                P[w].append(v)
    return S, P, sigma

def accumulate_edges(betweenness, S, P, sigma):
    delta = defaultdict(float)
    while S:
        w = S.pop()
        for v in P[w]:
            c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
            edge = tuple(sorted((int(v), int(w))))
            betweenness[edge] += c
            delta[v] += c
    return betweenness

def process_node_chunk(graph, nodes_chunk):
    local_betweenness = defaultdict(float)
    for node in nodes_chunk:
        S, P, sigma = bfs(graph, node)
        local_betweenness = accumulate_edges(local_betweenness, S, P, sigma)
    for edge in local_betweenness:
        local_betweenness[edge] /= 2.0  # Each edge is counted twice
    return dict(local_betweenness)

def compute_edge_betweenness_parallel(graph, nodes_chunks, num_workers=24):
    overall_betweenness = defaultdict(float)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_node_chunk, graph, chunk) for chunk in nodes_chunks]
        for future in as_completed(futures):
            chunk_betweenness = future.result()
            for edge, value in chunk_betweenness.items():
                overall_betweenness[edge] += value
    return overall_betweenness

def remove_highest_betweenness_edge(graph, betweenness):
    max_betweenness = max(betweenness.values())
    for edge, value in betweenness.items():
        if value == max_betweenness:
            graph[edge[0]].remove(edge[1])
            graph[edge[1]].remove(edge[0])
            return edge

def connected_components(graph):
    visited = set()
    components = []

    def dfs(v, component):
        stack = [v]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.append(node)
                stack.extend(set(graph[node]) - visited)

    for node in graph.keys():
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
    return components

def calculate_modularity(graph, components):
    m = sum(len(neighbors) for neighbors in graph.values()) / 2.0  # Total number of edges
    Q = 0.0
    for community in components:
        L_c = sum(len(graph[node]) for node in community) / 2.0
        D_c = sum(len(graph[node]) for node in community)
        Q += (L_c / m) - (D_c / (2 * m)) ** 2
    return Q




def girvan_newman_modularity_parallel(graph, chunk_size=100, num_workers=24):
    best_modularity = -1.0
    best_community_structure = None
    modularity_progression = []
    removed_edges = []
    splits = []

    with open('removed_edges_log.txt', 'w') as log_file:
        nodes = list(graph.keys())
        num_chunks = (len(nodes) + chunk_size - 1) // chunk_size
        nodes_chunks = [nodes[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
        
        iteration = 0
        no_change_counter = 0
        prev_num_communities = 0
        
        while len(graph) > 1 :
            betweenness = compute_edge_betweenness_parallel(graph, nodes_chunks, num_workers)
            edge_to_remove = remove_highest_betweenness_edge(graph, betweenness)
            if edge_to_remove:
                removed_edges.append(edge_to_remove)
                print(f"Removing edge: {edge_to_remove}")
                log_file.write(f"Removed edge: {edge_to_remove}\n")
            components = connected_components(graph)
            modularity = calculate_modularity(graph, components)
            modularity_progression.append(modularity)
            splits.append({node: i for i, community in enumerate(components) for node in community})

            num_communities = len(components)
            print(f"Iteration: {iteration + 1}, Number of communities = {num_communities}, Modularity = {modularity}")
            if num_communities == prev_num_communities:
                no_change_counter += 1
            else:
                no_change_counter = 0  # Reset counter if communities change

            if no_change_counter >= 70:
                print(f"Stopping as the number of communities hasn't changed in {no_change_counter} iterations.")
                break

            prev_num_communities = num_communities
            iteration += 1

            yield components, graph, modularity_progression, best_community_structure, splits

file_path = "wikivote.txt"
edges = import_wiki_vote_data(file_path)
graph = create_graph(edges)

for idx, (components, graph, modularity_progression, best_community_structure, splits) in enumerate(girvan_newman_modularity_parallel(graph)):
    print(f"Iteration {idx + 1}: Number of communities = {len(components)}")
    if len(components) == len(graph):
        break

#alg02 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

def import_last_fm_asia_data(file_path):
    df = pd.read_csv(file_path)
    
    # Check and clean NaN values
    if df.isna().sum().sum() > 0:
        df = df.dropna()
    
    # Convert to integers
    df['node_1'] = df['node_1'].astype(int)
    df['node_2'] = df['node_2'].astype(int)
    
    edges = df.to_numpy()
    return edges

def create_graph(edges):
    graph = defaultdict(set)
    for edge in edges:
        node1, node2 = edge
        graph[node1].add(node2)
        graph[node2].add(node1)
    return graph

def bfs(graph, s):
    S = []
    P = {v: [] for v in graph.keys()}
    sigma = defaultdict(int)
    D = {}
    sigma[s] = 1
    D[s] = 0
    Q = deque([s])
    while Q:
        v = Q.popleft()
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in graph[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:
                sigma[w] += sigmav
                P[w].append(v)
    return S, P, sigma

def accumulate_edges(betweenness, S, P, sigma):
    delta = defaultdict(float)
    while S:
        w = S.pop()
        for v in P[w]:
            c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
            edge = tuple(sorted((int(v), int(w))))
            betweenness[edge] += c
            delta[v] += c
    return betweenness

def process_node_chunk(graph, nodes_chunk):
    local_betweenness = defaultdict(float)
    for node in nodes_chunk:
        S, P, sigma = bfs(graph, node)
        local_betweenness = accumulate_edges(local_betweenness, S, P, sigma)
    for edge in local_betweenness:
        local_betweenness[edge] /= 2.0  # Each edge is counted twice
    return dict(local_betweenness)

def compute_edge_betweenness_parallel(graph, nodes_chunks, num_workers=24):
    overall_betweenness = defaultdict(float)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_node_chunk, graph, chunk) for chunk in nodes_chunks]
        for future in as_completed(futures):
            chunk_betweenness = future.result()
            for edge, value in chunk_betweenness.items():
                overall_betweenness[edge] += value
    return overall_betweenness

def remove_highest_betweenness_edge(graph, betweenness):
    max_betweenness = max(betweenness.values())
    for edge, value in betweenness.items():
        if value == max_betweenness:
            graph[edge[0]].remove(edge[1])
            graph[edge[1]].remove(edge[0])
            return edge

def connected_components(graph):
    visited = set()
    components = []

    def dfs(v, component):
        stack = [v]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.append(node)
                stack.extend(set(graph[node]) - visited)

    for node in graph.keys():
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
    return components

def calculate_modularity(graph, components):
    m = sum(len(neighbors) for neighbors in graph.values()) / 2.0  # Total number of edges
    Q = 0.0
    for community in components:
        L_c = sum(len(graph[node]) for node in community) / 2.0
        D_c = sum(len(graph[node]) for node in community)
        Q += (L_c / m) - (D_c / (2 * m)) ** 2
    return Q

def girvan_newman_modularity_parallel(graph, chunk_size=100, num_workers=24):
    best_modularity = -1.0
    previous_modularity = None
    best_community_structure = None
    modularity_progression = []
    removed_edges = []
 
    with open('removed_edges_log1.txt', 'w') as log_file:  # Log removed edges
        nodes = list(graph.keys())
        num_chunks = (len(nodes) + chunk_size - 1) // chunk_size
        nodes_chunks = [nodes[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
 
        iteration = 0
        while len(graph) > 1:  # Limit to 50 iterations
            betweenness = compute_edge_betweenness_parallel(graph, nodes_chunks, num_workers)
            edge_to_remove = remove_highest_betweenness_edge(graph, betweenness)
            if edge_to_remove:
                removed_edges.append(edge_to_remove)
                components = connected_components(graph)
                modularity = calculate_modularity(graph, components)
                print(f"Removing edge: {edge_to_remove}, Modularity: {modularity}")
                log_file.write(f"Removed edge: {edge_to_remove}, Modularity: {modularity}\n")  # Log the removed edge and modularity
            else:
                break  # Break if no edge to remove
            iteration += 1  # Increment iteration count
 
            yield components, graph, modularity_progression, best_community_structure

# Load the dataset and create edges
file_path = "/home/aayushjeevan/Downloads/lastfm_asia_edges.csv"  # Update the path to your dataset
edges = import_last_fm_asia_data(file_path)
graph = create_graph(edges)

# Run the Girvan-Newman algorithm
dendrogram_data = []

for idx, (components, graph, modularity_progression, best_community_structure) in enumerate(girvan_newman_modularity_parallel(graph)):
    print(f"Iteration {idx + 1}: Number of communities = {len(components)}")
    #dendrogram_data.append((len(components), modularity_progression[-1]))

    if len(components) == len(graph):  # Each node is its own community
        break
#algo3
import pandas as pd
from collections import defaultdict
import numpy as np

# Load the dataset
file_path = '/home/aayushjeevan/Downloads/lastfm_asia_edges.csv'
edges_df = pd.read_csv(file_path)

# Construct the graph as an adjacency list
graph = defaultdict(list)
for index, row in edges_df.iterrows():
    node1, node2 = row['node_1'], row['node_2']
    graph[node1].append(node2)
    graph[node2].append(node1)

def initialize_communities(graph):
    """
    Initialize each node to its own community.
    """
    return {node: node for node in graph}

def calculate_modularity_optimized(graph, communities, m):
    """
    Optimized calculation of modularity based on current community assignments.
    """
    modularity = 0.0
    community_intra_edges = defaultdict(int)
    community_total_degree = defaultdict(int)

    for node in graph:
        community = communities[node]
        community_total_degree[community] += len(graph[node])
        for neighbor in graph[node]:
            if communities[neighbor] == community:
                community_intra_edges[community] += 1

    for community in community_intra_edges:
        l_c = community_intra_edges[community] / 2  # Each edge counted twice
        d_c = community_total_degree[community]
        modularity += (l_c / m) - (d_c / (2 * m)) ** 2

    return modularity

def louvain_phase_one_single_iteration(graph, communities):
    """
    Phase 1: Modularity optimization by moving nodes between communities (optimized).
    """
    m = sum(len(neighbors) for neighbors in graph.values()) / 2
    best_communities = communities.copy()
    best_modularity = calculate_modularity_optimized(graph, communities, m)
    
    for node in graph:
        current_community = communities[node]
        best_community = current_community
        max_modularity_gain = 0

        # Temporarily remove the node from its community
        communities[node] = None
        
        # Calculate modularity gain for neighboring communities
        for neighbor in graph[node]:
            new_community = communities[neighbor]
            if new_community != current_community:
                communities[node] = new_community
                modularity_gain = calculate_modularity_optimized(graph, communities, m) - best_modularity
                if modularity_gain > max_modularity_gain:
                    max_modularity_gain = modularity_gain
                    best_community = new_community
        
        # Reassign node to the best community found
        communities[node] = best_community if best_community is not None else current_community
        
        if max_modularity_gain > 0:
            best_modularity += max_modularity_gain
            best_communities = communities.copy()

    return best_communities, best_modularity

def louvain_phase_two_aggregation(graph, communities):
    """
    Phase 2: Aggregation of communities into a new graph.
    """
    new_graph = defaultdict(list)
    new_communities = {}
    community_map = {}

    # Map each community to a unique node
    for node, community in communities.items():
        if community not in community_map:
            community_map[community] = len(community_map)
        new_node = community_map[community]
        new_communities[node] = new_node

    # Create the new aggregated graph
    for node in graph:
        new_node = new_communities[node]
        for neighbor in graph[node]:
            new_neighbor = new_communities[neighbor]
            if new_neighbor != new_node:
                new_graph[new_node].append(new_neighbor)

    return new_graph, community_map

# Initialize communities
communities = initialize_communities(graph)

# Run Phase 1 of the Louvain algorithm (only one iteration)
communities, modularity = louvain_phase_one_single_iteration(graph, communities)

# Calculate the number of communities formed after one iteration of Phase 1
num_communities = len(set(communities.values()))

# Output the results
print("Number of communities formed:", num_communities)
print("Modularity:", modularity)

# Aggregate the graph based on the communities formed
new_graph, community_map = louvain_phase_two_aggregation(graph, communities)
#alg0 4
import pandas as pd
from collections import defaultdict
import numpy as np

# Load the dataset
file_path = 'wikivote.txt'
edges_df = pd.read_csv(file_path, delimiter='\t', header=None, names=['node_1', 'node_2'])

# Construct the graph as an adjacency list
graph = defaultdict(list)
for index, row in edges_df.iterrows():
    node1, node2 = row['node_1'], row['node_2']
    graph[node1].append(node2)
    graph[node2].append(node1)

def initialize_communities(graph):
    """
    Initialize each node to its own community.
    """
    return {node: node for node in graph}

def calculate_modularity_optimized(graph, communities, m):
    """
    Optimized calculation of modularity based on current community assignments.
    """
    modularity = 0.0
    community_intra_edges = defaultdict(int)
    community_total_degree = defaultdict(int)

    for node in graph:
        community = communities[node]
        community_total_degree[community] += len(graph[node])
        for neighbor in graph[node]:
            if communities[neighbor] == community:
                community_intra_edges[community] += 1

    for community in community_intra_edges:
        l_c = community_intra_edges[community] / 2  # Each edge counted twice
        d_c = community_total_degree[community]
        modularity += (l_c / m) - (d_c / (2 * m)) ** 2

    return modularity

def louvain_phase_one_single_iteration(graph, communities):
    """
    Phase 1: Modularity optimization by moving nodes between communities (optimized).
    """
    m = sum(len(neighbors) for neighbors in graph.values()) / 2
    best_communities = communities.copy()
    best_modularity = calculate_modularity_optimized(graph, communities, m)
    
    for node in graph:
        current_community = communities[node]
        best_community = current_community
        max_modularity_gain = 0

        # Temporarily remove the node from its community
        communities[node] = None
        
        # Calculate modularity gain for neighboring communities
        for neighbor in graph[node]:
            new_community = communities[neighbor]
            if new_community != current_community:
                communities[node] = new_community
                modularity_gain = calculate_modularity_optimized(graph, communities, m) - best_modularity
                if modularity_gain > max_modularity_gain:
                    max_modularity_gain = modularity_gain
                    best_community = new_community
        
        # Reassign node to the best community found
        communities[node] = best_community if best_community is not None else current_community
        
        if max_modularity_gain > 0:
            best_modularity += max_modularity_gain
            best_communities = communities.copy()

    return best_communities, best_modularity

def louvain_phase_two_aggregation(graph, communities):
    """
    Phase 2: Aggregation of communities into a new graph.
    """
    new_graph = defaultdict(list)
    new_communities = {}
    community_map = {}

    # Map each community to a unique node
    for node, community in communities.items():
        if community not in community_map:
            community_map[community] = len(community_map)
        new_node = community_map[community]
        new_communities[node] = new_node

    # Create the new aggregated graph
    for node in graph:
        new_node = new_communities[node]
        for neighbor in graph[node]:
            new_neighbor = new_communities[neighbor]
            if new_neighbor != new_node:
                new_graph[new_node].append(new_neighbor)

    return new_graph, community_map

# Initialize communities
communities = initialize_communities(graph)

# Run Phase 1 of the Louvain algorithm (only one iteration)
communities, modularity = louvain_phase_one_single_iteration(graph, communities)

# Calculate the number of communities formed after one iteration of Phase 1
num_communities = len(set(communities.values()))

# Output the results
print("Number of communities formed:", num_communities)
print("Modularity:", modularity)

# Aggregate the graph based on the communities formed
new_graph, community_map = louvain_phase_two_aggregation(graph, communities)
    

