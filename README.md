# Community Detection Algorithms on LastFM Asia and WikiVote Datasets

## Overview

This repository implements two popular community detection algorithms: the **Louvain Algorithm** and the **Girvan-Newman Algorithm**. The algorithms are applied to two datasets: **LastFM Asia** and **WikiVote**. The Louvain algorithm is optimized for modularity-based community detection, while the Girvan-Newman algorithm is designed to detect communities by progressively removing edges with the highest betweenness.



## Datasets

1. **LastFM Asia Dataset** (`lastfm_asia_edges.csv`):
   - Contains relationships (edges) between users (nodes) of LastFM from Asia.
   - The dataset is stored as a CSV file with two columns: `node_1` and `node_2`, which represent an undirected edge between two users.

2. **WikiVote Dataset** (`wikivote.txt`):
   - Contains voting data for Wikipedia administrators.
   - Each row contains two columns: `node_1` and `node_2`, representing a directed edge where `node_1` voted for `node_2`.

## Louvain Algorithm

### Overview

The **Louvain Algorithm** is used to detect communities by optimizing the modularity of a network. It works in two phases:
1. **Phase 1**: Optimize modularity by moving nodes between communities.
2. **Phase 2**: Aggregate nodes into new communities and repeat.

The implementation in this repository focuses on an optimized calculation of modularity during the community detection process.

### Code Explanation

- **Phase 1**: The algorithm iterates over each node, temporarily removing it from its community, and evaluates the modularity gain of assigning the node to a neighboring community. The best community assignment that maximizes modularity is kept.
  
- **Phase 2**: Communities formed in Phase 1 are aggregated into a new graph, where each community becomes a new "supernode." The algorithm is then applied to this aggregated graph.

### Usage

1. **LastFM Asia Dataset**:
   - Load the dataset using `lastfm_asia_edges.csv`.
   - Run one iteration of Phase 1 to optimize modularity.

2. **WikiVote Dataset**:
   - Load the dataset using `wikivote.txt`.
   - Similarly, run one iteration of Phase 1 for the WikiVote dataset.

## Girvan-Newman Algorithm

### Overview

The **Girvan-Newman Algorithm** detects communities by removing edges with the highest betweenness centrality. This is done iteratively until the network is divided into separate components (communities).

### Code Explanation

- **Edge Betweenness Calculation**: The edge with the highest betweenness is removed at each step. Betweenness centrality of an edge is computed by counting the number of shortest paths that pass through it.
  
- **Community Formation**: After removing each edge, the number of connected components is evaluated. If the number of communities changes, the edge betweenness is recalculated.

### Usage

1. **LastFM Asia Dataset**:
   - Load the dataset and apply the Girvan-Newman algorithm.
   - Print the number of communities after each step, the modularity, and the edges removed.

2. **WikiVote Dataset**:
   - Load the dataset and apply the Girvan-Newman algorithm similarly.
   - Keep track of the edges removed and their betweenness at each step.
 ## Output

### Louvain Algorithm and Girvan Newman Algorithm

For both the LastFM Asia and WikiVote datasets, the output will include:

- **Number of communities**: After running one iteration of the Louvain algorithm, the number of detected communities will be printed.
  
- **Modularity**: After each iteration, the modularity value will be computed and printed to indicate the quality of the community structure.




